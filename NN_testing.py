from z3 import *
from z3_clf_check import *
import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
import matplotlib.pyplot as plt
import sympy


class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        self.layer2 = torch.nn.Linear(n_hidden,n_output)
        # dont need this since we arent using a LQR 
        # self.control = torch.nn.Linear(n_input,1,bias=False)
        # self.control.weight = torch.nn.Parameter(lqr)

    def forward(self,x):
        # change this to a sigmoid function so we can z3 can solve it (z3 cant do trig)
        # sigmoid = torch.nn.Tanh()
        sigmoid = torch.nn.Sigmoid()
        h_1 = sigmoid(self.layer1(x))
        out = sigmoid(self.layer2(h_1))
        # only need to return out now, u will come from the mpc
        # u = self.control(x)
        # return out,u
        return out


#Dont need this function since we will define the flows as a diction of the states in the main learning loop
# def f_value(x,u):
#     # dynamics
#     y = []
#     G = 9.81  # gravity
#     L = 0.5   # length of the pole 
#     m = 0.15  # ball mass
#     b = 0.1   # friction
    
#     for r in range(0,len(x)): 
#         f = [ x[r][1], 
#               (m*G*L*np.sin(x[r][0])- b*x[r][1]) / (m*L**2)]
#         y.append(f) 
#     y = torch.tensor(y)
#     y[:,1] = y[:,1] + (u[:,0]/(m*L**2))
#     return y

'''
For learning 
'''
N = 500             # sample size
#updated input dimension to be the same as the states of the car, might drop V
D_in = 5            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)  
x_start = torch.Tensor(N, D_in).uniform_(-5, 5)           
x_0 = torch.zeros([1, D_in])

'''
For verifying 
'''
state_list = x, y, v, phi, theta = sympy.symbols('x y v phi theta')
cont_list = u1, u2= sympy.symbols('u1 u2')
vars_ = state_list + cont_list
#car params:

L = 2.5 #wheel base

epsilon = 0
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.1
ball_ub = 5

#define the flows:
flows = {
    x: v*sympy.cos(phi), #this is going to be a problem since z3 cant solve cos(), I think we can just make a piecewise function
    y: v*sympy.sin(phi), #same comment as above,
    v: u2,
    phi: v*sympy.tanh(u1)/L,
    theta: 1
}
#u bounds

start = timeit.default_timer()
#remove lqr references
model = Net(D_in,H1, D_out)
L = []
i = 0 
t = 0
# max_iters = 2000
learning_rate = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# while i < max_iters and not valid: 
# print(x)
#x is initialized to a tensor of len(states) unifromly sampled from -5 to 5 
V_candidate = model(x_start)
print(len(V_candidate))

X0 = model(x_0)

#get model params
w1 = model.layer1.weight.data.numpy()
w2 = model.layer2.weight.data.numpy()
b1 = model.layer1.bias.data.numpy()
b2 = model.layer2.bias.data.numpy()

#calculate the canidate V
# Candidate V
z1 = np.dot(state_list,w1.T)+b1

a1 = []
#will need to replace exp with the actual function(1/(1+(2.71828182846**(-x))
for j in range(0,len(z1)):
    a1.append(1/(1+(2.71828182846**z1[j])))
z2 = np.dot(a1,w2.T)+b2
V_learn = 1/(1 + (2.71828182846**z2.item(0)))

print(V_learn)

#check V @ 0 :
print("\nresult of 0 check is: ", CheckLyapunov_zero(state_list,V_learn))
#this isnt working right, hard coded bounds for now
print("\nresult of PD test is: ", CheckLyapunov_PD(state_list,V_learn, 0.5, 0.5))
#now need to figure out an approximation of sin and cos
# if we just bound theta +/- pi/6 its pretty close as sin(x) = x and cos(x) = 1 -(x^2)/2


# f = f_value(x,u)
# still dont know what circle turning is
# Circle_Tuning = Tune(x)
# Compute lie derivative of V : L_V = ∑∂V/∂xᵢ*fᵢ
# L_V = torch.diagonal(torch.mm(torch.mm(torch.mm(dtanh(V_candidate),model.layer2.weight)\
#                     *dtanh(torch.tanh(torch.mm(x,model.layer1.weight.t())+model.layer1.bias)),model.layer1.weight),f.t()),0)

# # With tuning term 
# Lyapunov_risk = (F.relu(-V_candidate)+ 1.5*F.relu(L_V+0.5)).mean()\
#             +2.2*((Circle_Tuning-6*V_candidate).pow(2)).mean()+(X0).pow(2) 
# # Without tuning term
# #         Lyapunov_risk = (F.relu(-V_candidate)+ 1.5*F.relu(L_V+0.5)).mean()+ 1.2*(X0).pow(2)


# print(i, "Lyapunov Risk=",Lyapunov_risk.item()) 
# L.append(Lyapunov_risk.item())
# optimizer.zero_grad()
# Lyapunov_risk.backward()
# optimizer.step() 



# Falsification
# if i % 10 == 0:
    # u_NN = (q.item(0)*x1 + q.item(1)*x2) 
    # f = [ x2,
    #         (m*G*l*sin(x1) + u_NN - b*x2) /(m*l**2)]


#     print('===========Verifying==========')        
#     start_ = timeit.default_timer() 
#     result= CheckLyapunov(vars_, f, V_learn, ball_lb, ball_ub, config,epsilon)
#     stop_ = timeit.default_timer() 

#     if (result): 
#         print("Not a Lyapunov function. Found counterexample: ")
#         print(result)
#         x = AddCounterexamples(x,result,10)
#     else:  
#         valid = True
#         print("Satisfy conditions!!")
#         print(V_learn, " is a Lyapunov function.")
#     t += (stop_ - start_)
#     print('==============================') 
# i += 1

# stop = timeit.default_timer()


# np.savetxt("w1.txt", model.layer1.weight.data, fmt="%s")
# np.savetxt("w2.txt", model.layer2.weight.data, fmt="%s")
# np.savetxt("b1.txt", model.layer1.bias.data, fmt="%s")
# np.savetxt("b2.txt", model.layer2.bias.data, fmt="%s")
# np.savetxt("q.txt", model.control.weight.data, fmt="%s")

# print('\n')
# print("Total time: ", stop - start)
# print("Verified time: ", t)

# out_iters+=1



# epsilon = -0.00001
# start_ = timeit.default_timer() 
# result = CheckLyapunov(vars_, f, V_learn, ball_lb, ball_ub, config, epsilon)
# stop_ = timeit.default_timer() 

# if (result): 
# print("Not a Lyapunov function. Found counterexample: ")
# else:  
# print("Satisfy conditions with epsilon= ",epsilon)
# print(V_learn, " is a Lyapunov function.")
# t += (stop_ - start_)


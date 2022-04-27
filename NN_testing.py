from copy import deepcopy


from z3 import *
from z3_clf_check import *
import torch 
import torch.nn.functional as F
import numpy as np
import timeit 
import matplotlib.pyplot as plt
import sympy
import Time_based_MPC


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
        sigmoid = torch.nn.Tanh()
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
x_start = torch.Tensor(N, D_in).uniform_(-3.14, 3.14)           
# x_0 = torch.zeros([1, D_in])

'''
For verifying 
'''
state_list = x, y, v, phi, theta = sympy.symbols('x y v phi theta')
cont_list = u1, u2= sympy.symbols('u1 u2')
vars_ = state_list + cont_list
#define control bounds
u_bounds = [-3.14/4, 3.14/4, 0.99,1.01]

#car params:
L = 2.5 #wheel base

epsilon = 0
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.1
ball_ub = 5

#define the flows:
#might not need this if we are just pluggin in the values
flows = {
    x: v*sympy.cos(phi), #this is going to be a problem since z3 cant solve cos(), I think we can just make a piecewise function
                            #what if I just approximate this with the piecewise function y=x [x<= pi/2] and y = -x +pi [pi/2<x<pi]
    y: v*sympy.sin(phi), #same comment as above,
    v: u2,
    phi: v*sympy.tan(u1)/L,
    theta: 1
}
#remove lqr references
model = Net(D_in,H1, D_out)
L = []
i = 0 
t = 0
# max_iters = 2000
learning_rate = 0.1#0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# while i < max_iters and not valid: 
# print(x)
#x is initialized to a tensor of len(states) unifromly sampled from -5 to 5 
V_candidate = model(x_start)
print(len(V_candidate))
print(x_start)
# X0 = model(x_0)

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
    # a1.append(1/(1+(2.71828182846**(-z1[j]))))
    a1.append(((2.71828182846**z1[j])-(2.71828182846**-z1[j]))/((2.71828182846**z1[j])+(2.71828182846**-z1[j])))
z2 = np.dot(a1,w2.T)+b2
# V_learn = 1/(1 + (2.71828182846**(-z2.item(0))))
V_learn = ((2.71828182846**z2.item(0))-(2.71828182846**-z2.item(0)))/((2.71828182846**z2.item(0))+(2.71828182846**-z2.item(0)))

print(V_learn)
v_0 = CheckLyapunov_zero(state_list,V_learn)
#check V @ 0 :
print("\nresult of 0 check is: ", v_0)
#this isnt working right, hard coded bounds for now
print("\nresult of PD test is: ", CheckLyapunov_PD(state_list,V_learn, 0.1, 3.14, v_0))


#lie derivative testing
for i in range(100):
    #get sum number of random x samples:
    #rightnow this is generating 500 samples of this with is not necessay
    x_0 = torch.Tensor(1, D_in).uniform_(-3.14, 3.14) 
    print("x_0 :", x_0)
    #get controls
    u_0 = Time_based_MPC.get_first_control(x_0[0])

    print("u_0 = ", u_0)
    #need to set u_0 of 1 to 1 since V always == 1
    u_0[1] = 1.0
    #check if L_V is negative
    L_V = 0
    for state in state_list:
        L_V += sympy.diff(V_learn,state) * flows[state]

    for idx,state in enumerate(state_list):
        L_V = L_V.subs(state,x_0[0][idx])
    for idx, cont in enumerate(cont_list):
        L_V = L_V.subs(cont, u_0[idx])
    
    if L_V > 0:
        print("x_O: ", x_0)
        print("u_0: ", u_0)
        print("Lie V: ", L_V)
        break

print(L_V)

#need to sample about CE here
#we will do 10 CE for now
#init some size of samples we will just do 0.5 for now
num_samps = 10
samp_range = 0.5

x_samp = np.random.uniform(np.array(x_0[0]) - samp_range, np.array(x_0[0]) + samp_range, [num_samps, len(np.array(x_0[0]))])

# print(np.array(x_0[0]))
temp = np.array(x_0[0])
x_samp = np.append(x_samp, np.array([temp]), axis = 0)
x_samp = x_samp.astype(float)
#get controls for each x_samp:
#hard coding this to 
u_samp = np.empty((0,len(cont_list)), np.float64)
for x in x_samp:
    # print(np.array(Time_based_MPC.get_first_control(x)))
    u_samp = np.append(u_samp, np.array([Time_based_MPC.get_first_control(x)]), axis = 0)
print(x_samp)
test_x_samp = deepcopy(x_samp)
print(u_samp)
#and then conver values into tensors
#compute Lyapunov risk
#pos risk
pos_vals = []

x_samp = torch.tensor([x_samp], dtype = torch.float)
#pass the sampled x through the network
# model_out = model(x_samp)


#i should be able to just pass the x_samps through the network
for sample_states in test_x_samp:
    pos_check = deepcopy(V_learn)
    for idx,state in enumerate(state_list):
        pos_check = pos_check.subs(state,sample_states[idx])
    pos_vals.append(pos_check)

#issue, these should be the same but their not, so we are calculating our network equation wrong
model_out = model(x_samp)
print("network values: ", model_out)
print("calulated values: ", pos_vals)

# train!
for j in range(1000):
    ###############################
    #caluclate Lyapunov Risk
    ################################
    model_out = model(x_samp)
    pos_risk = F.relu(-model_out)
    #cacluate zero risk
    zero_tesn = torch.zeros([1, D_in])
    zero_risk = model(zero_tesn).pow(2)
    # print("zero risk is: ", zero_risk)

    #calculate L_V risk

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
    for k in range(0,len(z1)):
        # a1.append(1/(1+(2.71828182846**(-z1[j]))))
        a1.append(((2.71828182846**z1[k])-(2.71828182846**-z1[k]))/((2.71828182846**z1[k])+(2.71828182846**-z1[k])))
    z2 = np.dot(a1,w2.T)+b2
    # V_learn = 1/(1 + (2.71828182846**(-z2.item(0))))
    V_learn = ((2.71828182846**z2.item(0))-(2.71828182846**-z2.item(0)))/((2.71828182846**z2.item(0))+(2.71828182846**-z2.item(0)))

    x_samp_np = deepcopy(np.array(x_samp[0]))
    # print("x_sample: ", x_samp_np)
    # print("u_samp: ", u_samp)
    L_V = 0
    L_V_samp = []
    #compute Lie Derivative
    for state in state_list:
        L_V += sympy.diff(V_learn,state) * flows[state]
    #check if it is negative for the sampled values:
    for i in range(len(x_samp_np)):
        L_V_cp = deepcopy(L_V)
        for idx,state in enumerate(state_list):
            L_V_cp = L_V_cp.subs(state,x_samp_np[i][idx])
        for idx, cont in enumerate(cont_list):
            L_V_cp = L_V_cp.subs(cont, u_samp[i][idx])
        L_V_samp.append(L_V_cp)
    # print("computed L_V: ", L_V_samp)


    #why are we adding 0.5 to the L_V_samp caclulation
    #convert lists to tensor
    pos_check_tens = torch.tensor(model_out,dtype = torch.float)
    LV_check_tens = torch.tensor(L_V_samp,dtype = torch.float)
    # print("P check: ", pos_check_tens)
    if j%10 == 0:
        print("LV_Check: ", LV_check_tens)
        if max(LV_check_tens) < 0:
        break

    #changed coeff term on the lie derivative part
    Lyapunov_risk = (F.relu(-pos_check_tens)+ 1.5*F.relu(LV_check_tens+0.5)).mean()+ 1.2*zero_risk.pow(2)
    # print(Lyapunov_risk)
    

    print(j, "Lyapunov Risk=",Lyapunov_risk.item()) 
    L.append(Lyapunov_risk.item())
    optimizer.zero_grad()
    Lyapunov_risk.backward()
    optimizer.step() 
    



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


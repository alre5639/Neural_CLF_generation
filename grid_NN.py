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
from utils import *
from torchinfo import summary
import math
import random

####################
#create NN
#####################

class Net(torch.nn.Module):
    
    def __init__(self,n_input,n_hidden,n_output):
        super(Net, self).__init__()
        torch.manual_seed(2)
        self.layer1 = torch.nn.Linear(n_input, n_hidden)
        # self.layer2 = torch.nn.Linear(n_hidden,n_hidden)
        self.layer3 = torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        sigmoid = torch.nn.Tanh()
        h_1 = sigmoid(self.layer1(x))
        # h_2 = sigmoid(self.layer2(h_1))
        out = sigmoid(self.layer3(h_1))
        return out


'''
For learning 
'''
N = 500             # sample size
#updated input dimension to be the same as the states of the car, might drop V
D_in = 3            # input dimension
H1 = 6              # hidden dimension
D_out = 1           # output dimension
torch.manual_seed(10)          

'''
For verifying 
'''
state_list = x, y, phi= sympy.symbols('x y phi')
u1 = sympy.symbols('u1')
cont_list = (u1,)
vars_ = state_list + cont_list
#define control bounds
# u_bounds = [-3.14/4, 3.14/4, 0,0]

#car params:
L = 2.5 #wheel base

epsilon = 0
# Checking candidate V within a ball around the origin (ball_lb ≤ sqrt(∑xᵢ²) ≤ ball_ub)
ball_lb = 0.1
ball_ub = 5

#define the flows:
#might not need this if we are just pluggin in the values
flows = {
    x: 1*sympy.cos(phi), #this is going to be a problem since z3 cant solve cos(), I think we can just make a piecewise function                            #what if I just approximate this with the piecewise function y=x [x<= pi/2] and y = -x +pi [pi/2<x<pi]
    y: 1*sympy.sin(phi), #same comment as above,
    phi: 1*sympy.tan(u1)/L,
}
#remove lqr references
model = Net(D_in,H1, D_out)
L = []
i = 0 
t = 0
# max_iters = 2000
learning_rate = 0.1#0.01
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
summary(model)

###########################
#Start the grid search:
############################

zero_tesn = torch.zeros([1, D_in])
x_samp = np.empty((0,3), float)
u_samp = []
#look for counter examples:
for dead in range(10):
    #sample a radius to find a point
    r  = random.uniform(1,3)
    #get random x coordinate
    x_1 = random.uniform(-r,r)
    y_1 = math.sqrt(r**2 - x_1**2)
    #the heading will be directly at the goal for now
    heading = math.atan2(-y_1,-x_1)

    #convert from tensor to np array
    if dead != 0:
        x_samp = np.array(x_samp[0])
    print(x_samp)
    x_samp = np.append(x_samp,np.array([[x_1,y_1,heading]]), axis = 0)
    #get associated control
    u_samp.append(Time_based_MPC.get_first_control([x_1,y_1,heading]))
    x_samp = torch.tensor([x_samp], dtype = torch.float)


    print("x samples:", x_samp)
    print("associated u's", u_samp)

    ###############################
    #Train the model
    ################################
    for j in range(1000):

        #caclulate V and Lie V
        # V_learn = two_Hlayer_NN_to_sympy(model, state_list)
        V_learn = NN_to_sympy(model, state_list)
        Lie_V = calc_LV(V_learn, flows, state_list)

        x_samp_np = deepcopy(np.array(x_samp[0]))
        L_V_samp = []
        
        #compute Lie Derivative value
        for i in range(len(x_samp_np)):
            L_V_cp = deepcopy(Lie_V)
            for idx,state in enumerate(state_list):
                L_V_cp = L_V_cp.subs(state,x_samp_np[i][idx])
            for idx, cont in enumerate(cont_list):
                L_V_cp = L_V_cp.subs(cont, u_samp[i][idx])
            L_V_samp.append(L_V_cp)

        #compute result of V(x)
        V_out = model(x_samp)
        #compute V(0)
        zero_tesn = torch.zeros([1, D_in])
        zero_risk = model(zero_tesn).pow(2)

        #convert to tensors
        pos_check_tens = torch.tensor(V_out,dtype = torch.float)
        LV_check_tens = torch.tensor(L_V_samp,dtype = torch.float)
        
        #compute Lyapunov risk
        Lyapunov_risk = (F.relu (-pos_check_tens)+ 1.5*F.relu(LV_check_tens+0.5)).mean()+ 1.2*zero_risk.pow(2)

        # print("P check: ", pos_check_tens)
        if j%2 == 0:
            print("LV_Check: ", LV_check_tens)
            print("pos check: ", pos_check_tens[0])
            print("V(0)=", model(zero_tesn))
            if max(LV_check_tens) < 0 and min(pos_check_tens[0]) > 0:
                print("\n\nFound CLF that satisfies CE's\n\n")
                break
        
        #print the lyapunov risk and then train the network
        print(j, "Lyapunov Risk=",Lyapunov_risk.item()) 
        L.append(Lyapunov_risk.item())
        optimizer.zero_grad()
        Lyapunov_risk.backward()
        optimizer.step() 
        

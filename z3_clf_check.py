from z3 import *
from utils import *
import sympy
import numpy as np

def CheckLyapunov_zero(x, V):
    for state in x:
        V = V.subs(state,0)
    return V

#need to update this to conver V from sympy to z3 expression
def CheckLyapunov_PD(x, V, ball_lb, ball_ub):    
    '''
    CheckLyapunov_PD() checks that the candidate CLF is positive definite in the provided region

    x is the states of the system [z3 Reals]
    V is the canditate control lyapunov function [function in terms of z3_reals]
    ball_lb is the lower bound of the lyapunov region (if ball_lb = 0.5, then clf conditions are not check when x < 0.5)
    ball_ub is the upper bound of the lyapunov region0
    '''
    
    #we will use the ball method for now, but may need to change it later since we may need elispoidal bounds for some systems (x and)

    #conver V from sympy to z3
    z3_var_list, z3_exp = sympy_to_z3(x, V)

    #create the solver
    s = Solver()
    #add the contraints
    for idx, state in enumerate(z3_var_list):
        #for each state, add contraint that we are in the ball region
        s.add(state*state >= ball_lb*ball_lb, state*state <= ball_ub*ball_ub)

    #add contraint that CLF is (+)
    s.add(z3_exp < 0)

    if s.check() == unsat:
        print("lyapunov Function is PD inbetween b_lb and b_up")
        return None
    else:
        print("found counterexample: ", s.model())
        return s.model()

def Check_LD_Lyapunov_ND(x, u, f, V, var_list, ball_lb, ball_ub, u_bounds, epsilon):    
    '''
    Check_LD_Lyapunov_ND() checks that the candidate CLF < epsilon in the provided region

    x is the states of the system
    u is the controls of the system
    f is the dynamics (flows) of the system. f is a dictionary of states and associated dynamics'
    V is the canditate control lyapunov function (sympy expression)
    var list is the concatinated list of states and controls (for z3 conversion)
    ball_lb is the lower bound of the lyapunov region (if ball_lb = 0.5, then clf conditions are not check when x < 0.5)
    ball_ub is the upper bound of the lyapunov region
    u_bound is a array of boundarys of controls should be 2xlen(u)
    epsilon is the minimum slope of the lie derivative
    '''   
    #take lie derivative of V
    L_V = 0
    for state in x:
        L_V += sympy.diff(V,state) * f[state]
    
    print("\nLie Derivative is: ", L_V)

    #conver lie derivative to z3
    z3_var_list, z3_exp = sympy_to_z3(var_list, L_V)
    #get z3 states
    z3_states = sympy_vars_to_z3_vars(x)
    #get z3 controls
    z3_cont = sympy_vars_to_z3_vars(u)



    #check CLF condition (3)
    #for all x in X there exists a u in U for which L_V < 0
    s = z3.Solver()
    #add state boundary contraints
    for idx, state in enumerate(z3_states):
        #for each state, add contraint that we are in the ball region
        s.add(state*state >= ball_lb*ball_lb, state*state <= ball_ub*ball_ub)
    #add the control boundary contraints
    # for idx, control in enumerate(z3_cont):
    #     #for each state, add contraint that we are in the ball region
    #     s.add(control > u_bounds[idx*2], control < u_bounds[(idx*2) + 1])

    # Add the negation of what we want to prove
    #this part might not work iteratively
    # s.add(z3.Not(z3.Exists(z3_cont, z3_exp < 0)))

    #not sure how to itereate this and contraint, for now we will just hard code it
    s.add(z3.Not(z3.Exists(z3_cont, z3.And(z3_cont[0] > 0, z3_cont[0] < 5,z3_cont[1] > 0, z3_cont[1] < 5, z3_exp < 0))))

    if s.check() == unsat:
            print("\nlie derivative is N_D between the boundary")
            return None
    else:
        print("\nfound counterexample: ", s.model())
        return s.model()


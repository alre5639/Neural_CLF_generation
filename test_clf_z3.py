from z3_clf_check import *
import sympy
#########################
#check CLF is 0 # X = [0]
#########################
state_list = x, y = sympy.symbols('x y')

#u bounds
u_bounds = [-5,5,-5,5]
#dummy V
V = x**2 + y**2 - 3

print("\nresult of 0 check is: ", CheckLyapunov_zero(state_list,V))

#########################
#CHECK CLF is PD
#########################

V = x**2 + y**2 - 1

#generate bounds as a ball
b_lb = 0.5
b_ub = 5

CheckLyapunov_PD(state_list, V, b_lb, b_ub)




#########################
#CHECK LD is ND
#########################
state_list = x, y = sympy.symbols('x y')
cont_list = u1, u2 = sympy.symbols('u1 u2')

#make flows dictiony
flows = {
    x: 1+ u1,
    y: 17*u2
}
#u bounds
u_bounds = [-5,5,-5,5]
#dummy V
V = x**2 + y**2 - 3
#concat var list
var_list = state_list + cont_list

Check_LD_Lyapunov_ND(state_list, cont_list, flows, V, var_list, 0.5, 5, u_bounds, 0)
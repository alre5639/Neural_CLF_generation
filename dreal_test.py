from dreal import *
import matplotlib.pyplot as plt
import math


x = Variable("x")
y = Variable("y")
# z = Variable("z")

# f_sat = And(0 <= x, x <= 10,
#             0 <= y, y <= 10,
#             0 <= z, z <= 10,
#             sin(x) + cos(y) == z)

#check to see if -5 + x^2 is positive on the interval [0:10]
f_sat = And(0 <= x, x <=10,
			y < 0,
			-5 + x**2 == y)



result = CheckSatisfiability(f_sat, 0.001)
print(result)

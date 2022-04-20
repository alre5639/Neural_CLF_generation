# Modified from: 
# https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together

import z3
import sympy

def sympy_to_z3(sympy_var_list, sympy_exp):
    '''
    convert a sympy expression to a z3 expression. This returns (z3_vars, z3_expression)
    '''

    z3_vars = []
    z3_var_map = {}

    for var in sympy_var_list:
        name = var.name
        z3_var = z3.Real(name)
        z3_var_map[name] = z3_var
        z3_vars.append(z3_var)

    result_exp = _sympy_to_z3_rec(z3_var_map, sympy_exp)

    return z3_vars, result_exp

def sympy_vars_to_z3_vars(sympy_var_list):
    z3_vars = []
    z3_var_map = {}

    for var in sympy_var_list:
        name = var.name
        z3_var = z3.Real(name)
        z3_var_map[name] = z3_var
        z3_vars.append(z3_var)

    return z3_vars

def _sympy_to_z3_rec(var_map, e):
    '''
    recursive call for sympy_to_z3()
    '''

    rv = None

    if not isinstance(e, sympy.core.Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, sympy.core.Symbol):
        rv = var_map.get(e.name)

        if rv == None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, sympy.core.Number):
        rv = float(e)
    elif isinstance(e, sympy.core.Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, sympy.core.Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, sympy.core.Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])

        if exponent == 0.5:
            # sqrt
            rv = sympy.core.Sqrt(term)
        else:
            rv = term**exponent

    if rv == None:
        raise RuntimeError("Type '" + str(type(e)) + "' is not yet implemented for convertion to a z3 expresion. " + \
                            "Subexpression was '" + str(e) + "'.")

    return rv

# if __name__ == "__main__":
#     # define the variable list for the conversion and symbols for Sympy
#     var_list = x,y = sympy.symbols('x y')
#     # expression to differentiate
#     # gfg_exp = x**2 * y**2
#     # gfg_exp = -x**2 + y + 1
#     gfg_exp = (x**2) - 1 + y
#     print(f"Before diff: {gfg_exp}")
#     # differentiate wrt to x
#     dif_x = sympy.diff(gfg_exp, x)
#     print(f"After differentiation wrt x: {dif_x}")
#     # differentiate wrt to y
#     dif_y = sympy.diff(gfg_exp, y)
#     print(f"After differentiation wrt y: {dif_y}")

#     # convert to z3 formatted variables
#     z3_vars, z3_exp = sympy_to_z3(var_list, gfg_exp)
#     print(f"z3_x: {z3_vars[0]}")
#     print(f"z3_y: {z3_vars[1]}")

#     z3_x = z3_vars[0]
#     z3_y = z3_vars[1]

#     # Example 1 - from Sympy documentation
#     s = z3.Solver()
#     # s.add(z3_exp == 0) # add a constraint with converted expression
#     # s.add(z3_y >= 0) # add an extra constraint
#     # result = s.check()
#     # if result == z3.sat:
#     #     m = s.model()

#     #     print(f"SAT at x={m[z3_x]}, y={m[z3_y]}")
#     # else:
#     #     print("UNSAT")

#     # Example 2 -  from Stackoverflow
#     # https://stackoverflow.com/questions/71862652/dreal4-forall-smt/71877195?noredirect=1#comment127032644_71877195
#     s = z3.Solver()
#     s.add(z3_x < 5)
#     s.add(z3_x > -5)

#     # Add the negation of what we want to prove
#     s.add(z3.Not(z3.Exists([z3_y], z3.And(z3_y > -20, z3_y < 25, z3_exp < 0))))

#     r = s.check()
#     if r == z3.sat:
#         print("Counter-example:")
#         print(s.model())
#     else:
#         print("Solver said: ", r)
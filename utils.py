# Modified from: 
# https://stackoverflow.com/questions/22488553/how-to-use-z3py-and-sympy-together

import z3
import sympy
import numpy as np

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

def approx_sin(x):
    '''
    approx_sin(x) is a piece wise linear appromxiation of sin valid between -4 and 4
    '''
    if x >= -1 and x <= 1:
        return x
    elif x > 1 and x <= 2.142:
        return 1
    elif x > 2.142 and x < 4:
        return 3.14159 - x
    elif x < -1 and x >= -2.142:
        return -1
    elif x < -2.142 and x > -4:
        return -1.0 * 3.14159  -x
    else:
        raise ValueError("x is out of range for approx_sin()")

def approx_cos(x):
    '''
    approx_cos(x) is a piece wise linear appromxiation of vos valid between -4 and 4
    '''
    if x >= -0.571 and x <= 0.571:
        return 1
    elif x > 0.571 and x <= 2.571:
        return (3.14159 /2) - x
    elif x > 2.571 and x < 4:
        return -1
    elif x < -0.571 and x >= -2.571:
        return -3.14159/2 + x
    elif x < -2.571 and x > -4:
        return -1.0
    else:
        raise ValueError("x is out of range for approx_cos()")
def utils_tanh(x):
    ((2.718281828**x) - (2.718281828**(-x)))/((2.718281828**x) + (2.718281828**(-x)))

def NN_to_sympy(w1,w2,b1,b2, state_list):
    z1 = np.dot(state_list,w1.T)+b1

    a1 = []
    #will need to replace exp with the actual function(1/(1+(2.71828182846**(-x))
    for j in range(0,len(z1)):
        # a1.append(1/(1+(2.71828182846**(-z1[j]))))
        a1.append(((2.71828182846**z1[j])-(2.71828182846**-z1[j]))/((2.71828182846**z1[j])+(2.71828182846**-z1[j])))
    z2 = np.dot(a1,w2.T)+b2
    # V_learn = 1/(1 + (2.71828182846**(-z2.item(0))))
    V_learn = ((2.71828182846**z2.item(0))-(2.71828182846**-z2.item(0)))/((2.71828182846**z2.item(0))+(2.71828182846**-z2.item(0)))
    return V_learn
def two_Hlayer_NN_to_sympy(model, state_list):
    w1 = model.layer1.weight.data.numpy()
    w2 = model.layer2.weight.data.numpy()
    w3 = model.layer3.weight.data.numpy()
    b1 = model.layer1.bias.data.numpy()
    b2 = model.layer2.bias.data.numpy()
    b3 = model.layer3.bias.data.numpy()

    # Candidate V
    z1 = np.dot(state_list,w1.T)+b1

    a1 = []
    a2 = []
    for j in range(0,len(z1)):
        a1.append(sympy.tanh(z1[j]))
    z2 = np.dot(a1,w2.T)+b2
    for j in range(0,len(z2)):
        a2.append(sympy.tanh(z2[j]))
    z3 = np.dot(a2,w3.T)+b3
    V_learn = sympy.tanh(z3.item(0))
    return V_learn


def calc_LV(V, f, state_list):
    L_V = 0
    for state in state_list:
        L_V += sympy.diff(V,state) * f[state]
    return L_V
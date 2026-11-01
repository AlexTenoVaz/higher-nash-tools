import sympy as sp

def order_t(expr, parameter):
    expr = sp.simplify(expr)

    if expr == 0:
        return sp.oo

    leading_term = sp.simplify(
        expr.as_leading_term(parameter)
    )

    exponent = sp.sympify(
        leading_term.as_powers_dict().get(parameter, 0)
    )

    if exponent.is_integer is not True:
        raise ValueError(
            "The leading exponent must be an integer."
        )

    return int(exponent)

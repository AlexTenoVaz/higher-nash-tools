import sympy as sp


def order_t(expr, parameter):
    """
    Compute the order of vanishing of an expression at parameter = 0.

    For a nonzero polynomial in `parameter`, this is the smallest
    exponent with nonzero coefficient.

    Parameters
    ----------
    expr : sympy.Expr
        Expression in the parameter.
    parameter : sympy.Symbol
        Parameter with respect to which the order is computed.

    Returns
    -------
    int or sympy.oo
        The order of vanishing. Returns +oo for the zero expression.
    """
    expr = sp.expand(expr)

    if expr == 0:
        return sp.oo

    poly = sp.Poly(expr, parameter)

    return min(exponent[0] for exponent in poly.monoms())
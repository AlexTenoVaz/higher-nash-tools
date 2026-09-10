import sympy as sp

from source.order import order_t


def leading_exponents(curve, parameter):
    """
    Extract the leading exponents of a parameterized curve.

    For each coordinate gamma_i(t), compute

        e_i = ord_t(gamma_i(t)).

    These exponents are the vector used to construct the
    weighted-order matrix.

    Parameters
    ----------
    curve : list or tuple of sympy.Expr
        Parameterized curve.
    parameter : sympy.Symbol
        Parameter with respect to which the orders are computed.

    Returns
    -------
    list
        List of leading exponents.
    """
    return [
        order_t(coordinate, parameter)
        for coordinate in curve
    ]
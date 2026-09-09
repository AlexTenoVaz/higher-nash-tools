import sympy as sp


def weighted_order(polynomial, weights):
    """
    Compute the weighted order of a polynomial.

    For

        g = sum(c_alpha * x^alpha),

    and a weight vector e, the weighted order is

        w_e(g) = min_{c_alpha != 0} <alpha, e>.

    Parameters
    ----------
    polynomial : sympy.Expr
        Polynomial whose weighted order is computed.
    weights : tuple or list of numbers
        Weight vector e.

    Returns
    -------
    number
        The weighted order of the polynomial.

    Notes
    -----
    The zero polynomial has weighted order +oo.
    """
    polynomial = sp.expand(polynomial)

    if polynomial == 0:
        return sp.oo

    variables = sorted(polynomial.free_symbols, key=lambda symbol: symbol.name)

    if len(weights) != len(variables):
        raise ValueError(
            "The number of weights must match the number of variables."
        )

    polynomial = sp.Poly(polynomial, *variables)

    weighted_orders = [
        sum(exponent * weight for exponent, weight in zip(monomial, weights))
        for monomial in polynomial.monoms()
    ]

    return min(weighted_orders)
import sympy as sp

from source.order import order_t


t = sp.Symbol("t")


def test_order_of_nonzero_polynomial():
    assert order_t(3*t**2 + 5*t**7, t) == 2


def test_order_of_constant():
    assert order_t(7, t) == 0


def test_order_of_zero():
    assert order_t(0, t) == sp.oo


def test_order_after_cancellation():
    expr = t**3 - t**3 + 4*t**5
    assert order_t(expr, t) == 5
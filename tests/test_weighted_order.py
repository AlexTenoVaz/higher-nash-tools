import sympy as sp

from source.weighted_order import weighted_order


x, y, z = sp.symbols("x y z")


def test_weighted_order_of_monomial_sum():
    f = x**2 + y**3 + z**4

    assert weighted_order(f, (2, 3, 4)) == 4


def test_weighted_order_of_single_monomial():
    f = x**2 * y**3 * z

    assert weighted_order(f, (1, 2, 3)) == 11


def test_weighted_order_of_zero():
    assert weighted_order(0, (1, 2, 3)) == sp.oo


def test_weighted_order_ignores_coefficients():
    f = 7*x**2 + 100*y**3

    assert weighted_order(f, (1, 1)) == 2
    
def test_weighted_order_of_constant_with_explicit_variables():
    assert weighted_order(1, (1, 1, 1), variables=(x, y, z)) == 0
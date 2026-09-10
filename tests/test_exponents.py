import sympy as sp

from source.exponents import leading_exponents


t = sp.Symbol("t")


def test_leading_exponents_of_polynomial_curve():
    curve = [
        t**2 + t**5,
        3*t**4 + t**6,
        t**7
    ]

    assert leading_exponents(curve, t) == [2, 4, 7]


def test_zero_coordinate_has_infinite_order():
    curve = [
        t**2,
        0,
        t**5
    ]

    assert leading_exponents(curve, t) == [2, sp.oo, 5]


def test_e6_descended_curve():
    curve = [
        sp.I*t**8*(-t**2 - 2*t - 1),
        t**6*(t + 1),
        t**4*(t + 1)
    ]

    assert leading_exponents(curve, t) == [8, 6, 4]
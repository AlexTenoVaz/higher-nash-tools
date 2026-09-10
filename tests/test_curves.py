import sympy as sp

from source.curves import curves_on_divisor, descend_curve


x, y, z, t = sp.symbols("x y z t")


def test_curves_on_e6_divisor():
    divisor = x + sp.I*z

    curve_1, curve_2 = curves_on_divisor(
        divisor=divisor,
        chart=1,
        fixed_polynomial=t**2,
        polynomial_1=t + 1,
        polynomial_2=t - 1
    )

    expected_curve_1 = [
        -sp.I*(t + 1),
        t**2,
        t + 1
    ]

    expected_curve_2 = [
        sp.I*(1 - t),
        t**2,
        t - 1
    ]

    assert all(
        sp.simplify(a - b) == 0
        for a, b in zip(curve_1, expected_curve_1)
    )

    assert all(
        sp.simplify(a - b) == 0
        for a, b in zip(curve_2, expected_curve_2)
    )


def test_descend_e6_curve():
    curve = [
        -sp.I*(t + 1),
        t**2,
        t + 1
    ]

    descended = descend_curve(
        curve,
        [2, 1, 1]
    )

    expected = [
        sp.I*t**8*(-t**2 - 2*t - 1),
        t**6*(t + 1),
        t**4*(t + 1)
    ]

    assert descended == expected


def test_descend_is_applied_from_last_blowup_to_first():
    curve = [t, t**2, t**3]

    descended = descend_curve(
        curve,
        [0]
    )

    assert descended == [
        t,
        t**3,
        t**4
    ]
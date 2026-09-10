import sympy as sp

from source.grassmannian import grassmannian_separation


x, y, z, t = sp.symbols("x y z t")


def test_separating_pair_is_found():
    curve_1 = [t, t, t]
    curve_2 = [t, 2*t, t]

    minors = [
        x,
        y,
    ]

    result = grassmannian_separation(
        curve_1,
        curve_2,
        minors,
        t
    )

    assert result["separated"] is True
    assert result["pair"] == (0, 1)
    assert result["limits_curve_1"] == 1
    assert result["limits_curve_2"] == 2


def test_curves_not_separated_by_given_minors():
    curve_1 = [t, t, t]
    curve_2 = [t, t, 2*t]

    minors = [
        x,
        y,
    ]

    result = grassmannian_separation(
        curve_1,
        curve_2,
        minors,
        t
    )

    assert result["separated"] is False
    assert result["pair"] is None


def test_zero_denominator_is_skipped():
    curve_1 = [t, t, 0]
    curve_2 = [t, 2*t, 0]

    minors = [
        z,
        x,
        y,
    ]

    result = grassmannian_separation(
        curve_1,
        curve_2,
        minors,
        t
    )

    assert result["separated"] is True
    assert result["pair"] == (1, 2)
    assert result["limits_curve_1"] == 1
    assert result["limits_curve_2"] == 2
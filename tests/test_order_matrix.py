import sympy as sp

from source.order_matrix import order_matrix


x, y, z = sp.symbols("x y z")


def test_order_matrix_shape():
    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(1, 1, 1),
        jac_order=3
    )

    assert O.shape == (10, 19)


def test_order_matrix_first_row():
    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(1, 1, 1),
        jac_order=3
    )

    assert O[0, 0] == 1
    assert O[0, 1] == 2
    assert O[0, 2] == 3
    assert O[0, 3] == 0
    assert O[0, 4] == sp.oo
    assert O[0, 5] == 1
    assert O[0, 8] == 2
    assert O[0, 12] == 0
    assert O[0, 18] == 1


def test_zero_jacobian_entries_have_infinite_weighted_order():
    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(1, 1, 1),
        jac_order=3
    )

    assert O[0, 4] == sp.oo
    assert O[1, 0] == sp.oo


def test_nontrivial_weights():
    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(2, 3, 4),
        jac_order=3
    )

    assert O[0, 0] == 2
    assert O[0, 1] == 6
    assert O[0, 2] == 12
    assert O[0, 3] == 0
    assert O[0, 5] == 3
    assert O[0, 8] == 8
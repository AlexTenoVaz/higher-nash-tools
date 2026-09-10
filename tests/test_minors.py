import sympy as sp

from source.higher_jacobian_matrix import jacobian_matrix, x, y, z
from source.minors import maximal_minor


def test_small_minor():
    J = sp.Matrix([
        [1, 2],
        [3, 4],
    ])

    assert maximal_minor(J, [0, 1]) == -2


def test_e6_minor_from_real_jacobian():
    f = x**2 + y**3 + z**4

    J = jacobian_matrix(
        f,
        jac_order_local=3
    )

    columns = [3, 4, 6, 9, 10, 11, 12, 13, 14, 16]

    minor = maximal_minor(J, columns)

    expected = (
    512*x**9
    + 2304*x**7*y**3
    + 1728*x**5*y**6
    )

    assert sp.expand(minor - expected) == 0


def test_invalid_number_of_columns():
    J = sp.Matrix([
        [1, 2],
        [3, 4],
    ])

    try:
        maximal_minor(J, [0])
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for a non-maximal column set."
    )
from sympy import Matrix

from source.higher_jacobian_matrix import (
    x,
    y,
    z,
    jacobian_matrix,
)


def test_e6_higher_jacobian_order_3():
    f = x**2 + y**3 + z**4

    J = jacobian_matrix(
        f,
        jac_order_local=3
    )

    expected = Matrix([
        [2*x, 3*y**2, 4*z**3, 1, 0, 3*y, 0, 0, 6*z**2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4*z],
        [0, 0, 0, 2*x, 3*y**2, 0, 4*z**3, 0, 0, 1, 0, 3*y, 0, 0, 0, 0, 6*z**2, 0, 0],
        [0, 0, 0, 0, 2*x, 3*y**2, 0, 4*z**3, 0, 0, 1, 0, 3*y, 0, 0, 0, 0, 6*z**2, 0],
        [0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 4*z**3, 0, 0, 0, 0, 1, 0, 3*y, 0, 0, 6*z**2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 0, 0, 4*z**3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 0, 0, 4*z**3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 0, 0, 4*z**3, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 0, 4*z**3, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 0, 4*z**3, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2*x, 3*y**2, 4*z**3]
    ])

    assert J == expected
    assert J.shape == (10, 19)


def test_custom_indices_are_respected():
    f = x**2 + y**3 + z**4

    J = jacobian_matrix(
        f,
        row_vectors=[(0, 0, 0)],
        col_vectors=[(1, 0, 0)],
        variables_list=[x, y, z],
        number_of_vars=3,
        jac_order_local=3
    )

    assert J == Matrix([[2*x]])
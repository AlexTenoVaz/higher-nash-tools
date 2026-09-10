import sympy as sp

from source.higher_jacobian_matrix import jacobian_matrix, x, y, z
from source.weighted_order import weighted_order


def order_matrix(
        polynomial,
        weights,
        jac_order,
        variables=None,
        row_vectors=None,
        col_vectors=None
        ):
    """
    Construct the weighted-order matrix of a higher-order Jacobian.

    For each entry J_{beta,alpha} of the higher-order Jacobian,

        O_{beta,alpha} = w_e(J_{beta,alpha}),

    where e = weights.

    Parameters
    ----------
    polynomial : sympy.Expr
        Polynomial defining the higher-order Jacobian.
    weights : tuple or list
        Weight vector e.
    jac_order : int
        Order m of the higher-order Jacobian.
    variables : tuple or list of sympy.Symbol, optional
        Variables corresponding to the entries of `weights`.
    row_vectors : list of tuple, optional
        Row multi-indices.
    col_vectors : list of tuple, optional
        Column multi-indices.

    Returns
    -------
    sympy.Matrix
        Matrix of weighted orders.

    Notes
    -----
    Zero entries of the higher-order Jacobian have weighted order +oo.
    """
    if variables is None:
        default_variables = (x, y, z)
        variables = list(default_variables[:len(weights)])
    else:
        variables = list(variables)

    if len(weights) != len(variables):
        raise ValueError(
            "The number of weights must match the number of variables."
        )

    jacobian = jacobian_matrix(
        polynomial,
        row_vectors=row_vectors,
        col_vectors=col_vectors,
        variables_list=variables,
        number_of_vars=len(variables),
        jac_order_local=jac_order
    )

    return sp.Matrix(jacobian).applyfunc(
        lambda entry: weighted_order(
            entry,
            weights,
            variables=variables
        )
    )
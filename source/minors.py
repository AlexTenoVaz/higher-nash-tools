import sympy as sp


def maximal_minor(jacobian, columns):
    """
    Compute the maximal minor determined by a set of columns.

    Parameters
    ----------
    jacobian : sympy.Matrix
        Higher-order Jacobian matrix.
    columns : iterable of int
        Zero-based column indices.

    Returns
    -------
    sympy.Expr
        Determinant of the submatrix formed by all rows and
        the selected columns.

    Raises
    ------
    ValueError
        If the number of selected columns is different from
        the number of rows of the Jacobian.
    """
    J = sp.Matrix(jacobian)
    columns = list(columns)

    if len(columns) != J.rows:
        raise ValueError(
            "A maximal minor must select exactly one column per row."
        )

    return sp.expand(J[:, columns].det(method="domain-ge"))
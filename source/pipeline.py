import sympy as sp

from source.higher_jacobian_matrix import jacobian_matrix
from source.exponents import leading_exponents
from source.order_matrix import order_matrix
from source.matching import minimum_cost_matching
from source.optimal_columns import enumerate_optimal_column_sets
from source.minors import maximal_minor
from source.grassmannian import grassmannian_separation

def build_pipeline(
        polynomial,
        curve_1,
        curve_2,
        parameter,
        jac_order
        ):
    """
    Build the computational data used in the higher Nash pipeline.

    Parameters
    ----------
    polynomial : sympy.Expr
        Polynomial defining the hypersurface.
    curve_1, curve_2 : list or tuple of sympy.Expr
        Descended parameterized curves.
    parameter : sympy.Symbol
        Parameter of the curves.
    jac_order : int
        Order of the higher-order Jacobian.

    Returns
    -------
    dict
        Dictionary containing:
        - higher_jacobian
        - exponents_1
        - exponents_2
        - order_matrix
        - initial_columns
        - optimal_cost
        - optimal_column_sets
    """
    exponents_1 = leading_exponents(
        curve_1,
        parameter
    )

    exponents_2 = leading_exponents(
        curve_2,
        parameter
    )

    if exponents_1 != exponents_2:
        raise ValueError(
            "The two curves must have the same leading exponents."
        )

    jacobian = jacobian_matrix(
        polynomial,
        jac_order_local=jac_order
    )

    weighted_orders = order_matrix(
        polynomial,
        weights=exponents_1,
        jac_order=jac_order
    )

    initial_columns, optimal_cost = minimum_cost_matching(
        weighted_orders
    )

    optimal_column_sets = enumerate_optimal_column_sets(
        weighted_orders,
        initial_columns,
        optimal_cost
    )

    return {
        "higher_jacobian": sp.Matrix(jacobian),
        "exponents_1": exponents_1,
        "exponents_2": exponents_2,
        "order_matrix": sp.Matrix(weighted_orders),
        "initial_columns": initial_columns,
        "optimal_cost": optimal_cost,
        "optimal_column_sets": optimal_column_sets,
    }

def _try_separation_with_minor(
        minor,
        previous_minors,
        curve_1,
        curve_2,
        parameter
        ):
    """
    Test a new minor against previously computed nonzero minors.
    """
    for previous_minor in previous_minors:
        result = grassmannian_separation(
            curve_1,
            curve_2,
            [previous_minor, minor],
            parameter
        )

        if result["separated"]:
            return result

    return None

def separate_curves(
        pipeline_data,
        curve_1,
        curve_2,
        parameter,
        additional_minors=None
        ):
    """
    Apply the Grassmannian separation test.

    Exact maximal minors are computed incrementally. The computation
    stops as soon as a pair of minors separates the two curves.

    Parameters
    ----------
    pipeline_data : dict
        Result returned by build_pipeline.
    curve_1, curve_2 : list or tuple of sympy.Expr
        Descended parameterized curves.
    parameter : sympy.Symbol
        Parameter of the curves.
    additional_minors : iterable of sympy.Expr, optional
        Additional minors selected by the geometric argument.

    Returns
    -------
    dict
        Separation result.
    """
    jacobian = pipeline_data["higher_jacobian"]
    optimal_column_sets = pipeline_data["optimal_column_sets"]

    computed_minors = []

    for columns in optimal_column_sets:
        minor = maximal_minor(
            jacobian,
            columns
        )

        if minor == 0:
            continue

        result = _try_separation_with_minor(
            minor,
            computed_minors,
            curve_1,
            curve_2,
            parameter
        )

        if result is not None:
            return result

        computed_minors.append(minor)

    if additional_minors is not None:
        for minor in additional_minors:
            if minor == 0:
                continue

            result = _try_separation_with_minor(
                minor,
                computed_minors,
                curve_1,
                curve_2,
                parameter
            )

            if result is not None:
                return result

            computed_minors.append(minor)

    return grassmannian_separation(
        curve_1,
        curve_2,
        computed_minors,
        parameter
    )
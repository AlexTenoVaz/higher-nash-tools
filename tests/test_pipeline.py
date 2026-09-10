import sympy as sp

from source.curves import curves_on_divisor, descend_curve
from source.exponents import leading_exponents
from source.order_matrix import order_matrix
from source.matching import minimum_cost_matching
from source.optimal_columns import enumerate_optimal_column_sets
from source.minors import maximal_minor
from source.grassmannian import grassmannian_separation


x, y, z, t = sp.symbols("x y z t")


def test_e6_automatic_pipeline():
    # Polynomial
    f = x**2 + y**3 + z**4

    # Curves on the chosen exceptional divisor
    divisor = x + sp.I*z

    curve_1, curve_2 = curves_on_divisor(
        divisor=divisor,
        chart=1,
        fixed_polynomial=t**2,
        polynomial_1=t + 1,
        polynomial_2=t - 1
    )

    # Descend both curves
    descended_1 = descend_curve(
        curve_1,
        [2, 1, 1]
    )

    descended_2 = descend_curve(
        curve_2,
        [2, 1, 1]
    )

    # Extract leading exponents
    exponents_1 = leading_exponents(
        descended_1,
        t
    )

    exponents_2 = leading_exponents(
        descended_2,
        t
    )

    assert exponents_1 == [8, 6, 4]
    assert exponents_2 == [8, 6, 4]

    # Construct weighted-order matrix
    O = order_matrix(
        f,
        weights=exponents_1,
        jac_order=5
    )

    assert O.shape == (35, 55)

    # Minimum-cost matching
    initial_columns, optimal_cost = minimum_cost_matching(O)

    assert len(initial_columns) == 35
    assert optimal_cost == 226

    # Enumerate all optimal column sets
    optimal_sets = enumerate_optimal_column_sets(
        O,
        initial_columns,
        optimal_cost
    )

    canonical_sets = {
        tuple(sorted(columns))
        for columns in optimal_sets
    }

    assert len(optimal_sets) == 51
    assert len(canonical_sets) == 51

    assert all(
        len(columns) == 35
        for columns in optimal_sets
    )

    for columns in optimal_sets:
        _, cost = minimum_cost_matching(
            O,
            excluded_columns=set(range(O.cols)) - set(columns)
        )

        assert cost == 226
        
def test_a2_geometric_pipeline():
    """
    Integrated A2 test.

    The minimum-cost candidates are obtained automatically.
    The geometrically selected minor from the thesis is then
    added explicitly before applying the Grassmannian test.
    """
    f = x**2 + y**2 + z**3

    I = sp.I

    # Curves on the exceptional divisor from the thesis.
    curve_1_on_divisor = [
        I + sp.Symbol("t"),
        1,
        -sp.Symbol("t") * (sp.Symbol("t") + 2*I)
    ]

    curve_2_on_divisor = [
        -I - sp.Symbol("t"),
        -1,
        -sp.Symbol("t") * (sp.Symbol("t") + 2*I)
    ]

    t_local = sp.Symbol("t")

    curve_1_on_divisor = [
        I + t_local,
        1,
        -t_local * (t_local + 2*I)
    ]

    curve_2_on_divisor = [
        -I - t_local,
        -1,
        -t_local * (t_local + 2*I)
    ]

    # Descend through the blow-up.
    curve_1 = descend_curve(
        curve_1_on_divisor,
        [2]
    )

    curve_2 = descend_curve(
        curve_2_on_divisor,
        [2]
    )

    # Leading exponents.
    e1 = leading_exponents(
        curve_1,
        t_local
    )

    e2 = leading_exponents(
        curve_2,
        t_local
    )

    assert e1 == [1, 1, 1]
    assert e2 == [1, 1, 1]

    # Weighted-order matrix.
    O = order_matrix(
        f,
        weights=e1,
        jac_order=2
    )

    assert O.shape == (4, 9)

    # Automatic minimum-cost matching.
    initial_columns, optimal_cost = minimum_cost_matching(O)

    assert optimal_cost == 3
    assert len(initial_columns) == 4

    # Enumerate all minimum-cost column sets.
    optimal_sets = enumerate_optimal_column_sets(
        O,
        initial_columns,
        optimal_cost
    )

    assert len(optimal_sets) == 2

    # Compute the corresponding exact minors.
    J = sp.Matrix(
        sp.sympify(
            __import__(
                "source.higher_jacobian_matrix",
                fromlist=["jacobian_matrix"]
            ).jacobian_matrix(
                f,
                jac_order_local=2
            )
        )
    )

    automatic_minors = [
        maximal_minor(J, columns)
        for columns in optimal_sets
    ]

    assert len(automatic_minors) == 2
    assert all(
        minor != 0
        for minor in automatic_minors
    )

    # The geometrically selected minor used in the thesis.
    geometric_minor = maximal_minor(
        J,
        [0, 3, 4, 6]
    )

    assert sp.factor(geometric_minor) == 16*x**4

    # The minimum-cost candidates alone do not separate these curves.
    automatic_result = grassmannian_separation(
        curve_1,
        curve_2,
        automatic_minors,
        t_local
    )

    assert automatic_result["separated"] is False

    # Add the geometrically relevant minor and repeat.
    all_minors = automatic_minors + [
        geometric_minor
    ]

    result = grassmannian_separation(
        curve_1,
        curve_2,
        all_minors,
        t_local
    )

    assert result["separated"] is True

    assert sp.simplify(
        result["limits_curve_1"]
        - result["limits_curve_2"]
    ) != 0
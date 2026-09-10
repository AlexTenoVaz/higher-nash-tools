import sympy as sp

from source.curves import curves_on_divisor, descend_curve
from source.exponents import leading_exponents
from source.order_matrix import order_matrix
from source.matching import minimum_cost_matching
from source.optimal_columns import enumerate_optimal_column_sets


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
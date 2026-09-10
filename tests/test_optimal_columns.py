import sympy as sp

from source.matching import minimum_cost_matching
from source.optimal_columns import enumerate_optimal_column_sets


def test_enumerate_all_optimal_column_sets():
    O = sp.Matrix([
        [0, 0, 0],
        [0, 0, 0],
    ])

    initial_columns, optimal_cost = minimum_cost_matching(O)

    optimal_sets = enumerate_optimal_column_sets(
        O,
        initial_columns,
        optimal_cost
    )

    optimal_sets = sorted(optimal_sets)

    assert optimal_sets == [
        [0, 1],
        [0, 2],
        [1, 2],
    ]


def test_optimal_column_sets_have_same_cost():
    O = sp.Matrix([
        [0, 0, 1],
        [1, 0, 0],
    ])

    initial_columns, optimal_cost = minimum_cost_matching(O)

    optimal_sets = enumerate_optimal_column_sets(
        O,
        initial_columns,
        optimal_cost
    )

    assert initial_columns in optimal_sets

    for columns in optimal_sets:
        assert len(columns) == O.rows

        selected = O[:, columns]

        _, cost = minimum_cost_matching(
            selected
        )

        assert cost == optimal_cost
        
def test_optimal_columns_with_real_order_matrix():
    from source.order_matrix import order_matrix

    x, y, z = sp.symbols("x y z")

    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(2, 3, 4),
        jac_order=3
    )

    initial_columns, optimal_cost = minimum_cost_matching(O)

    optimal_sets = enumerate_optimal_column_sets(
        O,
        initial_columns,
        optimal_cost
    )

    assert initial_columns == [
        3, 4, 6, 9, 10, 11, 12, 13, 14, 16
    ]

    assert optimal_cost == 18

    assert optimal_sets == [
        [3, 4, 6, 9, 10, 11, 12, 13, 14, 16]
    ]
    
def test_e6_m5_finds_51_optimal_column_sets():
    from source.order_matrix import order_matrix

    x, y, z = sp.symbols("x y z")

    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(8, 6, 4),
        jac_order=5
    )

    initial_columns, optimal_cost = minimum_cost_matching(O)

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

        assert cost == optimal_cost == 226
import sympy as sp

from source.matching import minimum_cost_matching


def test_zero_cost_edges_are_valid():
    O = sp.Matrix([
        [0, 1, sp.oo],
        [2, 0, 3],
    ])

    columns, cost = minimum_cost_matching(O)

    assert columns == [0, 1]
    assert cost == 0


def test_infinite_entries_are_forbidden():
    O = sp.Matrix([
        [0, 1, sp.oo],
        [2, 0, 3],
    ])

    columns, cost = minimum_cost_matching(
        O,
        excluded_columns={1}
    )

    assert columns == [0, 2]
    assert cost == 3


def test_no_complete_matching():
    O = sp.Matrix([
        [0, sp.oo],
        [sp.oo, sp.oo],
    ])

    try:
        minimum_cost_matching(O)
    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError when no complete matching exists."
    )

def test_matching_with_real_order_matrix():
    from source.order_matrix import order_matrix

    x, y, z = sp.symbols("x y z")

    f = x**2 + y**3 + z**4

    O = order_matrix(
        f,
        weights=(2, 3, 4),
        jac_order=3
    )

    columns, cost = minimum_cost_matching(O)

    assert columns == [3, 4, 6, 9, 10, 11, 12, 13, 14, 16]
    assert cost == 18
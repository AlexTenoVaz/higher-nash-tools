from itertools import combinations

import sympy as sp

from source.matching import minimum_cost_matching


def enumerate_optimal_column_sets(
        order_matrix,
        initial_columns,
        optimal_cost
        ):
    """
    Enumerate column sets attaining a prescribed minimum matching cost.

    Parameters
    ----------
    order_matrix : sympy.Matrix or array-like
        Order matrix.
    initial_columns : iterable of int
        Initial optimal column set.
    optimal_cost : int
        Minimum matching cost associated with initial_columns.

    Returns
    -------
    list of list of int
        All discovered optimal column sets.

    Notes
    -----
    This follows Algorithm 8 of the thesis:
    starting from an optimal column set, non-empty subsets of the
    current column set are excluded and the minimum-cost matching
    problem is solved again.
    """
    O = sp.Matrix(order_matrix)

    initial_columns = tuple(sorted(initial_columns))

    if len(initial_columns) != O.rows:
        raise ValueError(
            "The initial column set must contain one column per row."
        )

    results = [list(initial_columns)]
    visited = {initial_columns}
    queue = [initial_columns]

    while queue:
        current = queue.pop(0)

        for size in range(1, len(current) + 1):
            for excluded_subset in combinations(current, size):
                try:
                    new_columns, new_cost = minimum_cost_matching(
                        O,
                        excluded_columns=excluded_subset
                    )
                except ValueError:
                    continue

                new_columns = tuple(sorted(new_columns))

                if (
                    len(new_columns) == O.rows
                    and new_cost == optimal_cost
                    and new_columns not in visited
                ):
                    visited.add(new_columns)
                    results.append(list(new_columns))
                    queue.append(new_columns)

    return results
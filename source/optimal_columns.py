import sympy as sp

from source.matching import minimum_cost_matching


def enumerate_optimal_column_sets(
        order_matrix,
        initial_columns,
        optimal_cost
        ):
    """
    Enumerate all column sets attaining the prescribed minimum matching cost.

    The search keeps the exclusion set as part of the state. At each state,
    one optimal column set is obtained. Excluding each of its columns creates
    subproblems that partition the remaining optimal solutions. Memoization
    avoids solving the same exclusion problem more than once.
    """
    O = sp.Matrix(order_matrix)
    rows = O.rows

    initial_columns = tuple(sorted(initial_columns))

    if len(initial_columns) != rows:
        raise ValueError(
            "The initial column set must contain one column per row."
        )

    results = set()
    visited_exclusions = set()
    cache = {}

    def solve(excluded_columns):
        excluded_columns = tuple(sorted(set(excluded_columns)))

        if excluded_columns not in cache:
            try:
                cache[excluded_columns] = minimum_cost_matching(
                    O,
                    excluded_columns=excluded_columns
                )
            except ValueError:
                cache[excluded_columns] = None

        return cache[excluded_columns]

    def explore(excluded_columns):
        excluded_columns = tuple(sorted(set(excluded_columns)))

        if excluded_columns in visited_exclusions:
            return

        visited_exclusions.add(excluded_columns)

        solution = solve(excluded_columns)

        if solution is None:
            return

        columns, cost = solution

        if cost != optimal_cost:
            return

        columns = tuple(sorted(columns))
        results.add(columns)

        # Every other optimal solution differs from this one by omitting
        # at least one of its columns. Recursing on each such exclusion
        # therefore reaches every remaining optimal solution.
        for column in columns:
            explore(excluded_columns + (column,))

    explore(())

    return [
        list(columns)
        for columns in sorted(results)
    ]
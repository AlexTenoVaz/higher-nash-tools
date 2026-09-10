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

    The search follows the exclusion strategy of Algorithm 8:
    starting from an optimal column set, columns are progressively
    excluded and the minimum-cost matching problem is solved again.

    Branches for which the minimum cost becomes larger than the
    prescribed optimal cost are discarded, since removing additional
    columns cannot decrease the minimum cost.
    """
    O = sp.Matrix(order_matrix)

    initial_columns = tuple(sorted(initial_columns))

    if len(initial_columns) != O.rows:
        raise ValueError(
            "The initial column set must contain one column per row."
        )

    results = [list(initial_columns)]

    discovered = {initial_columns}
    processed = set()

    # Each element of the queue is an optimal column set whose
    # descendants still have to be explored.
    queue = [initial_columns]

    # Cache matching computations for exclusion sets.
    cache = {}

    def solve_with_exclusions(excluded_columns):
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

    def explore_column_set(columns):
        """
        Explore all relevant exclusions of one optimal column set.

        The recursion is indexed by position, so every subset of
        `columns` is visited at most once.
        """
        columns = tuple(columns)

        if columns in processed:
            return

        processed.add(columns)

        def explore_subsets(start, excluded):
            excluded = tuple(excluded)

            for index in range(start, len(columns)):
                new_excluded = excluded + (columns[index],)

                solution = solve_with_exclusions(new_excluded)

                if solution is None:
                    continue

                new_columns, new_cost = solution

                # Removing more columns cannot improve the minimum cost.
                # Therefore this entire branch can be discarded.
                if new_cost > optimal_cost:
                    continue

                new_columns = tuple(sorted(new_columns))

                if (
                    new_cost == optimal_cost
                    and new_columns not in discovered
                ):
                    discovered.add(new_columns)
                    results.append(list(new_columns))
                    queue.append(new_columns)

                # Continue exploring larger exclusion sets.
                explore_subsets(
                    index + 1,
                    new_excluded
                )

        explore_subsets(0, ())

    while queue:
        current = queue.pop(0)
        explore_column_set(current)

    return results
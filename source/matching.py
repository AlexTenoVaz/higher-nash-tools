import networkx as nx
import sympy as sp


def minimum_cost_matching(order_matrix, excluded_columns=None):
    """
    Compute a minimum-cost matching for an order matrix.

    The matching assigns every row to a distinct column.
    Entries equal to +oo are treated as unavailable edges.
    Entries equal to zero are valid edges with zero cost.

    Parameters
    ----------
    order_matrix : sympy.Matrix or array-like
        Matrix of weighted orders.
    excluded_columns : iterable of int, optional
        Zero-based column indices that must be excluded.

    Returns
    -------
    columns : list of int
        Zero-based column indices used by the optimal matching.
    cost : int
        Total cost of the matching.

    Raises
    ------
    ValueError
        If the matrix has no complete matching.
    """
    O = sp.Matrix(order_matrix)
    rows, cols = O.shape

    if excluded_columns is None:
        excluded_columns = set()
    else:
        excluded_columns = set(excluded_columns)

    graph = nx.DiGraph()

    source = "s"
    sink = "t"

    graph.add_node(source, demand=-rows)
    graph.add_node(sink, demand=rows)

    row_nodes = [f"r{i}" for i in range(rows)]
    col_nodes = [f"c{j}" for j in range(cols)]

    for node in row_nodes + col_nodes:
        graph.add_node(node, demand=0)

    for row in row_nodes:
        graph.add_edge(
            source,
            row,
            capacity=1,
            weight=0
        )

    for col in col_nodes:
        graph.add_edge(
            col,
            sink,
            capacity=1,
            weight=0
        )

    for i in range(rows):
        for j in range(cols):
            if j in excluded_columns:
                continue

            cost = sp.sympify(O[i, j])

            if cost == sp.oo:
                continue

            if not cost.is_integer:
                raise ValueError(
                    "Order matrix entries must be integers or +oo."
                )

            graph.add_edge(
                f"r{i}",
                f"c{j}",
                capacity=1,
                weight=int(cost)
            )

    try:
        cost, flow = nx.network_simplex(graph)
    except nx.NetworkXUnfeasible as exc:
        raise ValueError(
            "No complete matching exists for the given order matrix."
        ) from exc

    matching_columns = []

    for i in range(rows):
        row = f"r{i}"

        for column, flow_value in flow[row].items():
            if flow_value > 0 and column.startswith("c"):
                matching_columns.append(int(column[1:]))

    matching_columns.sort()

    if len(matching_columns) != rows:
        raise ValueError(
            "No complete matching exists for the given order matrix."
        )

    return matching_columns, int(cost)
import sympy as sp

from source.higher_jacobian_matrix import x, y, z, t


def curves_on_divisor(
        divisor,
        chart,
        fixed_polynomial=None,
        polynomial_1=None,
        polynomial_2=None,
        variables=None
        ):
    """
    Construct two parameterized curves in a chosen chart.

    Parameters
    ----------
    divisor : sympy.Expr
        Equation defining the divisor in the chosen chart.
    chart : int
        Zero-based index of the chart coordinate.
    fixed_polynomial : sympy.Expr, optional
        Parameterized value assigned to the chart coordinate.
    polynomial_1, polynomial_2 : sympy.Expr, optional
        Two parameterized values assigned to one of the remaining
        coordinates.
    variables : list or tuple of sympy.Symbol, optional
        Variables of the divisor.

    Returns
    -------
    tuple of list
        Two parameterized curves.
    """
    if variables is None:
        variables = [x, y, z]
    else:
        variables = list(variables)

    if len(variables) != 3:
        raise ValueError(
            "This implementation currently requires exactly three variables."
        )

    if not 0 <= chart < len(variables):
        raise ValueError(
            "chart must be a valid zero-based variable index."
        )

    if fixed_polynomial is None:
        fixed_polynomial = t**2

    if polynomial_1 is None:
        polynomial_1 = t + 1

    if polynomial_2 is None:
        polynomial_2 = t - 1

    non_chart_variables = [
        variable
        for index, variable in enumerate(variables)
        if index != chart
    ]

    solve_variable = non_chart_variables[0]
    parameterized_variable = non_chart_variables[1]

    curves = []

    for parameterized_value in (polynomial_1, polynomial_2):
        substitution = {
            variables[chart]: fixed_polynomial,
            parameterized_variable: parameterized_value,
        }

        solutions = sp.solve(
            divisor.subs(substitution),
            solve_variable
        )

        if not solutions:
            raise ValueError(
                "No symbolic solution was found for the chosen curve data."
            )

        solve_value = solutions[0]

        curve = [None, None, None]
        curve[chart] = fixed_polynomial

        for index, variable in enumerate(variables):
            if index == chart:
                continue

            if variable == parameterized_variable:
                curve[index] = parameterized_value
            elif variable == solve_variable:
                curve[index] = solve_value

        curves.append(curve)

    return curves[0], curves[1]


def descend_curve(curve, blowup_sequence):
    """
    Descend a parameterized curve through a sequence of blow-ups.

    The transformations are applied from the last blow-up to the
    first one.

    Parameters
    ----------
    curve : list of sympy.Expr
        Parameterized curve in a resolution chart.
    blowup_sequence : list of int
        Zero-based chart indices.

    Returns
    -------
    list of sympy.Expr
        Descended parameterized curve.
    """
    descended_curve = list(curve)

    for chart_index in reversed(blowup_sequence):
        if not 0 <= chart_index < len(descended_curve):
            raise ValueError(
                "Every blow-up index must be a valid coordinate index."
            )

        multiplier = descended_curve[chart_index]

        for index in range(len(descended_curve)):
            if index != chart_index:
                descended_curve[index] = sp.expand(
                    descended_curve[index] * multiplier
                )
                descended_curve[index] = sp.simplify(
                    descended_curve[index]
                )

    return descended_curve
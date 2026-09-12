import sympy as sp

from source.higher_jacobian_matrix import x, y, z
from source.curves import curves_on_divisor, descend_curve
from source.minors import maximal_minor
from source.pipeline import (
    build_pipeline,
    separate_curves,
)


def test_e6_pipeline_api():
    t = sp.symbols("t")

    f = x**2 + y**3 + z**4

    divisor = x + sp.I*z

    curve_1_on_divisor, curve_2_on_divisor = curves_on_divisor(
        divisor=divisor,
        chart=1,
        fixed_polynomial=t**2,
        polynomial_1=t + 1,
        polynomial_2=t - 1
    )

    curve_1 = descend_curve(
        curve_1_on_divisor,
        [2, 1, 1]
    )

    curve_2 = descend_curve(
        curve_2_on_divisor,
        [2, 1, 1]
    )

    result = build_pipeline(
        f,
        curve_1,
        curve_2,
        t,
        jac_order=5
    )

    assert result["exponents_1"] == [8, 6, 4]
    assert result["exponents_2"] == [8, 6, 4]

    assert result["order_matrix"].shape == (35, 55)

    assert result["optimal_cost"] == 226

    assert len(result["optimal_column_sets"]) == 66


def test_a2_pipeline_api_with_geometric_minor():
    t = sp.symbols("t")
    I = sp.I

    f = x**2 + y**2 + z**3

    curve_1 = [
        -t*(t + I)*(t + 2*I),
        -t*(t + 2*I),
        -t*(t + 2*I)
    ]

    curve_2 = [
        t*(t + I)*(t + 2*I),
        t*(t + 2*I),
        -t*(t + 2*I)
    ]

    result = build_pipeline(
        f,
        curve_1,
        curve_2,
        t,
        jac_order=2
    )

    assert result["exponents_1"] == [1, 1, 1]
    assert result["exponents_2"] == [1, 1, 1]

    assert result["order_matrix"].shape == (4, 9)

    assert result["optimal_cost"] == 3

    assert len(result["optimal_column_sets"]) == 2

    # The minimum-cost candidates alone do not separate
    # the two curves.
    automatic_result = separate_curves(
        result,
        curve_1,
        curve_2,
        t
    )

    assert automatic_result["separated"] is False

    # Additional geometrically selected minor from the A2
    # argument in the thesis.
    geometric_minor = maximal_minor(
        result["higher_jacobian"],
        [0, 3, 4, 6]
    )

    assert sp.factor(geometric_minor) == 16*x**4

    # Adding the geometrically relevant minor gives
    # Grassmannian separation.
    final_result = separate_curves(
        result,
        curve_1,
        curve_2,
        t,
        additional_minors=[geometric_minor]
    )

    assert final_result["separated"] is True
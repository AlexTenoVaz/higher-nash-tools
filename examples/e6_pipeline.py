import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp

from source.higher_jacobian_matrix import x, y, z
from source.curves import curves_on_divisor, descend_curve
from source.pipeline import build_pipeline

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

curve_1 = descend_curve(curve_1_on_divisor, [2, 1, 1])
curve_2 = descend_curve(curve_2_on_divisor, [2, 1, 1])

print("Curve 1:")
print(curve_1)

print("\nCurve 2:")
print(curve_2)

result = build_pipeline(
    f,
    curve_1,
    curve_2,
    t,
    jac_order=5
)

print("\nLeading exponents:")
print(result["exponents_1"])

print("\nOrder matrix shape:")
print(result["order_matrix"].shape)

print("\nOptimal matching cost:")
print(result["optimal_cost"])

print("\nNumber of optimal column sets:")
print(len(result["optimal_column_sets"]))

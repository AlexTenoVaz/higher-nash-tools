# Higher Nash Tools
Computational tools for the study of higher Nash modifications and singularities.
This repository contains:

- Higher Jacobian matrices
- Curves on exceptional divisors
- Curve descent algorithms
- Order matrices
- Minimum-cost matching
- Enumeration of optimal minors
- Examples for ADE singularities

## Quick Example

```python
from jacobian import *

f = x**2 + y**3 + z**4

J = jacobian_matrix(
    f,
    jac_order_local=5
)

print(J)
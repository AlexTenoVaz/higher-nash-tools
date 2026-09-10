import sympy as sp

from source.higher_jacobian_matrix import x, y, z


def _substitute_curve(expression, curve, variables):
    """Evaluate an expression along a parameterized curve."""
    substitution = {
        variable: coordinate
        for variable, coordinate in zip(variables, curve)
    }

    return sp.simplify(
        expression.subs(substitution)
    )


def grassmannian_separation(
        curve_1,
        curve_2,
        minors,
        parameter,
        variables=None
        ):
    """
    Test whether two parameterized curves are separated by
    a pair of candidate maximal minors.

    Parameters
    ----------
    curve_1, curve_2 : list or tuple of sympy.Expr
        Parameterized curves.
    minors : list of sympy.Expr
        Candidate maximal minors.
    parameter : sympy.Symbol
        Parameter used in the curves.
    variables : list or tuple of sympy.Symbol, optional
        Variables appearing in the minors and curves.
        Defaults to (x, y, z).

    Returns
    -------
    dict
        Dictionary with the separation result.

        If a separating pair is found:

        {
            "separated": True,
            "pair": (p, q),
            "limits_curve_1": L1,
            "limits_curve_2": L2
        }

        Otherwise:

        {
            "separated": False,
            "pair": None,
            "limits_curve_1": None,
            "limits_curve_2": None
        }

    Notes
    -----
    A pair (M_p, M_q) separates the two curves when M_p is
    nonzero away from t=0 near the origin and

        lim_{t->0} M_q(gamma_1(t)) / M_p(gamma_1(t))

    and

        lim_{t->0} M_q(gamma_2(t)) / M_p(gamma_2(t))

    are finite and different.
    """
    if variables is None:
        variables = (x, y, z)
    else:
        variables = tuple(variables)

    if len(curve_1) != len(curve_2):
        raise ValueError(
            "The two curves must have the same dimension."
        )

    if len(curve_1) != len(variables):
        raise ValueError(
            "The curve dimension must match the number of variables."
        )

    if not minors:
        return {
            "separated": False,
            "pair": None,
            "limits_curve_1": None,
            "limits_curve_2": None,
        }

    for p, minor_p in enumerate(minors):
        mp1 = _substitute_curve(
            minor_p,
            curve_1,
            variables
        )

        mp2 = _substitute_curve(
            minor_p,
            curve_2,
            variables
        )

        # A denominator that is identically zero on either curve
        # cannot define the required affine chart.
        if mp1 == 0 or mp2 == 0:
            continue

        for q, minor_q in enumerate(minors):
            if p == q:
                continue

            mq1 = _substitute_curve(
                minor_q,
                curve_1,
                variables
            )

            mq2 = _substitute_curve(
                minor_q,
                curve_2,
                variables
            )

            try:
                l1 = sp.simplify(
                    sp.limit(
                        mq1 / mp1,
                        parameter,
                        0
                    )
                )

                l2 = sp.simplify(
                    sp.limit(
                        mq2 / mp2,
                        parameter,
                        0
                    )
                )
            except (ValueError, NotImplementedError):
                continue

            if (
                l1.is_finite is True
                and l2.is_finite is True
                and sp.simplify(l1 - l2) != 0
            ):
                return {
                    "separated": True,
                    "pair": (p, q),
                    "limits_curve_1": l1,
                    "limits_curve_2": l2,
                }

    return {
        "separated": False,
        "pair": None,
        "limits_curve_1": None,
        "limits_curve_2": None,
    }
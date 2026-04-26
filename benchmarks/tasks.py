"""Benchmark task catalogue for the cross-CAS comparison driver.

PA-4 — Cross-CAS benchmark driver.

Each task class exposes:
  - ``name: str``
  - ``size_params: list[int]``  (the "size" axis — degree, dimension, …)
  - ``run_alkahest(size) -> object``
  - ``run_sympy(size) -> object``   (optional; skipped if SymPy not installed)
  - ``expected_result(size) -> object | None``  (for correctness cross-checks)

Add new tasks by subclassing :class:`BenchTask`.
"""

from __future__ import annotations

import abc
import math
from typing import Any


class BenchTask(abc.ABC):
    """Abstract base class for a benchmark task."""

    name: str = ""
    size_params: list[int] = [5, 10, 20]

    @abc.abstractmethod
    def run_alkahest(self, size: int) -> Any:
        """Run the task using Alkahest and return the result."""

    def run_sympy(self, size: int) -> Any:  # noqa: ARG002
        """Run the task using SymPy (optional)."""
        raise NotImplementedError

    def expected_result(self, size: int) -> Any:  # noqa: ARG002
        """Return the expected result for a correctness cross-check."""
        return None


# ---------------------------------------------------------------------------
# Task 1 — Polynomial differentiation: d/dx of x^n + x^{n-1} + … + 1
# ---------------------------------------------------------------------------


class DegreeNPolyDiff(BenchTask):
    """Differentiate a degree-N polynomial."""

    name = "poly_diff"
    size_params = [10, 50, 100, 200]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        # Build 1 + x + x^2 + … + x^size
        terms = [p.integer(1)]
        for k in range(1, size + 1):
            terms.append(x ** k)
        poly = sum(terms[1:], terms[0])
        result = alkahest.diff(poly, x)
        return result.value

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        poly = sum(x**k for k in range(size + 1))
        return sp.diff(poly, x)

    def expected_result(self, size: int) -> Any:
        # Leading coefficient should be `size`
        return size


# ---------------------------------------------------------------------------
# Task 2 — Polynomial GCD
# ---------------------------------------------------------------------------


class DegreeNPolyGCD(BenchTask):
    """GCD of two degree-N polynomials that share a common factor."""

    name = "poly_gcd"
    size_params = [5, 10, 20, 40]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        # Build (x-1)^size  and  (x-1)^(size//2) * (x+1)^(size//2)
        # GCD should be (x-1)^(size//2)
        one = p.integer(1)
        xm1 = p.add([x, p.mul([p.integer(-1), one])])  # x - 1
        # use poly conversion
        poly_a = alkahest.UniPoly.from_symbolic(
            alkahest.poly_normal(x ** size - one, [x]).value
            if hasattr(alkahest, "poly_normal")
            else x ** size,
            x,
        )
        poly_b = alkahest.UniPoly.from_symbolic(
            alkahest.poly_normal(x ** (size // 2) - one, [x]).value
            if hasattr(alkahest, "poly_normal")
            else x ** (size // 2),
            x,
        )
        return alkahest.UniPoly.gcd(poly_a, poly_b)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        a = x**size - 1
        b = x ** (size // 2) - 1
        return sp.gcd(a, b)


# ---------------------------------------------------------------------------
# Task 3 — Rational simplification
# ---------------------------------------------------------------------------


class RationalSimplification(BenchTask):
    """Simplify (x^n - 1) / (x - 1) to 1 + x + … + x^{n-1}."""

    name = "rational_simplify"
    size_params = [5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        numer_uni = alkahest.UniPoly.from_coefficients([p.integer(-1)] + [p.integer(0)] * (size - 1) + [p.integer(1)], x)
        denom_uni = alkahest.UniPoly.from_coefficients([p.integer(-1), p.integer(1)], x)
        rf = alkahest.RationalFunction(numer_uni, denom_uni)
        return rf

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        expr = (x**size - 1) / (x - 1)
        return sp.simplify(expr)


# ---------------------------------------------------------------------------
# Task 4 — Trigonometric identity simplification: sin²(x) + cos²(x) → 1
# ---------------------------------------------------------------------------


class TrigIdentitySimplify(BenchTask):
    """Simplify N copies of sin²(x) + cos²(x) down to N."""

    name = "trig_identity"
    size_params = [1, 5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        expr = p.integer(0)
        for _ in range(size):
            sin2 = alkahest.sin(x) ** 2
            cos2 = alkahest.cos(x) ** 2
            expr = expr + sin2 + cos2
        return alkahest.simplify_trig(expr)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        expr = sum(sp.sin(x) ** 2 + sp.cos(x) ** 2 for _ in range(size))
        return sp.trigsimp(expr)

    def expected_result(self, size: int) -> Any:
        return size


# ---------------------------------------------------------------------------
# Task 5 — Jacobian of an N×N system
# ---------------------------------------------------------------------------


class JacobianNxN(BenchTask):
    """Compute the Jacobian of an N-variable polynomial system."""

    name = "jacobian_nxn"
    size_params = [3, 5, 8, 10]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        xs = [p.symbol(f"x{i}") for i in range(size)]
        # f_i = x_i^2 + x_{i-1} * x_i  (wraps around)
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        mat = alkahest.Matrix(fns)
        return alkahest.jacobian(mat, xs)

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        xs = [sp.Symbol(f"x{i}") for i in range(size)]
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        return sp.Matrix(fns).jacobian(xs)


# ---------------------------------------------------------------------------
# Task 6 — Ball arithmetic: rigorous sin(cos(x)) at a narrow interval
# ---------------------------------------------------------------------------


class BallSinCos(BenchTask):
    """Rigorous evaluation of sin(cos(x)) at x ∈ [1 ± ε], varying ε."""

    name = "ball_sin_cos"
    size_params = [1, 10, 100, 1000]  # size = 1/ε factor (larger = tighter ball)

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        expr = alkahest.sin(alkahest.cos(x))
        radius = 1.0 / size
        ball = alkahest.ArbBall.from_midpoint_radius(1.0, radius, 128)
        ev = alkahest.interval_eval(expr, {x: ball})
        return ev

    def expected_result(self, size: int) -> Any:
        import math

        return math.sin(math.cos(1.0))


# ---------------------------------------------------------------------------
# Task 7 — JIT compilation and evaluation of a polynomial
# ---------------------------------------------------------------------------


class ODEJITCompile(BenchTask):
    """Compile a degree-N polynomial and evaluate it at 10^6 points."""

    name = "poly_jit_eval"
    size_params = [5, 10, 20]

    def run_alkahest(self, size: int) -> Any:
        import numpy as np
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        terms = [x**k for k in range(size + 1)]
        poly = terms[0]
        for t in terms[1:]:
            poly = poly + t
        compiled = alkahest.compile_expr(poly, [x])
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return alkahest.numpy_eval(compiled, xs)

    def run_sympy(self, size: int) -> Any:
        import numpy as np
        import sympy as sp  # type: ignore[import]

        x = sp.Symbol("x")
        poly = sum(x**k for k in range(size + 1))
        fn = sp.lambdify(x, poly, "numpy")
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return fn(xs)


# ---------------------------------------------------------------------------
# Task 8 — V1-4 polynomial system solver (circle ∩ line)
# ---------------------------------------------------------------------------


class SolveCircleLine(BenchTask):
    """Solve the intersection of an N-circle with the line y = x.

    Each size yields a 2-equation, 2-variable system whose Gröbner basis
    triangularises to a univariate quadratic.  The circle radius scales
    with ``size`` so the basis computation has meaningfully different
    coefficients per run; the number of real solutions stays at 2.
    """

    name = "solve_circle_line"
    size_params = [1, 2, 5, 10]

    def run_alkahest(self, size: int) -> Any:
        import alkahest

        p = alkahest.ExprPool()
        x = p.symbol("x")
        y = p.symbol("y")
        neg_one = p.integer(-1)
        r2 = p.integer(size * size)
        # x² + y² - r²
        eq1 = x ** 2 + y ** 2 + neg_one * r2
        # y - x
        eq2 = y + neg_one * x
        return alkahest.solve([eq1, eq2], [x, y])

    def run_sympy(self, size: int) -> Any:
        import sympy as sp  # type: ignore[import]

        x, y = sp.symbols("x y")
        return sp.solve([x ** 2 + y ** 2 - size ** 2, y - x], [x, y], dict=True)

    def expected_result(self, size: int) -> Any:
        return 2  # number of real solutions


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ALL_TASKS: list[BenchTask] = [
    DegreeNPolyDiff(),
    TrigIdentitySimplify(),
    JacobianNxN(),
    BallSinCos(),
    ODEJITCompile(),
    SolveCircleLine(),
]

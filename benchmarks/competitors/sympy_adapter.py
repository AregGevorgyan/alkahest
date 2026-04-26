"""SymPy adapter for cross-CAS benchmarks (V1-13)."""

from __future__ import annotations

from typing import Any

from .base import CASAdapter


class SymPyAdapter(CASAdapter):
    """Adapter wrapping SymPy."""

    name = "SymPy"

    def is_available(self) -> bool:
        try:
            import sympy  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Task-named benchmark methods (match tasks.py task names) ─────────────

    def bench_poly_diff(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = sum(x ** k for k in range(size + 1))
        return str(sp.diff(expr, x))

    def bench_trig_identity(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = sum(sp.sin(x) ** 2 + sp.cos(x) ** 2 for _ in range(size))
        return str(sp.trigsimp(expr))

    def bench_jacobian_nxn(self, size: int) -> Any:
        import sympy as sp
        xs = [sp.Symbol(f"x{i}") for i in range(size)]
        fns = [xs[i] ** 2 + xs[(i - 1) % size] * xs[i] for i in range(size)]
        jac = sp.Matrix(fns).jacobian(xs)
        return [[str(jac[i, j]) for j in range(size)] for i in range(size)]

    def bench_ball_sin_cos(self, size: int) -> Any:
        import mpmath
        mpmath.mp.prec = 128
        iv = mpmath.iv
        radius = 1.0 / size
        x_iv = iv.mpf([1.0 - radius, 1.0 + radius])
        result = iv.sin(iv.cos(x_iv))
        return str(result)

    def bench_poly_jit_eval(self, size: int) -> Any:
        import sympy as sp
        import numpy as np
        x = sp.Symbol("x")
        poly = sum(x ** k for k in range(size + 1))
        fn = sp.lambdify(x, poly, "numpy")
        xs = np.linspace(0.0, 1.0, 1_000_000)
        return fn(xs)

    def bench_solve_circle_line(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        return [str(s) for s in sp.solve(
            [x ** 2 + y ** 2 - size ** 2, y - x], [x, y], dict=True
        )]

    # ── Legacy names ─────────────────────────────────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.integrate(x ** size, x))

    def bench_simplify(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        expr = (sp.sin(x) ** 2 + sp.cos(x) ** 2) ** size
        return str(sp.simplify(expr))

    def bench_diff(self, size: int) -> Any:
        return self.bench_poly_diff(size)

    def bench_groebner(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        polys = [x ** 2 + y ** 2 - 1, x - y]
        return [str(p) for p in sp.groebner(polys, [x, y], order="lex")]

    def bench_poly_gcd(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return str(sp.gcd(x ** size - 1, x ** (size // 2) - 1))

    def bench_jacobian(self, size: int) -> Any:
        import sympy as sp
        x, y = sp.symbols("x y")
        exprs = [sp.sin(x * y), sp.cos(x + y), sp.exp(x - y)]
        jac = sp.Matrix(exprs).jacobian([x, y])
        return [[str(jac[i, j]) for j in range(2)] for i in range(3)]

    def bench_polynomial_solve(self, size: int) -> Any:
        import sympy as sp
        x = sp.Symbol("x")
        return [str(s) for s in sp.solve(x ** 2 - size, x)]

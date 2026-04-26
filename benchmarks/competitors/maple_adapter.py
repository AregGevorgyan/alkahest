"""Maple adapter for cross-CAS benchmarks (V1-13).

Communicates with Maple via subprocess (``maple -q``).
Requires Maple to be installed and ``maple`` on PATH.

Usage:
    from benchmarks.competitors import MapleAdapter
    adapter = MapleAdapter()
    if adapter.is_available():
        print(adapter.bench_integrate(10))
"""

from __future__ import annotations

import subprocess
import textwrap
from typing import Any

from .base import CASAdapter


def _run_maple(code: str, timeout: float = 30.0) -> str:
    """Execute ``code`` in a non-interactive Maple session and capture output."""
    script = textwrap.dedent(f"""\
        interface(quiet=true):
        {code}
        quit:
    """)
    result = subprocess.run(
        ["maple", "-q"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout.strip()


class MapleAdapter(CASAdapter):
    """Adapter wrapping Maple via subprocess."""

    name = "Maple"

    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["maple", "-v"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── Task-named benchmark methods ─────────────────────────────────────────

    def bench_poly_diff(self, size: int) -> Any:
        poly = " + ".join(f"x^{k}" for k in range(size + 1))
        return _run_maple(f"diff({poly}, x);")

    def bench_trig_identity(self, size: int) -> Any:
        terms = " + ".join(["sin(x)^2 + cos(x)^2"] * size)
        return _run_maple(f"simplify({terms});")

    def bench_jacobian_nxn(self, size: int) -> Any:
        vars_list = ", ".join(f"x{i}" for i in range(size))
        fns = "[" + ", ".join(
            f"x{i}^2 + x{(i - 1) % size}*x{i}" for i in range(size)
        ) + "]"
        return _run_maple(
            f"with(VectorCalculus): Jacobian({fns}, [{vars_list}]);"
        )

    def bench_solve_circle_line(self, size: int) -> Any:
        return _run_maple(
            f"solve({{x^2 + y^2 = {size}^2, y = x}}, {{x, y}});"
        )

    # ── Legacy names ─────────────────────────────────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        return _run_maple(f"int(x^{size}, x);")

    def bench_simplify(self, size: int) -> Any:
        return _run_maple(f"simplify((sin(x)^2 + cos(x)^2)^{size});")

    def bench_diff(self, size: int) -> Any:
        return self.bench_poly_diff(size)

    def bench_groebner(self, size: int) -> Any:
        return _run_maple(
            "with(Groebner): Basis([x^2+y^2-1, x-y], tdeg(x,y));"
        )

    def bench_poly_gcd(self, size: int) -> Any:
        return _run_maple(f"gcd(x^{size}-1, x^{size//2}-1);")

    def bench_polynomial_solve(self, size: int) -> Any:
        return _run_maple(f"solve(x^2={size}, x);")

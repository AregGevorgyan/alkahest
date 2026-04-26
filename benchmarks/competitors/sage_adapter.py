"""SageMath adapter for cross-CAS benchmarks (V1-13).

Requires SageMath to be installed as a Python package (``pip install sagemath-standard``
on Ubuntu) or the ``sage`` command to be available.

We use the SageMath Python API directly (``from sage.all import *``) which is
available when SageMath is installed as a Python package.  If only the ``sage``
command-line is available, we fall back to subprocess execution.
"""

from __future__ import annotations

import subprocess
import json
from typing import Any

from .base import CASAdapter


class SageAdapter(CASAdapter):
    """Adapter wrapping SageMath."""

    name = "SageMath"

    def is_available(self) -> bool:
        # Try Python API first (faster)
        try:
            import sage.all  # noqa: F401
            return True
        except ImportError:
            pass
        # Fall back to CLI
        try:
            result = subprocess.run(
                ["sage", "--version"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _sage(self, code: str) -> str:
        """Execute ``code`` in a Sage session and return printed output."""
        try:
            import sage.all as sage
            result = eval(code, {"sage": sage, **vars(sage)})  # noqa: S307
            return str(result)
        except Exception:
            pass
        # CLI fallback
        script = f"print({code})"
        result = subprocess.run(
            ["sage", "--python", "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()

    def bench_integrate(self, size: int) -> Any:
        try:
            from sage.all import SR, var, integrate
            x = var("x")
            return str(integrate(x ** size, x))
        except ImportError:
            return self._sage(f"var('x'); integrate(x^{size}, x)")

    def bench_diff(self, size: int) -> Any:
        try:
            from sage.all import var, sum as sage_sum, diff
            x = var("x")
            expr = sage_sum(x ** k for k in range(size + 1))
            return str(diff(expr, x))
        except ImportError:
            return self._sage(
                f"var('x'); diff(sum(x^k for k in range({size+1})), x)"
            )

    def bench_groebner(self, size: int) -> Any:
        try:
            from sage.all import QQ, PolynomialRing
            R = PolynomialRing(QQ, ["x", "y"], order="lex")
            x, y = R.gens()
            I = R.ideal([x ** 2 + y ** 2 - 1, x - y])
            return [str(g) for g in I.groebner_basis()]
        except ImportError:
            return self._sage(
                "R.<x,y> = QQ[]; I = R.ideal(x^2+y^2-1, x-y); I.groebner_basis()"
            )

    def bench_polynomial_solve(self, size: int) -> Any:
        try:
            from sage.all import var, solve
            x = var("x")
            return [str(s) for s in solve(x ** 2 == size, x)]
        except ImportError:
            return self._sage(f"var('x'); solve(x^2=={size}, x)")

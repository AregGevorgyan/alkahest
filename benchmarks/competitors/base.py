"""Base class for CAS competitor adapters (V1-13)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskResult:
    """Result of running a single benchmark task."""

    system: str
    task: str
    size: int
    wall_ms: float | None
    result: Any = None
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "task": self.task,
            "size": self.size,
            "wall_ms": self.wall_ms,
            "ok": self.ok,
            "error": self.error,
        }


class CASAdapter:
    """Abstract base class for a CAS competitor adapter.

    Sub-class and implement:
    - ``name: str`` — display name, e.g. ``"SymPy"``
    - ``is_available() -> bool`` — True if the system can be imported/launched
    - ``integrate(expr_str, var_str) -> str`` — symbolic integration
    - ``simplify(expr_str) -> str`` — algebraic simplification
    - ``diff(expr_str, var_str) -> str`` — differentiation
    - ``groebner(polys, vars) -> list[str]`` — Gröbner basis
    - ``poly_gcd(p, q, var) -> str`` — polynomial GCD
    - ``jacobian(exprs, vars) -> list[list[str]]`` — Jacobian matrix

    Methods that are not supported should raise ``NotImplementedError``.
    """

    name: str = "unknown"

    def is_available(self) -> bool:
        return False

    def _time(self, fn, *args, **kwargs) -> tuple[Any, float]:
        """Run ``fn(*args, **kwargs)`` and return ``(result, wall_ms)``."""
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return result, wall_ms

    def run_task(self, task_name: str, size: int = 10) -> TaskResult:
        """Dispatch to the appropriate method and record timing."""
        method = getattr(self, f"bench_{task_name}", None)
        if method is None:
            return TaskResult(
                system=self.name,
                task=task_name,
                size=size,
                wall_ms=None,
                ok=False,
                error="not_implemented",
            )
        try:
            result, wall_ms = self._time(method, size)
            return TaskResult(
                system=self.name,
                task=task_name,
                size=size,
                wall_ms=wall_ms,
                result=result,
                ok=True,
            )
        except Exception as e:
            return TaskResult(
                system=self.name,
                task=task_name,
                size=size,
                wall_ms=None,
                ok=False,
                error=str(e),
            )

    # ── Benchmark methods keyed by task name (override in sub-classes) ───────
    # Names must match the ``name`` attribute of tasks in ``tasks.py``.

    def bench_poly_diff(self, size: int) -> Any:
        raise NotImplementedError

    def bench_trig_identity(self, size: int) -> Any:
        raise NotImplementedError

    def bench_jacobian_nxn(self, size: int) -> Any:
        raise NotImplementedError

    def bench_ball_sin_cos(self, size: int) -> Any:
        raise NotImplementedError

    def bench_poly_jit_eval(self, size: int) -> Any:
        raise NotImplementedError

    def bench_solve_circle_line(self, size: int) -> Any:
        raise NotImplementedError

    # ── Legacy names kept for backwards compat ───────────────────────────────

    def bench_integrate(self, size: int) -> Any:
        raise NotImplementedError

    def bench_simplify(self, size: int) -> Any:
        raise NotImplementedError

    def bench_diff(self, size: int) -> Any:
        raise NotImplementedError

    def bench_groebner(self, size: int) -> Any:
        raise NotImplementedError

    def bench_poly_gcd(self, size: int) -> Any:
        raise NotImplementedError

    def bench_jacobian(self, size: int) -> Any:
        raise NotImplementedError

    def bench_polynomial_solve(self, size: int) -> Any:
        raise NotImplementedError

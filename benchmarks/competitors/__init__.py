"""Cross-CAS competitor adapters (V1-13).

Each adapter wraps a CAS system with a uniform `.run_task(task_name, size)` interface.
Missing systems degrade gracefully — their results are omitted from reports.

Available adapters
------------------
- :class:`SymPyAdapter`      — always available if ``sympy`` is installed
- :class:`SageAdapter`       — available if SageMath Python API is installed
- :class:`MathematicaAdapter`— available if ``wolframclient`` + Wolfram Engine is installed
- :class:`MapleAdapter`      — available if Maple is accessible via subprocess

Usage
-----
::

    from benchmarks.competitors import get_available_adapters
    for adapter in get_available_adapters():
        result = adapter.integrate("x**2", "x")
        print(adapter.name, result)
"""

from __future__ import annotations

from .base import CASAdapter, TaskResult  # noqa: F401
from .sympy_adapter import SymPyAdapter  # noqa: F401
from .sage_adapter import SageAdapter  # noqa: F401
from .mathematica_adapter import MathematicaAdapter  # noqa: F401
from .maple_adapter import MapleAdapter  # noqa: F401
from .symengine_adapter import SymEngineAdapter  # noqa: F401


def get_available_adapters() -> list[CASAdapter]:
    """Return competitor adapters whose backend is importable / runnable.

    SymPy is excluded here because it is already benchmarked via the tasks'
    ``run_sympy()`` methods in ``cas_comparison.py``; including it here would
    double-count it when ``--competitors`` is active.
    """
    candidates: list[CASAdapter] = [
        SageAdapter(),
        MathematicaAdapter(),
        MapleAdapter(),
        SymEngineAdapter(),
    ]
    return [a for a in candidates if a.is_available()]


__all__ = [
    "CASAdapter",
    "TaskResult",
    "SymPyAdapter",
    "SageAdapter",
    "MathematicaAdapter",
    "MapleAdapter",
    "SymEngineAdapter",
    "get_available_adapters",
]

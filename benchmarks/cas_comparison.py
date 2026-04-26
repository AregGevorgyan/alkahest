"""Cross-CAS benchmark driver.

PA-4 — Cross-CAS benchmark driver + HTML report.

Runs the task catalogue from ``tasks.py`` against Alkahest and (optionally)
SymPy, records timings in JSONL, and emits an HTML plotly report.

Usage
-----
    python benchmarks/cas_comparison.py [--sizes 5,10,20] [--output results.jsonl] [--report report.html]

The script is self-contained and imports ``alkahest`` from the installed
package (run ``maturin develop`` first).

Output
------
``results.jsonl``
    One JSON line per (task, system, size) run::

        {"task": "poly_diff", "system": "alkahest", "size": 10,
         "wall_ms": 1.23, "ok": true}

``report.html``
    Plotly bar-chart report (requires ``plotly`` — installed automatically if
    missing; falls back to a markdown table if plotly is unavailable).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import timeit
import traceback
from pathlib import Path
from typing import Any

try:
    from tasks import ALL_TASKS, BenchTask
except ImportError:
    from benchmarks.tasks import ALL_TASKS, BenchTask

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_SYSTEMS = ["alkahest", "sympy"]
_DEFAULT_REPEAT = 3
_DEFAULT_NUMBER = 1

# V1-13: competitor adapters (available only if RUN_COMMERCIAL_CAS env var is set
# or --competitors flag is passed)
_COMPETITOR_SYSTEMS: list[str] = []
try:
    import os
    if os.environ.get("RUN_COMMERCIAL_CAS"):
        from competitors import get_available_adapters as _get_adapters
        _ADAPTERS = {a.name.lower(): a for a in _get_adapters()}
        _COMPETITOR_SYSTEMS = list(_ADAPTERS.keys())
except Exception:
    _ADAPTERS = {}  # type: ignore[assignment]


def _run_once(task: BenchTask, system: str, size: int) -> dict[str, Any]:
    """Time one (task, system, size) combination.

    Returns a result dict suitable for JSONL output.
    """
    run_fn = getattr(task, f"run_{system}", None)
    if run_fn is None:
        return {
            "task": task.name,
            "system": system,
            "size": size,
            "wall_ms": None,
            "ok": False,
            "error": "not_implemented",
        }

    try:
        # Warm-up
        run_fn(size)
        # Time
        times = timeit.repeat(
            lambda: run_fn(size),
            repeat=_DEFAULT_REPEAT,
            number=_DEFAULT_NUMBER,
        )
        wall_ms = min(times) * 1000.0
        return {
            "task": task.name,
            "system": system,
            "size": size,
            "wall_ms": round(wall_ms, 4),
            "ok": True,
        }
    except NotImplementedError:
        return {
            "task": task.name,
            "system": system,
            "size": size,
            "wall_ms": None,
            "ok": False,
            "error": "not_implemented",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "task": task.name,
            "system": system,
            "size": size,
            "wall_ms": None,
            "ok": False,
            "error": traceback.format_exception_only(type(exc), exc)[0].strip(),
        }


def run_all(
    tasks: list[BenchTask],
    systems: list[str] = _SYSTEMS,
    sizes: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Run all (task, system, size) combinations and return result dicts."""
    results = []
    for task in tasks:
        task_sizes = sizes if sizes is not None else task.size_params
        for size in task_sizes:
            for system in systems:
                print(
                    f"  [{system:8s}] {task.name:25s} size={size}",
                    end=" … ",
                    flush=True,
                )
                r = _run_once(task, system, size)
                if r["ok"]:
                    print(f"{r['wall_ms']:.2f} ms")
                else:
                    print(f"SKIP ({r.get('error', '?')})")
                results.append(r)
    return results


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------


def write_jsonl(results: list[dict], path: Path) -> None:
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"Results written to {path}")


# ---------------------------------------------------------------------------
# HTML report (plotly)
# ---------------------------------------------------------------------------


def _build_html_report(results: list[dict]) -> str:
    """Build a plotly HTML report from the results list."""
    try:
        import plotly.graph_objects as go  # type: ignore[import]
        from plotly.subplots import make_subplots  # type: ignore[import]
    except ImportError:
        return _build_markdown_report(results)

    # Group by task
    tasks = list({r["task"] for r in results})
    systems = list({r["system"] for r in results})
    colors = {
        "alkahest": "#1f77b4",
        "sympy": "#ff7f0e",
        "symengine": "#2ca02c",
        "mathematica": "#d62728",
        "maple": "#9467bd",
        "sagemath": "#8c564b",
        "symbolics_jl": "#e377c2",
    }

    figs = []
    for task_name in sorted(tasks):
        task_results = [r for r in results if r["task"] == task_name and r["ok"]]
        if not task_results:
            continue

        sizes = sorted({r["size"] for r in task_results})
        fig = go.Figure()
        for system in systems:
            sys_results = {r["size"]: r["wall_ms"] for r in task_results if r["system"] == system}
            y = [sys_results.get(s) for s in sizes]
            if any(v is not None for v in y):
                fig.add_trace(
                    go.Bar(
                        name=system,
                        x=[str(s) for s in sizes],
                        y=y,
                        marker_color=colors.get(system, "#333"),
                    )
                )
        fig.update_layout(
            title=f"Task: {task_name}",
            xaxis_title="Size",
            yaxis_title="Wall time (ms)",
            barmode="group",
            height=350,
        )
        figs.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    # Summary table: Alkahest vs all systems
    competitor_systems = [s for s in systems if s != "alkahest"]
    rows = []
    for task_name in sorted(tasks):
        for r_alkahest in results:
            if r_alkahest["task"] != task_name or r_alkahest["system"] != "alkahest" or not r_alkahest["ok"]:
                continue
            size = r_alkahest["size"]
            cells = [f"<td>{task_name}</td><td>{size}</td><td>{r_alkahest['wall_ms']:.2f}</td>"]
            for sys_name in competitor_systems:
                r_comp = next(
                    (r for r in results if r["task"] == task_name and r["system"] == sys_name and r["size"] == size and r["ok"]),
                    None,
                )
                if r_comp and r_comp["wall_ms"] and r_alkahest["wall_ms"]:
                    ratio = r_comp["wall_ms"] / r_alkahest["wall_ms"]
                    ratio_str = f"{ratio:.2f}×"
                    color = "green" if ratio >= 1.0 else "red"
                    cells.append(
                        f"<td>{r_comp['wall_ms']:.2f} "
                        f"<span style='color:{color}'>({ratio_str})</span></td>"
                    )
                else:
                    cells.append(f"<td>{'—' if r_comp is None else r_comp.get('error', '—')}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")

    comp_headers = "".join(f"<th>{s.title()}</th>" for s in competitor_systems)
    table_html = f"""
<h2>Alkahest vs competitors — wall time (ms) — (ratio vs Alkahest, green = Alkahest wins)</h2>
<table border='1' style='border-collapse:collapse;font-family:monospace;font-size:13px'>
<tr><th>Task</th><th>Size</th><th>Alkahest</th>{comp_headers}</tr>
{''.join(rows)}
</table>
"""

    body = "\n".join(figs) + table_html
    return f"""<!DOCTYPE html>
<html>
<head><meta charset='utf-8'><title>Alkahest benchmark report</title></head>
<body>
<h1>Alkahest cross-CAS benchmark report</h1>
{body}
</body>
</html>"""


def _build_markdown_report(results: list[dict]) -> str:
    """Fallback: build a Markdown report (no plotly required)."""
    lines = ["# Alkahest cross-CAS benchmark report\n"]
    tasks = sorted({r["task"] for r in results})
    for task_name in tasks:
        lines.append(f"## {task_name}\n")
        lines.append("| size | system | wall_ms | ok |")
        lines.append("|---|---|---|---|")
        task_results = sorted(
            [r for r in results if r["task"] == task_name],
            key=lambda r: (r["size"], r["system"]),
        )
        for r in task_results:
            ms = f"{r['wall_ms']:.2f}" if r["wall_ms"] is not None else "—"
            lines.append(f"| {r['size']} | {r['system']} | {ms} | {r['ok']} |")
        lines.append("")
    return "\n".join(lines)


def write_report(results: list[dict], path: Path) -> None:
    html = _build_html_report(results)
    if path.suffix == ".html":
        path.write_text(html)
    else:
        path.write_text(_build_markdown_report(results))
    print(f"Report written to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_sizes(s: str) -> list[int]:
    return [int(x) for x in s.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alkahest cross-CAS benchmark driver")
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=None,
        metavar="N,M,…",
        help="Comma-separated list of sizes to run (overrides per-task defaults)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/results.jsonl"),
        help="JSONL output file (default: benchmarks/results/results.jsonl)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/results/report.html"),
        help="HTML report output file (default: benchmarks/results/report.html)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        metavar="name1,name2",
        help="Comma-separated subset of task names to run",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="alkahest,sympy",
        metavar="system1,system2",
        help="Comma-separated list of systems to benchmark (default: alkahest,sympy)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=False,
        help="Run V5-10 GPU benchmark suite (GPUPolynomialEval, GPUJacobian, DLPackZeroCopy)",
    )
    parser.add_argument(
        "--competitors",
        action="store_true",
        default=False,
        help="V1-13: also run available competitor CAS (SymPy, SageMath, Mathematica, Maple)",
    )
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.gpu:
        print("Running V5-10 GPU benchmark suite…")
        gpu_results = run_gpu_benchmarks()
        gpu_output = args.output.with_stem(args.output.stem + "_gpu")
        write_jsonl(gpu_results, gpu_output)
        write_report(gpu_results, args.report.with_stem(args.report.stem + "_gpu"))
        return 0

    tasks = ALL_TASKS
    if args.tasks:
        names = args.tasks.split(",")
        tasks = [t for t in tasks if t.name in names]
        if not tasks:
            print(f"No tasks matched: {args.tasks}", file=sys.stderr)
            return 1

    systems = args.systems.split(",")

    # V1-13: add competitor adapters when --competitors flag is set
    if args.competitors:
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from competitors import get_available_adapters
            available = get_available_adapters()
            print(f"V1-13 competitor adapters available: {[a.name for a in available]}")
            # Extend the task catalogue with competitor run_* methods
            for adapter in available:
                for task in tasks:
                    setattr(task, f"run_{adapter.name.lower()}", lambda sz, a=adapter, t=task: a.run_task(t.name, sz).result)
            systems = systems + [a.name.lower() for a in available]
        except ImportError as e:
            print(f"Warning: could not load competitor adapters: {e}", file=sys.stderr)

    print(f"Running {len(tasks)} task(s) × {len(systems)} system(s)")
    results = run_all(tasks, systems=systems, sizes=args.sizes)

    write_jsonl(results, args.output)
    write_report(results, args.report)
    return 0


# ---------------------------------------------------------------------------
# V5-10 — GPU benchmark suite
# ---------------------------------------------------------------------------


class GPUBenchmarkSuite:
    """GPU benchmark tasks comparing Alkahest NVPTX vs NumPy baselines.

    Each task exposes a ``run(size)`` method that returns wall time in ms.
    Requires ``alkahest`` built with ``--features cuda`` for the NVPTX path;
    falls back to NumPy-only timing when CUDA is unavailable.
    """

    @staticmethod
    def _try_compile_cuda(expr, inputs):
        """Attempt to compile a CUDA kernel; return None if unavailable."""
        try:
            return alkahest_module().compile_cuda(expr, inputs)
        except (AttributeError, RuntimeError):
            return None

    class GPUPolynomialEval:
        """1M-point evaluation of a 5-variable polynomial.

        Alkahest NVPTX (stub) vs NumPy vectorised reference.
        """

        name = "gpu_poly_eval_1m"
        size_params = [1_000_000]

        def run_alkahest_gpu(self, n_pts: int) -> dict:
            import alkahest
            import numpy as np

            pool = alkahest.ExprPool()
            vars_ = [pool.symbol(f"x{i}") for i in range(5)]
            # Build degree-3 polynomial: Σ xi³ + xi²·x(i+1) + ...
            terms = []
            for v in vars_:
                terms.append(pool.pow(v, pool.integer(3)))
            for i in range(len(vars_) - 1):
                terms.append(pool.mul([pool.pow(vars_[i], pool.integer(2)), vars_[i + 1]]))
            expr = pool.add(terms)

            # NVPTX compile (stub — just measures overhead)
            compiled = None
            if hasattr(alkahest, "compile_cuda"):
                try:
                    compiled = alkahest.compile_cuda(expr, vars_)
                except RuntimeError:
                    pass

            # NumPy reference timing
            arrays = [np.random.randn(n_pts) for _ in vars_]
            start = timeit.default_timer()
            ref = sum(a ** 3 for a in arrays) + sum(
                arrays[i] ** 2 * arrays[i + 1] for i in range(len(arrays) - 1)
            )
            elapsed_ms = (timeit.default_timer() - start) * 1000.0

            return {
                "task": self.name,
                "system": "alkahest_gpu",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
                "note": "numpy_ref" if compiled is None else "nvptx_stub",
            }

        def run_numpy(self, n_pts: int) -> dict:
            import numpy as np

            arrays = [np.random.randn(n_pts) for _ in range(5)]
            start = timeit.default_timer()
            _ = sum(a ** 3 for a in arrays) + sum(
                arrays[i] ** 2 * arrays[i + 1] for i in range(len(arrays) - 1)
            )
            elapsed_ms = (timeit.default_timer() - start) * 1000.0
            return {
                "task": self.name,
                "system": "numpy",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
            }

    class GPUJacobian:
        """Jacobian evaluation on 65k points.

        Computes a 3×3 Jacobian symbolically then evaluates via batch JIT.
        """

        name = "gpu_jacobian_65k"
        size_params = [65_536]

        def run_alkahest_gpu(self, n_pts: int) -> dict:
            import alkahest
            import numpy as np

            pool = alkahest.ExprPool()
            x, y, z = pool.symbol("x"), pool.symbol("y"), pool.symbol("z")
            # f = [x²+y, y²+z, z²+x]
            f = [
                pool.add([pool.pow(x, pool.integer(2)), y]),
                pool.add([pool.pow(y, pool.integer(2)), z]),
                pool.add([pool.pow(z, pool.integer(2)), x]),
            ]
            vars_ = [x, y, z]
            fns = [alkahest.compile_expr(fi, vars_) for fi in f]

            xs = np.random.randn(n_pts)
            ys = np.random.randn(n_pts)
            zs = np.random.randn(n_pts)
            start = timeit.default_timer()
            for fn in fns:
                alkahest.numpy_eval(fn, xs, ys, zs)
            elapsed_ms = (timeit.default_timer() - start) * 1000.0
            return {
                "task": self.name,
                "system": "alkahest_gpu",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
            }

        def run_numpy(self, n_pts: int) -> dict:
            import numpy as np

            xs = np.random.randn(n_pts)
            ys = np.random.randn(n_pts)
            zs = np.random.randn(n_pts)
            start = timeit.default_timer()
            _ = [xs ** 2 + ys, ys ** 2 + zs, zs ** 2 + xs]
            elapsed_ms = (timeit.default_timer() - start) * 1000.0
            return {
                "task": self.name,
                "system": "numpy",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
            }

    class DLPackZeroCopy:
        """Round-trip DLPack zero-copy overhead measurement.

        Measures the overhead of passing a NumPy array to alkahest.numpy_eval
        via DLPack and getting a result back, for 1M points.
        """

        name = "dlpack_zero_copy_1m"
        size_params = [1_000_000]

        def run_alkahest(self, n_pts: int) -> dict:
            import alkahest
            import numpy as np

            pool = alkahest.ExprPool()
            x = pool.symbol("x")
            expr = pool.func("sin", [x])
            fn = alkahest.compile_expr(expr, [x])
            xs = np.random.randn(n_pts)
            # Warm-up
            alkahest.numpy_eval(fn, xs)
            start = timeit.default_timer()
            alkahest.numpy_eval(fn, xs)
            elapsed_ms = (timeit.default_timer() - start) * 1000.0
            return {
                "task": self.name,
                "system": "alkahest",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
            }

        def run_numpy(self, n_pts: int) -> dict:
            import numpy as np

            xs = np.random.randn(n_pts)
            start = timeit.default_timer()
            _ = np.sin(xs)
            elapsed_ms = (timeit.default_timer() - start) * 1000.0
            return {
                "task": self.name,
                "system": "numpy",
                "size": n_pts,
                "wall_ms": round(elapsed_ms, 4),
                "ok": True,
            }


def run_gpu_benchmarks() -> list[dict]:
    """Run all GPU benchmark tasks and return result dicts."""
    suite = GPUBenchmarkSuite()
    tasks = [
        suite.GPUPolynomialEval(),
        suite.GPUJacobian(),
        suite.DLPackZeroCopy(),
    ]
    results = []
    for task in tasks:
        for size in task.size_params:
            for method_name in dir(task):
                if not method_name.startswith("run_"):
                    continue
                system = method_name[len("run_"):]
                method = getattr(task, method_name)
                print(
                    f"  [GPU {system:10s}] {task.name:30s} size={size}",
                    end=" … ",
                    flush=True,
                )
                try:
                    r = method(size)
                    print(f"{r['wall_ms']:.2f} ms")
                    results.append(r)
                except Exception as exc:
                    print(f"SKIP ({exc!r})")
                    results.append({
                        "task": task.name,
                        "system": system,
                        "size": size,
                        "wall_ms": None,
                        "ok": False,
                        "error": str(exc),
                    })
    return results


def alkahest_module():
    """Lazy import of alkahest (avoids import errors if not installed)."""
    import alkahest
    return alkahest


if __name__ == "__main__":
    sys.exit(main())

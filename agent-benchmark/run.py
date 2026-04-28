"""Agent benchmark CLI.

Uses LiteLLM so any supported provider works out of the box.
Set the matching API key environment variable before running.

Usage examples
--------------
    # Anthropic (default)
    ANTHROPIC_API_KEY=sk-... python agent-benchmark/run.py

    # OpenAI
    OPENAI_API_KEY=sk-... python agent-benchmark/run.py --model gpt-4o

    # Google Gemini
    GEMINI_API_KEY=... python agent-benchmark/run.py --model gemini/gemini-1.5-pro

    # Local Ollama (no key needed)
    python agent-benchmark/run.py --model ollama/llama3

    # Only alkahest and sympy, easy tasks only
    python agent-benchmark/run.py --skills alkahest,sympy --difficulty 1

    # Specific tasks, debug mode
    python agent-benchmark/run.py --tasks diff_sin_x2,trig_identity --debug

    # Preview prompts without calling the API
    python agent-benchmark/run.py --dry-run

    # Save results to a custom path
    python agent-benchmark/run.py --output my_results.jsonl --report my_report.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root or from agent-benchmark/
sys.path.insert(0, str(Path(__file__).parent))

from harness import (
    SKILL_PATHS,
    build_report,
    dry_run,
    run_benchmark,
)
from tasks import TASK_BY_NAME, TASKS, get_tasks

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_DEFAULT_OUTPUT = Path(__file__).parent / "results" / "results.jsonl"
_DEFAULT_REPORT = Path(__file__).parent / "results" / "report.md"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI agent benchmark: alkahest-skill vs sympy-skill vs mathematica-skill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skills",
        default=",".join(SKILL_PATHS.keys()),
        metavar="s1,s2",
        help=(
            f"Comma-separated skills to benchmark "
            f"(default: {','.join(SKILL_PATHS.keys())})"
        ),
    )
    parser.add_argument(
        "--tasks",
        default=None,
        metavar="t1,t2",
        help="Comma-separated task names (default: all tasks)",
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Filter tasks by difficulty level (1=easy, 2=medium, 3=hard)",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=(
            f"LiteLLM model string (default: {_DEFAULT_MODEL}). "
            "Examples: gpt-4o, gemini/gemini-1.5-pro, ollama/llama3, "
            "openrouter/mistralai/mistral-7b-instruct"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"JSONL results file (default: {_DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_DEFAULT_REPORT,
        help=f"Markdown report file (default: {_DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=45,
        metavar="SECONDS",
        help="Per-task code execution timeout in seconds (default: 45)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts without calling the API",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print extracted agent code before execution",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available task names and exit",
    )
    parser.add_argument(
        "--list-skills",
        action="store_true",
        help="List available skill names and exit",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.list_tasks:
        print("Available tasks:")
        for t in TASKS:
            print(f"  {t.name:35s}  difficulty={t.difficulty}  category={t.category}")
        return 0

    if args.list_skills:
        print("Available skills:")
        for name, path in SKILL_PATHS.items():
            exists = "OK" if path.exists() else "MISSING"
            print(f"  {name:15s}  {path}  [{exists}]")
        return 0

    # Resolve skills
    skill_names = [s.strip() for s in args.skills.split(",") if s.strip()]
    unknown = [s for s in skill_names if s not in SKILL_PATHS]
    if unknown:
        print(f"ERROR: unknown skills: {unknown}", file=sys.stderr)
        print(f"Available: {list(SKILL_PATHS.keys())}", file=sys.stderr)
        return 1

    # Resolve tasks
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown_tasks = [n for n in task_names if n not in TASK_BY_NAME]
        if unknown_tasks:
            print(f"ERROR: unknown tasks: {unknown_tasks}", file=sys.stderr)
            return 1
        tasks = get_tasks(task_names)
    else:
        tasks = get_tasks()

    # Filter by difficulty
    if args.difficulty is not None:
        tasks = [t for t in tasks if t.difficulty == args.difficulty]
        if not tasks:
            print(f"No tasks at difficulty {args.difficulty}", file=sys.stderr)
            return 1

    print(f"Skills : {skill_names}")
    print(f"Tasks  : {[t.name for t in tasks]}")
    print(f"Model  : {args.model}")
    print()

    if args.dry_run:
        dry_run(skill_names, tasks)
        return 0

    results = run_benchmark(
        skill_names=skill_names,
        tasks=tasks,
        model=args.model,
        output_path=args.output,
        code_timeout=args.timeout,
        debug=args.debug,
    )

    report = build_report(results)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print()
    print(report)
    print(f"Results  → {args.output}")
    print(f"Report   → {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

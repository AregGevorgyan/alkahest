"""Agent benchmark harness.

Drives an AI agent through each task in the catalogue via LiteLLM, which
supports 100+ providers with a single unified interface.  For each (skill,
task) pair the harness:

  1. Loads the skill markdown file as system context.
  2. Sends the task prompt to the configured model.
  3. Extracts the first Python code block from the response.
  4. Executes the code in a subprocess with a timeout.
  5. Calls the task's verify() function on the captured stdout.
  6. Records a result dict.

Model strings follow LiteLLM conventions:
  claude-haiku-4-5-20251001      → Anthropic (needs ANTHROPIC_API_KEY)
  gpt-4o                         → OpenAI   (needs OPENAI_API_KEY)
  gemini/gemini-1.5-pro          → Google   (needs GEMINI_API_KEY)
  ollama/llama3                  → Ollama   (local, no key needed)
  openrouter/mistralai/...       → OpenRouter (needs OPENROUTER_API_KEY)

Prompt caching is enabled automatically for Anthropic models; other
providers receive an equivalent plain-string system message.

Environment
-----------
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / …
                      provider key for the chosen model
  WOLFRAM_KERNEL      optional; path to WolframKernel for Mathematica tasks
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from tasks import AgentTask

# ---------------------------------------------------------------------------
# Skill file paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent

SKILL_PATHS: dict[str, Path] = {
    "alkahest": _REPO_ROOT / "alkahest-skill" / "alkahest.md",
    "sympy": Path(__file__).parent / "skills" / "sympy.md",
    "mathematica": Path(__file__).parent / "skills" / "mathematica.md",
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert mathematical computing assistant.\n"
    "You have been given a skill guide for a specific computer algebra library.\n"
    "When asked to solve a math problem, you MUST write a complete, self-contained "
    "Python script using exactly that library.\n"
    "Enclose your script in a single ```python ... ``` code block.\n"
    "Do not include any text after the closing ``` of the code block.\n"
    "The script must print exactly: ANSWER: <value>  as its last output line."
)

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def extract_code(response_text: str) -> str | None:
    """Return the first Python code block from a model response, or None."""
    m = _CODE_BLOCK_RE.search(response_text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def execute_code(code: str, timeout: int = 45) -> tuple[str, str | None]:
    """Run *code* in a subprocess.

    Returns ``(stdout, error_message)``.  *error_message* is None on success.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(code)
        tmp_path = fh.name

    env = os.environ.copy()
    # propagate WOLFRAM_KERNEL if set
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        stdout = result.stdout
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()
            return stdout, f"exit {result.returncode}: {err[-1] if err else ''}"
        return stdout, None
    except subprocess.TimeoutExpired:
        return "", f"timeout after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return "", f"subprocess error: {exc}"
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Agent call (LiteLLM — provider-agnostic)
# ---------------------------------------------------------------------------

def _is_anthropic_model(model: str) -> bool:
    """True when the model string routes to Anthropic via LiteLLM."""
    m = model.lower()
    return m.startswith("claude") or m.startswith("anthropic/")


def call_agent(
    skill_text: str,
    task_prompt: str,
    model: str,
) -> tuple[str, int]:
    """Call any LiteLLM-supported model with the skill and task prompt.

    Returns ``(response_text, total_tokens)``.
    Anthropic models receive prompt caching on the skill block; all others
    get an equivalent plain-string system message.
    """
    try:
        import litellm
    except ImportError as exc:
        raise SystemExit("litellm not found — run: pip install litellm") from exc

    skill_section = f"## Skill guide\n\n{skill_text}"

    if _is_anthropic_model(model):
        # LiteLLM forwards cache_control to Anthropic's API transparently.
        system_content: Any = [
            {"type": "text", "text": _SYSTEM_PROMPT},
            {"type": "text", "text": skill_section, "cache_control": {"type": "ephemeral"}},
        ]
    else:
        system_content = f"{_SYSTEM_PROMPT}\n\n{skill_section}"

    response = litellm.completion(
        model=model,
        max_tokens=4096,
        temperature=0,
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": task_prompt},
        ],
    )

    text = response.choices[0].message.content or ""
    usage = response.usage
    total_tokens = (usage.total_tokens if usage and usage.total_tokens else 0)
    return text, total_tokens


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def benchmark_task(
    skill_name: str,
    skill_text: str,
    task: AgentTask,
    model: str,
    code_timeout: int = 45,
    debug: bool = False,
) -> dict[str, Any]:
    """Run one (skill, task) pair and return a result dict."""
    result: dict[str, Any] = {
        "skill": skill_name,
        "task": task.name,
        "category": task.category,
        "difficulty": task.difficulty,
        "model": model,
        "ok": False,
        "answer_correct": False,
        "code_extracted": False,
        "stdout": "",
        "error": None,
        "tokens": 0,
        "wall_ms": 0.0,
    }

    t0 = time.perf_counter()
    try:
        response_text, tokens = call_agent(skill_text, task.prompt, model)
        result["tokens"] = tokens

        code = extract_code(response_text)
        if code is None:
            result["error"] = "no_code_block"
            return result
        result["code_extracted"] = True

        if debug:
            print(f"\n[DEBUG] Generated code for {task.name}:\n{code}\n")

        stdout, exec_error = execute_code(code, timeout=code_timeout)
        result["stdout"] = stdout.strip()

        if exec_error:
            result["error"] = exec_error
            return result

        result["ok"] = True
        result["answer_correct"] = task.verify(stdout)

    except Exception as exc:  # noqa: BLE001
        result["error"] = traceback.format_exception_only(type(exc), exc)[0].strip()
    finally:
        result["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_benchmark(
    skill_names: list[str],
    tasks: list[AgentTask],
    model: str,
    output_path: Path,
    code_timeout: int = 45,
    debug: bool = False,
) -> list[dict[str, Any]]:
    """Run all (skill × task) combinations via LiteLLM.

    Set the appropriate provider key in the environment before calling
    (e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY).
    Results are streamed to *output_path* (JSONL) as they complete.
    """
    try:
        import litellm  # noqa: F401
    except ImportError as exc:
        raise SystemExit("litellm not found — run: pip install litellm") from exc

    # Cache skill texts (avoid re-reading files on every task)
    skill_texts: dict[str, str] = {}
    for name in skill_names:
        path = SKILL_PATHS.get(name)
        if path is None or not path.exists():
            print(f"WARNING: skill '{name}' not found at {path} — skipping", file=sys.stderr)
            continue
        skill_texts[name] = path.read_text(encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    with output_path.open("w", encoding="utf-8") as fh:
        for skill_name, skill_text in skill_texts.items():
            for task in tasks:
                label = f"[{skill_name:12s}] {task.name:30s}"
                print(f"  {label}", end=" … ", flush=True)
                r = benchmark_task(
                    skill_name, skill_text, task, model,
                    code_timeout=code_timeout, debug=debug,
                )
                status = "OK" if r["answer_correct"] else ("EXEC_ERR" if not r["ok"] else "WRONG")
                print(f"{status:8s}  {r['wall_ms']:.0f} ms  {r['tokens']} tok")
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                all_results.append(r)

    return all_results


# ---------------------------------------------------------------------------
# Dry-run (prints prompts without calling the API)
# ---------------------------------------------------------------------------

def dry_run(skill_names: list[str], tasks: list[AgentTask]) -> None:
    for skill_name in skill_names:
        path = SKILL_PATHS.get(skill_name)
        skill_text = path.read_text() if path and path.exists() else "(skill not found)"
        for task in tasks:
            print("=" * 72)
            print(f"SKILL: {skill_name}   TASK: {task.name}")
            print("-" * 72)
            print("SYSTEM PROMPT (truncated):")
            print(_SYSTEM_PROMPT[:200] + "…")
            print()
            print("SKILL (first 300 chars):")
            print(skill_text[:300] + "…")
            print()
            print("USER PROMPT:")
            print(task.prompt)
            print()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_report(results: list[dict]) -> str:
    """Build a markdown summary table from result dicts."""
    if not results:
        return "No results.\n"

    skills = sorted({r["skill"] for r in results})
    tasks = sorted({r["task"] for r in results})

    lines = ["# Agent Benchmark Report\n"]

    # ── Per-skill accuracy summary ──────────────────────────────────────────
    lines.append("## Accuracy by skill\n")
    lines.append("| Skill | Correct | Total | % |")
    lines.append("|---|---|---|---|")
    for skill in skills:
        sub = [r for r in results if r["skill"] == skill]
        correct = sum(1 for r in sub if r["answer_correct"])
        lines.append(f"| {skill} | {correct} | {len(sub)} | {100*correct/len(sub):.0f}% |")
    lines.append("")

    # ── Full results table ──────────────────────────────────────────────────
    lines.append("## Full results\n")
    header = "| Task | Category | Difficulty |" + "".join(f" {s} |" for s in skills)
    sep = "|---|---|---|" + "".join("---|" for _ in skills)
    lines.append(header)
    lines.append(sep)
    for task_name in tasks:
        row = [task_name]
        first = next((r for r in results if r["task"] == task_name), None)
        row.append(first["category"] if first else "")
        row.append(str(first["difficulty"]) if first else "")
        for skill in skills:
            r = next(
                (r for r in results if r["task"] == task_name and r["skill"] == skill),
                None,
            )
            if r is None:
                row.append("—")
            elif r["answer_correct"]:
                row.append(f"✓ ({r['wall_ms']:.0f} ms)")
            elif not r["ok"]:
                row.append(f"ERR: {r['error'] or '?'}"[:30])
            else:
                row.append(f"✗ `{(r['stdout'] or '')[:20]}`")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ── Token usage ─────────────────────────────────────────────────────────
    lines.append("## Token usage\n")
    lines.append("| Skill | Total tokens | Mean per task |")
    lines.append("|---|---|---|")
    for skill in skills:
        sub = [r for r in results if r["skill"] == skill]
        total = sum(r["tokens"] for r in sub)
        mean = total / len(sub) if sub else 0
        lines.append(f"| {skill} | {total:,} | {mean:.0f} |")

    return "\n".join(lines) + "\n"

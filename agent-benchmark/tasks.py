"""Agent benchmark task catalogue.

Each task is a math problem expressed as a natural-language prompt.
The agent is expected to write a self-contained Python script that prints:

    ANSWER: <value>

as its last (or only) output line.  The verifier checks that line.

Tasks are CAS-agnostic: the agent chooses which library to use based on the
skill it was given (alkahest, sympy, or mathematica).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _answer_line(output: str) -> str | None:
    """Return the value after the last 'ANSWER: ' prefix, or None."""
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("ANSWER:"):
            return line[len("ANSWER:"):].strip()
    return None


def _approx(output: str, expected: float, tol: float = 1e-4) -> bool:
    ans = _answer_line(output)
    if ans is None:
        return False
    try:
        return abs(float(ans) - expected) <= tol
    except (ValueError, TypeError):
        return False


def _exact_int(output: str, expected: int) -> bool:
    ans = _answer_line(output)
    if ans is None:
        return False
    try:
        return int(ans.strip()) == expected
    except (ValueError, TypeError):
        return False


def _contains(output: str, *substrings: str) -> bool:
    ans = (_answer_line(output) or "").lower()
    return any(s.lower() in ans for s in substrings)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

_TASK_PREAMBLE = (
    "Write a complete, self-contained Python script that solves the problem below.\n"
    "Your script must print exactly one line in the format:\n\n"
    "    ANSWER: <value>\n\n"
    "Do not print anything else after that line.  The <value> must be a number "
    "or short keyword that can be verified programmatically.\n\n"
    "Problem:\n"
)


@dataclass
class AgentTask:
    """One benchmark task for an AI agent."""

    name: str
    category: str
    difficulty: int   # 1 = easy, 2 = medium, 3 = hard
    problem: str      # math statement (appended after preamble)
    verify: Callable[[str], bool]
    hint: str = ""    # expected answer string, shown only in debug mode


    @property
    def prompt(self) -> str:
        return _TASK_PREAMBLE + self.problem


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

TASKS: list[AgentTask] = [

    # ── Differentiation ──────────────────────────────────────────────────────

    AgentTask(
        name="diff_sin_x2",
        category="differentiation",
        difficulty=1,
        problem=(
            "Differentiate sin(x**2) with respect to x.\n"
            "Evaluate the derivative numerically at x = 1.0.\n"
            "Print the result as a float rounded to 8 decimal places."
        ),
        verify=lambda out: _approx(out, 2 * math.cos(1.0), tol=1e-4),
        hint=f"{2 * math.cos(1.0):.8f}",
    ),

    AgentTask(
        name="diff_poly_leading",
        category="differentiation",
        difficulty=1,
        problem=(
            "Build the polynomial p(x) = 1 + x + x**2 + x**3 + ... + x**20 (degree 20).\n"
            "Differentiate it with respect to x.\n"
            "What is the leading coefficient of the derivative polynomial?\n"
            "Print that integer."
        ),
        verify=lambda out: _exact_int(out, 20),
        hint="20",
    ),

    AgentTask(
        name="gradient_sum",
        category="differentiation",
        difficulty=2,
        problem=(
            "Compute the gradient of f(x, y, z) = x**2 + 2*y**2 + 3*z**2 "
            "with respect to [x, y, z].\n"
            "Evaluate each partial derivative at (x=1, y=2, z=3) and print "
            "the sum of the three values as a float."
        ),
        # grad = [2x, 4y, 6z]; at (1,2,3) = [2, 8, 18]; sum = 28
        verify=lambda out: _approx(out, 28.0, tol=1e-3),
        hint="28.0",
    ),

    # ── Integration ──────────────────────────────────────────────────────────

    AgentTask(
        name="integrate_x2_definite",
        category="integration",
        difficulty=1,
        problem=(
            "Compute the definite integral of x**2 from x = 0 to x = 1.\n"
            "Print the result as a float."
        ),
        verify=lambda out: _approx(out, 1 / 3, tol=1e-4),
        hint="0.33333333",
    ),

    AgentTask(
        name="integrate_sin_definite",
        category="integration",
        difficulty=1,
        problem=(
            "Compute the definite integral of sin(x) from x = 0 to x = pi.\n"
            "Print the result as a float."
        ),
        verify=lambda out: _approx(out, 2.0, tol=1e-4),
        hint="2.0",
    ),

    AgentTask(
        name="risch_nonelementary",
        category="integration",
        difficulty=3,
        problem=(
            "Attempt to compute the antiderivative of exp(x**2) with respect to x.\n"
            "If the result involves a special function (erf, erfi, etc.) or "
            "the CAS raises an error indicating no elementary antiderivative exists, "
            "print: ANSWER: nonelementary\n"
            "Otherwise print the antiderivative expression as a string."
        ),
        verify=lambda out: _contains(out, "nonelementary", "erf", "erfi", "dawson"),
        hint="nonelementary",
    ),

    # ── Simplification ───────────────────────────────────────────────────────

    AgentTask(
        name="trig_identity",
        category="simplification",
        difficulty=1,
        problem=(
            "Simplify the expression sin(x)**2 + cos(x)**2 using your CAS.\n"
            "Print the integer result."
        ),
        verify=lambda out: _exact_int(out, 1),
        hint="1",
    ),

    AgentTask(
        name="log_exp_simplify",
        category="simplification",
        difficulty=1,
        problem=(
            "Simplify log(exp(x)) symbolically.\n"
            "Evaluate the simplified expression at x = 7.0 and print the float."
        ),
        verify=lambda out: _approx(out, 7.0, tol=1e-4),
        hint="7.0",
    ),

    AgentTask(
        name="trig_sum_simplify",
        category="simplification",
        difficulty=2,
        problem=(
            "Build the expression: sum of (sin(x)**2 + cos(x)**2) repeated 15 times.\n"
            "Simplify the result.\n"
            "Print the integer value."
        ),
        verify=lambda out: _exact_int(out, 15),
        hint="15",
    ),

    # ── Polynomial operations ────────────────────────────────────────────────

    AgentTask(
        name="poly_gcd_eval",
        category="polynomial",
        difficulty=2,
        problem=(
            "Compute the GCD of the polynomials (x**6 - 1) and (x**4 - 1) in x.\n"
            "Evaluate the resulting GCD polynomial at x = 3.0 and print the float."
        ),
        # gcd(x^6-1, x^4-1) = x^2-1  (gcd of degrees 6,4 is 2)
        # at x=3: 9-1 = 8.0
        verify=lambda out: _approx(out, 8.0, tol=1e-3),
        hint="8.0",
    ),

    AgentTask(
        name="poly_eval",
        category="polynomial",
        difficulty=1,
        problem=(
            "Evaluate the polynomial x**10 + x**5 + 1 at x = 2.0.\n"
            "Print the result as a float."
        ),
        verify=lambda out: _approx(out, 1057.0, tol=0.5),
        hint="1057.0",
    ),

    # ── Solving ──────────────────────────────────────────────────────────────

    AgentTask(
        name="solve_circle_line",
        category="solving",
        difficulty=2,
        problem=(
            "Solve the real-valued system:\n"
            "  x**2 + y**2 = 2\n"
            "  y = x\n"
            "How many real solutions does this system have?\n"
            "Print that integer."
        ),
        verify=lambda out: _exact_int(out, 2),
        hint="2",
    ),

    AgentTask(
        name="solve_quadratic_count",
        category="solving",
        difficulty=1,
        problem=(
            "Solve x**2 - 5*x + 6 = 0 over the reals.\n"
            "How many distinct real solutions are there?\n"
            "Print that integer."
        ),
        verify=lambda out: _exact_int(out, 2),
        hint="2",
    ),

    # ── Linear algebra ───────────────────────────────────────────────────────

    AgentTask(
        name="jacobian_entry",
        category="linear_algebra",
        difficulty=2,
        problem=(
            "Compute the Jacobian matrix of the vector function\n"
            "  f(x, y) = [x**2 + y,  x * y**2]\n"
            "with respect to [x, y].\n"
            "Evaluate the entry at row 1, column 0 (0-indexed) at x=2.0, y=3.0.\n"
            "Print the float."
        ),
        # J = [[2x, 1], [y^2, 2xy]].  J[1][0] = y^2 = 9.0
        verify=lambda out: _approx(out, 9.0, tol=1e-3),
        hint="9.0",
    ),

    AgentTask(
        name="matrix_det",
        category="linear_algebra",
        difficulty=2,
        problem=(
            "Compute the symbolic determinant of the 3×3 matrix:\n"
            "  [[x,  1,  0],\n"
            "   [0,  x,  1],\n"
            "   [1,  0,  x]]\n"
            "Evaluate the determinant at x = 2.0 and print the float."
        ),
        # det = x^3 - x + 1 - 1 ... let me compute: cofactor expansion
        # det = x*(x*x - 0) - 1*(0*x - 1*1) + 0 = x^3 - (-1) = x^3 + 1
        # at x=2: 8+1=9
        verify=lambda out: _approx(out, 9.0, tol=1e-3),
        hint="9.0",
    ),

    # ── Numerics / rigorous arithmetic ───────────────────────────────────────

    AgentTask(
        name="ball_sin_cos",
        category="numerics",
        difficulty=3,
        problem=(
            "Compute sin(cos(1.0)) using interval or ball arithmetic with a "
            "ball/interval of radius 1e-8 centred at x = 1.0.\n"
            "Print the midpoint of the resulting interval as a float "
            "rounded to 8 decimal places."
        ),
        verify=lambda out: _approx(out, math.sin(math.cos(1.0)), tol=1e-4),
        hint=f"{math.sin(math.cos(1.0)):.8f}",
    ),

    AgentTask(
        name="jit_poly_sum",
        category="numerics",
        difficulty=2,
        problem=(
            "Build the polynomial p(x) = 1 + x + x**2 + ... + x**10.\n"
            "JIT-compile or lambdify it, then evaluate it at 1 000 000 evenly-spaced "
            "points in [0, 1].\n"
            "Print the mean value rounded to 6 decimal places."
        ),
        # integral of p(x) from 0 to 1 = sum_{k=0}^{10} 1/(k+1) ≈ 2.020...
        verify=lambda out: _approx(
            out,
            sum(1 / (k + 1) for k in range(11)),
            tol=1e-3,
        ),
        hint=f"{sum(1 / (k + 1) for k in range(11)):.6f}",
    ),
]

# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

TASK_BY_NAME: dict[str, AgentTask] = {t.name: t for t in TASKS}


def get_tasks(names: list[str] | None = None) -> list[AgentTask]:
    """Return all tasks, or the subset matching *names*."""
    if names is None:
        return list(TASKS)
    return [TASK_BY_NAME[n] for n in names if n in TASK_BY_NAME]

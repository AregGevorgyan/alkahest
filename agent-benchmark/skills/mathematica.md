# Mathematica / Wolfram Engine Agent Skill

Use this skill whenever you are writing Python code that calls Mathematica or the Wolfram Engine through the `wolframclient` library.

## Install & setup

```bash
# Python bridge
pip install wolframclient

# Wolfram Engine (free for non-commercial use)
# Download from: https://www.wolfram.com/engine/
# After install, activate once: wolframscript -activate
```

---

## Core mental model

All computation happens **inside a Wolfram Language (WL) kernel process**. Python sends WL expression strings (or `wl`-built objects) to the kernel, and the kernel returns Python-native values or WL expression trees. The session is a long-lived subprocess — start it once per script, reuse it for all evaluations.

```python
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr, wl

session = WolframLanguageSession()  # auto-discovers the kernel
session.start()

# Always terminate when done
try:
    result = session.evaluate(wlexpr("1 + 1"))   # Python int: 2
finally:
    session.terminate()
```

If the kernel is not on the PATH, pass the path explicitly:

```python
session = WolframLanguageSession("/usr/local/Wolfram/WolframEngine/14.3/Executables/WolframKernel")
```

---

## Evaluating expressions

```python
# String form — simplest
result = session.evaluate(wlexpr("D[Sin[x^2], x]"))
# Returns a WL expression object; use ToString for a readable string

result_str = session.evaluate(wlexpr("ToString[D[Sin[x^2], x]]"))
# Python str: "2 x Cos[x^2]"

# Numeric result
val = session.evaluate(wlexpr("N[D[Sin[x^2], x] /. x -> 1.0]"))
# Python float: 1.0806046117362795

# Multiple substitutions
val = session.evaluate(wlexpr("f[x_, y_] := x^2 + y; f[3, 4]"))
```

Use `wl` for programmatic expression construction:

```python
from wolframclient.language import wl
expr = wl.D(wl.Sin(wl.Power("x", 2)), "x")
result = session.evaluate(expr)
```

---

## Differentiation

```python
# D[expr, var]
session.evaluate(wlexpr("ToString[D[Sin[x^2], x]]"))
# "2 x Cos[x^2]"

# Higher order
session.evaluate(wlexpr("ToString[D[x^5, {x, 3}]]"))
# "60 x^2"

# Partial
session.evaluate(wlexpr("ToString[D[x^2*y + y^3, y]]"))

# Jacobian matrix
session.evaluate(wlexpr(
    "ToString[D[{x^2 + y, x*y^2}, {{x, y}}]]"
))
```

---

## Integration

```python
# Indefinite
session.evaluate(wlexpr("ToString[Integrate[x^2, x]]"))
# "x^3/3"

# Definite
session.evaluate(wlexpr("N[Integrate[x^2, {x, 0, 1}]]"))
# 0.3333...

# Non-elementary — Mathematica uses special functions
session.evaluate(wlexpr("ToString[Integrate[Exp[x^2], x]]"))
# "Sqrt[Pi] Erfi[x] / 2"

# Check if integral was computed (no unevaluated Integrate head)
result = session.evaluate(wlexpr("Integrate[Exp[x^2], x]"))
# If result still has head Integrate, no closed form was found
```

---

## Simplification

```python
session.evaluate(wlexpr("ToString[Simplify[Sin[x]^2 + Cos[x]^2]]"))
# "1"

session.evaluate(wlexpr("ToString[FullSimplify[Log[Exp[x]]]]"))
# "x"

session.evaluate(wlexpr("ToString[Expand[(x+1)^3]]"))
# "1 + 3 x + 3 x^2 + x^3"

session.evaluate(wlexpr("ToString[Factor[x^2 - 1]]"))
# "(-1 + x)(1 + x)"

session.evaluate(wlexpr("ToString[TrigExpand[Sin[x+y]]]"))
session.evaluate(wlexpr("ToString[TrigReduce[Sin[x]^2]]"))
```

Key simplification functions:
- `Simplify` — standard simplification
- `FullSimplify` — more powerful, slower
- `Expand` — distribute
- `Factor` — factor over ℤ
- `Cancel` — cancel GCDs in rational expressions
- `Apart` — partial fractions
- `Collect[expr, x]` — collect powers of x

---

## Numeric evaluation

```python
# N[expr] — 15 significant digits
session.evaluate(wlexpr("N[Pi]"))          # 3.141592653589793

# N[expr, n] — n significant digits
session.evaluate(wlexpr("N[Pi, 50]"))

# Substitute and evaluate
session.evaluate(wlexpr("N[Sin[x^2] /. x -> 1.5]"))

# Table of values
session.evaluate(wlexpr("Table[N[Sin[k/10]], {k, 0, 10}]"))
# Python list of floats
```

---

## Polynomial tools

```python
# GCD
session.evaluate(wlexpr("ToString[PolynomialGCD[x^4 - 1, x^2 - 1]]"))
# "(-1 + x^2)"

# LCM
session.evaluate(wlexpr("ToString[PolynomialLCM[x^2 - 1, x^2 - 2*x + 1]]"))

# Gröbner basis
session.evaluate(wlexpr(
    "ToString[GroebnerBasis[{x^2 + y^2 - 1, x - y}, {x, y}, MonomialOrder -> Lexicographic]]"
))

# Resultant
session.evaluate(wlexpr("ToString[Resultant[x^2 - 2, x - 1, x]]"))
```

---

## Solving equations

```python
# Real solutions
session.evaluate(wlexpr("Solve[x^2 + y^2 == 1 && y == x, {x, y}, Reals]"))
# List of rules: {{x -> -1/Sqrt[2], y -> -1/Sqrt[2]}, ...}

# All solutions (including complex)
session.evaluate(wlexpr("Solve[x^4 == 1, x]"))

# NSolve for numeric
session.evaluate(wlexpr("NSolve[x^5 - x - 1 == 0, x, Reals]"))

# Count solutions
result = session.evaluate(wlexpr("Length[Solve[x^2 + y^2 == 1 && y == x, {x, y}, Reals]]"))
# Python int: 2
```

---

## Interval / rigorous arithmetic

```python
# Interval arithmetic
session.evaluate(wlexpr("Sin[Cos[Interval[{0.9, 1.1}]]]"))
# Returns Interval[{min, max}]

# Extract bounds
lo, hi = session.evaluate(wlexpr("List @@ Sin[Cos[Interval[{0.9, 1.1}]]][[1]]"))

# Midpoint
mid = session.evaluate(wlexpr(
    "With[{iv = Sin[Cos[Interval[{0.99, 1.01}]]]}, Mean[First[iv]]]"
))
```

---

## Symbolic matrices

```python
session.evaluate(wlexpr("ToString[Det[{{1, x}, {x, 1}}]]"))
# "1 - x^2"

session.evaluate(wlexpr("ToString[Inverse[{{a, b}, {c, d}}]]"))

# Jacobian matrix  (list of functions, list of variables)
session.evaluate(wlexpr(
    "ToString[D[{x^2 + y, x*y^2}, {{x, y}}]]"
))

# Eigenvalues
session.evaluate(wlexpr("Eigenvalues[{{1, 2}, {3, 4}}]"))
```

---

## Key rules for agents

1. **Always start/stop the session.** Use `try/finally` to call `session.terminate()` even if an error occurs.
2. **`ToString[...]` for readable output.** Raw WL expression objects are hard to parse; wrap with `ToString` to get a Python string.
3. **`N[...]` for floating-point results.** Pure symbolic results (like `Sqrt[2]`) stay symbolic — wrap with `N` to get floats.
4. **WL uses `^` for exponentiation.** In WL strings: `x^2`, not `x**2`.
5. **WL uses `==` for equations.** In `Solve`/`FindRoot`, write `x^2 == 1`, not `x^2 - 1`.
6. **Substitution uses `/.`** (`ReplaceAll`): `Sin[x] /. x -> 1.0`.
7. **`Length[Solve[...]]` counts solutions.** Use it to avoid parsing solution lists.
8. **`FullSimplify` is stronger than `Simplify` but slower.** Prefer `Simplify` first; fall back to `FullSimplify` for stubborn expressions.
9. **Variable names must not collide with WL built-ins.** Avoid single-capital-letter names like `C`, `D`, `E`, `I`, `K`, `N`, `O`.
10. **Session kernel discovery.** If `WolframLanguageSession()` fails, pass the explicit kernel path or set `WOLFRAM_KERNEL` env var.

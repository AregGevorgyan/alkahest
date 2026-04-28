# SymPy Agent Skill

Use this skill whenever you are writing Python code that uses the `sympy` library.

## Install

```bash
pip install sympy
```

---

## Core mental model

SymPy expressions are **immutable Python objects**. Symbols are created with `sp.Symbol` or `sp.symbols`. Arithmetic operators (`+`, `-`, `*`, `**`, `/`) are overloaded — use them freely. There is no "pool" or interning step; raw Python ints work directly in expressions.

```python
import sympy as sp

x = sp.Symbol("x")
y = sp.Symbol("y")
z = sp.Symbol("z")

# Multiple symbols at once
a, b, c = sp.symbols("a b c")

# With assumptions
t = sp.Symbol("t", real=True, positive=True)
n = sp.Symbol("n", integer=True)
```

---

## Differentiation

```python
# First derivative
d = sp.diff(sp.sin(x**2), x)          # 2*x*cos(x**2)

# Higher-order derivative
d2 = sp.diff(x**5, x, 3)              # 60*x**2  (third derivative)

# Partial derivative
dp = sp.diff(x**2 * y, x)             # 2*x*y

# Jacobian of a vector function
fns = sp.Matrix([x**2 + y, x * y**2])
J = fns.jacobian([x, y])              # 2×2 Matrix
J[0, 1]                               # entry (row 0, col 1)
```

---

## Integration

```python
# Indefinite
i = sp.integrate(x**2, x)                     # x**3/3
i = sp.integrate(sp.sin(x), x)                # -cos(x)

# Definite
i = sp.integrate(x**2, (x, 0, 1))             # Rational(1, 3)
i = sp.integrate(sp.exp(-x), (x, 0, sp.oo))   # 1

# Non-elementary — SymPy returns in terms of special functions
i = sp.integrate(sp.exp(x**2), x)             # sqrt(pi)*erfi(x)/2
i = sp.integrate(sp.exp(-x**2), x)            # sqrt(pi)*erf(x)/2

# Check if result is elementary: if it contains Integral(), no closed form was found
result = sp.integrate(sp.exp(x**2), x)
if result.has(sp.Integral):
    print("No elementary antiderivative found")
```

---

## Simplification

```python
sp.simplify(expr)           # general — tries multiple strategies
sp.trigsimp(expr)           # sin²+cos²=1, etc.
sp.expand(expr)             # distribute / expand powers
sp.factor(expr)             # factor polynomials
sp.cancel(expr)             # cancel common factors in rational expressions
sp.apart(expr, x)           # partial fraction decomposition
sp.collect(expr, x)         # collect terms by powers of x
sp.radsimp(expr)            # rationalise denominators
sp.powsimp(expr)            # simplify exponents
sp.logcombine(expr)         # combine logarithms

# Examples
sp.trigsimp(sp.sin(x)**2 + sp.cos(x)**2)      # 1
sp.simplify(sp.log(sp.exp(x)))                 # x
sp.expand((x + 1)**3)                          # x**3 + 3*x**2 + 3*x + 1
sp.factor(x**2 - 1)                            # (x - 1)*(x + 1)
```

---

## Numeric evaluation

```python
# Substitute a value
val = expr.subs(x, 2)                  # returns SymPy expr, may still be symbolic

# Force float
val = float(expr.subs(x, 2))

# Multiple substitutions
val = expr.subs([(x, 1.0), (y, 2.0)])

# Arbitrary precision (N significant figures)
sp.N(sp.pi, 50)                        # 50-digit float

# Fast lambdify for vectorised numeric eval
import numpy as np
f = sp.lambdify(x, sp.sin(x**2), "numpy")
f(np.linspace(0, 1, 1_000_000))       # ndarray
```

---

## Polynomial tools

```python
# Polynomial object
p = sp.Poly(x**3 - 2*x + 1, x)
p.degree()          # 3
p.coeffs()          # [1, 0, -2, 1]  (leading first)
p.eval(2)           # numeric value at x=2

# GCD of two polynomials
sp.gcd(x**4 - 1, x**2 - 1, x)          # x**2 - 1

# LCM
sp.lcm(x**2 - 1, x**2 - 2*x + 1)

# Resultant
sp.resultant(x**2 - 1, x - 2, x)

# Gröbner basis
sp.groebner([x**2 + y**2 - 1, x - y], [x, y], order="lex")
```

---

## Solving equations

```python
# Single equation
sp.solve(x**2 - 2, x)                          # [-sqrt(2), sqrt(2)]

# System of equations
sp.solve([x**2 + y**2 - 1, y - x], [x, y])    # list of tuples

# dict=True for labelled solutions
sp.solve([x + y - 1, x - y], [x, y], dict=True)   # [{'x': 1/2, 'y': 1/2}]

# Reals only
sp.solve(x**4 - 1, x, domain="RR")

# Transcendental equations (may return conditions)
sp.solve(sp.exp(x) - 1, x)                     # [0]
```

---

## Symbolic matrices

```python
M = sp.Matrix([[1, x], [x, 1]])
M.det()             # 1 - x**2
M.inv()             # symbolic inverse
M.eigenvals()       # dict: eigenvalue -> multiplicity
M.eigenvects()      # list of (eigenvalue, multiplicity, eigenvectors)
M * M               # matrix multiply
M.T                 # transpose
M[0, 1]             # element access (row, col)
M.tolist()          # list[list]
```

---

## Available math functions

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`,
`exp`, `log`, `sqrt`, `Abs`, `sign`, `floor`, `ceiling`, `gamma`, `erf`, `erfc`, `erfi`,
`Heaviside`, `DiracDelta`, `Piecewise`, `pi`, `E`, `oo` (infinity), `zoo` (complex infinity), `nan`

---

## Key rules for agents

1. **Raw Python ints work.** `x**2 + 1` is valid — no interning needed.
2. **`simplify` is general but slow.** Prefer `trigsimp`, `expand`, `factor`, `cancel` when the structure is known.
3. **Integration may return `Integral` objects.** Check `result.has(sp.Integral)` to detect failure.
4. **`solve` returns a list.** For systems, it returns a list of tuples (or dicts when `dict=True`). The list may be empty if no solution exists.
5. **Numeric answers need `float()`.** `sp.Rational(1, 3)` is exact; wrap with `float()` for a decimal.
6. **`lambdify` for fast loops.** Do not evaluate symbolic expressions in a Python loop — use `lambdify` + NumPy.
7. **`sp.N(expr, n)` for precision.** Use for arbitrary-precision floating point; default is ~15 digits.
8. **Assumptions speed up simplification.** `sp.Symbol('x', real=True)` enables `sqrt(x**2) == x` etc.

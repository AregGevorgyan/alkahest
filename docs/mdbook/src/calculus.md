# Calculus

Alkahest supports symbolic differentiation and integration with full derivation logging.

## Differentiation

`diff(expr, var)` computes the symbolic derivative of `expr` with respect to `var`.

```python
from alkahest import diff, sin, cos, exp, log

pool = ExprPool()
x = pool.symbol("x")

# Polynomial
dr = diff(x**3 + pool.integer(2) * x, x)
print(dr.value)   # 3*x^2 + 2

# Chain rule
dr = diff(sin(x**2), x)
print(dr.value)   # 2*x*cos(x^2)

# Product rule
dr = diff(x * exp(x), x)
print(dr.value)   # exp(x) + x*exp(x)

# Logarithm
dr = diff(log(x**2 + pool.integer(1)), x)
print(dr.value)   # 2*x / (x^2 + 1)
```

### Registered primitives

Every primitive in the registry has a differentiation rule. The 23 currently registered primitives include:

`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`, `sign`, `erf`, `erfc`, `gamma`, `floor`, `ceil`, `round`, `min`, `max`

### Derivation log

The `DerivedResult` returned by `diff` records every rule application:

```python
dr = diff(sin(x**2), x)
for step in dr.steps:
    print(f"  {step['rule']:25s}  {step['before']}  →  {step['after']}")
```

## Forward-mode automatic differentiation

`diff_forward` computes the derivative using forward-mode AD (dual numbers). It produces the same result as `diff` but through a different computational path:

```python
from alkahest import diff, diff_forward

sym = diff(x**3, x)
fwd = diff_forward(x**3, x)
# fwd.value == sym.value
```

Forward mode is useful for checking that the symbolic rules agree with dual-number evaluation.

## Symbolic gradient

`symbolic_grad` differentiates with respect to multiple variables:

```python
from alkahest import symbolic_grad

pool = ExprPool()
x = pool.symbol("x")
y = pool.symbol("y")

expr = x**2 * y + sin(x * y)
grads = symbolic_grad(expr, [x, y])
# grads[0] = ∂/∂x = 2*x*y + y*cos(x*y)
# grads[1] = ∂/∂y = x^2 + x*cos(x*y)
```

For the traced-function gradient (composable with `jit`), see [Transformations](./transformations.md).

## Integration

`integrate(expr, var)` computes the symbolic antiderivative of `expr` with respect to `var`.

```python
from alkahest import integrate, sin, cos, exp

# Polynomials
r = integrate(x**3, x)
print(r.value)    # x^4/4

# Known functions
r = integrate(sin(x), x)
print(r.value)    # -cos(x)

r = integrate(exp(x), x)
print(r.value)    # exp(x)

r = integrate(x**pool.integer(-1), x)
print(r.value)    # log(x)
```

### Integration rules

The integration engine applies rules from a table of known forms (Risch subset):

- Power rule: `∫ xⁿ dx = xⁿ⁺¹/(n+1)` for integer `n ≠ -1`
- Logarithm: `∫ 1/x dx = log(x)`
- Exponential tower: `∫ exp(a*x + b) dx`, `∫ x * exp(x) dx`
- Linear substitution: `∫ f(a*x + b) dx`
- Trigonometric: `∫ sin(x) dx`, `∫ cos(x) dx`
- Standard table entries for `erf`, inverse trig, etc.

If no rule applies, `integrate` raises an `IntegrationError` with a remediation hint indicating what class of integrand would be needed (e.g. "algebraic extension required — see v1.1 algebraic Risch").

**Upcoming (v1.1):** Algebraic-function Risch (Trager's algorithm) will handle integrands involving `sqrt(P(x))` and other algebraic extensions.

### Verification

A common pattern is to verify an antiderivative by differentiating it back:

```python
antideriv = integrate(expr, x).value
check = simplify(diff(antideriv, x).value)
# check.value should equal expr
```

## Higher derivatives

Chain calls to `diff`:

```python
d2 = diff(diff(sin(x), x).value, x)
print(d2.value)   # -sin(x)
```

The derivation log of the outer `diff` does not include the inner steps. If you need the full trace, concatenate `dr1.steps + dr2.steps`.

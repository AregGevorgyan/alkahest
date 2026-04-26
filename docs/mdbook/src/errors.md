# Error handling

Alkahest uses a structured exception hierarchy. Every error carries a stable diagnostic code, a human-readable message, an optional source span, and an optional remediation hint.

## Exception hierarchy

```
AlkahestError (base)
├── ConversionError   (E-POLY-*)   — expression → polynomial/rational conversion
├── DomainError       (E-DOMAIN-*) — mathematical side conditions violated
├── DiffError         (E-DIFF-*)   — differentiation failed
├── IntegrationError  (E-INT-*)    — integration failed
├── MatrixError       (E-MAT-*)    — linear algebra errors
├── OdeError          (E-ODE-*)    — ODE construction or lowering
├── DaeError          (E-DAE-*)    — DAE structural analysis
├── SolverError       (E-SOLVE-*)  — polynomial system solving
├── JitError          (E-JIT-*)    — LLVM/JIT codegen
├── CudaError         (E-CUDA-*)   — CUDA kernel launch or driver
└── PoolError         (E-POOL-*)   — ExprPool misuse
```

## Error attributes

Every exception instance exposes:

| Attribute | Type | Description |
|---|---|---|
| `.code` | `str` | Stable diagnostic code, e.g. `"E-POLY-001"` |
| `.message` | `str` | Human-readable description |
| `.remediation` | `str \| None` | What the user should try |
| `.span` | `tuple[int, int] \| None` | Character offset range in source expression |

```python
import alkahest
from alkahest import ExprPool, UniPoly, ConversionError

pool = ExprPool()
x = pool.symbol("x")

try:
    # sin(x) cannot be represented as a polynomial
    p = UniPoly.from_symbolic(alkahest.sin(x), x)
except ConversionError as e:
    print(e.code)          # E-POLY-001
    print(e.message)       # "expression contains non-polynomial term: sin(x)"
    print(e.remediation)   # "Use Expr directly, or expand sin(x) as a series first"
```

## Common errors and remediations

### ConversionError (E-POLY-*)

Raised when an expression cannot be converted to a polynomial or rational function.

| Code | Cause | Remediation |
|---|---|---|
| `E-POLY-001` | Non-polynomial term (e.g. `sin`) | Use `Expr` directly; or expand as series |
| `E-POLY-002` | Non-integer exponent | Algebraic extension not yet supported |
| `E-POLY-003` | Symbolic exponent (variable in exponent) | Use `Expr.pow`, not `UniPoly` |

### DomainError (E-DOMAIN-*)

Raised when a mathematical side condition is violated.

| Code | Cause | Remediation |
|---|---|---|
| `E-DOMAIN-001` | Division by zero | Check denominator before dividing |
| `E-DOMAIN-002` | `log(0)` or `log(negative)` | Ensure argument is positive; use complex domain if needed |
| `E-DOMAIN-003` | `sqrt(negative)` | Use `AcbBall` or declare complex domain |

### IntegrationError (E-INT-*)

| Code | Cause | Remediation |
|---|---|---|
| `E-INT-001` | No integration rule matches | Result may not have an elementary antiderivative |
| `E-INT-002` | Algebraic extension required | Planned for v1.1 (algebraic Risch) |
| `E-INT-003` | Risch gave up (transcendental tower too deep) | Try numerical integration |

### SolverError (E-SOLVE-*)

| Code | Cause | Remediation |
|---|---|---|
| `E-SOLVE-001` | System is inconsistent | No solutions exist |
| `E-SOLVE-002` | High-degree univariate factor (> 2) | Symbolic solution not supported; use numerical solve |
| `E-SOLVE-003` | Gröbner basis did not terminate | Increase node/iteration limits |

## Catching errors by code

For programmatic error handling:

```python
try:
    result = alkahest.integrate(expr, x)
except alkahest.AlkahestError as e:
    if e.code.startswith("E-INT-"):
        print(f"Integration failed: {e.remediation}")
    else:
        raise
```

## Error taxonomy

Every error is classified on two independent axes: **subsystem** (determines the code prefix and exception class) and **cause** (informs the remediation hint).

### Subsystem axis

| Prefix | Class | Scope |
|---|---|---|
| `E-POLY-*` | `ConversionError` | Expression → polynomial/rational-function conversion |
| `E-DOMAIN-*` | `DomainError` | Side-condition violations (div-by-zero, log of 0, `sqrt` of negative) |
| `E-DIFF-*` | `DiffError` | Forward/reverse differentiation, unknown derivatives |
| `E-INT-*` | `IntegrationError` | Symbolic integration (Risch, heuristic, table) |
| `E-MAT-*` | `MatrixError` | Linear algebra (shape, singular, non-invertible) |
| `E-ODE-*` | `OdeError` | ODE construction, lowering, event handling |
| `E-DAE-*` | `DaeError` | DAE structural analysis (Pantelides, index reduction) |
| `E-SOLVE-*` | `SolverError` | Polynomial system solving, Gröbner basis |
| `E-JIT-*` | `JitError` | LLVM/Cranelift codegen and linking |
| `E-CUDA-*` | `CudaError` | NVPTX compile, kernel launch, driver/runtime failures |
| `E-POOL-*` | `PoolError` | `ExprPool` misuse (closed, cross-pool, persisted-handle mismatch) |
| `E-PARSE-*` | `ParseError` *(reserved)* | Parser integration — owns `span()` by default |
| `E-IO-*` | `IoError` *(reserved)* | Checkpoint/serde paths (`PoolPersistError`) |

### Cause axis

1. **User-input** — the expression or argument is outside the supported fragment. Always has a `remediation`; carries a `span` once parsing lands.
2. **Domain** — input is syntactically fine but violates a mathematical side condition. Remediation is "substitute a different value," not "reformulate."
3. **Unsupported** — the operation is not implemented for this case. Must name the missing capability so users can file a feature request.
4. **Resource/environment** — CUDA device absent, out-of-memory, JIT target mismatch, pool closed. Typically no `span`; remediation references the environment, not the expression.
5. **Internal invariant** — a bug. Should never reach users in release; in debug it carries a backtrace. Use `E-INTERNAL-001`.

### Adding a new error code

1. Does it fit an existing subsystem? Add a variant and a code one higher than the current max for that prefix.
2. Does it name a new subsystem? Add a prefix, a class, and an entry in `REGISTRY` in the same PR. Do not reuse prefixes across unrelated subsystems.
3. Write the `remediation` before the message — if you cannot say what the user should do, the taxonomy is telling you this is an internal bug, not a user error.

Users match on subsystem (the exception class); triagers filter on cause (the code suffix and remediation text).

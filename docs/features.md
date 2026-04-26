# Feature surface

Current stable feature surface as of v1.0.

## Core expression kernel

- Hash-consed DAG with structural equality as pointer comparison
- N-ary `Add` / `Mul` with AC normalization at construction
- Arbitrary-precision integers and rationals (FLINT/GMP)
- Symbol domains: `real`, `positive`, `nonnegative`, `integer`, `complex`
- Persistent pool: serialize to disk and reload across sessions (V1-14)
- Sharded pool for concurrent insertion (`--features parallel`)

## Simplification

- Rule-based fixpoint simplification (`simplify`)
- Domain-specific rule sets: trig (`simplify_trig`), log/exp (`simplify_log_exp`), expanded (`simplify_expanded`)
- Custom rule sets via `make_rule` / `simplify_with`
- E-graph equality saturation via egglog (`simplify_egraph`, `--features egraph`)
- Pluggable cost functions: `SizeCost`, `DepthCost`, `OpCost`, `StabilityCost`
- Phased saturation with `node_limit` / `iter_limit` config
- `collect_like_terms`, `poly_normal`
- Branch-cut-aware log/exp rewrites with `SideCondition` tracking
- Parallel simplification (`simplify_par`, `--features parallel`)

## Polynomial algebra (FLINT-backed)

- `UniPoly`: dense univariate polynomial arithmetic, GCD, degree, coefficients
- `MultiPoly`: sparse multivariate polynomial arithmetic, GCD, total degree
- `RationalFunction`: quotient with automatic GCD normalization
- Horner-form rewriting (`horner`)
- C code emission (`emit_c`)

## Calculus

- Symbolic differentiation (`diff`, `diff_forward`)
- Forward-mode automatic differentiation
- Reverse-mode automatic differentiation (`symbolic_grad`)
- Symbolic integration: power rule, log, exp tower, linear substitution, trig (Risch subset)
- **Upcoming (v1.1):** Algebraic-function Risch (Trager's algorithm)

## Transformations

- `trace` / `trace_fn` — symbolic function tracing
- `grad` — gradient transformation (symbolically differentiates a traced function)
- `jit` — LLVM JIT compilation of traced functions
- `CompiledTracedFn` for array-vectorised evaluation
- JAX-style pytree flattening (`flatten_exprs`, `unflatten_exprs`, `map_exprs`)
- Context manager (`alkahest.context(pool=..., simplify=...)`)

## Code generation

- LLVM JIT for native CPU code (`--features jit`)
- NVPTX (CUDA GPU) codegen for `sm_86` (`--features cuda`, 16.2× over CPU on RTX 3090)
- Custom `alkahest` MLIR dialect with three lowering targets: ArithMath, StableHLO, LLVM
- `to_stablehlo` — emit textual StableHLO MLIR for XLA/JAX
- Compilation result caching keyed by expression hash
- **Upcoming (v1.1):** AMD ROCm / `amdgcn` target

## Ball arithmetic

- `ArbBall`: real interval `[mid ± rad]` backed by Arb (FLINT)
- `AcbBall`: complex ball arithmetic
- `interval_eval`: rigorously evaluate a symbolic expression with ball inputs
- Guaranteed enclosures for all arithmetic and transcendental operations

## Numerical integration

- `compile_expr` + `eval_expr` for scalar evaluation
- `numpy_eval` for vectorised batch evaluation (NumPy, PyTorch, JAX arrays)
- DLPack support for zero-copy interop
- `to_jax` — register a symbolic expression as a JAX primitive with JVP and vmap rules

## Mathematical operations

- Symbolic matrices (`Matrix`), determinant, inverse, transpose
- Jacobian computation (`jacobian`)
- ODE representation and first-order lowering (`ODE`, `lower_to_first_order`)
- DAE structural analysis: Pantelides index reduction (`DAE`, `pantelides`)
- Acausal component modeling (`AcausalSystem`, `Port`, `resistor`)
- Sensitivity analysis: forward (`sensitivity_system`) and adjoint (`adjoint_system`)
- Hybrid systems with events (`HybridODE`, `Event`)
- Piecewise expressions and predicates

## Polynomial system solving (requires `--features groebner`)

- Gröbner basis: Buchberger F4 with product-criterion pruning
- Parallel F4 S-polynomial reduction via Rayon (`--features parallel`)
- CUDA Macaulay-matrix row reduction (`--features groebner-cuda`)
- Monomial orders: Lex, GrLex, GRevLex
- `solve` — symbolic solution of polynomial systems (linear + quadratic, exact symbolic output)
- Elimination ideals (`GroebnerBasis.eliminate`)

## Lean certificates (proof export)

- Derivation logs always on: ordered `RewriteStep` list with rule names and side conditions
- Lean 4 proof term export for: polynomial differentiation, trig differentiation, basic arithmetic rewrites
- Algorithmic certificates (witness-based): polynomial GCD, factoring (claims verified by Lean `ring_nf`)
- Lean CI: auto-generates proof corpus and verifies via lean compiler
- 20+ rule → Lean / Mathlib theorem mappings

## Primitive registry

- 23 registered primitives with full bundles: sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, exp, log, sqrt, abs, sign, erf, erfc, gamma, floor, ceil, round, min, max
- Each primitive: numerical evaluator, ball evaluator, forward/reverse diff, MLIR lowering, Lean theorem tag

## Error handling

- Structured exception hierarchy with stable codes (`E-POLY-*`, `E-DIFF-*`, etc.)
- Every exception: `.code`, `.message`, `.remediation`, `.span`
- 9 error subsystems: ConversionError, DomainError, DiffError, IntegrationError, MatrixError, OdeError, DaeError, JitError, CudaError, PoolError, SolverError

## Cross-CAS benchmarks

- Benchmark driver against SymPy, SymEngine, WolframEngine, Maple, SageMath
- HTML + JSONL reports via Criterion dashboard
- Nightly CI runs with `--competitors` flag

## Upcoming (v1.1)

- Algebraic-function Risch integration (Trager)
- AMD ROCm codegen
- PyPI wheel publishing
- E-graph default rule completeness (trig + log/exp out of the box)
- Python bindings: `ExprPool.save_to/load_from`, `GroebnerBasis.compute`, symbolic `solve` output
- LaTeX / Unicode pretty-printing
- String expression parsing

## Planned (v2.0+)

- Modular / CRT framework
- Resultants and subresultant PRS
- Sparse interpolation
- Real root isolation
- Hermite and Smith normal forms
- LLL lattice reduction and PSLQ
- Polynomial factorization (CZ, Berlekamp, Zassenhaus, van Hoeij)
- F5 / signature-based Gröbner
- Cylindrical algebraic decomposition (real QE)
- Creative telescoping / Zeilberger summation
- Primary decomposition
- Homotopy continuation solver
- Limits (Gruntz algorithm)
- Series expansion
- Eigenvalues and eigenvectors
- Difference equations (rsolve)
- Diophantine equations
- Integer number theory (FLINT bindings)
- Noncommutative algebra

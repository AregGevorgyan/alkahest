# Lean certificates

Alkahest can export machine-checkable Lean 4 proofs for a subset of computations.

## Three levels of evidence

**Derivation logs** — always on, always cheap. Records every rewrite rule applied, with rule name and arguments. Human-readable; machine-parseable; forms the basis for Lean export.

**Lean certificate export** — for computations expressible as sequences of rewrites tagged with Lean theorem names. The library emits a `.lean` file containing a proof term. Lean checks it independently.

**Algorithmic certificates** — for operations where rewrite sequences do not work (polynomial factoring, integration by the Risch algorithm). The library emits a verifiable witness instead. For factoring, the witness is the claimed factorization, which Lean verifies by multiplying out.

## Theorem mapping

Every primitive in the registry is tagged with a Lean 4 / Mathlib theorem name:

| Primitive rule | Mathlib theorem |
|---|---|
| `diff_sin` | `Real.hasDerivAt_sin` |
| `diff_exp` | `Real.hasDerivAt_exp` |
| `diff_log` | `Real.hasDerivAt_log` |
| `diff_chain` | `HasDerivAt.comp` |
| `diff_add` | `HasDerivAt.add` |
| `diff_mul` | `HasDerivAt.mul` |
| `add_zero` | `add_zero` |
| `mul_one` | `mul_one` |

The full mapping lives in `alkahest-core/src/lean/`.

## Exporting a certificate

```python
from alkahest import diff, sin

pool = ExprPool()
x = pool.symbol("x", "real")

dr = diff(sin(x**2), x)

# The certificate is in dr.certificate when Lean export is enabled
if dr.certificate:
    with open("proof.lean", "w") as f:
        f.write(dr.certificate)
```

The emitted `.lean` file imports Mathlib and contains a proof term that Lean can verify:

```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

-- Alkahest certificate: d/dx sin(x²) = 2*x*cos(x²)
theorem alkahest_diff_sin_sq (x : ℝ) :
    HasDerivAt (fun x => Real.sin (x ^ 2)) (2 * x * Real.cos (x ^ 2)) x := by
  have h1 : HasDerivAt (fun x => x ^ 2) (2 * x) x := ...
  exact (Real.hasDerivAt_sin _).comp x h1
```

## Lean CI

The CI pipeline (`.github/workflows/lean.yml`) runs on every change to `lean/`-tagged files:

1. Generates proof files via `tests/lean_corpus.py`
2. Compiles them with the `lean` compiler (with Mathlib cached)
3. Fails the build if any proof does not typecheck

## Coverage

Lean export works for computations that decompose into the tagged primitive set. Operations that currently produce certificates:

- Polynomial differentiation (all degrees)
- Trigonometric differentiation (`sin`, `cos`, `tan`, and chain compositions)
- Exponential and logarithm differentiation
- Basic arithmetic rewrites (`add_zero`, `mul_one`, `mul_comm`, etc.)

Operations that use algorithmic certificates (witness-based):

- Polynomial factoring — the claimed factorization is verified by `ring_nf` in Lean
- Polynomial GCD — verified by showing `gcd` divides both inputs and is a linear combination

Operations without Lean certificates (Lean theorem not yet mapped, or algorithm not expressible as rewrites):

- Integration by the Risch algorithm
- E-graph extraction steps

**Upcoming (v2.0):** Deeper Mathlib coverage including limits (`Filter.Tendsto`), series (`HasSum`), and real algebraic geometry (`Polynomial.roots`).

## Side conditions in proofs

Side conditions (domain constraints, branch cut restrictions) are propagated into the emitted Lean proof as hypotheses. A certificate that depends on `x > 0` will include `hx : 0 < x` as an assumption in the proof term. This makes the trust boundary explicit: the certificate is only verifiable when the side conditions hold.

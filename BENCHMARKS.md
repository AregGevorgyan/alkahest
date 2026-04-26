# Alkahest Benchmarks

Two complementary suites cover the full stack: **Rust criterion benchmarks**
for precise CPU timing, and a **Python script** that adds PyO3-boundary
overhead and `tracemalloc` peak-heap measurements.

---

## Rust benchmarks (criterion)

Located at `alkahest-core/benches/alkahest_bench.rs`. Uses
[criterion 0.5](https://docs.rs/criterion) with `html_reports`.

### Run

```bash
# Full suite — all groups, HTML report in target/criterion/
cargo bench -p alkahest-core

# One group only
cargo bench -p alkahest-core -- simplify

# Smoke pass (correctness check, no timing)
cargo bench -p alkahest-core -- --test

# Quick pass (3 s per bench instead of 5 s)
cargo bench -p alkahest-core -- --measurement-time 3
```

The HTML report is written to `target/criterion/report/index.html`.

### Benchmark groups

| Group | What it measures |
|---|---|
| `intern` | ExprPool hash-consing throughput — cached symbols, unique integer interning, `build_add3`, structural-sharing verification |
| `simplify` | Simplification engine — `x+0`, constant folding, polynomials degree 1–4, fixpoint-detection on already-simplified expressions |
| `diff` | Symbolic differentiation — polynomials degree 1–4, `sin(x²)`, `exp(sin(x))`, `log(poly)` |
| `unipoly` | FLINT-backed `UniPoly` — `from_symbolic` at degrees 2/4/8, degree-4 multiplication, GCD |
| `multipoly` | Sparse `MultiPoly` — univariate and bivariate `from_symbolic`, bivariate multiplication |
| `memory` | Per-operation heap bytes via a counting `GlobalAlloc`; validates that the pool doesn't grow on a second identical-expression build |
| `log_overhead` | `DerivationLog` step count after `diff` and `simplify` — measures logging cost separate from computation |

### Memory measurement

`alkahest_bench.rs` replaces the default allocator with a counting wrapper
(`CountingAllocator`) that tracks cumulative bytes allocated and number of
`alloc` calls via two `AtomicU64` globals. The `bench_memory` group uses
`iter_custom` to snapshot counters before and after each operation and
passes the deltas through `criterion::black_box` so they appear in
criterion output without being optimised away.

The `memory/hash_consing_second_build` case asserts `pool.len()` is
unchanged after rebuilding an identical expression tree — a regression test
for the intern table's structural-sharing guarantee.

### Comparing baselines

```bash
# Save a baseline
cargo bench -p alkahest-core -- --save-baseline before_change

# Make your change, then compare
cargo bench -p alkahest-core -- --baseline before_change
```

Criterion will print `Performance has regressed` / `improved` for each benchmark.

---

## Python benchmarks

`benchmarks/python_bench.py` measures the full PyO3 call path including
Python object construction, Rust GIL acquisition, and return-value wrapping.

### Dependencies

```bash
pip install hypothesis   # already in dev deps; nothing extra needed
```

### Run

```bash
# Full suite (~30 s)
python benchmarks/python_bench.py

# Quick smoke pass (~2 s)
python benchmarks/python_bench.py --quick
```

### What is measured

Each case reports:

| Column | Meaning |
|---|---|
| `Mean (µs)` | Minimum-of-repeats wall-clock time per iteration in microseconds |
| `Peak (KiB)` | Peak heap allocated by a single call (Python's `tracemalloc`) |
| `Notes` | Operation-specific annotation — step count for `simplify`/`diff`, etc. |

Cases covered:

- **intern** — `symbol()` 100×, `integer()` 100 unique values, hash-consing verify
- **simplify** — `x+0`, const fold, polynomials degree 1–4 (with step counts)
- **diff** — polynomials degree 1–4, `sin(x²)` (with step counts)
- **unipoly** — `from_symbolic` at degree 2/4/8, degree-4 multiplication
- **multipoly** — bivariate `from_symbolic`

### Interpreting results

The `Peak (KiB)` column is the peak of a **single call** measured by
`tracemalloc`. It reflects Python-side allocations; Rust-side heap traffic
from `rug`/FLINT is not visible here (use the Rust `memory` group or
Valgrind for that).

The `steps=N` annotation on `simplify`/`diff` rows gives the number of
`RewriteStep` entries in the returned `DerivedResult.steps` list — a proxy
for derivation-log overhead.

---

## Profiling beyond benchmarks

### Flame graph (Linux)

```bash
cargo install flamegraph
# Record a 10-second profile of the full bench suite
sudo cargo flamegraph -p alkahest-core --bench alkahest_bench -- --bench
# Open flamegraph.svg in a browser
```

### Valgrind Massif (heap profile)

```bash
cargo build -p alkahest-core --profile bench
valgrind --tool=massif --pages-as-heap=yes \
  ./target/release/deps/alkahest_bench-* --bench simplify
ms_print massif.out.* | head -60
```

### perf stat

```bash
perf stat -e cache-misses,cache-references,instructions \
  cargo bench -p alkahest-core -- simplify 2>&1
```

---

## Nightly deep run (CI)

The CI nightly job (`.github/workflows/ci.yml`) runs the full proptest suite
with `PROPTEST_CASES=100000` and the hypothesis suite with
`HYPOTHESIS_MAX_EXAMPLES=10000`. To reproduce locally:

```bash
PROPTEST_CASES=100000 cargo test --all --release
HYPOTHESIS_MAX_EXAMPLES=10000 python -m pytest tests/test_properties.py -v
```

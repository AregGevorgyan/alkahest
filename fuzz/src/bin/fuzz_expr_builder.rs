/// Fuzz target: expression construction and simplification.
///
/// Takes an arbitrary byte sequence and uses it to build a polynomial
/// expression in the alkahest kernel, then calls simplify(). Any panic
/// (not just wrong answer) is an AFL hit. The goal is to find inputs
/// that trigger panics, assertion failures, or infinite loops in the
/// simplification engine.
///
/// Run manually (requires cargo-afl):
///   cargo afl build --manifest-path fuzz/Cargo.toml --bin fuzz_expr_builder
///   cargo afl fuzz -i fuzz/in/expr_builder -o fuzz/out/expr_builder \
///       target/debug/fuzz_expr_builder
use afl::fuzz;
use alkahest_core::{simplify, Domain, ExprPool};

fn main() {
    fuzz!(|data: &[u8]| {
        if data.len() < 2 {
            return;
        }
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        // First byte: number of terms (capped at 8 to keep inputs manageable).
        let n_terms = (data[0] as usize % 8) + 1;

        // Remaining bytes: alternating (degree, coefficient) pairs.
        let mut terms = vec![];
        for i in 0..n_terms {
            let byte_idx = 1 + i * 2;
            if byte_idx + 1 >= data.len() {
                break;
            }
            let degree = data[byte_idx] as i32 % 8; // degrees 0–7
            let coeff = data[byte_idx + 1] as i64 - 128; // coefficients −128…127

            if coeff == 0 {
                continue;
            }
            let c_id = pool.integer(coeff);
            if degree == 0 {
                terms.push(c_id);
            } else {
                let xpow = pool.pow(x, pool.integer(degree));
                terms.push(pool.mul(vec![c_id, xpow]));
            }
        }

        if terms.is_empty() {
            return;
        }
        let expr = if terms.len() == 1 {
            terms[0]
        } else {
            pool.add(terms)
        };

        // Must not panic, must return a valid ExprId.
        let result = simplify(expr, &pool);
        let _ = std::hint::black_box(result.value);
    });
}

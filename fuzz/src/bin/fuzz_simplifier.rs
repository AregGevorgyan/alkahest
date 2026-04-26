/// Fuzz target: simplification idempotence.
///
/// Builds an expression from the input bytes and asserts
/// `simplify(simplify(e)) == simplify(e)`. A violation means the
/// simplification engine is not at fixpoint after one pass, which is a
/// correctness bug.
///
/// Run manually (requires cargo-afl):
///   cargo afl build --manifest-path fuzz/Cargo.toml --bin fuzz_simplifier
///   cargo afl fuzz -i fuzz/in/simplifier -o fuzz/out/simplifier \
///       target/debug/fuzz_simplifier
use afl::fuzz;
use alkahest_core::{simplify, Domain, ExprPool};

fn build_expr(
    pool: &ExprPool,
    x: alkahest_core::ExprId,
    data: &[u8],
) -> Option<alkahest_core::ExprId> {
    if data.len() < 2 {
        return None;
    }
    let n_terms = (data[0] as usize % 8) + 1;
    let mut terms = vec![];
    for i in 0..n_terms {
        let idx = 1 + i * 2;
        if idx + 1 >= data.len() {
            break;
        }
        let degree = data[idx] as i32 % 8;
        let coeff = data[idx + 1] as i64 - 128;
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
    match terms.len() {
        0 => None,
        1 => Some(terms[0]),
        _ => Some(pool.add(terms)),
    }
}

fn main() {
    fuzz!(|data: &[u8]| {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let Some(expr) = build_expr(&pool, x, data) else {
            return;
        };

        let r1 = simplify(expr, &pool);
        let r2 = simplify(r1.value, &pool);

        assert_eq!(
            r1.value, r2.value,
            "simplify idempotence violated: simplify(simplify(e)) != simplify(e)"
        );
    });
}

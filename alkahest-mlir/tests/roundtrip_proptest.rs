//! V1-5 acceptance: 1000 random-expression round-trip test.
//!
//! Emits each expression as high-level `alkahest` dialect text, parses it
//! back into an `ExprId`, and checks that the two evaluate to the same
//! number at several sampled points.

use alkahest_core::jit::eval_interp;
use alkahest_core::kernel::{Domain, ExprId, ExprPool};
use alkahest_mlir::{emit_dialect, parse_dialect, EmitOptions};
use proptest::prelude::*;
use std::collections::HashMap;

#[derive(Debug, Clone)]
enum ExprTree {
    Var(u8), // 0 → x, 1 → y
    Const(i32),
    Add(Box<ExprTree>, Box<ExprTree>),
    Mul(Box<ExprTree>, Box<ExprTree>),
    Pow(Box<ExprTree>, u8), // exponent 1..=5
    Call(&'static str, Box<ExprTree>),
}

fn expr_tree_strategy() -> impl Strategy<Value = ExprTree> {
    let leaf = prop_oneof![
        (0u8..=1).prop_map(ExprTree::Var),
        (-5i32..=5).prop_map(ExprTree::Const),
    ];
    leaf.prop_recursive(4, 40, 3, |inner| {
        prop_oneof![
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| ExprTree::Add(Box::new(a), Box::new(b))),
            (inner.clone(), inner.clone())
                .prop_map(|(a, b)| ExprTree::Mul(Box::new(a), Box::new(b))),
            (inner.clone(), 1u8..=5).prop_map(|(a, n)| ExprTree::Pow(Box::new(a), n)),
            (prop_oneof![Just("sin"), Just("cos"), Just("exp")], inner,)
                .prop_map(|(name, a)| ExprTree::Call(name, Box::new(a))),
        ]
    })
}

fn build(tree: &ExprTree, x: ExprId, y: ExprId, pool: &ExprPool) -> ExprId {
    match tree {
        ExprTree::Var(0) => x,
        ExprTree::Var(_) => y,
        ExprTree::Const(n) => pool.integer(*n),
        ExprTree::Add(a, b) => {
            let a = build(a, x, y, pool);
            let b = build(b, x, y, pool);
            pool.add(vec![a, b])
        }
        ExprTree::Mul(a, b) => {
            let a = build(a, x, y, pool);
            let b = build(b, x, y, pool);
            pool.mul(vec![a, b])
        }
        ExprTree::Pow(a, n) => {
            let a = build(a, x, y, pool);
            pool.pow(a, pool.integer(*n as i32))
        }
        ExprTree::Call(name, a) => {
            let a = build(a, x, y, pool);
            pool.func(*name, vec![a])
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        .. ProptestConfig::default()
    })]
    #[test]
    fn dialect_roundtrip_preserves_semantics(tree in expr_tree_strategy()) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = build(&tree, x, y, &pool);

        let text = emit_dialect(expr, &[x, y], &EmitOptions::default(), &pool);
        let back = parse_dialect(&text, &[x, y], &pool)
            .expect("parser should accept its own emitter's output");

        // Sample a few points; skip any where the original or round-trip
        // produces non-finite floats (e.g. `exp(large)` overflow).
        let samples = [(0.25f64, 1.5), (1.0, -0.5), (-1.25, 2.0)];
        for (xv, yv) in samples {
            let mut env = HashMap::new();
            env.insert(x, xv);
            env.insert(y, yv);
            let va = eval_interp(expr, &env, &pool).unwrap_or(f64::NAN);
            let vb = eval_interp(back, &env, &pool).unwrap_or(f64::NAN);
            if !va.is_finite() || !vb.is_finite() {
                continue;
            }
            let rel = (va - vb).abs() / (1.0 + va.abs());
            prop_assert!(
                rel < 1e-9,
                "mismatch at (x={xv}, y={yv}): orig={va}, roundtrip={vb}"
            );
        }
    }
}

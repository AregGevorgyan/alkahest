//! Minimal parser for the high-level `alkahest` dialect text.
//!
//! Supports only what the emitter in [`crate::emit`] produces, which is
//! enough to close the ExprId → MLIR → ExprId round-trip exercised by
//! V1-5's test plan.  This is *not* a general MLIR parser.

use alkahest_core::kernel::{Domain, ExprId, ExprPool};
use std::collections::HashMap;

use crate::ops::AlkahestOp;

/// Parse a high-level `alkahest` dialect module back into an expression.
///
/// `inputs` is the symbolic-expression interpretation of `%arg0`, `%arg1`, ….
/// Returns `None` if the module can't be parsed (malformed text, unsupported
/// ops, mismatched `%arg` count, missing terminator).
pub fn parse_dialect(text: &str, inputs: &[ExprId], pool: &ExprPool) -> Option<ExprId> {
    let mut env: HashMap<String, ExprId> = HashMap::new();
    for (i, &id) in inputs.iter().enumerate() {
        env.insert(format!("%arg{i}"), id);
    }

    let mut result: Option<ExprId> = None;

    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        // Strip trailing `// …` comments (emitter never uses them inside ops
        // but the round-trip corpus might).
        let line = match line.split_once("//") {
            Some((lhs, _)) => lhs.trim_end(),
            None => line,
        };
        if line.is_empty() {
            continue;
        }

        // func.return %v : f64
        if let Some(rest) = line.strip_prefix("func.return ") {
            let name = rest.split_whitespace().next()?;
            result = env.get(name).copied();
            continue;
        }

        // Skip module/func boilerplate.
        if line.starts_with("module")
            || line.starts_with("func.func")
            || line.starts_with("}")
            || line.starts_with("{")
        {
            continue;
        }

        // Op line: `%v0 = <op> <operands> [{ attrs }] : <sig>`
        let (lhs, rhs) = line.split_once('=')?;
        let ssa = lhs.trim().to_string();
        let rhs = rhs.trim();

        // Mnemonic is the first whitespace-separated token.
        let (mnem, rest) = match rhs.split_once(char::is_whitespace) {
            Some((m, r)) => (m, r),
            None => (rhs, ""),
        };

        let op = AlkahestOp::from_mnemonic(mnem)?;
        let emitted = parse_op(op, rest, &env, pool)?;
        env.insert(ssa, emitted);
    }

    result
}

fn parse_op(
    op: AlkahestOp,
    rest: &str,
    env: &HashMap<String, ExprId>,
    pool: &ExprPool,
) -> Option<ExprId> {
    match op {
        AlkahestOp::Sym => {
            // Treat as a free symbol.  Generate a fresh name; we don't expect
            // this in the round-trip path (inputs are in `env`) but we produce
            // something so the parser is total.
            Some(pool.symbol("_unresolved", Domain::Real))
        }
        AlkahestOp::Const => {
            // `alkahest.const {value = X : f64} : f64`
            let val = extract_attr_f64(rest, "value")?;
            Some(float_literal(val, pool))
        }
        AlkahestOp::Add | AlkahestOp::Mul => {
            let operands = extract_operands_before_sig(rest);
            let children: Option<Vec<ExprId>> =
                operands.iter().map(|s| env.get(s).copied()).collect();
            let children = children?;
            Some(match op {
                AlkahestOp::Add => pool.add(children),
                AlkahestOp::Mul => pool.mul(children),
                _ => unreachable!(),
            })
        }
        AlkahestOp::Pow => {
            let operands = extract_operands_before_sig(rest);
            if operands.len() != 2 {
                return None;
            }
            let base = env.get(&operands[0]).copied()?;
            let exp = env.get(&operands[1]).copied()?;
            Some(pool.pow(base, exp))
        }
        AlkahestOp::Call => {
            // `alkahest.call @name(%a, %b, …) : (f64, …) -> f64`
            let at = rest.find('@')?;
            let after_at = &rest[at + 1..];
            let paren_open = after_at.find('(')?;
            let name = after_at[..paren_open].trim().to_string();
            let paren_close = after_at.find(')')?;
            let args_str = &after_at[paren_open + 1..paren_close];
            let operands: Vec<String> = args_str
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let children: Option<Vec<ExprId>> =
                operands.iter().map(|s| env.get(s).copied()).collect();
            Some(pool.func(name, children?))
        }
        AlkahestOp::Horner | AlkahestOp::PolyEval => {
            // `alkahest.horner %x {coeffs = dense<[c0, c1, …]> : tensor<Nxf64>,
            //                     form = "horner"} : (f64) -> f64`
            let operands = extract_operands_before_attrs(rest);
            if operands.is_empty() {
                return None;
            }
            let x = env.get(&operands[0]).copied()?;
            let coeffs = extract_dense_coeffs(rest)?;
            Some(build_horner_expr(&coeffs, x, pool))
        }
        AlkahestOp::SeriesTaylor | AlkahestOp::IntervalEval | AlkahestOp::RationalFn => {
            // Parser support for these is a v1.1 follow-up; the round-trip
            // emitter does not produce them yet.
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Small parsing helpers
// ---------------------------------------------------------------------------

/// `operands, operands, … {attrs} : <sig>` → extract just the operands.
fn extract_operands_before_attrs(rest: &str) -> Vec<String> {
    // cut at first `{` or `:` — whichever is earlier
    let cut = match (rest.find('{'), rest.find(':')) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    };
    let head = match cut {
        Some(i) => &rest[..i],
        None => rest,
    };
    head.split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| s.starts_with('%'))
        .collect()
}

fn extract_operands_before_sig(rest: &str) -> Vec<String> {
    extract_operands_before_attrs(rest)
}

fn extract_attr_f64(rest: &str, key: &str) -> Option<f64> {
    // `{key = VALUE : f64}`
    let needle = format!("{key} =");
    let at = rest.find(&needle)?;
    let after = &rest[at + needle.len()..].trim_start();
    // Read up to `:` or `,` or `}`.
    let end = after.find([':', ',', '}']).unwrap_or(after.len());
    after[..end].trim().parse::<f64>().ok()
}

fn extract_dense_coeffs(rest: &str) -> Option<Vec<i64>> {
    // `dense<[a.0, b.0, c.0]>`
    let start = rest.find("dense<[")?;
    let after = &rest[start + "dense<[".len()..];
    let end = after.find(']')?;
    let inner = &after[..end];
    let mut out = Vec::new();
    for piece in inner.split(',') {
        let tok = piece.trim();
        if tok.is_empty() {
            continue;
        }
        let parsed = tok.parse::<f64>().ok()?;
        out.push(parsed as i64);
    }
    Some(out)
}

fn float_literal(val: f64, pool: &ExprPool) -> ExprId {
    // Prefer exact integer if possible.
    if val.fract() == 0.0 && val.abs() < (1u64 << 53) as f64 {
        pool.integer(val as i64)
    } else {
        pool.float(val, 64)
    }
}

fn build_horner_expr(coeffs: &[i64], x: ExprId, pool: &ExprPool) -> ExprId {
    if coeffs.is_empty() {
        return pool.integer(0_i32);
    }
    let n = coeffs.len();
    let mut acc = pool.integer(coeffs[n - 1]);
    for k in (0..n - 1).rev() {
        let xr = pool.mul(vec![x, acc]);
        acc = pool.add(vec![pool.integer(coeffs[k]), xr]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::{emit_dialect, EmitOptions};
    use alkahest_core::jit::eval_interp;
    use alkahest_core::kernel::{Domain, ExprPool};
    use std::collections::HashMap;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    fn assert_numerically_equivalent(
        a: ExprId,
        b: ExprId,
        vars: &[(ExprId, &[f64])],
        pool: &ExprPool,
    ) {
        // Cartesian product over assignments in `vars`.
        let n = vars.iter().map(|(_, xs)| xs.len()).product::<usize>();
        for idx in 0..n {
            let mut env: HashMap<ExprId, f64> = HashMap::new();
            let mut i = idx;
            for (v, xs) in vars {
                let pick = xs[i % xs.len()];
                env.insert(*v, pick);
                i /= xs.len();
            }
            let va = eval_interp(a, &env, pool).unwrap();
            let vb = eval_interp(b, &env, pool).unwrap();
            assert!(
                (va - vb).abs() < 1e-9,
                "mismatch at env={env:?}: {va} vs {vb}"
            );
        }
    }

    #[test]
    fn roundtrip_symbol() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let text = emit_dialect(x, &[x], &EmitOptions::default(), &pool);
        let back = parse_dialect(&text, &[x], &pool).unwrap();
        assert_eq!(back, x);
    }

    #[test]
    fn roundtrip_add() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let expr = pool.add(vec![x, y]);
        let text = emit_dialect(expr, &[x, y], &EmitOptions::default(), &pool);
        let back = parse_dialect(&text, &[x, y], &pool).unwrap();
        assert_numerically_equivalent(expr, back, &[(x, &[1.0, 2.0]), (y, &[3.0, 4.0])], &pool);
    }

    #[test]
    fn roundtrip_polynomial_via_horner() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        // x^3 - 2x + 5
        let x2 = pool.pow(x, pool.integer(2_i32));
        let x3 = pool.pow(x, pool.integer(3_i32));
        let expr = pool.add(vec![
            x3,
            pool.mul(vec![pool.integer(-2_i32), x]),
            pool.integer(5_i32),
        ]);
        let _ = x2;
        let text = emit_dialect(expr, &[x], &EmitOptions::default(), &pool);
        assert!(text.contains("alkahest.horner"), "{text}");
        let back = parse_dialect(&text, &[x], &pool).unwrap();
        assert_numerically_equivalent(expr, back, &[(x, &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.5])], &pool);
    }

    #[test]
    fn roundtrip_call() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let expr = pool.func("sin", vec![x]);
        let text = emit_dialect(expr, &[x], &EmitOptions::default(), &pool);
        let back = parse_dialect(&text, &[x], &pool).unwrap();
        assert_numerically_equivalent(expr, back, &[(x, &[0.0, 0.5, 1.0])], &pool);
    }

    #[test]
    fn roundtrip_mul_and_pow() {
        let pool = p();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        // x*y + x^2  (should *not* lift to Horner because it's multivariate)
        let expr = pool.add(vec![pool.mul(vec![x, y]), pool.pow(x, pool.integer(2_i32))]);
        let text = emit_dialect(expr, &[x, y], &EmitOptions::default(), &pool);
        let back = parse_dialect(&text, &[x, y], &pool).unwrap();
        assert_numerically_equivalent(
            expr,
            back,
            &[(x, &[-1.0, 0.5, 2.0]), (y, &[-2.0, 0.0, 3.0])],
            &pool,
        );
    }
}

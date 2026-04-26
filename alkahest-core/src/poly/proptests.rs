use super::multipoly::MultiPoly;
use super::rational::RationalFunction;
use super::unipoly::UniPoly;
use crate::kernel::{Domain, ExprId, ExprPool};
use proptest::prelude::*;
use std::collections::BTreeMap;

fn small_coeff() -> impl Strategy<Value = i64> {
    -20i64..=20i64
}

// ---------------------------------------------------------------------------
// MultiPoly helpers
// ---------------------------------------------------------------------------

/// Exponent vector for a bivariate polynomial (2 variables, degrees 0-3).
/// Trailing zeros are stripped so the result matches MultiPoly's invariant.
fn arb_exps() -> impl Strategy<Value = Vec<u32>> {
    proptest::collection::vec(0u32..=3u32, 0usize..=2usize).prop_map(|mut e| {
        while e.last() == Some(&0) {
            e.pop();
        }
        e
    })
}

fn arb_raw_terms() -> impl Strategy<Value = Vec<(Vec<u32>, i64)>> {
    proptest::collection::vec((arb_exps(), -10i64..=10i64), 0usize..=6usize)
}

fn build_multipoly(raw: Vec<(Vec<u32>, i64)>, x: ExprId, y: ExprId) -> MultiPoly {
    let mut terms: BTreeMap<Vec<u32>, rug::Integer> = BTreeMap::new();
    for (exp, coeff) in raw {
        if coeff == 0 {
            continue;
        }
        let entry = terms.entry(exp).or_insert_with(|| rug::Integer::from(0i64));
        *entry += coeff;
    }
    terms.retain(|_, v| *v != 0);
    MultiPoly {
        vars: vec![x, y],
        terms,
    }
}

// ---------------------------------------------------------------------------
// Properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn add_commutative(
        ca in proptest::collection::vec(small_coeff(), 1..=5),
        cb in proptest::collection::vec(small_coeff(), 1..=5),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let b = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cb) };
        prop_assert_eq!(&a + &b, &b + &a);
    }

    #[test]
    fn add_associative(
        ca in proptest::collection::vec(small_coeff(), 1..=4),
        cb in proptest::collection::vec(small_coeff(), 1..=4),
        cc in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let b = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cb) };
        let c = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cc) };
        prop_assert_eq!((&a + &b) + c.clone(), a + (&b + &c));
    }

    #[test]
    fn mul_commutative(
        ca in proptest::collection::vec(small_coeff(), 1..=4),
        cb in proptest::collection::vec(small_coeff(), 1..=4),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let b = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cb) };
        prop_assert_eq!(&a * &b, &b * &a);
    }

    #[test]
    fn mul_associative(
        ca in proptest::collection::vec(small_coeff(), 1..=3),
        cb in proptest::collection::vec(small_coeff(), 1..=3),
        cc in proptest::collection::vec(small_coeff(), 1..=3),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let b = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cb) };
        let c = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cc) };
        prop_assert_eq!((&a * &b) * c.clone(), a * (&b * &c));
    }

    #[test]
    fn distributive(
        ca in proptest::collection::vec(small_coeff(), 1..=3),
        cb in proptest::collection::vec(small_coeff(), 1..=3),
        cc in proptest::collection::vec(small_coeff(), 1..=3),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let b = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cb) };
        let c = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&cc) };
        // a * (b + c) == a*b + a*c
        let lhs = &a * &(&b + &c);
        let rhs = &(&a * &b) + &(&a * &c);
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn add_zero_identity(
        ca in proptest::collection::vec(small_coeff(), 1..=5),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let z = UniPoly::zero(x);
        prop_assert_eq!(&a + &z, a.clone());
    }

    #[test]
    fn mul_one_identity(
        ca in proptest::collection::vec(small_coeff(), 1..=5),
    ) {
        use crate::flint::FlintPoly;
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let a = UniPoly { var: x, coeffs: FlintPoly::from_coefficients(&ca) };
        let one = UniPoly::constant(x, 1);
        prop_assert_eq!(&a * &one, a.clone());
    }
}

// ---------------------------------------------------------------------------
// MultiPoly property tests
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn multipoly_add_commutative(ta in arb_raw_terms(), tb in arb_raw_terms()) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let b = build_multipoly(tb, x, y);
        prop_assert_eq!(a.clone() + b.clone(), b + a);
    }

    #[test]
    fn multipoly_add_associative(
        ta in arb_raw_terms(),
        tb in arb_raw_terms(),
        tc in arb_raw_terms(),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let b = build_multipoly(tb, x, y);
        let c = build_multipoly(tc, x, y);
        prop_assert_eq!((a.clone() + b.clone()) + c.clone(), a + (b + c));
    }

    #[test]
    fn multipoly_mul_commutative(ta in arb_raw_terms(), tb in arb_raw_terms()) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let b = build_multipoly(tb, x, y);
        prop_assert_eq!(a.clone() * b.clone(), b * a);
    }

    #[test]
    fn multipoly_distributive(
        ta in arb_raw_terms(),
        tb in arb_raw_terms(),
        tc in arb_raw_terms(),
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let b = build_multipoly(tb, x, y);
        let c = build_multipoly(tc, x, y);
        // a * (b + c) == a*b + a*c
        let lhs = a.clone() * (b.clone() + c.clone());
        let rhs = (a.clone() * b) + (a * c);
        prop_assert_eq!(lhs, rhs);
    }

    #[test]
    fn multipoly_add_zero_identity(ta in arb_raw_terms()) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let zero = MultiPoly::zero(vec![x, y]);
        prop_assert_eq!(a.clone() + zero, a);
    }

    #[test]
    fn multipoly_mul_one_identity(ta in arb_raw_terms()) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);
        let a = build_multipoly(ta, x, y);
        let one = MultiPoly::constant(vec![x, y], 1);
        prop_assert_eq!(a.clone() * one, a);
    }

    #[test]
    fn multipoly_from_symbolic_round_trip(
        ca in -10i64..=10i64,
        cb in -10i64..=10i64,
        cc in -10i64..=10i64,
    ) {
        // Build ca + cb*x + cc*y symbolically and check from_symbolic recovers the terms.
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);
        let y = pool.symbol("y", Domain::Real);

        let mut addends: Vec<ExprId> = Vec::new();
        if ca != 0 { addends.push(pool.integer(ca)); }
        if cb != 0 { addends.push(pool.mul(vec![pool.integer(cb), x])); }
        if cc != 0 { addends.push(pool.mul(vec![pool.integer(cc), y])); }

        let expr = match addends.len() {
            0 => pool.integer(0_i32),
            1 => addends[0],
            _ => pool.add(addends),
        };

        let poly = MultiPoly::from_symbolic(expr, vec![x, y], &pool).unwrap();

        let got_const = poly.terms.get(&vec![]).cloned().unwrap_or(rug::Integer::from(0));
        let got_x = poly.terms.get(&vec![1u32]).cloned().unwrap_or(rug::Integer::from(0));
        let got_y = poly.terms.get(&vec![0u32, 1u32]).cloned().unwrap_or(rug::Integer::from(0));

        prop_assert_eq!(got_const, rug::Integer::from(ca));
        prop_assert_eq!(got_x, rug::Integer::from(cb));
        prop_assert_eq!(got_y, rug::Integer::from(cc));
    }
}

// ---------------------------------------------------------------------------
// RationalFunction property tests
// ---------------------------------------------------------------------------

/// Build a nonzero univariate poly from two small coefficients [c0, c1] (linear).
/// Returns None if both are zero.
fn nonzero_unipoly(c0: i64, c1: i64, x: ExprId) -> Option<MultiPoly> {
    if c0 == 0 && c1 == 0 {
        return None;
    }
    let mut terms = BTreeMap::new();
    if c0 != 0 {
        terms.insert(vec![], rug::Integer::from(c0));
    }
    if c1 != 0 {
        terms.insert(vec![1u32], rug::Integer::from(c1));
    }
    Some(MultiPoly {
        vars: vec![x],
        terms,
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// (a/b) * (b/c) == a/c for nonzero linear polynomials a, b, c.
    #[test]
    fn rational_mul_cancel(
        a0 in -5i64..=5i64, a1 in -3i64..=3i64,
        b0 in -5i64..=5i64, b1 in -3i64..=3i64,
        c0 in -5i64..=5i64, c1 in -3i64..=3i64,
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let a = match nonzero_unipoly(a0, a1, x) { Some(p) => p, None => return Ok(()) };
        let b = match nonzero_unipoly(b0, b1, x) { Some(p) => p, None => return Ok(()) };
        let c = match nonzero_unipoly(c0, c1, x) { Some(p) => p, None => return Ok(()) };

        let ab = match RationalFunction::new(a.clone(), b.clone()) { Ok(r) => r, Err(_) => return Ok(()) };
        let bc = match RationalFunction::new(b.clone(), c.clone()) { Ok(r) => r, Err(_) => return Ok(()) };
        let ac = match RationalFunction::new(a.clone(), c.clone()) { Ok(r) => r, Err(_) => return Ok(()) };

        let product = match ab * bc  { Ok(r) => r, Err(_) => return Ok(()) };
        prop_assert_eq!(product, ac, "(a/b)*(b/c) should equal a/c");
    }

    /// Additive identity: (a/b) + (zero/b) == (a/b).
    #[test]
    fn rational_add_zero_identity(
        a0 in -5i64..=5i64, a1 in -3i64..=3i64,
        b0 in -5i64..=5i64, b1 in -3i64..=3i64,
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let a = match nonzero_unipoly(a0, a1, x) { Some(p) => p, None => return Ok(()) };
        let b = match nonzero_unipoly(b0, b1, x) { Some(p) => p, None => return Ok(()) };

        let ab = match RationalFunction::new(a.clone(), b.clone()) { Ok(r) => r, Err(_) => return Ok(()) };
        let zero_over_b = match RationalFunction::new(MultiPoly::zero(vec![x]), b.clone()) {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };
        let sum = match ab.clone() + zero_over_b  { Ok(r) => r, Err(_) => return Ok(()) };
        prop_assert_eq!(sum, ab, "(a/b) + 0 should equal a/b");
    }

    /// Multiplication is commutative: (a/b) * (c/d) == (c/d) * (a/b).
    #[test]
    fn rational_mul_commutative(
        a0 in -5i64..=5i64, a1 in -3i64..=3i64,
        b0 in -5i64..=5i64, b1 in -3i64..=3i64,
        c0 in -5i64..=5i64, c1 in -3i64..=3i64,
        d0 in -5i64..=5i64, d1 in -3i64..=3i64,
    ) {
        let pool = ExprPool::new();
        let x = pool.symbol("x", Domain::Real);

        let a = match nonzero_unipoly(a0, a1, x) { Some(p) => p, None => return Ok(()) };
        let b = match nonzero_unipoly(b0, b1, x) { Some(p) => p, None => return Ok(()) };
        let c = match nonzero_unipoly(c0, c1, x) { Some(p) => p, None => return Ok(()) };
        let d = match nonzero_unipoly(d0, d1, x) { Some(p) => p, None => return Ok(()) };

        let ab = match RationalFunction::new(a.clone(), b.clone()) { Ok(r) => r, Err(_) => return Ok(()) };
        let cd = match RationalFunction::new(c.clone(), d.clone()) { Ok(r) => r, Err(_) => return Ok(()) };

        let lhs = match ab.clone() * cd.clone()  { Ok(r) => r, Err(_) => return Ok(()) };
        let rhs = match cd * ab  { Ok(r) => r, Err(_) => return Ok(()) };
        prop_assert_eq!(lhs, rhs, "(a/b)*(c/d) should equal (c/d)*(a/b)");
    }
}

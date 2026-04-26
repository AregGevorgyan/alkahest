//! Op catalog for the `alkahest` MLIR dialect.
//!
//! Each [`AlkahestOp`] corresponds to a named op.  The table here is the
//! single source of truth for mnemonic, arity, and result type shape
//! until the real ODS table-gen lands behind `mlir-native`.

/// Ops exposed by the `alkahest` dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlkahestOp {
    /// `alkahest.sym` — a symbolic function argument (opaque f64).
    Sym,
    /// `alkahest.const` — an f64 literal (integer/rational/float folded).
    Const,
    /// `alkahest.add` — n-ary floating-point addition.
    Add,
    /// `alkahest.mul` — n-ary floating-point multiplication.
    Mul,
    /// `alkahest.pow` — `base^exp`; integer `exp` stays symbolic so that
    /// the lowering pass can pick fast-path expansions.
    Pow,
    /// `alkahest.call` — call a registered primitive
    /// (`sin`, `cos`, `exp`, `log`, `sqrt`, …).
    Call,
    /// `alkahest.horner` — `a0 + x*(a1 + x*(a2 + …))` with a dense coefficient
    /// attribute.  Canonical form for polynomial evaluation.
    Horner,
    /// `alkahest.poly_eval` — naive polynomial evaluation; the canonicalize
    /// pass rewrites it into [`AlkahestOp::Horner`].
    PolyEval,
    /// `alkahest.series_taylor` — Taylor series of a callable around a point.
    SeriesTaylor,
    /// `alkahest.interval_eval` — evaluation on an `ArbBall` interval.
    IntervalEval,
    /// `alkahest.rational_fn` — ratio of two polynomials (shares
    /// `poly_eval`'s attribute shape for numerator / denominator).
    RationalFn,
}

impl AlkahestOp {
    /// Full MLIR mnemonic (with `alkahest.` prefix).
    pub fn mnemonic(self) -> &'static str {
        match self {
            AlkahestOp::Sym => "alkahest.sym",
            AlkahestOp::Const => "alkahest.const",
            AlkahestOp::Add => "alkahest.add",
            AlkahestOp::Mul => "alkahest.mul",
            AlkahestOp::Pow => "alkahest.pow",
            AlkahestOp::Call => "alkahest.call",
            AlkahestOp::Horner => "alkahest.horner",
            AlkahestOp::PolyEval => "alkahest.poly_eval",
            AlkahestOp::SeriesTaylor => "alkahest.series_taylor",
            AlkahestOp::IntervalEval => "alkahest.interval_eval",
            AlkahestOp::RationalFn => "alkahest.rational_fn",
        }
    }

    /// Parse a mnemonic (with or without `alkahest.` prefix) into an op.
    pub fn from_mnemonic(s: &str) -> Option<Self> {
        let short = s.strip_prefix("alkahest.").unwrap_or(s);
        Some(match short {
            "sym" => AlkahestOp::Sym,
            "const" => AlkahestOp::Const,
            "add" => AlkahestOp::Add,
            "mul" => AlkahestOp::Mul,
            "pow" => AlkahestOp::Pow,
            "call" => AlkahestOp::Call,
            "horner" => AlkahestOp::Horner,
            "poly_eval" => AlkahestOp::PolyEval,
            "series_taylor" => AlkahestOp::SeriesTaylor,
            "interval_eval" => AlkahestOp::IntervalEval,
            "rational_fn" => AlkahestOp::RationalFn,
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mnemonic_roundtrip() {
        for op in [
            AlkahestOp::Sym,
            AlkahestOp::Const,
            AlkahestOp::Add,
            AlkahestOp::Mul,
            AlkahestOp::Pow,
            AlkahestOp::Call,
            AlkahestOp::Horner,
            AlkahestOp::PolyEval,
            AlkahestOp::SeriesTaylor,
            AlkahestOp::IntervalEval,
            AlkahestOp::RationalFn,
        ] {
            assert_eq!(AlkahestOp::from_mnemonic(op.mnemonic()), Some(op));
        }
    }

    #[test]
    fn short_form_accepted() {
        assert_eq!(
            AlkahestOp::from_mnemonic("horner"),
            Some(AlkahestOp::Horner)
        );
    }
}

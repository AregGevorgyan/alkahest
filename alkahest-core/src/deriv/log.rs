use crate::kernel::{Domain, ExprId, ExprPool};
use std::fmt;

// ---------------------------------------------------------------------------
// SideCondition
// ---------------------------------------------------------------------------

/// A logical side-condition recorded alongside a rewrite step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SideCondition {
    /// The expression is assumed to be non-zero.
    NonZero(ExprId),
    /// The expression is assumed to be positive.
    Positive(ExprId),
    /// The expression is assumed to lie in the given domain.
    InDomain(ExprId, Domain),
}

impl SideCondition {
    pub fn display_with<'a>(&'a self, pool: &'a ExprPool) -> SideConditionDisplay<'a> {
        SideConditionDisplay { cond: self, pool }
    }
}

impl fmt::Display for SideCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SideCondition::NonZero(id) => write!(f, "nonzero({id:?})"),
            SideCondition::Positive(id) => write!(f, "positive({id:?})"),
            SideCondition::InDomain(id, d) => write!(f, "in_domain({id:?}, {d:?})"),
        }
    }
}

pub struct SideConditionDisplay<'a> {
    cond: &'a SideCondition,
    pool: &'a ExprPool,
}

impl fmt::Display for SideConditionDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cond {
            SideCondition::NonZero(id) => {
                write!(f, "{} ≠ 0", self.pool.display(*id))
            }
            SideCondition::Positive(id) => {
                write!(f, "{} > 0", self.pool.display(*id))
            }
            SideCondition::InDomain(id, d) => {
                write!(f, "{} ∈ {:?}", self.pool.display(*id), d)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RewriteStep
// ---------------------------------------------------------------------------

/// A single rule application: `before` rewrote to `after` via `rule_name`.
#[derive(Debug, Clone)]
pub struct RewriteStep {
    pub rule_name: &'static str,
    pub before: ExprId,
    pub after: ExprId,
    pub side_conditions: Vec<SideCondition>,
}

impl RewriteStep {
    pub fn simple(rule_name: &'static str, before: ExprId, after: ExprId) -> Self {
        RewriteStep {
            rule_name,
            before,
            after,
            side_conditions: vec![],
        }
    }

    pub fn with_conditions(
        rule_name: &'static str,
        before: ExprId,
        after: ExprId,
        side_conditions: Vec<SideCondition>,
    ) -> Self {
        RewriteStep {
            rule_name,
            before,
            after,
            side_conditions,
        }
    }
}

// ---------------------------------------------------------------------------
// DerivationLog
// ---------------------------------------------------------------------------

/// An ordered sequence of rewrite steps recording how an expression was transformed.
#[derive(Debug, Clone, Default)]
pub struct DerivationLog(pub Vec<RewriteStep>);

impl DerivationLog {
    pub fn new() -> Self {
        DerivationLog(Vec::new())
    }

    pub fn push(&mut self, step: RewriteStep) {
        self.0.push(step);
    }

    /// Append all steps from `other` after `self`.
    pub fn merge(mut self, other: DerivationLog) -> DerivationLog {
        self.0.extend(other.0);
        self
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn steps(&self) -> &[RewriteStep] {
        &self.0
    }

    /// Pool-aware display: shows full expression text for before/after.
    pub fn display_with<'a>(&'a self, pool: &'a ExprPool) -> LogDisplay<'a> {
        LogDisplay { log: self, pool }
    }
}

/// Compact display without a pool: shows rule names and ExprId numbers.
impl fmt::Display for DerivationLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "(no steps)");
        }
        for (i, step) in self.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(
                f,
                "step {}: {} ({:?} → {:?})",
                i + 1,
                step.rule_name,
                step.before,
                step.after
            )?;
        }
        Ok(())
    }
}

/// Pool-aware display: shows full expression text for before/after.
pub struct LogDisplay<'a> {
    log: &'a DerivationLog,
    pool: &'a ExprPool,
}

impl fmt::Display for LogDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.log.0.is_empty() {
            return write!(f, "(no steps)");
        }
        for (i, step) in self.log.0.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(
                f,
                "step {}: {} applied to {} → {}",
                i + 1,
                step.rule_name,
                self.pool.display(step.before),
                self.pool.display(step.after)
            )?;
            for cond in &step.side_conditions {
                write!(f, "  [{}]", cond.display_with(self.pool))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DerivedExpr<T>
// ---------------------------------------------------------------------------

/// The return type of every expression-transforming operation.
/// Carries the resulting value plus the log of all steps that produced it.
#[derive(Debug, Clone)]
pub struct DerivedExpr<T> {
    pub value: T,
    pub log: DerivationLog,
}

impl<T> DerivedExpr<T> {
    /// Wrap a value with an empty log (no steps taken).
    pub fn new(value: T) -> Self {
        DerivedExpr {
            value,
            log: DerivationLog::new(),
        }
    }

    /// Wrap with a single log step.
    pub fn with_step(value: T, step: RewriteStep) -> Self {
        let mut log = DerivationLog::new();
        log.push(step);
        DerivedExpr { value, log }
    }

    /// Wrap with a pre-built log.
    pub fn with_log(value: T, log: DerivationLog) -> Self {
        DerivedExpr { value, log }
    }

    /// Transform the value without adding steps.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> DerivedExpr<U> {
        DerivedExpr {
            value: f(self.value),
            log: self.log,
        }
    }

    /// Monadic chain: call `f` which may itself produce steps; merge the logs.
    pub fn and_then<U, F: FnOnce(T) -> DerivedExpr<U>>(self, f: F) -> DerivedExpr<U> {
        let next = f(self.value);
        DerivedExpr {
            value: next.value,
            log: self.log.merge(next.log),
        }
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};

    fn pool_and_x() -> (ExprPool, ExprId) {
        let p = ExprPool::new();
        let x = p.symbol("x", Domain::Real);
        (p, x)
    }

    #[test]
    fn log_push_and_len() {
        let (p, x) = pool_and_x();
        let one = p.integer(1_i32);
        let mut log = DerivationLog::new();
        assert!(log.is_empty());
        log.push(RewriteStep::simple("test_rule", x, one));
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn log_merge_order() {
        let (p, x) = pool_and_x();
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        let mut a = DerivationLog::new();
        a.push(RewriteStep::simple("rule_a", x, one));
        let mut b = DerivationLog::new();
        b.push(RewriteStep::simple("rule_b", one, two));
        let merged = a.merge(b);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.steps()[0].rule_name, "rule_a");
        assert_eq!(merged.steps()[1].rule_name, "rule_b");
    }

    #[test]
    fn display_without_pool() {
        let (p, x) = pool_and_x();
        let one = p.integer(1_i32);
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("add_zero", x, one));
        let s = log.to_string();
        assert!(s.contains("step 1"), "should have step 1: {s}");
        assert!(s.contains("add_zero"), "should mention rule: {s}");
    }

    #[test]
    fn display_with_pool() {
        let (p, x) = pool_and_x();
        let one = p.integer(1_i32);
        let mut log = DerivationLog::new();
        log.push(RewriteStep::simple("diff_identity", x, one));
        let s = log.display_with(&p).to_string();
        assert!(s.contains("diff_identity"), "{s}");
        assert!(s.contains('x'), "{s}");
        assert!(s.contains('1'), "{s}");
    }

    #[test]
    fn side_condition_display() {
        let (p, x) = pool_and_x();
        let cond = SideCondition::NonZero(x);
        let s = cond.display_with(&p).to_string();
        assert!(s.contains('x'), "{s}");
        assert!(s.contains('0'), "{s}");
    }

    #[test]
    fn derived_expr_map() {
        let d: DerivedExpr<i32> = DerivedExpr::new(5);
        let doubled = d.map(|v| v * 2);
        assert_eq!(doubled.value, 10);
        assert!(doubled.log.is_empty());
    }

    #[test]
    fn derived_expr_and_then_merges_logs() {
        let (p, x) = pool_and_x();
        let one = p.integer(1_i32);
        let two = p.integer(2_i32);
        let d = DerivedExpr::with_step(x, RewriteStep::simple("step_a", x, one));
        let result =
            d.and_then(|_| DerivedExpr::with_step(two, RewriteStep::simple("step_b", one, two)));
        assert_eq!(result.value, two);
        assert_eq!(result.log.len(), 2);
        assert_eq!(result.log.steps()[0].rule_name, "step_a");
        assert_eq!(result.log.steps()[1].rule_name, "step_b");
    }
}

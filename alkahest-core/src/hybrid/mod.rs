//! Phase 20 — Event handling for hybrid dynamical systems.
//!
//! A *hybrid* system alternates between continuous-time dynamics (governed by
//! an ODE) and instantaneous discrete events.  An event fires when a
//! *zero-crossing condition* `g(t, y) = 0` is detected.  At that instant,
//! the state is reset according to a *reset map*.
//!
//! This module provides:
//! - `Event` — a zero-crossing condition plus a reset map `y⁺ = r(y⁻)`.
//! - `HybridODE` — an ODE together with a list of events.
//! - Utilities for working with the guard conditions and reset maps.
//!
//! # Example
//!
//! A bouncing ball:
//! ```text
//! dy/dt = [v, -g]          (position y[0], velocity y[1])
//! Event: y[0] = 0 (ball hits ground)
//!   reset: y[0]⁺ = y[0]⁻, y[1]⁺ = -e * y[1]⁻   (elastic bounce)
//! ```

use crate::kernel::{ExprData, ExprId, ExprPool};
use crate::ode::ODE;
use crate::simplify::engine::simplify;

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A zero-crossing event: the event fires when `condition(t, y)` crosses zero.
///
/// The `reset_map` specifies new values for each state variable after the event:
/// `reset_map[i] = (state_var, new_value_expr)`.
#[derive(Clone, Debug)]
pub struct Event {
    /// Descriptive name (for diagnostics)
    pub name: String,
    /// Guard condition: event fires when this crosses zero (from − to +)
    pub condition: ExprId,
    /// Reset map: `(variable, new_value_expression)` pairs
    pub reset_map: Vec<(ExprId, ExprId)>,
    /// Direction of crossing: +1 = rising, -1 = falling, 0 = both
    pub direction: i8,
}

impl Event {
    /// Create a new event with a zero-crossing condition and reset map.
    pub fn new(name: &str, condition: ExprId, reset_map: Vec<(ExprId, ExprId)>) -> Self {
        Event {
            name: name.to_string(),
            condition,
            reset_map,
            direction: 0,
        }
    }

    /// Create an event that only fires on rising zero-crossings.
    pub fn rising(mut self) -> Self {
        self.direction = 1;
        self
    }

    /// Create an event that only fires on falling zero-crossings.
    pub fn falling(mut self) -> Self {
        self.direction = -1;
        self
    }

    /// Return the set of state variables modified by this event's reset map.
    pub fn modified_vars(&self) -> impl Iterator<Item = ExprId> + '_ {
        self.reset_map.iter().map(|(v, _)| *v)
    }

    /// Apply the reset map to a (symbolic) state vector.
    ///
    /// Returns a new state vector where each variable in `reset_map` is
    /// replaced by the corresponding new-value expression.
    pub fn apply_reset(&self, state: &[ExprId], state_vars: &[ExprId]) -> Vec<ExprId> {
        let reset: std::collections::HashMap<ExprId, ExprId> =
            self.reset_map.iter().cloned().collect();
        state
            .iter()
            .zip(state_vars.iter())
            .map(|(&s, &var)| reset.get(&var).copied().unwrap_or(s))
            .collect()
    }

    /// Display the event.
    pub fn display(&self, pool: &ExprPool) -> String {
        let dir = match self.direction {
            1 => "↑",
            -1 => "↓",
            _ => "↕",
        };
        let cond = pool.display(self.condition);
        let resets: Vec<String> = self
            .reset_map
            .iter()
            .map(|(v, e)| format!("{} := {}", pool.display(*v), pool.display(*e)))
            .collect();
        format!(
            "event '{}' {} [{}]: {}",
            self.name,
            dir,
            cond,
            if resets.is_empty() {
                "no reset".to_string()
            } else {
                resets.join(", ")
            }
        )
    }
}

// ---------------------------------------------------------------------------
// HybridODE
// ---------------------------------------------------------------------------

/// A hybrid dynamical system: continuous ODE plus a list of discrete events.
#[derive(Clone, Debug)]
pub struct HybridODE {
    /// The continuous-time dynamics
    pub continuous: ODE,
    /// List of events (may fire during integration)
    pub events: Vec<Event>,
}

impl HybridODE {
    /// Create a new hybrid system.
    pub fn new(ode: ODE) -> Self {
        HybridODE {
            continuous: ode,
            events: vec![],
        }
    }

    /// Add an event to the system.
    pub fn add_event(mut self, event: Event) -> Self {
        self.events.push(event);
        self
    }

    /// Return all distinct zero-crossing guard conditions.
    pub fn guards(&self) -> Vec<ExprId> {
        self.events.iter().map(|e| e.condition).collect()
    }

    /// Check whether any guard condition structurally depends on state variable `var`.
    pub fn guard_depends_on(&self, var: ExprId, pool: &ExprPool) -> bool {
        self.guards()
            .iter()
            .any(|&g| structurally_contains(g, var, pool))
    }

    /// Return all state variables modified by any event.
    pub fn reset_targets(&self) -> Vec<ExprId> {
        let mut targets: Vec<ExprId> = Vec::new();
        for event in &self.events {
            for v in event.modified_vars() {
                if !targets.contains(&v) {
                    targets.push(v);
                }
            }
        }
        targets
    }

    /// Display the hybrid system.
    pub fn display(&self, pool: &ExprPool) -> String {
        let mut lines = vec!["Continuous dynamics:".to_string()];
        lines.push(self.continuous.display(pool));
        if !self.events.is_empty() {
            lines.push("Events:".to_string());
            for ev in &self.events {
                lines.push(format!("  {}", ev.display(pool)));
            }
        }
        lines.join("\n")
    }

    /// Simplify all guard conditions and reset expressions.
    pub fn simplify_events(&self, pool: &ExprPool) -> HybridODE {
        let events = self
            .events
            .iter()
            .map(|ev| {
                let cond = simplify(ev.condition, pool).value;
                let reset_map = ev
                    .reset_map
                    .iter()
                    .map(|&(v, e)| (v, simplify(e, pool).value))
                    .collect();
                Event {
                    name: ev.name.clone(),
                    condition: cond,
                    reset_map,
                    direction: ev.direction,
                }
            })
            .collect();
        HybridODE {
            continuous: self.continuous.clone(),
            events,
        }
    }
}

// ---------------------------------------------------------------------------
// Zero-crossing detection utilities
// ---------------------------------------------------------------------------

/// Evaluate the sign of a guard condition expression (−1, 0, or +1) given
/// a symbolic substitution for state variables.
///
/// This is a *structural* analysis — it checks whether the expression is
/// provably zero, positive, or negative given only the structure of the
/// symbols and not numeric values.
pub fn guard_sign_structure(condition: ExprId, pool: &ExprPool) -> GuardStructure {
    match pool.get(condition) {
        ExprData::Integer(n) => {
            if n.0 == 0 {
                GuardStructure::Zero
            } else if n.0 > 0 {
                GuardStructure::Positive
            } else {
                GuardStructure::Negative
            }
        }
        _ => GuardStructure::Unknown,
    }
}

/// Structural sign information for a guard condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardStructure {
    Zero,
    Positive,
    Negative,
    Unknown,
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn structurally_contains(expr: ExprId, needle: ExprId, pool: &ExprPool) -> bool {
    if expr == needle {
        return true;
    }
    let children = pool.with(expr, |data| match data {
        ExprData::Add(args) | ExprData::Mul(args) | ExprData::Func { args, .. } => args.clone(),
        ExprData::Pow { base, exp } => vec![*base, *exp],
        _ => vec![],
    });
    children
        .into_iter()
        .any(|c| structurally_contains(c, needle, pool))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Domain, ExprPool};
    use crate::ode::ODE;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn event_creation() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        // Event: y = 0  (no reset)
        let ev = Event::new("hit_ground", y, vec![]);
        assert_eq!(ev.condition, y);
        assert!(ev.reset_map.is_empty());
    }

    #[test]
    fn event_apply_reset() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let v = pool.symbol("v", Domain::Real);
        let neg_v = pool.mul(vec![pool.integer(-1_i32), v]);
        // On bounce: v ← -v  (elastic, no energy loss)
        let ev = Event::new("bounce", y, vec![(v, neg_v)]);
        let state = vec![y, v]; // current state [y, v]
        let state_vars = vec![y, v];
        let new_state = ev.apply_reset(&state, &state_vars);
        // y unchanged, v → -v
        assert_eq!(new_state[0], y);
        assert_eq!(new_state[1], neg_v);
    }

    #[test]
    fn hybrid_ode_basic() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let v = pool.symbol("v", Domain::Real);
        let g = pool.symbol("g", Domain::Real);
        let t = pool.symbol("t", Domain::Real);

        let neg_g = pool.mul(vec![pool.integer(-1_i32), g]);
        let ode = ODE::new(vec![y, v], vec![v, neg_g], t, &pool).unwrap();

        let neg_v = pool.mul(vec![pool.integer(-1_i32), v]);
        let bounce = Event::new("bounce", y, vec![(v, neg_v)]).falling();

        let hybrid = HybridODE::new(ode).add_event(bounce);
        assert_eq!(hybrid.events.len(), 1);
        assert!(hybrid.guard_depends_on(y, &pool));
        assert!(!hybrid.guard_depends_on(v, &pool));
    }

    #[test]
    fn hybrid_ode_guards() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let z = pool.symbol("z", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let ode = ODE::new(vec![y, z], vec![y, z], t, &pool).unwrap();
        let ev1 = Event::new("ev1", y, vec![]);
        let ev2 = Event::new("ev2", z, vec![]);
        let hybrid = HybridODE::new(ode).add_event(ev1).add_event(ev2);
        let guards = hybrid.guards();
        assert_eq!(guards, vec![y, z]);
    }

    #[test]
    fn guard_sign_constant() {
        let pool = p();
        let pos = pool.integer(5_i32);
        let neg = pool.integer(-3_i32);
        let zero = pool.integer(0_i32);
        assert_eq!(guard_sign_structure(pos, &pool), GuardStructure::Positive);
        assert_eq!(guard_sign_structure(neg, &pool), GuardStructure::Negative);
        assert_eq!(guard_sign_structure(zero, &pool), GuardStructure::Zero);
    }

    #[test]
    fn reset_targets() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let v = pool.symbol("v", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let ode = ODE::new(vec![y, v], vec![v, y], t, &pool).unwrap();
        let neg_v = pool.mul(vec![pool.integer(-1_i32), v]);
        let ev = Event::new("bounce", y, vec![(v, neg_v)]);
        let hybrid = HybridODE::new(ode).add_event(ev);
        let targets = hybrid.reset_targets();
        assert_eq!(targets, vec![v]);
    }

    #[test]
    fn simplify_events() {
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let zero = pool.integer(0_i32);
        // condition = y + 0  → should simplify to y
        let cond = pool.add(vec![y, zero]);
        let ev = Event::new("ev", cond, vec![]);
        let ode = ODE::new(vec![y], vec![y], t, &pool).unwrap();
        let hybrid = HybridODE::new(ode).add_event(ev);
        let simplified = hybrid.simplify_events(&pool);
        assert_eq!(simplified.events[0].condition, y);
    }
}

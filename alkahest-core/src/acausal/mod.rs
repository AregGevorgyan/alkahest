//! Phase 18 — Acausal component modelling.
//!
//! Models physical systems as networks of *components* connected through
//! *ports*.  Each port has a *potential* (across) variable and a *flow*
//! (through) variable.  Kirchhoff-style connection equations enforce:
//!
//! - **Potential continuity**: potentials at connected ports are equal.
//! - **Flow balance**: flows at connected ports sum to zero (Kirchhoff's
//!   current law / Newton's third law for force).
//!
//! After all components and connections are specified, `flatten` collects all
//! equations into a single `DAE` ready for the Pantelides analyser (Phase 17).
//!
//! # Example (resistor-capacitor circuit)
//!
//! ```text
//! Resistor: v_R = R * i_R
//! Capacitor: C * dv_C/dt = i_C
//! connect(Resistor.pin_p, Capacitor.pin_p)  →  v_R = v_C, i_R + i_C = 0
//! ```

use crate::dae::DAE;
use crate::kernel::{Domain, ExprId, ExprPool};

// ---------------------------------------------------------------------------
// Port
// ---------------------------------------------------------------------------

/// A connection port with a *potential* (across) and a *flow* (through) variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    /// Name of this port (for diagnostics)
    pub name: String,
    /// Potential (voltage, pressure, temperature, …)
    pub potential: ExprId,
    /// Flow (current, mass flow, heat flux, …)
    pub flow: ExprId,
}

impl Port {
    /// Create a new port, allocating fresh symbolic variables in `pool`.
    pub fn new(name: &str, pool: &ExprPool) -> Self {
        let potential = pool.symbol(format!("{name}.v"), Domain::Real);
        let flow = pool.symbol(format!("{name}.i"), Domain::Real);
        Port {
            name: name.to_string(),
            potential,
            flow,
        }
    }
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

/// A physical component with named ports and internal constitutive equations.
#[derive(Clone, Debug)]
pub struct Component {
    /// Component name (for diagnostics)
    pub name: String,
    /// External connection ports
    pub ports: Vec<Port>,
    /// Internal (constitutive) equations in the form `expr = 0`
    pub equations: Vec<ExprId>,
    /// Internal variables not exposed as ports
    pub internal_vars: Vec<ExprId>,
    /// Derivative symbols for internal differential variables
    pub internal_derivs: Vec<ExprId>,
}

impl Component {
    /// Create a new empty component.
    pub fn new(name: &str) -> Self {
        Component {
            name: name.to_string(),
            ports: vec![],
            equations: vec![],
            internal_vars: vec![],
            internal_derivs: vec![],
        }
    }

    /// Add a port to this component.
    pub fn add_port(mut self, port: Port) -> Self {
        self.ports.push(port);
        self
    }

    /// Add a constitutive equation (implicit form `expr = 0`).
    pub fn add_equation(mut self, eq: ExprId) -> Self {
        self.equations.push(eq);
        self
    }

    /// Add an internal variable with its derivative symbol.
    pub fn add_internal_var(mut self, var: ExprId, deriv: ExprId) -> Self {
        self.internal_vars.push(var);
        self.internal_derivs.push(deriv);
        self
    }

    /// Get a port by name.
    pub fn port(&self, name: &str) -> Option<&Port> {
        self.ports.iter().find(|p| p.name == name)
    }
}

// ---------------------------------------------------------------------------
// System builder
// ---------------------------------------------------------------------------

/// A connection between two ports (equality of potentials, zero-sum of flows).
#[derive(Clone, Debug)]
struct Connection {
    port_a: Port,
    port_b: Port,
}

/// A network of components and their connections.
#[derive(Clone, Debug, Default)]
pub struct System {
    components: Vec<Component>,
    connections: Vec<Connection>,
}

impl System {
    pub fn new() -> Self {
        System::default()
    }

    /// Add a component to the system.
    pub fn add_component(&mut self, component: Component) {
        self.components.push(component);
    }

    /// Connect port `a` to port `b`.
    ///
    /// This adds two connection equations:
    /// - `a.potential - b.potential = 0`
    /// - `a.flow + b.flow = 0`
    pub fn connect(&mut self, port_a: &Port, port_b: &Port) {
        self.connections.push(Connection {
            port_a: port_a.clone(),
            port_b: port_b.clone(),
        });
    }

    /// Flatten all component equations and connection equations into a single `DAE`.
    ///
    /// All port and internal variables become DAE variables.
    /// Derivative symbols are generated with the `d{name}/dt` convention.
    pub fn flatten(&self, time_var: ExprId, pool: &ExprPool) -> DAE {
        let mut equations: Vec<ExprId> = Vec::new();
        let mut variables: Vec<ExprId> = Vec::new();
        let mut derivatives: Vec<ExprId> = Vec::new();

        // Collect all component equations and variables
        for comp in &self.components {
            equations.extend_from_slice(&comp.equations);

            // Port variables
            for port in &comp.ports {
                add_var_if_new(port.potential, pool, &mut variables, &mut derivatives);
                add_var_if_new(port.flow, pool, &mut variables, &mut derivatives);
            }

            // Internal variables (with their explicit derivative symbols)
            for (&var, &deriv) in comp.internal_vars.iter().zip(comp.internal_derivs.iter()) {
                if !variables.contains(&var) {
                    variables.push(var);
                    derivatives.push(deriv);
                }
            }
        }

        // Generate connection equations
        let neg_one = pool.integer(-1_i32);
        for conn in &self.connections {
            // potential_a - potential_b = 0
            let neg_b = pool.mul(vec![neg_one, conn.port_b.potential]);
            let pot_eq = pool.add(vec![conn.port_a.potential, neg_b]);
            equations.push(pot_eq);

            // flow_a + flow_b = 0
            let flow_eq = pool.add(vec![conn.port_a.flow, conn.port_b.flow]);
            equations.push(flow_eq);
        }

        DAE::new(equations, variables, derivatives, time_var)
    }
}

/// Add `var` to `variables` if not already present, creating a derivative symbol.
fn add_var_if_new(
    var: ExprId,
    pool: &ExprPool,
    variables: &mut Vec<ExprId>,
    derivatives: &mut Vec<ExprId>,
) {
    if variables.contains(&var) {
        return;
    }
    let deriv_name = pool.with(var, |d| match d {
        crate::kernel::ExprData::Symbol { name, .. } => format!("d{name}/dt"),
        _ => "d?/dt".to_string(),
    });
    let deriv = pool.symbol(&deriv_name, Domain::Real);
    variables.push(var);
    derivatives.push(deriv);
}

// ---------------------------------------------------------------------------
// Convenience constructors for standard components
// ---------------------------------------------------------------------------

/// Create a linear resistor: `v - R*i = 0`
pub fn resistor(name: &str, resistance: ExprId, pool: &ExprPool) -> Component {
    let port_p = Port::new(&format!("{name}.p"), pool);
    let port_n = Port::new(&format!("{name}.n"), pool);

    // Voltage across: v = port_p.v - port_n.v
    let v = pool.add(vec![
        port_p.potential,
        pool.mul(vec![pool.integer(-1_i32), port_n.potential]),
    ]);
    // Current: i = port_p.i  (positive into component)
    let i = port_p.flow;

    // Ohm's law: v - R*i = 0
    let ri = pool.mul(vec![resistance, i]);
    let eq = pool.add(vec![v, pool.mul(vec![pool.integer(-1_i32), ri])]);

    Component::new(name)
        .add_port(port_p)
        .add_port(port_n)
        .add_equation(eq)
}

/// Create a capacitor: `C * dv/dt - i = 0`
pub fn capacitor(name: &str, capacitance: ExprId, pool: &ExprPool) -> Component {
    let port_p = Port::new(&format!("{name}.p"), pool);
    let port_n = Port::new(&format!("{name}.n"), pool);

    // Voltage across: v = port_p.v - port_n.v
    let v_name = format!("{name}.vc");
    let v = pool.symbol(&v_name, Domain::Real);
    let dv_name = format!("d{v_name}/dt");
    let dv = pool.symbol(&dv_name, Domain::Real);

    // Constitutive: C*dv/dt - i = 0
    let i = port_p.flow;
    let c_dv = pool.mul(vec![capacitance, dv]);
    let eq = pool.add(vec![c_dv, pool.mul(vec![pool.integer(-1_i32), i])]);

    // Voltage definition: v - (port_p.potential - port_n.potential) = 0
    let neg_vn = pool.mul(vec![pool.integer(-1_i32), port_n.potential]);
    let v_def = pool.add(vec![
        v,
        pool.mul(vec![
            pool.integer(-1_i32),
            pool.add(vec![port_p.potential, neg_vn]),
        ]),
    ]);

    Component::new(name)
        .add_port(port_p)
        .add_port(port_n)
        .add_equation(eq)
        .add_equation(v_def)
        .add_internal_var(v, dv)
}

/// Create an ideal voltage source: `port_p.potential - port_n.potential - V = 0`
pub fn voltage_source(name: &str, voltage: ExprId, pool: &ExprPool) -> Component {
    let port_p = Port::new(&format!("{name}.p"), pool);
    let port_n = Port::new(&format!("{name}.n"), pool);

    let neg_v = pool.mul(vec![pool.integer(-1_i32), port_n.potential]);
    let diff_v = pool.add(vec![port_p.potential, neg_v]);
    let neg_voltage = pool.mul(vec![pool.integer(-1_i32), voltage]);
    let eq = pool.add(vec![diff_v, neg_voltage]);

    Component::new(name)
        .add_port(port_p)
        .add_port(port_n)
        .add_equation(eq)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::ExprPool;

    fn p() -> ExprPool {
        ExprPool::new()
    }

    #[test]
    fn port_creation() {
        let pool = p();
        let port = Port::new("pin1", &pool);
        assert_eq!(port.name, "pin1");
        // Variables are distinct
        assert_ne!(port.potential, port.flow);
    }

    #[test]
    fn resistor_has_one_equation() {
        let pool = p();
        let r_val = pool.symbol("R", crate::kernel::Domain::Real);
        let comp = resistor("R1", r_val, &pool);
        assert_eq!(comp.equations.len(), 1);
        assert_eq!(comp.ports.len(), 2);
    }

    #[test]
    fn capacitor_has_two_equations() {
        let pool = p();
        let c_val = pool.symbol("C", crate::kernel::Domain::Real);
        let comp = capacitor("C1", c_val, &pool);
        assert_eq!(comp.equations.len(), 2);
        assert_eq!(comp.internal_vars.len(), 1);
    }

    #[test]
    fn system_flatten_rc_circuit() {
        // Simple RC: source → resistor → capacitor → ground
        let pool = p();
        let r_val = pool.symbol("R", crate::kernel::Domain::Real);
        let c_val = pool.symbol("C", crate::kernel::Domain::Real);
        let v_src = pool.symbol("Vs", crate::kernel::Domain::Real);
        let t = pool.symbol("t", crate::kernel::Domain::Real);

        let res = resistor("R1", r_val, &pool);
        let cap = capacitor("C1", c_val, &pool);
        let src = voltage_source("V1", v_src, &pool);

        let src_p = src.port("V1.p").unwrap().clone();
        let src_n = src.port("V1.n").unwrap().clone();
        let res_p = res.port("R1.p").unwrap().clone();
        let res_n = res.port("R1.n").unwrap().clone();
        let cap_p = cap.port("C1.p").unwrap().clone();
        let cap_n = cap.port("C1.n").unwrap().clone();

        let mut sys = System::new();
        sys.add_component(src);
        sys.add_component(res);
        sys.add_component(cap);
        sys.connect(&src_p, &res_p);
        sys.connect(&res_n, &cap_p);
        sys.connect(&cap_n, &src_n);

        let dae = sys.flatten(t, &pool);
        // Should have equations from components + 2 per connection
        assert!(dae.n_equations() >= 3); // at least R + C + source equations
        assert!(dae.n_variables() >= 4); // at least 4 port variables
    }

    #[test]
    fn connect_generates_two_equations() {
        let pool = p();
        let t = pool.symbol("t", crate::kernel::Domain::Real);
        let r_val = pool.symbol("R", crate::kernel::Domain::Real);
        let comp1 = resistor("R1", r_val, &pool);
        let comp2 = resistor("R2", r_val, &pool);
        let port_a = comp1.port("R1.n").unwrap().clone();
        let port_b = comp2.port("R2.p").unwrap().clone();
        let mut sys = System::new();
        sys.add_component(comp1);
        sys.add_component(comp2);
        sys.connect(&port_a, &port_b);
        let dae = sys.flatten(t, &pool);
        // 2 component equations + 2 connection equations = 4
        assert_eq!(dae.n_equations(), 4);
    }
}

//! Phase 19 — Forward sensitivity analysis.
//!
//! Given an ODE `dy/dt = f(t, y, p)` and a parameter vector `p`, the
//! forward sensitivity equations are:
//!
//! ```text
//! dS_j/dt = (∂f/∂y) · S_j + ∂f/∂p_j
//! ```
//!
//! where `S_j = ∂y/∂p_j` is an m-vector of sensitivities.
//!
//! `sensitivity_system` returns an extended ODE whose state is `(y, S)`.

use crate::kernel::{Domain, ExprData, ExprId, ExprPool};
use crate::matrix::{jacobian, Matrix};
use crate::ode::{OdeError, ODE};
use crate::simplify::engine::simplify;

// ---------------------------------------------------------------------------
// Sensitivity system
// ---------------------------------------------------------------------------

/// Build the forward-sensitivity ODE for `ode` with respect to `params`.
///
/// Returns an extended `ODE` whose state vector is `[y_0…y_{m-1}, S_{0,0}…]`
/// where `S_{i,j} = ∂y_i/∂params[j]`.
///
/// # Errors
///
/// Returns an `OdeError` if differentiation of any RHS fails (e.g. an unknown
/// function that cannot be differentiated).
pub fn sensitivity_system(
    ode: &ODE,
    params: &[ExprId],
    pool: &ExprPool,
) -> Result<SensitivitySystem, OdeError> {
    let m = ode.order();
    let n_params = params.len();

    // Jacobian of f w.r.t. state variables: ∂f_i/∂y_j  (m × m)
    let jac_y = jacobian(&ode.rhs, &ode.state_vars, pool)
        .map_err(|e| OdeError::DiffError(e.to_string()))?;

    // Jacobian of f w.r.t. parameters: ∂f_i/∂p_j  (m × n_params)
    let jac_p = jacobian(&ode.rhs, params, pool).map_err(|e| OdeError::DiffError(e.to_string()))?;

    // Sensitivity variables: S_{i,j} = ∂y_i/∂p_j  (m × n_params matrix)
    // Stored column-major as separate ODE states
    let mut sens_vars: Vec<Vec<ExprId>> = Vec::new(); // sens_vars[j][i] = S_{i,j}
    let mut sens_derivs: Vec<Vec<ExprId>> = Vec::new();
    for (j, &param) in params.iter().enumerate().take(n_params) {
        let col_vars: Vec<ExprId> = (0..m)
            .map(|i| {
                let pname = pool.with(param, |d| match d {
                    ExprData::Symbol { name, .. } => name.clone(),
                    _ => format!("p{j}"),
                });
                let yname = pool.with(ode.state_vars[i], |d| match d {
                    ExprData::Symbol { name, .. } => name.clone(),
                    _ => format!("y{i}"),
                });
                pool.symbol(format!("dS_{yname}_{pname}"), Domain::Real)
            })
            .collect();
        let col_derivs: Vec<ExprId> = col_vars
            .iter()
            .map(|&v| {
                let name = pool.with(v, |d| match d {
                    ExprData::Symbol { name, .. } => format!("d{name}/dt"),
                    _ => "d?/dt".to_string(),
                });
                pool.symbol(name, Domain::Real)
            })
            .collect();
        sens_vars.push(col_vars);
        sens_derivs.push(col_derivs);
    }

    // Build sensitivity RHS: dS_j/dt = J_y · S_j + ∂f/∂p_j
    let mut extended_vars: Vec<ExprId> = ode.state_vars.clone();
    let mut extended_derivs: Vec<ExprId> = ode.derivatives.clone();
    let mut extended_rhs: Vec<ExprId> = ode.rhs.clone();

    let mut sens_rhs_matrix: Vec<Vec<ExprId>> = Vec::new(); // [j][i]

    for j in 0..n_params {
        // S_j is the j-th column of the sensitivity matrix, as a vector
        let s_j = Matrix::new(sens_vars[j].iter().map(|&v| vec![v]).collect())
            .expect("single-column matrix");

        // J_y · S_j  (m×m times m×1 = m×1)
        let jac_sj = jac_y.mul(&s_j, pool).expect("compatible shapes");

        // ∂f/∂p_j  (column j of jac_p, shape m×1)
        let df_dpj: Vec<ExprId> = (0..m).map(|i| jac_p.get(i, j)).collect();

        // dS_j/dt = J_y * S_j + ∂f/∂p_j
        let col_rhs: Vec<ExprId> = (0..m)
            .map(|i| {
                let jac_term = jac_sj.get(i, 0);
                let param_term = df_dpj[i];
                simplify(pool.add(vec![jac_term, param_term]), pool).value
            })
            .collect();

        sens_rhs_matrix.push(col_rhs.clone());

        // Append to extended system
        for i in 0..m {
            extended_vars.push(sens_vars[j][i]);
            extended_derivs.push(sens_derivs[j][i]);
            extended_rhs.push(col_rhs[i]);
        }
    }

    Ok(SensitivitySystem {
        extended_ode: ODE {
            state_vars: extended_vars,
            derivatives: extended_derivs,
            rhs: extended_rhs,
            time_var: ode.time_var,
            initial_conditions: ode.initial_conditions.clone(),
        },
        original_dim: m,
        n_params,
        param_vars: params.to_vec(),
        sensitivity_vars: sens_vars,
    })
}

// ---------------------------------------------------------------------------
// Adjoint sensitivity (reverse-mode)
// ---------------------------------------------------------------------------

/// Build the adjoint (reverse) sensitivity system for a scalar objective.
///
/// Given `ode` and a scalar objective `obj = g(y(T))`, the adjoint equations
/// are:
///
/// ```text
/// dλ/dt = -(∂f/∂y)ᵀ · λ
/// λ(T)  = ∂g/∂y(T)
/// ```
///
/// The gradient w.r.t. parameters is then:
///
/// ```text
/// ∂J/∂p = ∫₀ᵀ (∂f/∂p)ᵀ · λ dt
/// ```
///
/// This function returns the adjoint ODE (to integrate backward in time).
pub fn adjoint_system(
    ode: &ODE,
    objective_grad: &[ExprId], // ∂g/∂y_i  at terminal time
    pool: &ExprPool,
) -> Result<AdjointSystem, OdeError> {
    let m = ode.order();

    // Jacobian ∂f/∂y  (m × m)
    let jac_y = jacobian(&ode.rhs, &ode.state_vars, pool)
        .map_err(|e| OdeError::DiffError(e.to_string()))?;

    // Adjoint variables λ_i
    let lambda: Vec<ExprId> = (0..m)
        .map(|i| {
            let yname = pool.with(ode.state_vars[i], |d| match d {
                ExprData::Symbol { name, .. } => name.clone(),
                _ => format!("y{i}"),
            });
            pool.symbol(format!("lambda_{yname}"), Domain::Real)
        })
        .collect();

    let lambda_derivs: Vec<ExprId> = lambda
        .iter()
        .map(|&v| {
            let name = pool.with(v, |d| match d {
                ExprData::Symbol { name, .. } => format!("d{name}/dt"),
                _ => "d?/dt".to_string(),
            });
            pool.symbol(&name, Domain::Real)
        })
        .collect();

    // Adjoint RHS: dλ/dt = -(J_y)ᵀ · λ  (backward in time)
    let jac_y_t = jac_y.transpose();
    let lam_mat = Matrix::new(lambda.iter().map(|&v| vec![v]).collect()).expect("column matrix");
    let jac_lam = jac_y_t.mul(&lam_mat, pool).expect("compatible shapes");

    let neg_one = pool.integer(-1_i32);
    let adjoint_rhs: Vec<ExprId> = (0..m)
        .map(|i| simplify(pool.mul(vec![neg_one, jac_lam.get(i, 0)]), pool).value)
        .collect();

    // Terminal conditions: λ(T) = ∂g/∂y
    let terminal_conditions: Vec<(ExprId, ExprId)> = lambda
        .iter()
        .zip(objective_grad.iter())
        .map(|(&l, &g)| (l, g))
        .collect();

    let adjoint_ode = ODE {
        state_vars: lambda.clone(),
        derivatives: lambda_derivs,
        rhs: adjoint_rhs,
        time_var: ode.time_var,
        initial_conditions: terminal_conditions.clone(),
    };

    Ok(AdjointSystem {
        adjoint_ode,
        lambda_vars: lambda,
        terminal_conditions,
    })
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// The extended ODE system for forward sensitivity analysis.
#[derive(Clone, Debug)]
pub struct SensitivitySystem {
    /// Extended ODE: state = [y, S_0, S_1, …, S_{n-1}]
    pub extended_ode: ODE,
    /// Dimension of the original state
    pub original_dim: usize,
    /// Number of parameters
    pub n_params: usize,
    /// The parameter variables
    pub param_vars: Vec<ExprId>,
    /// `sensitivity_vars[j][i]` = ExprId for S_{i,j} = ∂y_i/∂p_j
    pub sensitivity_vars: Vec<Vec<ExprId>>,
}

impl SensitivitySystem {
    /// Get the sensitivity variable S_{i,j} = ∂y_i/∂p_j.
    pub fn get_sensitivity(&self, state_idx: usize, param_idx: usize) -> ExprId {
        self.sensitivity_vars[param_idx][state_idx]
    }

    /// Display the sensitivity system.
    pub fn display(&self, pool: &ExprPool) -> String {
        self.extended_ode.display(pool)
    }
}

/// The adjoint ODE system for reverse-mode sensitivity.
#[derive(Clone, Debug)]
pub struct AdjointSystem {
    /// Adjoint ODE: dλ/dt = -(∂f/∂y)ᵀ · λ, integrated backward
    pub adjoint_ode: ODE,
    /// Adjoint variables λ_i
    pub lambda_vars: Vec<ExprId>,
    /// Terminal conditions λ(T) = ∂g/∂y
    pub terminal_conditions: Vec<(ExprId, ExprId)>,
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
    fn sensitivity_linear_ode() {
        // dy/dt = a*y,  param = a
        // Sensitivity: dS/dt = a*S + y,  S(0) = 0
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let rhs = pool.mul(vec![a, y]);
        let ode = ODE::new(vec![y], vec![rhs], t, &pool).unwrap();
        let sys = sensitivity_system(&ode, &[a], &pool).unwrap();
        // Extended state: [y, S_{y,a}]
        assert_eq!(sys.extended_ode.order(), 2);
        assert_eq!(sys.original_dim, 1);
        assert_eq!(sys.n_params, 1);
    }

    #[test]
    fn sensitivity_constant_ode() {
        // dy/dt = p  (constant RHS), param = p
        // ∂f/∂y = 0, ∂f/∂p = 1
        // dS/dt = 0 * S + 1 = 1
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let p_sym = pool.symbol("p", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let ode = ODE::new(vec![y], vec![p_sym], t, &pool).unwrap();
        let sys = sensitivity_system(&ode, &[p_sym], &pool).unwrap();
        assert_eq!(sys.extended_ode.order(), 2);
        // dS/dt should simplify to 1
        let s_rhs = sys.extended_ode.rhs[1];
        assert_eq!(s_rhs, pool.integer(1_i32));
    }

    #[test]
    fn adjoint_system_basic() {
        // dy/dt = -y, objective ∂g/∂y = 1
        // Adjoint: dλ/dt = -(-1)*λ = λ
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let neg_y = pool.mul(vec![pool.integer(-1_i32), y]);
        let ode = ODE::new(vec![y], vec![neg_y], t, &pool).unwrap();
        let obj_grad = vec![pool.integer(1_i32)];
        let adj = adjoint_system(&ode, &obj_grad, &pool).unwrap();
        assert_eq!(adj.adjoint_ode.order(), 1);
        // dλ/dt = λ  (Jacobian is -1, negated → 1 * λ)
        let lam = adj.lambda_vars[0];
        let rhs = adj.adjoint_ode.rhs[0];
        assert_eq!(rhs, lam);
    }

    #[test]
    fn sensitivity_two_params() {
        // dy/dt = a*y + b,  params = [a, b]
        let pool = p();
        let y = pool.symbol("y", Domain::Real);
        let a = pool.symbol("a", Domain::Real);
        let b = pool.symbol("b", Domain::Real);
        let t = pool.symbol("t", Domain::Real);
        let rhs = pool.add(vec![pool.mul(vec![a, y]), b]);
        let ode = ODE::new(vec![y], vec![rhs], t, &pool).unwrap();
        let sys = sensitivity_system(&ode, &[a, b], &pool).unwrap();
        // Extended state: [y, S_{y,a}, S_{y,b}]
        assert_eq!(sys.extended_ode.order(), 3);
        assert_eq!(sys.n_params, 2);
    }
}

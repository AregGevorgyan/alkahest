/// Domain is part of a symbol's structural identity.
/// `Symbol("x", Real)` and `Symbol("x", Complex)` are distinct expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Domain {
    Real,
    Complex,
    Integer,
    Positive,
    NonNegative,
    NonZero,
}

impl std::fmt::Display for Domain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Domain::Real => write!(f, "Real"),
            Domain::Complex => write!(f, "Complex"),
            Domain::Integer => write!(f, "Integer"),
            Domain::Positive => write!(f, "Positive"),
            Domain::NonNegative => write!(f, "NonNegative"),
            Domain::NonZero => write!(f, "NonZero"),
        }
    }
}

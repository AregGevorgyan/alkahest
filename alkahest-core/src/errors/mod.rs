//! Structured error types with stable diagnostic codes and remediation hints.
//!
//! The canonical list of every code is in [`codes::REGISTRY`].
//!
//! # V1-3 — Full structured error hierarchy
//!
//! Every public error type in `alkahest-core` implements [`AlkahestError`], which
//! provides three machine-readable fields in addition to the `Display` message:
//!
//! - **`code()`** — a stable `&'static str` like `"E-POLY-001"`.  Suitable for
//!   `match` in user code and dictionary look-up in tool integrations.
//! - **`remediation()`** — a human-readable fix suggestion, or `None` if the
//!   error is self-explanatory.
//! - **`span()`** — optional `(start, end)` byte offsets into a source string
//!   for IDE diagnostics.  `None` until the parser is integrated.

pub mod codes;

/// Core trait shared by every `alkahest-core` error type.
pub trait AlkahestError: std::error::Error {
    /// Stable diagnostic code, e.g. `"E-DIFF-001"`.
    fn code(&self) -> &'static str;

    /// Optional human-readable fix suggestion.
    fn remediation(&self) -> Option<&'static str> {
        None
    }

    /// Optional source span `(start_byte, end_byte)` within the input text.
    fn span(&self) -> Option<(usize, usize)> {
        None
    }
}

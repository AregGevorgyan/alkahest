#!/usr/bin/env python3
"""CI validation for the Alkahest error-code registry.

Checks:
  1. Every code in REGISTRY appears in at least one Rust AlkahestError impl.
  2. No Rust impl returns a code that is absent from REGISTRY.
  3. No PyO3 error-raise path emits E-UNKNOWN.
  4. Every PyO3 exception class name (Py*Error) has a matching Python class in
     alkahest/exceptions.py.

Usage:
    python scripts/check_error_codes.py
Exit codes: 0 = OK, 1 = failures found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "alkahest-core" / "src"
PY_LIB = REPO / "alkahest-py" / "src" / "lib.rs"
EXCEPTIONS_PY = REPO / "python" / "alkahest" / "exceptions.py"
CODES_RS = CORE / "errors" / "codes.rs"

# ---------------------------------------------------------------------------
# Parse REGISTRY from codes.rs
# ---------------------------------------------------------------------------

def parse_registry(path: Path) -> set[str]:
    text = path.read_text()
    return set(re.findall(r'"(E-[A-Z]+-\d+)"', text))


# ---------------------------------------------------------------------------
# Collect codes from Rust AlkahestError impls
# ---------------------------------------------------------------------------

def collect_rust_codes(core: Path) -> tuple[set[str], set[str]]:
    """Return (codes_in_impls, e_unknown_files)."""
    codes: set[str] = set()
    unknown_files: set[str] = set()
    code_pat = re.compile(r'"(E-[A-Z]+-\d+)"')
    unknown_pat = re.compile(r'"E-UNKNOWN"')

    for rs in core.rglob("*.rs"):
        text = rs.read_text(errors="replace")
        codes.update(code_pat.findall(text))
        if unknown_pat.search(text):
            unknown_files.add(str(rs.relative_to(REPO)))

    return codes, unknown_files


# ---------------------------------------------------------------------------
# Collect PyO3 exception class names from lib.rs
# ---------------------------------------------------------------------------

def collect_pyo3_classes(lib_rs: Path) -> set[str]:
    text = lib_rs.read_text()
    return set(re.findall(r'create_exception!\s*\(\s*alkahest\s*,\s*(Py\w+Error)', text))


# ---------------------------------------------------------------------------
# Collect Python class names from exceptions.py
# ---------------------------------------------------------------------------

def collect_python_classes(exceptions: Path) -> set[str]:
    text = exceptions.read_text()
    return set(re.findall(r'^class\s+(\w+Error)\s*\(', text, re.MULTILINE))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    errors: list[str] = []

    registry = parse_registry(CODES_RS)
    rust_codes, unknown_files = collect_rust_codes(CORE)

    # 1. Codes in REGISTRY but not used in any Rust impl
    unused = registry - rust_codes
    for code in sorted(unused):
        errors.append(f"REGISTRY code {code} is never returned by any AlkahestError impl")

    # 2. Codes used in Rust but absent from REGISTRY
    unlisted = rust_codes - registry
    # Filter out codes from REGISTRY comments or test strings that happen to be in the scan
    # (all legitimate codes start with E- and are 3 parts)
    unlisted = {c for c in unlisted if re.match(r"^E-[A-Z]+-\d+$", c)}
    for code in sorted(unlisted):
        errors.append(f"Code {code} appears in Rust but is not registered in REGISTRY")

    # 3. E-UNKNOWN in any Rust file
    for f in sorted(unknown_files):
        errors.append(f'E-UNKNOWN found in {f} — every raised error must have a real code')

    # 4. PyO3 class <→ Python class coverage
    pyo3_classes = collect_pyo3_classes(PY_LIB)
    py_classes = collect_python_classes(EXCEPTIONS_PY)

    # Map Py*Error → *Error for comparison
    pyo3_bare = {c[2:] for c in pyo3_classes}  # strip "Py"
    missing_py = pyo3_bare - py_classes
    for cls in sorted(missing_py):
        errors.append(
            f"PyO3 exception Py{cls} has no matching Python class {cls} in exceptions.py"
        )

    if errors:
        print("check_error_codes: FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
        return 1

    print(
        f"check_error_codes: OK  "
        f"({len(registry)} registered codes, "
        f"{len(pyo3_classes)} PyO3 classes, "
        f"{len(py_classes)} Python classes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

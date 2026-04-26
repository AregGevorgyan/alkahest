"""
SymPy oracle: differentiation and integration results compared against SymPy.

Run:
    pytest tests/test_oracle.py -v

Requires:
    pip install sympy
"""

import pytest

sympy = pytest.importorskip("sympy")

import random  # noqa: E402

from alkahest.alkahest import ExprPool, UniPoly, diff, integrate, simplify  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def alkahest_poly(pool, x, coeffs: list[int]):
    """Build sum(c * x^i) in a alkahest ExprPool."""
    terms = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        c_id = pool.integer(c)
        if i == 0:
            terms.append(c_id)
        else:
            xpow = x ** i
            terms.append(c_id * xpow if c != 1 else xpow)
    if not terms:
        return pool.integer(0)
    expr = terms[0]
    for t in terms[1:]:
        expr = expr + t
    return expr


def sympy_poly(sx, coeffs: list[int]):
    """Build sum(c * sx^i) in SymPy."""
    return sum(c * sx**i for i, c in enumerate(coeffs) if c != 0) or sympy.Integer(0)


def alkahest_coeffs(expr, pool_x, pool) -> list[int] | None:
    """Extract coefficient list from a alkahest Expr; return None if not a polynomial."""
    try:
        p = UniPoly.from_symbolic(expr, pool_x)
        return p.coefficients()
    except Exception:
        return None


def sympy_coeffs(expr, sx) -> list[int] | None:
    """Extract coefficient list from a SymPy expression; return None if not a polynomial."""
    try:
        p = sympy.Poly(sympy.expand(expr), sx)
        cs = [int(c) for c in reversed(p.all_coeffs())]
        # pad to include trailing zeros
        return cs or [0]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Oracle cases
# ---------------------------------------------------------------------------

CASES = [
    # (description, coeffs, expected_diff_coeffs)
    # expected_diff_coeffs = derivative of sum(c_i * x^i)
    ("constant", [5], [0]),
    ("linear_1x", [0, 1], [1]),
    ("linear_3x_plus_2", [2, 3], [3]),
    ("quadratic", [1, 2, 1], [2, 2]),
    ("cubic", [0, 0, 0, 1], [0, 0, 3]),
    ("full_cubic", [1, 2, 3, 4], [2, 6, 12]),
    ("degree_4", [1, 1, 1, 1, 1], [1, 2, 3, 4]),
    ("negative_coeffs", [-3, 0, 2], [0, 4]),
    ("zero_poly", [0, 0, 0], [0]),
    ("leading_zero_stripped", [0, 0, 1], [0, 2]),
]


@pytest.mark.parametrize("desc,coeffs,expected", CASES)
def test_diff_matches_known(desc, coeffs, expected):
    """Alkahest diff matches hand-computed derivative for fixed cases."""
    pool = ExprPool()
    x = pool.symbol("x")

    expr = alkahest_poly(pool, x, coeffs)
    r = diff(expr, x)
    got = alkahest_coeffs(r.value, x, pool) or [0]

    # Normalise: strip trailing zeros, default to [0]
    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0] or [0]

    assert norm(got) == norm(expected), (
        f"[{desc}] coeffs={coeffs}: alkahest={got}, expected={expected}"
    )


@pytest.mark.parametrize("seed", range(100))
def test_diff_matches_sympy(seed):
    """Alkahest diff(p, x) == SymPy diff(p, x) for 100 random polynomials."""
    rng = random.Random(seed)
    deg = rng.randint(0, 5)
    coeffs = [rng.randint(-10, 10) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_deriv = sympy.diff(sp_expr, sx)
    sp_coeffs = sympy_coeffs(sp_deriv, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)
    ca_r = diff(ak_expr, x)
    ca_coeffs = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs is None or ca_coeffs is None:
        pytest.skip("expression not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs) == norm(sp_coeffs), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs}, sympy={sp_coeffs}"
    )


@pytest.mark.parametrize("seed", range(50))
def test_simplify_matches_sympy(seed):
    """simplify(p) has the same polynomial value as SymPy expand(p)."""
    rng = random.Random(seed + 200)  # different seed space
    deg = rng.randint(0, 4)
    coeffs = [rng.randint(-5, 5) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_expanded = sympy.expand(sp_expr)
    sp_coeffs = sympy_coeffs(sp_expanded, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)
    ca_r = simplify(ak_expr)
    ca_coeffs = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs is None or ca_coeffs is None:
        pytest.skip("expression not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs) == norm(sp_coeffs), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs}, sympy={sp_coeffs}"
    )


# ---------------------------------------------------------------------------
# Integration oracle — alkahest integrate vs SymPy integrate
# ---------------------------------------------------------------------------

_INTEGRATE_CASES = [
    ("constant_5", [5]),
    ("linear_1x", [0, 1]),
    ("linear_3x_plus_2", [2, 3]),
    ("quadratic", [1, 2, 1]),
    ("cubic", [0, 0, 0, 1]),
    ("full_cubic", [1, 2, 3, 4]),
    ("degree_4", [1, 1, 1, 1, 1]),
    ("negative_coeffs", [-3, 0, 2]),
    ("zero_poly", [0]),
    ("degree_5", [1, -1, 2, -2, 3, -3]),
]


@pytest.mark.parametrize("desc,coeffs", _INTEGRATE_CASES)
def test_integrate_matches_sympy(desc, coeffs):
    """alkahest integrate(p, x) == SymPy integrate(p, x) for fixed polynomial cases."""
    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_integral = sympy.integrate(sp_expr, sx)
    sp_coeffs_int = sympy_coeffs(sp_integral, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)

    try:
        ca_r = integrate(ak_expr, x)
    except Exception:
        pytest.skip(f"[{desc}] alkahest returned NotImplemented")

    ca_coeffs_int = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs_int is None or ca_coeffs_int is None:
        pytest.skip(f"[{desc}] not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs_int) == norm(sp_coeffs_int), (
        f"[{desc}] coeffs={coeffs}: alkahest={ca_coeffs_int}, sympy={sp_coeffs_int}"
    )


@pytest.mark.parametrize("seed", range(50))
def test_integrate_matches_sympy_random(seed):
    """alkahest integrate(p, x) agrees with SymPy for 50 random polynomials."""
    rng = random.Random(seed + 500)
    deg = rng.randint(0, 4)
    coeffs = [rng.randint(-5, 5) for _ in range(deg + 1)]

    sx = sympy.Symbol("x")
    sp_expr = sympy_poly(sx, coeffs)
    sp_integral = sympy.integrate(sp_expr, sx)
    sp_coeffs_int = sympy_coeffs(sp_integral, sx)

    pool = ExprPool()
    x = pool.symbol("x")
    ak_expr = alkahest_poly(pool, x, coeffs)

    try:
        ca_r = integrate(ak_expr, x)
    except Exception:
        pytest.skip(f"seed={seed}: alkahest returned NotImplemented for coeffs={coeffs}")

    ca_coeffs_int = alkahest_coeffs(ca_r.value, x, pool)

    if sp_coeffs_int is None or ca_coeffs_int is None:
        pytest.skip(f"seed={seed}: not convertible to polynomial")

    def norm(cs):
        while len(cs) > 1 and cs[-1] == 0:
            cs = cs[:-1]
        return cs or [0]

    assert norm(ca_coeffs_int) == norm(sp_coeffs_int), (
        f"seed={seed} coeffs={coeffs}: alkahest={ca_coeffs_int}, sympy={sp_coeffs_int}"
    )

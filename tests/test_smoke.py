"""Phase 0 smoke test: import alkahest and call the version function."""

import alkahest


def test_import():
    assert alkahest is not None


def test_version():
    v = alkahest.version()
    assert isinstance(v, str)
    assert len(v) > 0

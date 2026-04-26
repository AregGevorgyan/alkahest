#!/usr/bin/env python3
"""Enforce the V1-8 semver policy for the Python API.

Compares ``alkahest.__all__`` on the current ref against a baseline ref
(``origin/main`` in CI).  Removals and renames fail; additions are allowed
within the same major version.  Run from the repo root.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


INIT_PATH = Path("python/alkahest/__init__.py")


def extract_all(source: str) -> set[str]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        return {
                            elt.value
                            for elt in node.value.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                        }
    raise SystemExit(f"no __all__ found in {INIT_PATH}")


def git_show(ref: str, path: Path) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        # File did not exist on the baseline — treat as empty surface.
        return ""
    return result.stdout


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <baseline-ref>", file=sys.stderr)
        return 2

    baseline_ref = argv[1]
    head_source = INIT_PATH.read_text()
    baseline_source = git_show(baseline_ref, INIT_PATH)

    head_all = extract_all(head_source)
    baseline_all = extract_all(baseline_source) if baseline_source else set()

    removed = sorted(baseline_all - head_all)
    added = sorted(head_all - baseline_all)

    if removed:
        print("ERROR: symbols removed from alkahest.__all__ without a major bump:")
        for name in removed:
            print(f"  - {name}")
        print()
        print("Policy (V1-8): within a major version, __all__ is append-only.")
        print("If the removal is intentional, open a 2.0 milestone and add")
        print("a deprecation shim re-exporting the old name.")
        return 1

    if added:
        print("Added (OK — additive):")
        for name in added:
            print(f"  + {name}")
    else:
        print("No API surface changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

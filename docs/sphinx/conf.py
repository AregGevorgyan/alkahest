"""Sphinx config for the Alkahest Python API reference (V1-11 scaffold)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../python"))

project = "Alkahest"
author = "Alkahest Contributors"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

html_theme = "furo"
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# Configuration file for the Sphinx documentation builder.
#
# Hyperion HALO Production Documentation
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the package importable from the docs directory
sys.path.insert(0, os.path.abspath("../.."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "Hyperion HALO"
copyright = "2024, Hyperion HALO Team"
author = "Hyperion HALO Team"
release = "3.0.0"
version = "3.0"

# ---------------------------------------------------------------------------
# Extensions
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",       # Google / NumPy docstring styles
    "sphinx.ext.viewcode",       # link to source
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch":  ("https://pytorch.org/docs/stable", None),
    "numpy":  ("https://numpy.org/doc/stable", None),
}

# ---------------------------------------------------------------------------
# Autodoc
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "private-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# ---------------------------------------------------------------------------
# HTML theme
# ---------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
todo_include_todos = True

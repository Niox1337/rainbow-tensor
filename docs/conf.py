"""Sphinx configuration for the rainbow-tensor documentation."""

import os
import sys
from datetime import date

# The package is pure standard library at import time, so adding the source
# tree to the path is enough for autodoc; no install step is needed.
sys.path.insert(0, os.path.abspath("../src"))

project = "rainbow-tensor"
author = "Zhixiang Feng"
copyright = f"{date.today().year}, {author}"

try:
    from rainbow_tensor import __version__ as release
except Exception:
    release = "0.7.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

myst_enable_extensions = ["colon_fence", "deflist"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = "rainbow-tensor"
html_static_path = ["_static"]

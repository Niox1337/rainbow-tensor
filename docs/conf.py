"""Sphinx configuration for the rainbow-tensor documentation."""

import json
import sys
from datetime import date
from importlib import import_module
from pathlib import Path

# Resolve this checkout independently of the directory used to start Sphinx.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

project = "rainbow-tensor"
author = "Zhixiang Feng"
copyright = f"{date.today().year}, {author}"

release = import_module("rainbow_tensor").__version__
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


def write_capability_reference(app, exception):
    """Publish the installed operation contract beside a successful HTML build."""
    if exception is not None or app.builder.format != "html":
        return
    from rainbow_tensor import capabilities

    target = Path(app.outdir) / "_static" / "capabilities.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(capabilities(), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def setup(app):
    """Generate the machine reference from the same package used by autodoc."""
    app.connect("build-finished", write_capability_reference)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

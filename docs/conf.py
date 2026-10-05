"""Sphinx configuration for connectome-analysis."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "Connectome Analysis"
author = "Open Brain Institute"
copyright = "2023-2025 Open Brain Institute / EPFL"
release = "1.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "obi_sphinx_theme"
html_title = "Connectome Analysis"

# Sort API members alphabetically by name (matches the previous docs and makes
# the sidebar/function listings easier to scan).
autodoc_member_order = "alphabetical"
napoleon_google_docstring = False

# Headings already live under their module's page, so don't repeat the full
# dotted module path (e.g. show ``run_batch_model_building`` instead of
# ``connalysis.modelling.modelling.run_batch_model_building``).
add_module_names = False

# Some existing NumPy docstrings contain legacy Markdown-style links and math.
# Keep them renderable without making those legacy formatting warnings fatal.
suppress_warnings = ["docutils"]
napoleon_numpy_docstring = True


def _strip_signature(app, what, name, obj, options, signature, return_annotation):
    """Show only the member name in API headings, not the full argument list.

    The complete parameter list is still documented in each function's
    ``Parameters`` section, so no information is lost. This keeps the headings
    short and readable for functions with many arguments.
    """
    if what in {"function", "method", "class"}:
        return ("", None)
    return (signature, return_annotation)


def setup(app):
    """Register local Sphinx customizations."""
    app.connect("autodoc-process-signature", _strip_signature)
    return {"parallel_read_safe": True, "parallel_write_safe": True}

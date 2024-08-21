# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import re
import sys

from sphinx.ext import autodoc

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "geneformer"
copyright = "2024, Christina Theodoris"
author = "Christina Theodoris"
release = "0.1.0"
repository_url = "https://huggingface.co/ctheodoris/Geneformer"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

templates_path = ["_templates"]
exclude_patterns = [
    "**.ipynb_checkpoints",
]
autoclass_content = "both"


class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)


autodoc.ClassDocumenter = MockedClassDocumenter
add_module_names = False


def process_signature(app, what, name, obj, options, signature, return_annotation):
    # loop through each line in the docstring and replace path with
    # the generic path text
    signature = re.sub(r"PosixPath\(.*?\)", "FILEPATH", signature)
    return (signature, None)


def setup(app):
    app.connect("autodoc-process-signature", process_signature)


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sphinx = False
html_static_path = ["_static"]
html_logo = "_static/gf_logo.png"
html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 3,
    "logo_only": True,
}
html_css_files = [
    "css/custom.css",
]
html_show_sourcelink = False

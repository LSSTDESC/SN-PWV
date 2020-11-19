# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from pathlib import Path

import guzzle_sphinx_theme

package_source_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(package_source_path))

# -- Project information -----------------------------------------------------

project = 'SNAT_SIM'
copyright = '2020, Dark Energy Science Collaboration'
author = 'DESC'

# -- General configuration ---------------------------------------------------
master_doc = 'index'

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_copybutton',
    'nbsphinx',
    'nbsphinx_link'
]

# Syntax highlighting style
pygments_style = 'sphinx'
highlight_language = 'python3'

# Settings for rendering jupyter notebooks
nbsphinx_execute = 'never'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']
html_static_path = ['./static/']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom_style.css',
]

# -- Options for HTML output -------------------------------------------------

html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'
html_title = 'Supernova Atmospheric Simulation'
html_short_title = 'SNAT-SIM'
html_sidebars = {
    '**': ['logo-text.html', 'searchbox.html', 'globaltoc.html']
}


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)
    app.add_stylesheet("css/theme.css")

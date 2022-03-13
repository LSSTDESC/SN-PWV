# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import sys
from datetime import datetime
from pathlib import Path

import guzzle_sphinx_theme
import sncosmo

package_source_path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(package_source_path))

# -- Project information -----------------------------------------------------

project = 'SNAT_SIM'
copyright = f'{datetime.now().year}, Dark Energy Science Collaboration'
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
    'sphinx_copybutton',
    'nbsphinx',
    'nbsphinx_link',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',  # Napoleon need to be loaded BEFORE sphinx_autodoc_typehints
    'sphinx_autodoc_typehints',
    'sphinxarg.ext'
]

# Syntax highlighting style
pygments_style = 'sphinx'
highlight_language = 'python3'

# Settings for rendering jupyter notebooks
nbsphinx_execute = 'never'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['templates']
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom_style.css',
]

# -- Options for HTML output -------------------------------------------------

autodoc_member_order = 'bysource'
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'
html_title = 'Supernova Atmospheric Simulation'
html_short_title = 'SNAT-SIM'
html_sidebars = {
    '**': ['logo-text.html', 'searchbox.html', 'globaltoc.html']
}

# Download sncosmo filter data ahead of time so download messages don't
# interfere with doctests later on
sncosmo.get_source('salt2')
sncosmo.get_bandpass('sdssu')
sncosmo.get_bandpass('sdssg')
sncosmo.get_bandpass('sdssr')
sncosmo.get_bandpass('sdssi')
sncosmo.get_bandpass('sdssz')
sncosmo.get_bandpass('standard::b')


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)

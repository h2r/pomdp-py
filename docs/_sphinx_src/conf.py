# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../pomdp_py'))
sys.path.insert(0, os.path.abspath('../../problems'))


# -- Project information -----------------------------------------------------

project = 'pomdp_py'
copyright = '2020, Kaiyu Zheng'
author = 'Kaiyu Zheng'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',  # pip install sphinxcontrib-bibtex
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',        
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['static']

html_sidebars = {
    '**': [
        'about.html',
        'localtoc.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html'
    ]
}

html_theme_options = {
    'description': 'A framework to build and solve POMDP problems.',
    'logo': 'logo.png',
    'github_user': 'h2r',
    'github_repo': 'pomdp-py',
    'github_button': True,
    'sidebar_collapse': True,
    'fixed_sidebar': True,
    'donate_url': "paypal.me/zkytony/10",
    'extra_nav_links': {
        "H2R lab": "https://h2r.cs.brown.edu/",
        "Kaiyu's homepage": "http://kaiyuzh.me",        
        # "bigAI initiative": "http://bigai.cs.brown.edu/"
    },
    # Colors
    'narrow_sidebar_bg': "#330000",
    'narrow_sidebar_fg': "#EFEFEF",
    'narrow_sidebar_link': "#CDCDCD",
}

html_favicon = "images/favicon.ico"

# Do not sort automodule classes alphebatically but by how they appear in source.
autodoc_member_order = 'bysource'

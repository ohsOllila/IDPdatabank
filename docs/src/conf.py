# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import importlib.metadata
import os
import re
import sys
from datetime import datetime

# Directory containing this conf.py
here = os.path.dirname(__file__)
# Repository root
repo_root = os.path.abspath(os.path.join(here, "..", ".."))

# Only on Read the Docs
if os.getenv("READTHEDOCS") == "True":
    rtd_root = os.environ.get("READTHEDOCS_REPOSITORY_PATH")
    if rtd_root:
        repo_root = rtd_root

# Importing fairmd.idp requires the Data folder of a cloned repository.
os.environ.setdefault("NMLDB_ROOT_PATH", repo_root)

# Add to path:
sys.path.insert(0, repo_root)
year = datetime.now().year

# -- Project information -----------------------------------------------------

_mtd = importlib.metadata.metadata("fairmd-idp")
__version__ = _mtd["Version"]
__author_email__ = _mtd.get("Author-email", "unknown")
__author__ = re.sub(r" ?<.*>$", "", __author_email__)
__url__ = next(
    (u.split(", ")[1] for u in _mtd.get_all("Project-URL", []) if u.startswith("repository")),
    "https://github.com/NMRLipids/IDPdatabank",
)

project = f"FAIRMD IDP v{__version__}"
author = __author_email__
copyright = f"""{year}, {author}
    OSI Approved: GNU General Public License v3 (GPLv3)
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version
   """


# The full version, including alpha/beta/rc tags
release = __version__
html_context = {
    "copyright_link": __url__ + "/blob/main/LICENSE",
    "repo_link": __url__,
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (like 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# Heavy or non-pip-installable dependencies that are not needed to render the
# API documentation.
autodoc_mock_imports = [
    "pymol",
    "libarchive",
    "mdtraj",
    "maicos",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["**tests**", "*__init__.py", "build", "auto_gen/fairmd.rst"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

autodoc_member_order = "bysource"

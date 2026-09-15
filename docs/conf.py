import os
import sys
import tomllib
from pathlib import Path

# Insert src/ for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# Parse metadata directly from pyproject.toml
pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    pyproject_data = tomllib.load(f)

# Extract standard project metadata
project_meta = pyproject_data.get("project", {})
project = project_meta.get("name", "calfcv")

authors_list = project_meta.get("authors", [])
author = ", ".join([a.get("name", "") for a in authors_list if "name" in a])
copyright = f"2026, {author}"

# Extract the organization/author email
contact_email = ""
for a in authors_list:
    if "email" in a:
        contact_email = a["email"]
        break

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
]

# External documentation links
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
}

autosummary_generate = True
numpydoc_show_class_members = False

html_theme = "pydata_sphinx_theme"
html_static_path: list[str] = ["_static"]

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"/plot_",
    "download_all_examples": False,
}

# Makefile for calfcv project and Sphinx documentation

# Variables for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs
BUILDDIR      = docs/_build
GENERATEDDIR  = docs/generated

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable.)
endif

# Internal variables.
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) $(SOURCEDIR)

.PHONY: help clean html linkcheck release

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to remove all generated build files and autosummary stubs"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  release    to stage, commit, tag, and push a release (usage: make release version=0.21.0)"

clean:
	rm -rf $(BUILDDIR) $(GENERATEDDIR)

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

linkcheck:
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

release:
ifndef version
	$(error Missing version. Usage: make release version=X.Y.Z)
endif
	git add src/calfcv/_version.py docs/changelog.rst pyproject.toml
	git commit -m "bump: release v$(version)"
	git tag -a v$(version) -m "Release v$(version)"
	git push origin main
	git push origin v$(version)

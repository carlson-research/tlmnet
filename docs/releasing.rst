Release Checklist & Workflow
============================

Follow these steps when preparing a new version release for ``calfcv``.

Pre-Release Verification
------------------------

Run local checks to ensure clean builds, full test coverage, strict linting, and warning-free documentation compilation:

.. code-block:: bash

   pytest tests/ --cov=calfcv --cov-report=term-missing
   pre-commit run --all-files
   make -C docs html

Version & Changelog Update
--------------------------

1. Update ``__version__`` in ``src/calfcv/_version.py``.
2. Move items from the ``[Unreleased]`` section into a new version header (e.g., ``[0.4.0] - 2026-09-12``) in ``CHANGELOG.md``.

Commit, Tag, and Publish
------------------------

Use the automated ``Makefile`` target to stage, commit, tag, and push the release to GitHub. Pushing an annotated version tag (``v*``) automatically triggers GitHub Actions to publish packages to PyPI and update GitHub Pages documentation.

.. code-block:: bash

   make release version=X.Y.Z

Manual Release Workflow
~~~~~~~~~~~~~~~~~~~~~~~

For reference, the automated ``make release`` target executes the following manual git sequence:

.. code-block:: bash

   # Stage and commit version bump
   git add src/calfcv/_version.py CHANGELOG.md
   git commit -m "bump: release vX.Y.Z"

   # Create annotated tag
   git tag -a vX.Y.Z -m "Release vX.Y.Z"

   # Push branch and tag
   git push origin main
   git push origin vX.Y.Z

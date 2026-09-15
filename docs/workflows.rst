GitHub Actions CI/CD Architecture
=================================

This document details the GitHub Actions workflows for ``calfcv``. The CI/CD pipeline is designed around three core principles: **empirical reliability**, **single-source dependency management**, and **automated quality control**.

Because ``calfcv`` is an open-source library running on standard GitHub-hosted runners, compute time is optimized to guarantee total API and packaging integrity without unnecessary build fragility.

Architectural Rationale
-----------------------

Single Source of Truth for Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rather than maintaining hardcoded ``pip install`` commands inside workflow YAML files, all workflows install testing and documentation tools using:

.. code-block:: bash

   pip install -e .[dev]

**Why:** This forces CI/CD to use the exact dependency specifications declared in ``pyproject.toml``. Upgrading a tool (such as ``pytest`` or ``mypy``) locally automatically updates the CI environment without requiring edits to ``.github/workflows/``.

Pre-Flight Verification on Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Publishing a package to PyPI is permanent—once a version tag is pushed, it cannot be overwritten.

**Why:** ``release-pypi.yml`` does not trust Git tags blindly. Before invoking ``build`` or uploading to PyPI, it executes a unit test check (``pytest tests/unit/``) and a metadata validation step (``twine check dist/*``). If any test fails or if ``README.md`` rendering is broken, the job terminates before touching PyPI.

Sphinx-Gallery Execution & Artifact Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generating the documentation gallery requires fitting ``calfcv`` estimators on datasets like 20 Newsgroups. Running full fits on every documentation commit risks external network timeouts when fetching raw datasets.

**Why:** ``docs.yml`` implements ``actions/cache`` on ``docs/_build`` and ``docs/auto_examples``, keyed against the hash of ``examples/**/*.py``. If an edit is made strictly to core code or docstrings without altering the gallery scripts, Sphinx reuses the previously generated HTML and plots, dramatically speeding up deployment while ensuring stability.

OpenID Connect (OIDC) Trusted Publishing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Password and static API token authentication for PyPI are omitted in favor of standard OIDC Trusted Publishing (``id-token: write``).

**Why:** PyPI verifies the cryptographic identity of the GitHub Actions runner directly. No long-lived secret tokens are stored in the repository settings, eliminating credential leakage risks.

Workflow Reference
------------------

CI/CD Pipeline (tests.yml)
~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Triggers:** Pull requests and pushes to ``main``.
* **Matrix:** Tests across supported runtime environments (Python 3.11, 3.12).

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Step
     - Command / Action
     - Purpose
   * - **Dependency Sync**
     - ``pip install -e .[dev]``
     - Installs package in editable mode along with dev tools.
   * - **Lint & Type Check**
     - ``pre-commit``, ``mypy src/``
     - Ensures code formatting standards and strict typing.
   * - **Test Execution**
     - ``pytest tests/unit/``, ``pytest tests/e2e/``
     - Runs unit tests with XML coverage and integration passes.
   * - **Artifact Retention**
     - ``actions/upload-artifact@v4``
     - Preserves ``coverage.xml`` for 14 days (preventing storage bloat).

PyPI & GitHub Release (release-pypi.yml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Triggers:** Tag pushes matching ``v*`` (e.g., ``v0.4.0``).

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Step
     - Command / Action
     - Purpose
   * - **Pre-Flight Check**
     - ``pytest tests/unit/``
     - Ensures tagged code passes unit tests.
   * - **Package Build**
     - ``python -m build``
     - Generates ``.whl`` and ``.tar.gz`` distributions.
   * - **Metadata Audit**
     - ``twine check dist/*``
     - Verifies package description and PyPI render compliance.
   * - **PyPI Upload**
     - ``pypa/gh-action-pypi-publish@release/v1``
     - Publishes distribution via OIDC.
   * - **GitHub Release**
     - ``softprops/action-gh-release@v2``
     - Drafts GitHub release notes and attaches binary assets.

Documentation Deployment (docs.yml)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Triggers:** Pushes to ``main``.

.. list-table::
   :widths: 25 30 45
   :header-rows: 1

   * - Step
     - Command / Action
     - Purpose
   * - **Gallery Cache**
     - ``actions/cache@v4``
     - Caches built example plots based on ``examples/**/*.py`` hashes.
   * - **Sphinx Build**
     - ``sphinx-build -b html docs docs/_build/html``
     - Compiles RST files and executes uncached gallery scripts.
   * - **Pages Deployment**
     - ``actions/deploy-pages@v4``
     - Pushes static HTML output to GitHub Pages.

Maintenance Guidelines
----------------------

.. note::

   * **Adding a new test dependency:** Add the dependency to ``[project.optional-dependencies] dev`` inside ``pyproject.toml``. Do **not** modify ``tests.yml``.
   * **Cleaning cache artifacts:** GitHub automatically handles cache eviction when approaching repository limits. Retention for testing artifacts is intentionally capped at 14 days to keep account-wide storage footprint minimal.

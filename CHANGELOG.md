# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-12

### Added
* **Project Restructuring:** Refactored repository layout to a modern `src/` architecture (`src/calfcv`) .
* **Documentation Suite:** Established Sphinx documentation using `pydata-sphinx-theme`, `numpydoc`, and `sphinx-gallery` engine .
* **Build System Consolidation:** Consolidated documentation and release build logic into the root `Makefile`.
* **Automated CI/CD Pipelines:** Configured GitHub Actions workflows for multi-Python matrix testing, automated GitHub Pages deployment, and OIDC Trusted Publisher PyPI release automation .
* **Dynamic Packaging Metadata:** Implemented direct `pyproject.toml` metadata parsing within Sphinx configuration .

## [0.1.0] - 2026-09-10

### Added
* **Core Estimators:** Initial implementation of `Calf` and `CalfCV` estimators providing Coarse Approximation Linear Function classification with coarse integer feature weighting .
* **Scikit-Learn Compatibility:** Full compliance with Scikit-Learn estimator conventions and pipeline integration standards .
* **Test Suite:** Established initial test suite verifying metric optimization, feature selection accuracy, and cross-validation procedures .

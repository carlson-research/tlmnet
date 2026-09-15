# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-09-15

### Added
* **Core Estimator:** Introduced `TlmMilpClassifier`, an exact Mixed-Integer Linear Programming (MILP) classifier for Quantized Statistical Learning with Ternary Linear Models ($w_j \in \{-1, 0, 1\}$).
* **Exact MILP Solver:** Integrated `scipy.optimize.milp` using decision variable splitting ($w_j = u_j - v_j$) and mutual exclusivity constraints.
* **Exact $L_0$ Cardinality Budgeting:** Support for strict feature selection upper bounds via the `max_features` parameter ($\sum |w_j| \le k$).
* **Native Sparse Matrix Support:** Sparse block constraint assembly (`coo_matrix`, `hstack`) capable of processing high-dimensional text datasets (`CSR`, `CSC`, `COO`) without dense conversion.
* **Scikit-Learn 1.6+ API Compliance:** Full integration with `Pipeline`, `OneVsRestClassifier`, confidence scoring via `decision_function`, and modern `__sklearn_tags__` (`input_tags.sparse = True`).
* **Timeout & Incumbent Recovery:** Optimization logic that automatically recovers the best feasible integer solution found if solver execution reaches `time_limit`.
* **Documentation & Example Gallery:** Sphinx documentation configured with `pydata-sphinx-theme` and executable example scripts in `sphinx-gallery`.
* **Build System & CI/CD:** Modern `src/` layout, consolidated `Makefile`, and GitHub Actions workflows for multi-Python testing, GitHub Pages deployment, and PyPI release automation.

# tlmnet

[![CI](https://github.com/carlson-research/tlmnet/actions/workflows/tests.yml/badge.svg)](https://github.com/carlson-research/tlmnet/actions)
[![PyPI Version](https://img.shields.io/pypi/v/tlmnet.svg)](https://pypi.org/project/tlmnet/)
[![Python Version](https://img.shields.io/pypi/pyversions/tlmnet.svg)](https://pypi.org/project/tlmnet/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://carlson-research.github.io/tlmnet/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/carlson-research/tlmnet/blob/main/LICENSE)

`tlmnet` provides exact Mixed-Integer Linear Programming (MILP) solvers for Quantized Statistical Learning with Ternary Linear Models (TLMs).

By restricting coefficients to discrete ternary values ($w_j \in \{-1, 0, 1\}$), it constructs globally optimal, sparse, and interpretable classifiers compliant with Scikit-Learn 1.6+ API standards.

## Features

* **Exact Discrete Weights:** Solves for ternary coefficients $w_j \in \{-1, 0, 1\}$ directly using branch-and-bound optimization via `scipy.optimize.milp`.
* **Exact $L_0$ Cardinality Constraints:** Enforces strict integer feature selection budgets ($\sum \vert{}w_j\vert{} \le k$) using `max_features`.
* **Native Sparse Data Support:** Efficiently processes high-dimensional `scipy.sparse` matrices (`CSR`, `CSC`, `COO`) using sparse block constraint assembly.
* **Scikit-Learn Compatibility:** Plugs directly into `Pipeline`, `OneVsRestClassifier`, `GridSearchCV`, and standard estimator workflows.
* **Timeout & Incumbent Recovery:** Automatically returns the best feasible integer solution found if the solver reaches `time_limit` before proving global optimality.

## Installation

Install `tlmnet` directly from PyPI:

```bash
pip install tlmnet
```

## Quick Start Example

```python
from tlmnet import TlmMilpClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=200, n_features=15, n_informative=5, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Build pipeline with standardization and exact MILP classification
clf = make_pipeline(
    StandardScaler(),
    TlmMilpClassifier(max_features=5, C=1.0, time_limit=30.0)
)
clf.fit(X_train, y_train)

# Inspect model properties
tlm_model = clf.named_steps["tlmmilpclassifier"]
print("Ternary Coefficients:", tlm_model.coef_)
print("Test Accuracy:", clf.score(X_test, y_test))

```

## License

This project is licensed under the BSD-3-Clause License.

## Authors

* **Author & Maintainer**: Rolf Carlson (rolf@hrolfrc.com)

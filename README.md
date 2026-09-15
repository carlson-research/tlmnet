# calfcv

[![CI](https://github.com/carlson-research/calfcv/actions/workflows/tests.yml/badge.svg)](https://github.com/carlson-research/calfcv/actions)
[![PyPI Version](https://img.shields.io/pypi/v/calfcv.svg)](https://pypi.org/project/calfcv/)
[![Python Version](https://img.shields.io/pypi/pyversions/calfcv.svg)](https://pypi.org/project/calfcv/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://carlson-research.github.io/calfcv/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/carlson-research/calfcv/blob/main/LICENSE)

A Python implementation of the Coarse Approximation Linear Function (CALF) algorithm for binomial classification and feature selection.
This package provides binary classification with parsimonious and interpretable feature selection.

## Features

* **Integer Weighting:** Assigns integer weights for
  interpretable linear models.
* **Cross-Validation:** Built in cross validation for
  automated  hyperparameter tuning and feature selection.
* **Sparse Data Support:** Processes high-dimensional
  `scipy.sparse` matrices with stable memory usage.
* **Multiclass and Multilabel:** Supports Scikit-Learn
  `OneVsRestClassifier`.
* **Scikit-Learn Compatibility:** Plugs directly into `Pipeline`, `GridSearchCV`, and standard estimator workflows.

## Installation

Use pip to install calfcv:

```bash
pip install calfcv
```

## Quick Start Example

Make a classification problem and train the classifier:

```python
from calfcv import CalfCV
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Make a classification problem
seed = 42
X, y = make_classification(
    n_samples=30,
    n_features=5,
    n_informative=2,
    n_redundant=2,
    n_classes=2,
    random_state=seed
)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# Train the classifier
cls = CalfCV().fit(X_train, y_train)

# Get the score on unseen data
print(cls.score(X_test, y_test))
# Output: 0.875
```

## Citation

If you use this package in your research, please cite:

> Jeffries, C.D., Ford, J.R., Tilson, J.L. et al. *A greedy regression algorithm with coarse weights offers novel advantages.* Sci Rep 12, 5440 (2022). https://doi.org/10.1038/s41598-022-09415-2

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

## Authors

* **CALF Algorithm**: Clark D. Jeffries, John R. Ford, Jeffrey L. Tilson, Diana O. Perkins, Darius M. Bost, Dayne L. Filer, and Kirk C. Wilhelmsen
* **CalfCV Python Package (`calfcv`)**: Rolf Carlson
  (rolf@hrolfrc.com)

"""
Univariate ROC-AUC Feature Sorting Transformer
==============================================

This module provides :class:`AUCSorter`, a Scikit-Learn compatible transformer
designed to reorder feature matrix columns by their univariate ROC-AUC strength
prior to downstream feature selection or classification.

Problem Context
---------------
1. Algorithmic Non-Determinism & Local Optima:
   Greedy forward selection algorithms (such as CALF) are sensitive to initial column
   traversal order. When multiple candidate features yield identical marginal AUC gains,
   arbitrary dataset ordering dictates selection, leading to run-to-run output variance
   and potential convergence on local optima.
2. Cross-Validation Target Leakage:
   Pre-sorting columns across an entire dataset prior to cross-validation introduces
   subtle target leakage. The column order encodes global target relationships from test
   folds, artificially inflating downstream cross-validated performance metrics.
3. Loss of Feature Lineage:
   Naive matrix permutations break column index mappings and strip feature names,
   complicating post-hoc model interpretation in clinical decision support (CDS) contexts.

Solutions & Architectural Design
--------------------------------
The :class:`AUCSorter` resolves these issues through a stateful estimator contract
(:class:`~sklearn.base.TransformerMixin`, :class:`~sklearn.base.BaseEstimator`):

* Strict CV Isolation:
  By binding the univariate AUC sorting logic inside `.fit(X, y)`, every cross-validation
  split computes an isolated, leak-free feature sequence derived solely from the
  training fold data.
* Stateful Index Preservation:
  During training, `.fit()` computes univariate AUCs in parallel and freezes the
  ranking in `self.order_`. Inference methods (`.transform(X)`) apply this locked
  permutation sequence seamlessly without requiring target `y`.
* Full Data-Type Agnosticism:
  Handles dense NumPy `ndarray`s, SciPy sparse matrices (`csr_matrix`/`csc_matrix`),
  and Pandas `DataFrame`s natively without forced type conversions or unnecessary
  memory duplication.
* Feature Name Lineage & Pipeline Integration:
  Fully implements `get_feature_names_out()` and supports `set_output(transform="pandas")`,
  allowing downstream estimators and pipelines to track exact feature identities
  across permutations.
"""

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from ._algorithm import fit_columns


class AUCSorter(TransformerMixin, BaseEstimator):
    """Sort feature matrix columns by univariate ROC-AUC strength.

    Parameters
    ----------
    grid : tuple or list, default=(-1, 1)
        Weight candidates used to evaluate single-column direction.
    n_jobs : int, default=-1
        Number of CPU cores for parallel univariate AUC calculations.

    Attributes
    ----------
    order_ : ndarray of shape (n_features,)
        The permuted column indices sorted by descending univariate AUC.
    n_features_in_ : int
        Number of features seen during `.fit()`.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `.fit()` if `X` is a DataFrame.
    """

    def __init__(self, grid=(-1, 1), n_jobs=-1):
        self.grid = grid
        self.n_jobs = n_jobs

    def _validate_input(self, X, y=None, reset=False):
        """Validate input data using scikit-learn utilities."""
        if y is not None:
            return validate_data(
                self, X=X, y=y, accept_sparse=["csr", "csc", "coo"], reset=reset
            )
        return validate_data(
            self, X=X, accept_sparse=["csr", "csc", "coo"], reset=reset
        )

    def fit(self, X, y=None):
        """Calculate univariate AUC for each column and lock sorted order.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data matrix.
        y : array-like of shape (n_samples,)
            Target vector (binary or multiclass).

        Returns
        -------
        self : object
            Fitted transformer.
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        X_validated, y = self._validate_input(X, y, reset=True)

        classes = unique_labels(y)
        if len(classes) < 2:
            raise ValueError(
                f"This solver needs samples of at least 2 classes in the data, "
                f"but the data contains only one class: {classes[0]}"
            )

        # Compute parallel univariate AUCs via _algorithm
        tups = fit_columns(X_validated, y, grid=self.grid, n_jobs=self.n_jobs)
        self.order_ = np.array([i for _, _, i in tups])

        return self

    def transform(self, X):
        """Reorder incoming matrix columns using the trained sequence.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to reorder.

        Returns
        -------
        X_sorted : {ndarray, sparse matrix, DataFrame} of shape (n_samples, n_features)
            Reordered data matrix.
        """
        check_is_fitted(self, "order_")
        X_validated = self._validate_input(X, reset=False)

        if issparse(X):
            return X_validated.tocsc()[:, self.order_]
        elif hasattr(X, "iloc"):
            return X.iloc[:, self.order_]
        return X_validated[:, self.order_]

    def get_feature_names_out(self, input_features=None):
        """Preserve and reorder feature names for Scikit-Learn pipelines.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names.

        Returns
        -------
        feature_names_out : ndarray of str
            Reordered feature names matching `self.order_`.
        """
        check_is_fitted(self, "order_")
        if input_features is None:
            if hasattr(self, "feature_names_in_"):
                input_features = self.feature_names_in_
            else:
                raise ValueError(
                    "input_features must be passed if X had no feature names during fit."
                )
        return np.asarray(input_features)[self.order_]

    def _more_tags(self):
        return {"requires_y": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        tags.input_tags.sparse = True
        return tags

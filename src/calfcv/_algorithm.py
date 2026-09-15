"""
CALF Algorithmic Engine
=======================

This module contains the core mathematical routines for the Coarse Approximation
Linear Function (CALF) algorithm.

The CALF algorithm operates across three primary stages:

1. Feature Ranking & Screening (Phase 1: `fit_columns`, `_column_task`):
   Evaluates each candidate feature's univariate predictive power across the discrete
   weight grid (e.g., {-1, 1}). Features are ranked in parallel by maximum individual
   AUC-ROC to prioritize the forward selection order.

2. Greedy Forward Weight Optimization (Phase 2: `fit_forward_selection`):
   Iteratively evaluates remaining unselected features. At each step, it selects the
   feature-weight pair that yields the maximum aggregate score AUC improvement.
   The loop terminates when no remaining feature improves cumulative AUC by at least
   `auc_tol` or when target metrics are met.

3. Coarse Linear Projection (Phase 3: `predict`):
   Computes raw linear decision scores Z = sum(w_j * X_j) using the sparse set of
   selected non-zero discrete weights.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score


def predict(X, w):
    """Phase 3: Compute raw linear decision scores from weights and features.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_selected_features)
        The input feature matrix containing only selected features.
    w : array-like of shape (n_selected_features,)
        The coarse non-zero weights assigned to the selected features.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Unscaled raw linear projections Z = sum(w_j * X_j).
    """
    if issparse(X):
        Z = X.multiply(w)
        y_pred = np.asarray(Z.sum(axis=1)).ravel()
    else:
        y_pred = np.sum(X * w, axis=1)
    return y_pred


def _column_task(i, V, y, grid):
    """Phase 1 Worker: Evaluate univariate prediction AUC for a single feature column.

    Parameters
    ----------
    i : int
        The column index to evaluate.
    V : ndarray of shape (n_samples,)
        A single feature column.
    y : array-like of shape (n_samples,)
        Target class vector.
    grid : array-like
        Candidate coarse weights (e.g., [-1, 1]).

    Returns
    -------
    best_auc : float
        Maximum univariate AUC achieved by column i.
    best_w : float
        Optimal candidate weight for column i.
    i : int
        The column index.
    """
    best_auc = -1.0
    best_w = None
    classes = np.unique(y)

    for w in grid:
        y_score = np.nan_to_num(V * w, copy=False)

        # Binary target: standard single ROC-AUC calculation
        if len(classes) == 2:
            auc = roc_auc_score(y_true=(y == classes[1]), y_score=y_score)
        # Multiclass target: OvR average
        else:
            auc = np.mean(
                [roc_auc_score(y_true=(y == c), y_score=y_score) for c in classes]
            )

        if auc >= best_auc:
            best_auc = auc
            best_w = w

    return best_auc, best_w, i


def fit_columns(X, y, grid, n_jobs=-1):
    """Phase 1 Manager: Rank all dataset columns in parallel by individual AUC.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training input features.
    y : array-like of shape (n_samples,)
        Target vector.
    grid : array-like
        Candidate weight grid.
    n_jobs : int, default=-1
        Number of parallel processing jobs.

    Returns
    -------
    candidates : list of tuples
        Sorted list of tuples in descending order of univariate AUC:
        (auc, weight, column_index).
    """
    is_sp = issparse(X)
    if is_sp:
        X = X.tocsc()

    def get_col(j):
        return X[:, j].toarray().ravel() if is_sp else X[:, j]

    candidates = Parallel(n_jobs=n_jobs)(
        delayed(_column_task)(i, get_col(i), y, grid) for i in range(X.shape[1])
    )
    return sorted(candidates, reverse=True)


def fit_forward_selection(X, y, grid, auc_tol=1e-6, order_col=False, verbose=False):
    """Phase 2 Core Engine: Greedy step-forward feature and discrete weight selection.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training input features.
    y : array-like of shape (n_samples,)
        Binary target vector.
    grid : array-like
        Candidate coarse weights (e.g., [-1, 1]).
    auc_tol : float, default=1e-6
        Minimum cumulative AUC improvement required to accept a new feature.
    order_col : bool, default=False
        If True, pre-ranks columns via `fit_columns` before forward selection.
    verbose : bool, default=False
        If True, prints progress logs during fitting.

    Returns
    -------
    auc : list of float
        Cumulative maximum AUC trajectory at each accepted step.
    weights : list of float
        Selected coarse non-zero weights corresponding to `index`.
    index : list of int
        Feature indices selected during forward selection.
    """
    n_samples, n_features = X.shape
    is_sp = issparse(X)
    if is_sp:
        X = X.tocsc()

    if order_col:
        tups = fit_columns(X, y, grid)
        col_order = [i for _, _, i in tups]
    else:
        col_order = range(n_features)

    count = 0
    weights = []
    auc = []
    index = []

    # Safely initialize cumulative scores to exact zeros
    U = np.zeros(n_samples)

    for i in col_order:
        V = X[:, i].toarray().ravel() if is_sp else X[:, i]

        candidates = []
        for w_idx, w in enumerate(grid):
            Z = U + V * w
            y_score = np.nan_to_num(Z, copy=False)
            candidates.append((roc_auc_score(y_true=y, y_score=y_score), w_idx, Z, w))

        best_candidate = max(candidates, key=lambda item: (item[0], item[1]))
        max_auc, _, next_U, w_c = best_candidate

        if not auc or max_auc > max(auc) + auc_tol:
            weights.append(w_c)
            index.append(i)
            U = next_U

            if auc and verbose:
                print(
                    f"Count {count} of {n_features} fit feature {i} "
                    f"feature auc: {round(max_auc, 4)} > max auc: {round(max(auc), 4)} "
                    f"weight: {w_c} selected features: {len(index)} auc tol: {auc_tol}"
                )

            auc.append(max_auc)
        else:
            if count % 100 == 0 and verbose:
                print(
                    f"Count {count} of {n_features} max auc: {round(max(auc), 4)} "
                    f"number of contributing features {len(index)}"
                )

        count += 1

        if auc and max(auc) >= 0.999:
            if verbose:
                print(
                    f"found {len(index)} features that contribute positive auc.\n"
                    "auc threshold reached, breaking ..."
                )
            break

    return auc, weights, index

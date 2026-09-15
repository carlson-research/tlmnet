import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data


class TlmMilpClassifier(ClassifierMixin, BaseEstimator):
    """Exact Ternary Linear Model Classifier using Mixed-Integer Linear Programming (MILP).

    Solves for w_j in {-1, 0, 1} using decision variable splitting w_j = u_j - v_j.
    Natively supports sparse data constraints via scipy.sparse block matrices.
    """

    def __init__(
        self, max_features=None, C=1.0, time_limit=60.0, mip_rel_gap=1e-4, verbose=False
    ):
        self.max_features = max_features
        self.C = C
        self.time_limit = time_limit
        self.mip_rel_gap = mip_rel_gap
        self.verbose = verbose

    def _more_tags(self):
        return {"poor_score": True, "binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = (
            True  # Tell Scikit-Learn we natively accept sparse matrices
        )
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        # 1. Validate data; natively accept sparse inputs (CSR/CSC/COO)
        X, y = validate_data(
            self, X=X, y=y, accept_sparse=["csr", "csc", "coo"], reset=True
        )

        # 2. Check target type
        y_type = type_of_target(y)
        if y_type == "unknown":
            raise ValueError("Unknown label type: target y must be binary.")
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of target is {y_type}."
            )

        self.classes_ = unique_labels(y)
        if len(self.classes_) < 2:
            raise ValueError(
                f"This solver needs samples of at least 2 classes in the data, "
                f"but the data contains only one class: {self.classes_[0]}"
            )

        n_samples, n_features = X.shape

        # Convert y to {-1, +1}
        y_binary = np.where(y == self.classes_[1], 1, -1)

        # Decision Variables: 2*p (weights) + n (slacks)
        c = np.zeros(2 * n_features + n_samples)
        c[2 * n_features :] = self.C

        integrality = np.concatenate([np.ones(2 * n_features), np.zeros(n_samples)])
        lb = np.zeros(2 * n_features + n_samples)
        ub = np.concatenate([np.ones(2 * n_features), np.full(n_samples, np.inf)])
        bounds = Bounds(lb, ub)

        constraints = []

        # 1. Mutually Exclusive Ternary Constraint (Sparse)
        row_exc = np.repeat(np.arange(n_features), 2)
        col_exc = np.arange(2 * n_features)
        data_exc = np.ones(2 * n_features)

        A_exc_left = sp.coo_matrix(
            (data_exc, (row_exc, col_exc)), shape=(n_features, 2 * n_features)
        )
        A_exc_right = sp.csr_matrix((n_features, n_samples))
        A_exclusive = sp.hstack([A_exc_left, A_exc_right])
        constraints.append(LinearConstraint(A_exclusive, -np.inf, 1))

        # 2. Classification Margin Constraint (Sparse)
        if not sp.issparse(X):
            X_sparse = sp.csr_matrix(X)
        else:
            X_sparse = X

        # Multiply rows of X by y efficiently
        Y_diag = sp.diags(y_binary, dtype=float)
        X_y = Y_diag @ X_sparse
        X_y = X_y.tocoo()

        # Interleave u and v coefficients
        row_u, col_u, data_u = X_y.row, 2 * X_y.col, X_y.data
        row_v, col_v, data_v = X_y.row, 2 * X_y.col + 1, -X_y.data

        row_margin = np.concatenate([row_u, row_v])
        col_margin = np.concatenate([col_u, col_v])
        data_margin = np.concatenate([data_u, data_v])

        A_margin_left = sp.coo_matrix(
            (data_margin, (row_margin, col_margin)), shape=(n_samples, 2 * n_features)
        )
        A_margin_right = sp.eye(n_samples, format="coo")
        A_margin = sp.hstack([A_margin_left, A_margin_right])

        constraints.append(LinearConstraint(A_margin, 1.0, np.inf))

        # 3. L0 Sparsity Budget (Sparse)
        if self.max_features is not None:
            A_sparse_left = sp.csr_matrix(np.ones((1, 2 * n_features)))
            A_sparse_right = sp.csr_matrix((1, n_samples))
            A_sparse_const = sp.hstack([A_sparse_left, A_sparse_right])
            constraints.append(LinearConstraint(A_sparse_const, 0, self.max_features))

        # Solve MILP with expanded options
        options = {
            "time_limit": self.time_limit,
            "mip_rel_gap": self.mip_rel_gap,
            "disp": self.verbose,
        }
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options=options,
        )

        # Status 0 = Optimal, Status 1 = Time Limit Reached (but res.x might exist!)
        if res.status in (0, 1) and res.x is not None:
            u_opt = np.round(res.x[: 2 * n_features : 2])
            v_opt = np.round(res.x[1 : 2 * n_features : 2])
        else:
            # Only fail to all zeros if it's proven infeasible (Status 2) or unbounded (Status 3)
            u_opt = np.zeros(n_features)
            v_opt = np.zeros(n_features)

        self.coef_ = u_opt - v_opt
        self.intercept_ = 0.0

        return self

    def decision_function(self, X):
        """Predict confidence scores for samples."""
        check_is_fitted(self)
        X = validate_data(self, X=X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return X @ self.coef_

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

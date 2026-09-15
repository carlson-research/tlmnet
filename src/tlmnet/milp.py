import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data


class TlmMilpClassifier(ClassifierMixin, BaseEstimator):
    """Exact Ternary Linear Model Classifier using Mixed-Integer Linear Programming (MILP).

    Solves for w_j in {-1, 0, 1} using decision variable splitting w_j = u_j - v_j.
    """

    def __init__(self, max_features=None, C=1.0, time_limit=60.0):
        self.max_features = max_features
        self.C = C
        self.time_limit = time_limit

    def _more_tags(self):
        return {"poor_score": True, "binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        tags.estimator_type = "classifier"
        return tags

    def fit(self, X, y):
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        # 1. Validate data FIRST via validate_data (sets self.n_features_in_ natively)
        X, y = validate_data(self, X=X, y=y, accept_sparse=False, reset=True)

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

        # Decision Variables:
        # u_j in {0, 1}, v_j in {0, 1} for j in 1..p (w_j = u_j - v_j)
        # xi_i >= 0 (slack variables for margin errors)
        # Total variables: 2*p + n
        c = np.zeros(2 * n_features + n_samples)
        c[2 * n_features :] = self.C  # Minimize sum of slacks

        # Variable Integrality: 1 for binary (u, v), 0 for continuous (slack xi)
        integrality = np.concatenate([np.ones(2 * n_features), np.zeros(n_samples)])

        # Bounds: u_j, v_j in {0, 1}, xi_i >= 0
        lb = np.zeros(2 * n_features + n_samples)
        ub = np.concatenate([np.ones(2 * n_features), np.full(n_samples, np.inf)])
        bounds = Bounds(lb, ub)

        # Constraints
        constraints = []

        # 1. Mutually Exclusive Ternary Constraint: u_j + v_j <= 1
        A_exclusive = np.zeros((n_features, 2 * n_features + n_samples))
        for j in range(n_features):
            A_exclusive[j, 2 * j] = 1
            A_exclusive[j, 2 * j + 1] = 1
        constraints.append(LinearConstraint(A_exclusive, -np.inf, 1))

        # 2. Classification Margin Constraint: y_i * (X_i @ (u - v)) + xi_i >= 1
        A_margin = np.zeros((n_samples, 2 * n_features + n_samples))
        for i in range(n_samples):
            for j in range(n_features):
                A_margin[i, 2 * j] = y_binary[i] * X[i, j]
                A_margin[i, 2 * j + 1] = -y_binary[i] * X[i, j]
            A_margin[i, 2 * n_features + i] = 1.0
        constraints.append(LinearConstraint(A_margin, 1.0, np.inf))

        # 3. Optional L0 Sparsity Budget: sum(u_j + v_j) <= max_features
        if self.max_features is not None:
            A_sparse = np.zeros((1, 2 * n_features + n_samples))
            A_sparse[0, : 2 * n_features] = 1
            constraints.append(LinearConstraint(A_sparse, 0, self.max_features))

        # Solve MILP
        res = milp(
            c=c, integrality=integrality, bounds=bounds, constraints=constraints
        )

        if not res.success:
            u_opt = np.zeros(n_features)
            v_opt = np.zeros(n_features)
        else:
            u_opt = np.round(res.x[: 2 * n_features : 2])
            v_opt = np.round(res.x[1 : 2 * n_features : 2])

        self.coef_ = u_opt - v_opt
        self.intercept_ = 0.0

        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)
        scores = X @ self.coef_
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])
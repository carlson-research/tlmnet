import time
import numpy as np
from scipy.sparse import issparse
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import minmax_scale
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import validate_data

from ._algorithm import predict, fit_forward_selection


class Calf(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Coarse approximation linear function.

    Calf fits a linear model with coefficients w = (w1, ..., wp)
    to maximize the AUC of the targets predicted by the linear function.

    Parameters
    ----------
    grid : tuple, list, or int, default=(-1, 1)
        The search grid for weight candidates.
    auc_tol : float, default=1e-6
        Tolerance above max AUC for inclusion of a feature index.
    order_col : bool, default=False
        Whether to order the columns by individual AUC prior to fitting.
    verbose : bool, default=False
        If True, print status messages.

    Attributes
    ----------
    coef_ : list of float
        Estimated coefficients for the linear fit problem. Only
        one target should be passed, and this is a 1D list of length
        n_features.
    auc_ : list of float
        The cumulative AUC up to each selected feature.
    weights_ : list of float
        The non-zero weights assigned to the selected features.
    feature_index_ : list of int
        The indices of the features that contribute positive AUC.
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    X_ : {ndarray, sparse matrix}
        The training input features.
    y_ : ndarray
        The target vector.
    fit_time_ : float
        The number of seconds to fit X to y.

    Notes
    -----
    The feature matrix must be centered at 0. This can be accomplished with
    sklearn.preprocessing.StandardScaler, or similar. No intercept is calculated.

    Examples
    --------
    >>> import numpy as np
    >>> from calfcv import Calf
    >>> from sklearn.datasets import make_classification as mc
    >>> X, y = mc(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    >>> np.round(X[0:3, :], 2)
    array([[ 1.23, -0.76],
           [ 0.7 , -1.38],
           [ 2.55,  2.5 ]])
    >>> y[0:3]
    array([0, 0, 1])
    >>> cls = Calf().fit(X, y)
    >>> cls.score(X, y)
    0.76
    """

    def __init__(self, grid=(-1, 1), auc_tol=1e-6, order_col=False, verbose=False):
        self.grid = [grid] if isinstance(grid, int) else grid
        self.auc_tol = auc_tol
        self.order_col = order_col
        self.verbose = verbose

    def _validate_input(self, X, reset=False):
        """Validate input data using scikit-learn utilities."""
        return validate_data(
            self, X=X, accept_sparse=["csr", "csc", "coo"], reset=reset
        )

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Binary target vector relative to X.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if y is None:
            raise ValueError("requires y to be passed, but the target y is None")

        # 1. Validate data FIRST. This natively catches NaNs and raises a clean ValueError.
        X, y = validate_data(
            self, X=X, y=y, accept_sparse=["csr", "csc", "coo"], reset=True
        )

        # 2. Check target type SECOND. NaNs are already filtered out by this point.
        y_type = type_of_target(y)
        if y_type == "unknown":
            raise ValueError("Unknown label type: target y must be binary.")
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported. The type of the target is {y_type}."
            )

        self.classes_ = unique_labels(y)

        # Catch Scikit-Learn's single-class tests
        if len(self.classes_) < 2:
            raise ValueError(
                f"This solver needs samples of at least 2 classes in the data, "
                f"but the data contains only one class: {self.classes_[0]}"
            )

        self.X_ = X
        self.y_ = y

        if self.verbose:
            print(f"fitting {X.shape[1]} features.")

        start = time.time()

        # Direct call to algorithmic core without redundant sparse branching
        self.auc_, self.weights_, self.feature_index_ = fit_forward_selection(
            X,
            y,
            grid=self.grid,
            auc_tol=self.auc_tol,
            order_col=self.order_col,
            verbose=self.verbose,
        )

        self.fit_time_ = time.time() - start

        self.coef_ = [0] * X.shape[1]
        for i, w in zip(self.feature_index_, self.weights_):
            self.coef_[i] = w

        return self

    def decision_function(self, X):
        """Identify confidence scores for the samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input features and samples to evaluate.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The decision vector scaled between -1 and 1.
        """
        check_is_fitted(self)
        X = self._validate_input(X, reset=False)

        # Fix 2: Ensure the matrix is sliceable
        if issparse(X):
            X = X.tocsr()

        scores = np.array(
            minmax_scale(
                predict(X[:, self.feature_index_], self.weights_), feature_range=(-1, 1)
            )
        )
        return scores

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The data matrix for which we want to get the predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Vector containing the predicted class labels for each sample.
        """
        check_is_fitted(self)
        X = self._validate_input(X, reset=False)

        if len(self.classes_) < 2:
            y_class = self.y_
        else:
            y_class = np.heaviside(self.decision_function(X), 0).astype(int)
            y_class = [self.classes_[x] for x in y_class]
        return np.array(y_class)

    def predict_proba(self, X):
        """Probability estimates for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Vector to be scored, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        class_prob : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
            To create the probabilities, Calf uses the expit (sigmoid) function.
        """
        check_is_fitted(self)
        X = self._validate_input(X, reset=False)
        y_proba = expit(self.decision_function(X))
        class_prob = np.column_stack((1 - y_proba, y_proba))
        return class_prob

    def transform(self, X):
        """Reduce X to the features that contribute positive AUC.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input features and samples.

        Returns
        -------
        X_r : {ndarray, sparse matrix} of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        check_is_fitted(self)
        X = self._validate_input(X, reset=False)

        # Fix 2: Ensure the matrix is sliceable
        if issparse(X):
            X = X.tocsr()

        return X[:, self.feature_index_]

    def fit_transform(self, X, y):
        """Fit to the data, then reduce X to the features that contribute positive AUC.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input features and samples.
        y : array-like of shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        X_r : {ndarray, sparse matrix} of shape (n_samples, n_selected_features)
            The input samples with only the selected features.
        """
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"poor_score": True, "non_deterministic": True, "binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.poor_score = True
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = True
        tags.non_deterministic = True
        tags.estimator_type = "classifier"
        return tags

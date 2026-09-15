import time
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.multiclass import unique_labels, type_of_target
from sklearn.utils.validation import check_is_fitted
from .calf import Calf

try:
    from sklearn.utils.validation import validate_data

    HAS_VALIDATE_DATA = True
except ImportError:
    HAS_VALIDATE_DATA = False


class CalfCV(ClassifierMixin, TransformerMixin, BaseEstimator):
    """Coarse approximation linear function with cross validation.

    CalfCV fits a linear model with coefficients w = (w1, ..., wp)
    to maximize the AUC of the targets predicted by the linear function.
    It optimizes weight selection and feature inclusion thresholds through
    an internal GridSearchCV pipeline.

    Parameters
    ----------
    grid : tuple, list of tuples, or int, default=(-1, 1)
        The candidate search grid(s) for weight candidates to optimize over.
        Pass a list of tuples (e.g., [(-1, 1), (-1, 0, 1)]) to evaluate multiple grids.
    auc_tol : float or list of floats, default=1e-6
        Tolerance above max AUC for inclusion of a feature index. Pass a list
        to evaluate multiple tolerances.
    order_col : bool or list of bools, default=False
        Whether to order the columns by individual AUC prior to fitting. Pass a
        list to evaluate both options.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy for GridSearchCV.
    n_jobs : int, default=None
        Number of jobs to run in parallel for GridSearchCV. -1 means using all processors.
    verbose : bool, default=False
        If True, print status messages.

    Attributes
    ----------
    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.
    best_coef_ : list of float
        Estimated coefficients for the linear fit problem from the best model.
        Only one target should be passed, and this is a 1D list of length
        n_features.
    best_score_ : float
        The best AUC score over the cross validation.
    best_auc_ : list of float
        The cumulative AUC up to each selected feature from the best model.
    classes_ : ndarray of shape (n_classes,)
        The unique class labels.
    X_ : {ndarray, sparse matrix}
        The training input features.
    y_ : ndarray
        The target vector.
    model_ : GridSearchCV
        The fitted grid search pipeline.
    fit_time_ : float
        The number of seconds to fit X to y.

    Notes
    -----
    The feature matrix must be centered at 0. If X is dense, a StandardScaler
    is automatically prepended to the GridSearchCV pipeline.

    Examples
    --------
    >>> import numpy as np
    >>> from calfcv import CalfCV
    >>> from sklearn.datasets import make_classification as mc
    >>> X, y = mc(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
    >>> np.round(X[0:3, :], 2)
    array([[ 1.23, -0.76],
           [ 0.7 , -1.38],
           [ 2.55,  2.5 ]])
    >>> y[0:3]
    array([0, 0, 1])
    >>> cls = CalfCV().fit(X, y)
    >>> cls.score(X, y)
    0.7
    """

    def __init__(
        self,
        grid=(-1, 1),
        auc_tol=1e-6,
        order_col=False,
        cv=None,
        n_jobs=None,
        verbose=False,
    ):
        self.grid = grid
        self.auc_tol = auc_tol
        self.order_col = order_col
        self.cv = cv
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the model according to the given training data and optimize hyperparameters.

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

        # 1. Validate data FIRST so validate_data catches NaNs cleanly
        X, y = validate_data(
            self, X=X, y=y, accept_sparse=["csr", "csc", "coo"], reset=True
        )

        # 2. Check target type SECOND
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
        self.X_ = X
        self.y_ = y

        # Robustly unpack parameters into lists for GridSearchCV combinatorial mapping
        if isinstance(self.grid, int):
            valid_grids = [self.grid]
        elif isinstance(self.grid, tuple) and all(
            isinstance(x, int) for x in self.grid
        ):
            valid_grids = [self.grid]
        elif isinstance(self.grid, (list, tuple)):
            valid_grids = list(self.grid)
        else:
            valid_grids = [self.grid]

        valid_auc_tols = (
            list(self.auc_tol)
            if isinstance(self.auc_tol, (list, tuple))
            else [self.auc_tol]
        )
        valid_order_cols = (
            list(self.order_col)
            if isinstance(self.order_col, (list, tuple))
            else [self.order_col]
        )

        parameter_grid = {
            "classifier__grid": valid_grids,
            "classifier__auc_tol": valid_auc_tols,
            "classifier__order_col": valid_order_cols,
            "classifier__verbose": [self.verbose],
        }

        steps = [("classifier", Calf())]
        if not issparse(X):
            steps.insert(0, ("scaler", StandardScaler()))

        self.model_ = GridSearchCV(
            estimator=Pipeline(steps=steps),
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=self.cv,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )

        start = time.time()
        self.model_.fit(X, y)
        self.fit_time_ = time.time() - start

        self.best_params_ = self.model_.best_params_
        self.best_score_ = self.model_.best_score_
        self.best_coef_ = self.model_.best_estimator_["classifier"].coef_
        self.best_auc_ = self.model_.best_estimator_["classifier"].auc_

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
            The decision vector generated by the best pipeline estimator.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return self.model_.decision_function(X)

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
        X = validate_data(self, X=X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return self.model_.predict(X)

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
            Returns the probability of the sample for each class in the model.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return self.model_.predict_proba(X)

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
        X = validate_data(self, X=X, accept_sparse=["csr", "csc", "coo"], reset=False)
        return self.model_.transform(X)

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
        # Delegating to self.transform natively handles the reset=False validation step
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

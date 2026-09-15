from sklearn.utils.estimator_checks import parametrize_with_checks
from calfcv import Calf, CalfCV, AUCSorter


@parametrize_with_checks([Calf(), CalfCV(), AUCSorter()])
def test_all_estimators(estimator, check):
    """Validates that Calf, CalfCV, and AUCSorter adhere strictly to the scikit-learn

    BaseEstimator, ClassifierMixin, and TransformerMixin API conventions.
    """
    check(estimator)

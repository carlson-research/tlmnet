from sklearn.utils.estimator_checks import parametrize_with_checks
from tlmnet import TlmMilpClassifier


@parametrize_with_checks([TlmMilpClassifier()])
def test_all_estimators(estimator, check):
    """Validates that TlmMilpClassifier adheres strictly to
    scikit-learn API conventions.
    """
    check(estimator)

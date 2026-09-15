import pytest
import numpy as np
from sklearn.datasets import make_classification
from tlmnet import TlmMilpClassifier

def test_tlm_milp_classifier_basic():
    X, y = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=42)
    clf = TlmMilpClassifier(max_features=3, C=1.0)
    clf.fit(X, y)

    # Verify fitted attributes
    assert hasattr(clf, "coef_")
    assert len(clf.coef_) == 10
    # Ensure weights are strictly ternary {-1, 0, 1}
    assert set(clf.coef_).issubset({-1.0, 0.0, 1.0})
    # Ensure L0 sparsity budget was respected
    assert np.sum(np.abs(clf.coef_)) <= 3

    # Verify predictions
    preds = clf.predict(X)
    assert preds.shape == (50,)
    assert set(preds).issubset(set(np.unique(y)))

def test_tlm_milp_classifier_multiclass_error():
    X = np.random.randn(30, 5)
    y = np.array([0, 1, 2] * 10)  # 3 classes
    clf = TlmMilpClassifier()
    with pytest.raises(ValueError, match="Only binary classification is supported"):
        clf.fit(X, y)
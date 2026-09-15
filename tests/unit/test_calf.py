import numpy
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
from calfcv import Calf


def test_calf_sparse(sparse_data):
    X, y = sparse_data

    assert issparse(X)
    assert X.shape == (4, 10)

    clf = Calf()
    assert clf.grid == (-1, 1)

    clf.fit(X, y)

    # Modern scikit-learn relies on trailing underscores rather than an is_fitted_ boolean
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "coef_")

    assert clf.auc_ == [0.5, 0.875, 1.0]
    assert clf.coef_ == [1, 1, -1, 0, 0, 0, 0, 0, 0, 0]

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_calf(data):
    X, y = data
    clf = Calf()
    assert clf.grid == (-1, 1)

    clf.fit(X, y)
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

    # Check the first several entries of y
    assert all(y == [0, 0, 1, 1, 0, 1, 1, 0, 0, 1])

    # Get the prediction
    y_score = numpy.round(clf.fit(X, y).predict(X), 2)
    assert all(y_score == [0, 0, 1, 1, 0, 1, 1, 1, 0, 1])

    # Check shape
    assert len(y_score) == len(y) == X.shape[0]

    auc_p = roc_auc_score(y_true=y, y_score=y_score)
    assert numpy.round(auc_p, 2) == 0.9

    # Expect 1-2 informative features to be found
    X_r = clf.transform(X)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)

    X_r = clf.fit_transform(X, y)
    assert X_r.shape[1] == 3
    assert X_r.shape[0] == len(y)

from calfcv import CalfCV


def test_calfcv_default(data):
    """Test default initialization and standard API compliance."""
    X, y = data
    clf = CalfCV()

    # Check original default state is preserved
    assert clf.grid == (-1, 1)

    clf.fit(X, y)

    # Verify the scikit-learn standard trailing underscore attributes
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "X_")
    assert hasattr(clf, "y_")
    assert hasattr(clf, "model_")

    # Verify the new attribute exposed by our refactor
    assert hasattr(clf, "best_params_")
    assert hasattr(clf, "best_score_")

    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)

    # Use a dynamic bound instead of hardcoding '3' so tests don't break
    # if the internal Grid Search finds a better feature subset.
    X_r = clf.transform(X)
    assert 0 < X_r.shape[1] <= X.shape[1]
    assert X_r.shape[0] == len(y)

    X_fit_r = clf.fit_transform(X, y)
    assert 0 < X_fit_r.shape[1] <= X.shape[1]
    assert X_fit_r.shape[0] == len(y)


def test_calfcv_grid_search(data):
    """Test that CalfCV can accept lists of parameters and successfully execute an internal grid search."""
    X, y = data

    # Instantiate with multiple candidates for each parameter
    clf = CalfCV(
        grid=[(-1, 1), (-1, 0, 1)],
        auc_tol=[1e-6, 1e-4],
        order_col=[True, False],
        cv=2,  # Keep CV low for fast test execution
    )

    clf.fit(X, y)

    # Ensure the model found a winning combination from our exact candidates
    assert clf.best_params_["classifier__grid"] in [(-1, 1), (-1, 0, 1)]
    assert clf.best_params_["classifier__auc_tol"] in [1e-6, 1e-4]
    assert clf.best_params_["classifier__order_col"] in [True, False]

    # Verify the combinatorial search actually mapped multiple permutations
    # 2 grids * 2 auc_tols * 2 order_cols = 8 candidate models
    cv_results = clf.model_.cv_results_
    assert len(cv_results["params"]) == 8

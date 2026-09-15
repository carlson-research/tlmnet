import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline

from calfcv.auc_sorter import AUCSorter


@pytest.fixture
def synthetic_data():
    """Dataset where:
    - Col 1 has perfect AUC (1.0)
    - Col 2 has moderate AUC (0.75)
    - Col 0 has baseline random AUC (0.50)
    """
    y = np.array([0, 0, 1, 1])
    X = np.array(
        [
            [0.5, 0.1, 0.1],  # Sample 0 (Class 0)
            [0.5, 0.2, 0.6],  # Sample 1 (Class 0)
            [0.5, 0.8, 0.4],  # Sample 2 (Class 1)
            [0.5, 0.9, 0.9],  # Sample 3 (Class 1)
        ]
    )
    return X, y


def test_auc_sorter_dense(synthetic_data):
    X, y = synthetic_data
    sorter = AUCSorter(grid=(-1, 1), n_jobs=1)

    sorter.fit(X, y)

    # Col 1 (1.0 AUC) -> Col 2 (0.75 AUC) -> Col 0 (0.50 AUC)
    assert hasattr(sorter, "order_")
    np.testing.assert_array_equal(sorter.order_, np.array([1, 2, 0]))

    X_trans = sorter.transform(X)
    assert X_trans.shape == X.shape
    np.testing.assert_array_equal(X_trans[:, 0], X[:, 1])
    np.testing.assert_array_equal(X_trans[:, 1], X[:, 2])
    np.testing.assert_array_equal(X_trans[:, 2], X[:, 0])


def test_auc_sorter_sparse(synthetic_data):
    X, y = synthetic_data
    X_sparse = csr_matrix(X)

    sorter = AUCSorter(grid=(-1, 1), n_jobs=1)
    sorter.fit(X_sparse, y)

    X_trans = sorter.transform(X_sparse)

    assert hasattr(sorter, "order_")
    np.testing.assert_array_equal(sorter.order_, np.array([1, 2, 0]))
    assert isinstance(X_trans, type(X_sparse.tocsc()))
    np.testing.assert_array_equal(X_trans.toarray(), X[:, sorter.order_])


def test_auc_sorter_dataframe_and_feature_names(synthetic_data):
    X, y = synthetic_data
    feature_names = ["feat_low", "feat_high", "feat_mid"]
    df = pd.DataFrame(X, columns=feature_names)

    sorter = AUCSorter(grid=(-1, 1), n_jobs=1)
    sorter.fit(df, y)

    # Check input feature name tracking
    assert hasattr(sorter, "feature_names_in_")
    np.testing.assert_array_equal(
        sorter.feature_names_in_, np.array(feature_names, dtype=object)
    )

    # Check output feature name reordering
    names_out = sorter.get_feature_names_out()
    expected_names = np.array(["feat_high", "feat_mid", "feat_low"])
    np.testing.assert_array_equal(names_out, expected_names)

    # Check custom feature names argument
    custom_out = sorter.get_feature_names_out(["a", "b", "c"])
    np.testing.assert_array_equal(custom_out, np.array(["b", "c", "a"]))

    # Check DataFrame transformation preserving iloc slicing
    df_trans = sorter.transform(df)
    assert isinstance(df_trans, pd.DataFrame)
    np.testing.assert_array_equal(df_trans.columns, expected_names)


def test_auc_sorter_not_fitted_errors(synthetic_data):
    X, _ = synthetic_data
    sorter = AUCSorter()

    with pytest.raises(NotFittedError):
        sorter.transform(X)

    with pytest.raises(NotFittedError):
        sorter.get_feature_names_out()


def test_auc_sorter_missing_feature_names_error(synthetic_data):
    X, y = synthetic_data
    sorter = AUCSorter().fit(X, y)  # Fit on raw numpy array (no feature_names_in_)

    with pytest.raises(ValueError, match="input_features must be passed"):
        sorter.get_feature_names_out()


def test_auc_sorter_in_pipeline(synthetic_data):
    X, y = synthetic_data
    pipeline = make_pipeline(AUCSorter(grid=(-1, 1)))

    X_trans = pipeline.fit_transform(X, y)
    np.testing.assert_array_equal(X_trans[:, 0], X[:, 1])


def test_auc_sorter_pandas_output_config(synthetic_data):
    X, y = synthetic_data
    feature_names = ["c0", "c1", "c2"]
    df = pd.DataFrame(X, columns=feature_names)

    sorter = AUCSorter()
    sorter.set_output(transform="pandas")
    sorter.fit(df, y)

    df_trans = sorter.transform(df)
    assert isinstance(df_trans, pd.DataFrame)
    np.testing.assert_array_equal(df_trans.columns, np.array(["c1", "c2", "c0"]))

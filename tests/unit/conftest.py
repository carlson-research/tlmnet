import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import TfidfVectorizer


@pytest.fixture
def sparse_data():
    """Make a sparse classification problem for visual inspection."""
    text = [
        "It was the best of times",
        "it was the worst of times",
        "it was the age of wisdom",
        "it was the age of foolishness",
    ]

    X = TfidfVectorizer().fit_transform(text)
    # samples 0 and 2 are positive
    # samples 1 and 3 are negative
    y = [1, 0, 1, 0]

    return X, y


@pytest.fixture
def data():
    """Make a dense classification problem for visual inspection."""
    X, y = make_classification(
        n_samples=10,
        n_features=3,
        n_informative=2,
        n_redundant=1,
        n_classes=2,
        hypercube=True,
        random_state=8,
    )
    return X, y

User Guide
==========

Overview
--------

``calfcv`` implements the Coarse Approximation Linear Function (CALF) algorithm integrated with Cross-Validation[cite: 7].

Mathematical Background
-----------------------

Instead of optimizing continuous weights via gradient descent or :math:`L_1 / L_2` shrinkage penalties, ``calfcv`` uses a greedy step-forward selection routine that assigns discrete weight values (:math:`\{-1, 0, 1\}`) to selected variables, optimizing target metrics such as the AUC-ROC or :math:`t`-statistic directly[cite: 7].

Pipeline Patterns & Preprocessing
---------------------------------

Because ``Calf`` relies on direct addition and subtraction of feature values, **data scaling is strictly required**.

Dense Data (Standardization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For dense numeric datasets (e.g., biological or financial data), features must be centered at zero. You should always assemble ``Calf`` or ``CalfCV`` within a Scikit-Learn pipeline using ``StandardScaler``.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from calfcv import CalfCV

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('calf', CalfCV(cv=5, n_jobs=-1))
    ])

Sparse Text Data (TF-IDF)
^^^^^^^^^^^^^^^^^^^^^^^^^
For high-dimensional text data (e.g., IMDB reviews or 20 Newsgroups), using ``StandardScaler(with_mean=True)`` will destroy matrix sparsity and crash your memory. Instead, pair ``Calf`` with ``TfidfVectorizer``. The algorithm natively handles SciPy sparse matrices and uses non-zero TF-IDF frequencies directly.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from calfcv import Calf

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(sublinear_tf=True, max_df=0.5)),
        ('calf', Calf(order_col=True))
    ])


Estimator Selection: Calf vs. CalfCV
------------------------------------

* **``Calf`` (Base Estimator):** Best for massive sparse matrices (like 100,000+ text features) where cross-validation would be computationally prohibitive. It executes a single greedy forward-selection pass using fixed hyperparameters.
* **``CalfCV`` (Cross-Validation Wrapper):** Best for dense tabular datasets. It automatically splits the data, runs parallel grid searches over candidate weights and early-stopping tolerances (``auc_tol``), and refits the best parameters on the full dataset.


Handling Multiclass Targets
---------------------------

``Calf`` is fundamentally a binary classifier. Natively passing a target vector with more than two classes will result in an error. To perform multiclass classification, you must wrap the estimator in Scikit-Learn's ``OneVsRestClassifier``.

.. code-block:: python

    from sklearn.multiclass import OneVsRestClassifier
    from calfcv import Calf

    # This creates one binary Calf model per class
    ovr_clf = OneVsRestClassifier(Calf(order_col=True))


Feature Selection & Early Stopping (auc_tol)
--------------------------------------------

``Calf`` does not use all available features. It evaluates all unselected columns and greedily appends the single feature that provides the highest cumulative ROC-AUC sum.

The algorithm prevents overfitting using the ``auc_tol`` parameter. If the best remaining feature cannot improve the cumulative AUC by at least ``auc_tol``, the selection loop terminates automatically. This acts as a built-in feature selector, often reducing thousands of columns down to a sparse scorecard of 10 to 50 highly interpretable variables.

User Guide
==========

Overview
--------

``tlmnet`` provides exact optimization for Quantized Statistical Learning with Ternary Linear Models (TLMs). The core estimator, ``TlmMilpClassifier``, optimizes linear models whose feature weights are restricted to discrete ternary states:

.. math::

   w_j \in \{-1, 0, 1\}

Instead of relying on greedy heuristic approximations or continuous :math:`L_1` shrinkage penalties, ``tlmnet`` uses Mixed-Integer Linear Programming (MILP) via ``scipy.optimize.milp`` to identify globally optimal weight vectors.

Mathematical Background
-----------------------

Decision Variable Splitting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To express non-convex ternary weight constraints within a linear programming framework, each weight :math:`w_j` is decomposed into two binary indicator variables:

.. math::

   w_j = u_j - v_j, \quad \text{where } u_j, v_j \in \{0, 1\}

To prevent simultaneous activation (:math:`u_j=1` and :math:`v_j=1`), the model enforces mutual exclusivity:

.. math::

   u_j + v_j \le 1 \quad \forall j \in \{1, \dots, p\}

Soft-Margin MILP Formulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a dataset :math:`(X, y)` with binary targets :math:`y_i \in \{-1, +1\}`, ``TlmMilpClassifier`` minimizes total classification slack penalties :math:`\sum_{i=1}^n \xi_i`:

.. math::

   \min_{u, v, \xi} C \sum_{i=1}^n \xi_i

subject to the classification margin constraints:

.. math::

   y_i \left( \sum_{j=1}^p X_{ij} (u_j - v_j) \right) + \xi_i \ge 1, \quad \xi_i \ge 0

Exact L0 Feature Budgeting
^^^^^^^^^^^^^^^^^^^^^^^^^^

An exact upper bound on feature selection cardinality can be imposed via the ``max_features`` parameter, enforcing a strict :math:`L_0` constraint:

.. math::

   \sum_{j=1}^p (u_j + v_j) \le k

Pipeline Patterns & Preprocessing
---------------------------------

Feature Standardization
^^^^^^^^^^^^^^^^^^^^^^^

Because feature values directly multiply discrete weights :math:`w_j \in \{-1, 0, 1\}`, feature scaling heavily influences margin penalties. Dense numeric features should be standardized to zero mean and unit variance using ``StandardScaler``.

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from tlmnet import TlmMilpClassifier

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('tlm', TlmMilpClassifier(max_features=5, C=1.0))
    ])

Sparse Data Support
^^^^^^^^^^^^^^^^^^^

``TlmMilpClassifier`` natively processes sparse data representations. By leveraging ``scipy.sparse`` block matrix assembly, it constructs the constraint equations directly from ``CSR``, ``CSC``, or ``COO`` matrices without ever converting the dataset to a dense array. This makes it highly efficient for high-dimensional text classification pipelines using ``TfidfVectorizer``.

Hyperparameter Configuration
----------------------------

* **``max_features`` (int or None, default=None):** Maximum number of non-zero ternary weights allowed in the model (:math:`L_0` cardinality constraint). If ``None``, feature inclusion is unconstrained.
* **``C`` (float, default=1.0):** Penalty parameter for classification margin slacks. Higher values penalize misclassifications more heavily.
* **``time_limit`` (float, default=60.0):** Maximum allowable time in seconds allocated to the underlying MILP branch-and-bound solver. If the solver reaches this limit before proving global optimality, it automatically recovers and returns the best incumbent integer solution found.
* **``mip_rel_gap`` (float, default=1e-4):** Relative MIP gap tolerance for early branch-and-bound termination. Allows the solver to stop once the incumbent solution is within this percentage of the theoretical optimal bound.
* **``verbose`` (bool, default=False):** Enables real-time solver output logging during optimization to monitor branch-and-bound progress.

Handling Multiclass Targets
---------------------------

``TlmMilpClassifier`` strictly supports binary classification targets. To use ternary linear models on multiclass problems, wrap the estimator in Scikit-Learn's ``OneVsRestClassifier`` or ``OneVsOneClassifier``.

.. code-block:: python

    from sklearn.multiclass import OneVsRestClassifier
    from tlmnet import TlmMilpClassifier

    ovr_clf = OneVsRestClassifier(TlmMilpClassifier(max_features=10, C=1.0))
    ovr_clf.fit(X_train, y_train)

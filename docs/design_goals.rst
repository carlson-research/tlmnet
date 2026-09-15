Design Goals & Mathematical Formulation
=======================================

``tlmnet`` provides exact Mixed-Integer Linear Programming (MILP) solvers for Quantized Statistical Learning with Ternary Linear Models (TLMs), enforcing discrete ternary weights :math:`w_j \in \{-1, 0, 1\}` while maintaining strict Scikit-Learn API compliance.

Core Mathematical Formulation
-----------------------------

* **Ternary Decision Variable Splitting:** To optimize discrete ternary weights within a linear programming framework, ``tlmnet`` decomposes each weight into binary components:

  .. math::

     w_j = u_j - v_j, \quad \text{where } u_j, v_j \in \{0, 1\} \text{ and } u_j + v_j \le 1

  This guarantees mutually exclusive ternary states: :math:`+1` (:math:`u_j=1, v_j=0`), :math:`-1` (:math:`u_j=0, v_j=1`), or :math:`0` (:math:`u_j=0, v_j=0`).

* **Exact :math:`L_0` Sparsity Budgeting:** Unlike greedy heuristics or continuous :math:`L_1` (Lasso) approximations, ``tlmnet`` enforces exact feature cardinality directly via linear constraints:

  .. math::

     \sum_{j=1}^p (u_j + v_j) \le k

  Setting ``max_features=k`` guarantees that at most :math:`k` features receive non-zero ternary weights.

* **Global Margin Optimization:** The solver minimizes soft-margin classification error penalties subject to the ternary constraint system using ``scipy.optimize.milp``:

  .. math::

     \min_{u, v, \xi} \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i (X_i (u - v)) + \xi_i \ge 1, \quad \xi_i \ge 0

  This guarantees exact, globally optimal binary linear classifiers without relying on greedy forward selection or gradient descent approximations.

Architectural Integration
-------------------------

* **Scikit-Learn Estimator Contract:** Implements ``ClassifierMixin`` and ``BaseEstimator`` with support for Scikit-Learn 1.6+ estimator tags, feature validation, and seamless pipeline integration.
* **Deterministic Execution:** Eliminates stochastic seed dependence by leveraging branch-and-bound integer optimization drivers to produce repeatable weight vectors across executions.

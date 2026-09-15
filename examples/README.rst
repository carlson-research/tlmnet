=============================
tlmnet Examples Gallery
=============================

This gallery provides practitioner-focused examples demonstrating how to leverage
``TlmMilpClassifier`` for exact ternary linear modeling, strict :math:`L_0` feature
cardinality budgeting, and sparse text classification.

Together, these scripts illustrate how exact Mixed-Integer Linear Programming (MILP)
replaces greedy heuristic approximations, yielding globally optimal integer-weighted
models compliant with Scikit-Learn 1.6+ workflows.

Gallery Highlights
------------------

* **Exact L0 Feature Selection Budgeting:**
  Demonstrates how to enforce hard integer cardinality constraints using the ``max_features`` parameter on the Breast Cancer dataset to build ultra-sparse, highly interpretable scorecards.
  *(See: Exact L0 Sparsity Budgeting on Breast Cancer Dataset)*

* **Multiclass One-vs-Rest Text Classification:**
  Demonstrates how ``TlmMilpClassifier`` integrates seamlessly with Scikit-Learn's ``OneVsRestClassifier`` to perform exact multi-class document classification.
  *(See: Multilabel Text Document Classification with Exact MILP One-vs-Rest)*

* **Runtime Scaling & Solver Performance:**
  Quantifies branch-and-bound solve times across varying feature dimensions (:math:`p`) and sample sizes (:math:`n`).
  *(See: MILP Solve Time vs. Feature & Sample Dimensions)*

* **Decision Boundaries & Surface Visualization:**
  Visualizes the discrete, quantized decision surfaces created by ternary weight vectors :math:`w_j \in \{-1, 0, 1\}` on 2D feature spaces.
  *(See: Decision Boundaries & Margin Scores)*

* **Sparse Text Matrix Classification:**
  Shows how to pair ``TfidfVectorizer`` natively with ``TlmMilpClassifier`` using sparse SciPy block matrices.
  *(See: Sparse Text Classification with Vocabulary Limits)*

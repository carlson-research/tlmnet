# tlmnet Project Roadmap

This document outlines the planned feature additions, mathematical extensions, and architectural milestones for `tlmnet`.

---

## Short-Term Milestones (v0.2.0) — Core Estimator Enhancements

* **`sample_weight` Support:** Extend `fit(X, y, sample_weight=None)` to scale margin slack penalties ($\min C \sum_i w_i \xi_i$) in the MILP objective, enabling exact ternary modeling on imbalanced datasets.
* **Warm-Starting (`warm_start=True`):** Reuse incumbent MIP solutions across `GridSearchCV` hyperparameter sweeps (`C`, `max_features`) to prune branch-and-bound search trees and accelerate cross-validation.
* **Integer Intercept Quantization:** Introduce an `intercept_bounds` hyperparameter to restrict the bias term $b$ to discrete integers ($b \in \{-B, \dots, B\}$), yielding 100% integer-weighted scorecards.
* **Feature Group Sparsity:** Support group structure constraints so that $L_0$ cardinality budgeting can act on entire feature blocks rather than single columns.

---

## Mid-Term Milestones (v0.3.0) — Estimators & Deployment Helpers

* **Exact Ternary Regressor (`TlmMilpRegressor`):** Implement exact ternary linear regression ($y = Xw + b, w_j \in \{-1, 0, 1\}$) minimizing $L_1$ absolute deviation or $L_\infty$ maximum error.
* **Scorecard Exporters:** Add zero-dependency export methods (`to_sql()`, `to_c()`, `to_python()`) that translate fitted ternary weights into SQL `CASE WHEN` statements or native C functions for high-throughput edge deployment.
* **Calibrated Probabilities:** Incorporate piecewise linear approximations of log-loss into the MILP objective to output calibrated probabilities via `predict_proba`.

---

## Long-Term Vision (v1.0.0) — Multi-Solver Ecosystem via MathOpt

* **MathOpt Backend Integration:** Introduce an optional solver abstraction via Google OR-Tools' MathOpt API.
* **Commercial Solver Routing:** While retaining `scipy.optimize.milp` as the zero-dependency default for standard users, allow seamless routing to high-performance solvers (Gurobi, CPLEX, SCIP, XPRESS) for massive datasets.
* **Enterprise Scaling:** Evaluate commercial/pro options for enterprise-grade solver integrations based on community feedback and production demand.

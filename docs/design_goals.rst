Design Goals
============

``calfcv`` provides a mathematically faithful Python implementation of the Coarse Approximation Linear Function (CALF) algorithm ([Jeffries2022]_) while enforcing modern Scikit-Learn engineering contracts.

Core Mathematical Lineage
-------------------------

* **Coarse Integer Weighting:** Unlike standard linear models that optimize continuous floating-point coefficients, CALF restricts weights to discrete coarse values. ``calfcv`` evaluates candidate grid vectors :math:`w \in \{-1, +1\}` at each iterative step, generating interpretable integer combinations.
* **Greedy Forward Selection:** At iteration :math:`k`, ``calfcv`` scans all unselected features to identify the single predictor and weight candidate that maximizes the rank-sum Area Under the Receiver Operating Characteristic Curve (AUC) on normalized inputs. Iteration terminates when candidate additions fail to exceed the convergence threshold ``auc_tol``.
* **Collinearity Filtering:** Linearly dependent or redundant features produce marginal AUC gains equal to or lower than the current composite score. The rank-sum objective inherently excludes redundant predictors, selecting only the primary informative feature.

Architectural Modernization & Guardrails
----------------------------------------

* **Standardization Contract:** CALF requires zero-mean, unit-variance input features for valid coarse weighting. ``calfcv`` integrates directly with Scikit-Learn's preprocessing API, ensuring data standardization is performed inside cross-validation folds via ``Pipeline`` and ``CalfCV`` to eliminate data leakage.

References
----------

.. [Jeffries2022] Jeffries, C.D., Ford, J.R., Tilson, J.L. *et al.* (2022).
   A greedy regression algorithm with coarse weights offers novel advantages.
   *Sci Rep* 12, 5440. https://doi.org/10.1038/s41598-022-09415-2

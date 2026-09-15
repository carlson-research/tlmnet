=============================
CalfCV Examples Gallery
=============================

This gallery provides practitioner-focused examples demonstrating how to leverage
``CalfCV`` for automated hyperparameter tuning, dynamic noise suppression, and
highly interpretable feature selection.

Together, these scripts illustrate how ``CalfCV`` trades upfront computational
time for robust generalization, yielding simple integer-weighted models that
match the predictive power of continuous optimization techniques.

Gallery Highlights
------------------

* **Dynamic Noise Suppression:**
  Demonstrates how ``CalfCV`` dynamically prunes pure noise features without requiring a hardcoded feature count target (``k``), protecting downstream classifiers from overfitting.
  *(See: CALF as a Supervised Feature Selection Preprocessor)*

* **Clinical Interpretability vs. Continuous Weights:**
  Validates the estimator on real-world medical data. Compares ``CalfCV`` against L1 (Lasso) and RFE, showing how discrete ±1 weights create a highly interpretable aggregate scorecard with competitive ROC-AUC.
  *(See: Feature Selection Sparsity: CALF vs. L1 & RFE)*

* **Runtime vs. Performance Benchmarking:**
  Quantifies the computational cost of the internal grid search. Illustrates that the modest increase in fit time buys optimal predictive generalization and extreme model sparsity.
  *(See: Runtime vs. Classifier Performance Trade-offs)*

* **Decision Calibration & Probabilities:**
  Proves that despite utilizing discrete integer weights, ``CalfCV`` produces calibrated probability estimates via a sigmoid transformation, allowing for standard precision-recall threshold tuning.
  *(See: Decision Threshold Calibration)*

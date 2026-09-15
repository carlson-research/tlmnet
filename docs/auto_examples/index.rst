:orphan:

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


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When a classifier generates continuous decision scores or probabilities, changing the decision threshold (the cutoff point where a score becomes a hard 0 or 1 classification) shifts classification metrics.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_decision_thresholds_thumb.png
    :alt:

  :doc:`/auto_examples/plot_decision_thresholds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Decision Threshold Calibration: Precision, Recall, and F1 Trade-offs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example visualizes the internal greedy forward-selection mechanics of CalfCV.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_cumulative_auc_by_feature_thumb.png
    :alt:

  :doc:`/auto_examples/plot_cumulative_auc_by_feature`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cumulative AUC by Feature: Forward Selection Trajectory</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates how Calf performs multi-class text classification using a One-vs-Rest strategy (~sklearn.multiclass.OneVsRestClassifier).">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_classify_newsgroups_thumb.png
    :alt:

  :doc:`/auto_examples/plot_classify_newsgroups`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multilabel Text Document Classification with CALF One-vs-Rest</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example compares the coarse feature selection of CalfCV against L1-penalized Logistic Regression (Lasso) and Recursive Feature Elimination (RFE) on the Wisconsin Breast Cancer dataset.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_feature_selection_breast_cancer_thumb.png
    :alt:

  :doc:`/auto_examples/plot_feature_selection_breast_cancer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Feature Selection Sparsity: CALF vs. L1 & RFE</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Predict positive and negative sentiment on the IMDB database using Calf.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_sentiment_imdb_thumb.png
    :alt:

  :doc:`/auto_examples/plot_sentiment_imdb`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Sentiment Analysis of High-Dimensional IMDB Reviews</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example benchmarks execution time (fit time) against downstream ROC-AUC and feature selection sparsity across different preprocessing strategies on a noisy dataset.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_runtime_vs_performance_thumb.png
    :alt:

  :doc:`/auto_examples/plot_runtime_vs_performance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Runtime vs. Classifier Performance Trade-offs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This example demonstrates using Calf and CalfCV as a dimensionality reduction preprocessor inside a Scikit-Learn pipeline.">

.. only:: html

  .. image:: /auto_examples/images/thumb/sphx_glr_plot_algorithm_leverage_thumb.png
    :alt:

  :doc:`/auto_examples/plot_algorithm_leverage`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CALF as a Supervised Feature Selection Preprocessor</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_examples/plot_decision_thresholds
   /auto_examples/plot_cumulative_auc_by_feature
   /auto_examples/plot_classify_newsgroups
   /auto_examples/plot_feature_selection_breast_cancer
   /auto_examples/plot_sentiment_imdb
   /auto_examples/plot_runtime_vs_performance
   /auto_examples/plot_algorithm_leverage



.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_

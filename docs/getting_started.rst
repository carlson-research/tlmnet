Getting Started
===============

Installation
------------

Install ``tlmnet`` directly from PyPI:

.. code-block:: bash

   pip install tlmnet

Basic Usage
-----------

``tlmnet`` provides an exact MILP classifier compliant with the Scikit-Learn Estimator API.

Because feature values directly multiply discrete weights, feature scaling heavily influences margin penalties. It is highly recommended to wrap the estimator in a standardization pipeline:

.. code-block:: python

   from tlmnet import TlmMilpClassifier
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler

   X, y = make_classification(n_samples=100, n_features=10, n_informative=5, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

   clf = make_pipeline(
       StandardScaler(),
       TlmMilpClassifier(max_features=5, C=1.0)
   )
   clf.fit(X_train, y_train)

   tlm_model = clf.named_steps["tlmmilpclassifier"]
   print("Ternary Coefficients:", tlm_model.coef_)
   print("Test Score:", clf.score(X_test, y_test))

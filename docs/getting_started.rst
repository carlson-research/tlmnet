Getting Started
===============

Installation
------------

Install ``calfcv`` directly from PyPI:

.. code-block:: bash

   pip install calfcv

Basic Usage
-----------

``calfcv`` exposes standard Scikit-Learn estimator classes:

.. code-block:: python

   from calfcv import CalfCV
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=500, n_features=20, random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

   model = CalfCV()
   model.fit(X_train, y_train)
   print("Selected features:", model.weight_vec_)

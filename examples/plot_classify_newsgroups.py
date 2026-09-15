"""
===================================================
Sparse Text Classification with Vocabulary Limits
===================================================

Shows how to pair ``TfidfVectorizer`` with ``TlmMilpClassifier`` by constraining
vocabulary dimension to prevent exponential MILP solver time, visualizing the
resulting ternary text weights.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from tlmnet import TlmMilpClassifier

# Fetch binary text dataset subset
categories = ["alt.atheism", "sci.space"]
newsgroups = fetch_20newsgroups(
    subset="train", categories=categories, remove=("headers", "footers", "quotes")
)

# Constrain vocabulary to top 25 terms to keep MILP dimension tractable
pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(max_features=25, stop_words="english", sublinear_tf=True),
        ),
        ("tlm", TlmMilpClassifier(max_features=8, C=1.0)),
    ]
)

pipeline.fit(newsgroups.data, newsgroups.target)

# Retrieve selected words and their discrete ternary weights
feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
weights = pipeline.named_steps["tlm"].coef_

# Filter down to only the non-zero features selected by the MILP solver
mask = weights != 0
active_words = feature_names[mask]
active_weights = weights[mask]

# Sort by weight for cleaner plotting
sort_idx = np.argsort(active_weights)
active_words = active_words[sort_idx]
active_weights = active_weights[sort_idx]

# Plot the ternary weights
plt.figure(figsize=(8, 4))
colors = ["#d62728" if w < 0 else "#2ca02c" for w in active_weights]
plt.barh(active_words, active_weights, color=colors, edgecolor="black")

plt.xlabel("Ternary Weight (-1 or +1)")
plt.title("Exact MILP Selected Terms (Text Classification)")
plt.xticks([-1, 0, 1])
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

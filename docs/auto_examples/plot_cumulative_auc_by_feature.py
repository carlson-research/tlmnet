"""
====================================================================
Cumulative AUC by Feature: Forward Selection Trajectory
====================================================================

This example visualizes the internal greedy forward-selection mechanics
of ``CalfCV``.

At each step, ``CalfCV`` evaluates all unselected features and appends the
single feature (and coarse ±1 weight direction) that yields the highest
cumulative ROC-AUC sum. The process terminates automatically when the
AUC gain falls below ``auc_tol``.

The weights (+1, -1) assigned at each step are displayed directly above
the data points, while the corresponding feature added at each step is
labeled along the X-axis.
"""

# %%
# Imports and Model Fitting
# -------------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from calfcv import CalfCV

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
feature_names = X.columns.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = CalfCV(
    grid=[(-1, 1), (-1, 0, 1)],
    auc_tol=[1e-4, 1e-3, 1e-2],
    order_col=[True, False],
    cv=5,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

# %%
# Extract Forward-Selection Steps
# -------------------------------
best_calf = clf.model_.best_estimator_["classifier"]

# Because of the patch to `_utils.py`, these three arrays are now
# guaranteed to be the exact same length.
cumulative_aucs = best_calf.auc_
selected_indices = best_calf.feature_index_
assigned_weights = best_calf.weights_

steps = np.arange(1, len(cumulative_aucs) + 1)

# Format X-axis tick labels as "1. Feature Name"
x_labels = [
    f"{step}. {feature_names[idx]}" for step, idx in zip(steps, selected_indices)
]

# %%
# Plot Clean Trajectory with Minimal Point Annotations
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Plot cumulative AUC curve
ax.plot(
    steps,
    cumulative_aucs,
    marker="o",
    markersize=8,
    color="#1f77b4",
    linewidth=2.5,
    label="Cumulative Training ROC-AUC",
)

# Minimal point annotations: Just display '+1' or '-1' directly above vertex
for step, auc, weight in zip(steps, cumulative_aucs, assigned_weights):
    weight_str = f"+{int(weight)}" if weight > 0 else f"{int(weight)}"
    ax.annotate(
        weight_str,
        (step, auc),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="#1f77b4" if weight > 0 else "#d62728",
    )

# Highlight early-stopping cutoff line
optimal_tol = clf.best_params_["classifier__auc_tol"]
ax.axhline(
    y=cumulative_aucs[-1],
    color="gray",
    linestyle="--",
    linewidth=1.5,
    label=f"Early Stopping Plateau (auc_tol={optimal_tol})",
)

# Configure crisp X-axis tick labels
ax.set_xticks(steps)
ax.set_xticklabels(x_labels, rotation=35, ha="right", fontsize=9)

ax.set_ylabel("Cumulative ROC-AUC", fontsize=11)
ax.set_title(
    "CALF Forward Selection: Cumulative AUC by Feature Step",
    fontsize=12,
    fontweight="bold",
)
ax.grid(True, linestyle="--", alpha=0.5)
ax.legend(loc="lower right")

# Padding for point labels
ymin, ymax = ax.get_ylim()
ax.set_ylim(ymin, ymax + 0.02)

plt.tight_layout()
plt.show()

# %%
# Step-by-Step Selection Summary
# ------------------------------
summary_df = pd.DataFrame(
    {
        "Step": steps,
        "Feature Added": [feature_names[i] for i in selected_indices],
        "Coarse Weight": [f"{int(w):+d}" for w in assigned_weights],
        "Cumulative AUC": np.round(cumulative_aucs, 4),
    }
)

print("\n" + "=" * 55)
print("GREEDY FEATURE SELECTION STEP SUMMARY")
print("=" * 55)
print(summary_df.to_string(index=False))

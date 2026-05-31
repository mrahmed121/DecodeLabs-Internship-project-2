"""
================================================================================
  DecodeLabs AI Engineering Internship — Project 2: Data Classification Using AI
  Week 2 Task | KNN Classification on the Iris Benchmark Dataset
  Author  : AI Engineer Intern
  Framework: Scikit-Learn | IPO Pipeline (Input → Process → Output)
================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import warnings

warnings.filterwarnings("ignore")

# ── Third-Party: Data & ML ─────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==============================================================================
#  SECTION 0 — GLOBAL CONFIG
# ==============================================================================

RANDOM_STATE = 42  # Seed for reproducibility across all random ops
TEST_SIZE = 0.20  # 80 / 20 train-test split
STANDARD_K = 5  # Strict instruction: Standard tuning K=5
K_RANGE = range(1, 31)  # Candidate K values evaluated by the Elbow Method
FIGURE_DPI = 150
DIVIDER = "=" * 72


# ==============================================================================
#  INPUT & PROCESS PHASE — Loading, Splitting & The Gatekeeper Rule
# ==============================================================================

def load_split_and_scale_data():
    """
    Load data, split it FIRST (to prevent data leakage), and then apply StandardScaler.
    """
    print(DIVIDER)
    print("  INPUT & PROCESS PHASE — Loading, Splitting & Scaling")
    print(DIVIDER)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    iris = load_iris()
    X, y = iris.data, iris.target

    print(f"\n  Dataset      : Iris Benchmark  (UCI / Fisher 1936)")
    print(f"  Samples      : {X.shape[0]}  |  Features : {X.shape[1]}")
    print(f"  Target Classes : {list(iris.target_names)}\n")

    # ── 2. Split (80/20) ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,  # Break sequence/order bias
        stratify=y  # Preserve class proportions
    )

    print(f"  [Split Success] Training: {len(X_train)} samples | Test: {len(X_test)} samples")

    # ── 3. Scale (The Gatekeeper Rule) ────────────────────────────────────────
    # Crucial: Fit only on Train, then transform Train and Test separately
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("  [Gatekeeper Rule Applied] StandardScaler → Data Leakage Prevented.")
    print(f"  Post-scale mean (Train Feature 0): {X_train_scaled[:, 0].mean():.2f} (Target: 0.0)")
    print(f"  Post-scale var  (Train Feature 0): {X_train_scaled[:, 0].var():.2f} (Target: 1.0)\n")

    return X_train_scaled, X_test_scaled, y_train, y_test, iris


# ==============================================================================
#  PROCESS PHASE — Model Training & Elbow Evaluation
# ==============================================================================

def evaluate_elbow(X_train_scaled, X_test_scaled, y_train, y_test):
    """Evaluate error rates across different K values for visualization."""
    error_rates = []
    for k in K_RANGE:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)
        preds = knn_temp.predict(X_test_scaled)
        error_rates.append(1 - accuracy_score(y_test, preds))
    return error_rates


def train_knn(X_train_scaled, y_train):
    """Instantiate and Fit the KNN model using strict instruction K=5."""
    print(DIVIDER)
    print("  PROCESS PHASE — Scikit-Learn Workflow (Instantiate → Fit → Predict)")
    print(DIVIDER)

    # ── Phase a: INSTANTIATE ──────────────────────────────────────────────────
    model = KNeighborsClassifier(
        n_neighbors=STANDARD_K,  # Strict Guideline K=5
        metric="euclidean",
        weights="uniform"
    )
    print(f"\n  [a] INSTANTIATE → KNeighborsClassifier(n_neighbors={STANDARD_K})")

    # ── Phase b: FIT ──────────────────────────────────────────────────────────
    model.fit(X_train_scaled, y_train)
    print(f"  [b] FIT         → model.fit(X_train, y_train)  ✓\n")

    return model


# ==============================================================================
#  OUTPUT PHASE — Validation & Metrics
# ==============================================================================

def evaluate_model(model, X_test_scaled, y_test, iris):
    """Run predictions and print evaluation metrics (Accuracy, CM, F1)."""

    # ── Phase c: PREDICT ──────────────────────────────────────────────────────
    predictions = model.predict(X_test_scaled)
    print(f"  [c] PREDICT     → predictions = model.predict(X_test)  ✓\n")

    print(DIVIDER)
    print("  OUTPUT PHASE — Validation & Metrics")
    print(DIVIDER)

    # ── 1. Accuracy ───────────────────────────────────────────────────────────
    accuracy = accuracy_score(y_test, predictions)
    print(f"\n  ➤ Overall Accuracy : {accuracy * 100:.2f}% (Beware of Accuracy Mirage on imbalanced data)\n")

    # ── 2. Confusion Matrix ───────────────────────────────────────────────────
    cm = confusion_matrix(y_test, predictions)
    print("  ➤ Confusion Matrix (Rows=Actual, Columns=Predicted):")
    print(cm)
    print("  * Diagonals = True Positives | Non-diagonals = FP & FN Errors\n")

    # ── 3. Classification Report (F1-Score) ───────────────────────────────────
    report = classification_report(y_test, predictions, target_names=iris.target_names)
    print("  ➤ Classification Report (Precision, Recall, F1-Score):")
    print(report)

    return predictions, cm, accuracy


# ==============================================================================
#  SECTION 5 — VISUALISATION
# ==============================================================================

def plot_results(error_rates, cm, class_names):
    """Generate side-by-side diagnostic plots."""
    fig = plt.figure(figsize=(14, 5), facecolor="#0f1117")
    fig.suptitle("DecodeLabs | Project 2: KNN Iris Classifier", fontsize=13, fontweight="bold", color="#e8eaf6")

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # ── Elbow Curve ───────────────────────────────────────────────────────────
    ax1.set_facecolor("#1a1d27")
    ax1.plot(list(K_RANGE), error_rates, color="#7986cb", marker="o", markersize=5)
    ax1.axvline(x=STANDARD_K, color="#ff8a65", linestyle="--", label=f"Blueprint K = {STANDARD_K}")
    ax1.set_title("Elbow Method Evaluation", color="#e8eaf6", pad=10)
    ax1.set_xlabel("K (Number of Neighbours)", color="#9e9e9e")
    ax1.set_ylabel("Error Rate", color="#9e9e9e")
    ax1.tick_params(colors="#9e9e9e")
    ax1.legend()
    ax1.grid(True, color="#2e3250", linewidth=0.6)

    # ── Confusion Matrix Heatmap ──────────────────────────────────────────────
    ax2.set_facecolor("#1a1d27")
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title("Confusion Matrix Heatmap", color="#e8eaf6", pad=10)
    ax2.set_xlabel("Predicted Label", color="#9e9e9e")
    ax2.set_ylabel("True Label", color="#9e9e9e")
    ax2.tick_params(colors="#9e9e9e")

    plt.tight_layout()
    # Saved in current directory to avoid FileNotFoundError
    plt.savefig("decodelabs_knn_diagnostics.png", dpi=FIGURE_DPI, bbox_inches="tight", facecolor="#0f1117")
    print("  [Plot saved] → decodelabs_knn_diagnostics.png locally.\n")
    plt.show()


# ==============================================================================
#  SECTION 6 — MAIN ORCHESTRATOR
# ==============================================================================

def main():
    print("\n" + DIVIDER)
    print("  DecodeLabs AI Internship — Project 2: Data Classification Using AI")
    print("  KNN Iris Classifier | IPO Pipeline                         Week 2")
    print(DIVIDER + "\n")

    X_train_scaled, X_test_scaled, y_train, y_test, iris = load_split_and_scale_data()

    error_rates = evaluate_elbow(X_train_scaled, X_test_scaled, y_train, y_test)

    model = train_knn(X_train_scaled, y_train)

    predictions, cm, accuracy = evaluate_model(model, X_test_scaled, y_test, iris)

    plot_results(error_rates, cm, iris.target_names)

    print(DIVIDER)
    print("  PIPELINE COMPLETE")
    print(f"  Model      : KNeighborsClassifier(n_neighbors={STANDARD_K})")
    print(f"  Accuracy   : {accuracy * 100:.2f}%")
    print(DIVIDER + "\n")


if __name__ == "__main__":
    main()
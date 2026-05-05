"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
MODULE 2 (continued) — MODEL EVALUATION & COMPARISON
Loads saved metrics and generates visualisations:
  • Comparison bar chart (accuracy / F1)
  • Per-model confusion matrices
  • ROC curves
  • ANN training history plot
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless backend for Streamlit compatibility
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc as sklearn_auc,
    classification_report, accuracy_score, f1_score
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── Seaborn style ─────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = [
    "#4C9BE8", "#E87B4C", "#4CE8A0", "#E84C9B",
    "#A04CE8", "#E8D14C", "#4CE8D1",
]


# ═════════════════════════════════════════════════════════════════════════════
# LOAD METRICS
# ═════════════════════════════════════════════════════════════════════════════
def load_metrics() -> list[dict]:
    path = os.path.join(MODEL_DIR, "all_metrics.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No metrics found. Run  python train.py  first."
        )
    with open(path) as f:
        return json.load(f)


def metrics_to_dataframe(metrics: list[dict]) -> pd.DataFrame:
    rows = []
    for m in metrics:
        rows.append({
            "Model":     m["model"],
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1-Score":  m["f1_score"],
            "ROC-AUC":   m["roc_auc"] if m["roc_auc"] != "N/A" else None,
        })
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Comparison Bar Chart
# ═════════════════════════════════════════════════════════════════════════════
def plot_comparison(df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight="bold")

    for ax, metric in zip(axes, ["Accuracy", "F1-Score"]):
        sorted_df = df.sort_values(metric, ascending=False)
        bars = ax.barh(
            sorted_df["Model"], sorted_df[metric],
            color=PALETTE[:len(sorted_df)], edgecolor="white", height=0.6,
        )
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"{metric} by Model", fontsize=13, fontweight="bold")
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        ax.axvline(0.8, color="#ff4444", linestyle="--", alpha=0.5, label="0.80 baseline")
        ax.legend(fontsize=9)
        ax.invert_yaxis()

    plt.tight_layout()
    path = save_path or os.path.join(ASSETS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved  →  %s", path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Confusion Matrix for Best Model
# ═════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(
    cm: list,
    class_names: list,
    model_name:  str,
    save_path:   str = None,
) -> plt.Figure:
    cm_array = np.array(cm)
    # Show max 20 classes (truncate for readability)
    n = min(20, len(class_names))
    cm_array    = cm_array[:n, :n]
    short_names = [c[:14] for c in class_names[:n]]

    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n - 2)))
    sns.heatmap(
        cm_array, annot=True, fmt="d",
        xticklabels=short_names, yticklabels=short_names,
        cmap="Blues", linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    path = save_path or os.path.join(ASSETS_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved  →  %s", path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — ROC Curves (best ML model)
# ═════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(
    model,
    X_test:      np.ndarray,
    y_test:      np.ndarray,
    n_classes:   int,
    class_names: list,
    model_name:  str,
    save_path:   str = None,
) -> plt.Figure:
    try:
        y_prob = model.predict_proba(X_test)
    except Exception:
        log.warning("Model %s does not support predict_proba; skipping ROC.", model_name)
        return None

    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    n_show = min(10, n_classes)   # show max 10 classes for clarity

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", n_show)

    for i in range(n_show):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc     = sklearn_auc(fpr, tpr)
        label       = f"{class_names[i]} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=cmap(i), lw=1.8, label=label, alpha=0.85)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()

    path = save_path or os.path.join(ASSETS_DIR, "roc_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved  →  %s", path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4 — ANN Training History
# ═════════════════════════════════════════════════════════════════════════════
def plot_ann_history(save_path: str = None) -> plt.Figure:
    history_path = os.path.join(MODEL_DIR, "ann_history.json")
    if not os.path.exists(history_path):
        log.warning("ANN history not found; skipping plot.")
        return None

    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ANN Training History", fontsize=15, fontweight="bold")

    epochs = range(1, len(history["loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["loss"],     label="Train Loss", color="#4C9BE8", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss",   color="#E87B4C", lw=2, linestyle="--")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["accuracy"],     label="Train Acc", color="#4CE8A0", lw=2)
    axes[1].plot(epochs, history["val_accuracy"], label="Val Acc",   color="#E84C9B", lw=2, linestyle="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    path = save_path or os.path.join(ASSETS_DIR, "ann_history.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved  →  %s", path)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FULL EVALUATION PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
def run_evaluation(
    X_test:      np.ndarray = None,
    y_test:      np.ndarray = None,
    n_classes:   int        = None,
    class_names: list       = None,
) -> dict:
    """
    Load metrics and generate all evaluation plots.
    X_test / y_test are optional — only needed for ROC curve generation.
    """
    log.info("=" * 60)
    log.info("MODULE 2 — MODEL EVALUATION")
    log.info("=" * 60)

    metrics = load_metrics()
    df      = metrics_to_dataframe(metrics)

    log.info("\n%s", df.to_string(index=False))

    # ── Comparison chart ──────────────────────────────────────────────────
    plot_comparison(df)

    # ── Confusion matrix for best ML model ────────────────────────────────
    if class_names:
        best_ml = max(
            [m for m in metrics if m["model"] not in ("ANN (Keras)", "LSTM (Keras)")],
            key=lambda m: m["f1_score"],
        )
        plot_confusion_matrix(
            best_ml["confusion_matrix"], class_names, best_ml["model"]
        )

    # ── ROC curves ────────────────────────────────────────────────────────
    if X_test is not None and y_test is not None and class_names:
        best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
        if os.path.exists(best_model_path):
            best_clf = joblib.load(best_model_path)
            meta_path = os.path.join(MODEL_DIR, "best_model_meta.json")
            with open(meta_path) as f:
                meta = json.load(f)
            plot_roc_curves(
                best_clf, X_test, y_test,
                n_classes, class_names, meta["best_model_name"],
            )

    # ── ANN history ───────────────────────────────────────────────────────
    plot_ann_history()

    log.info("✅ Evaluation complete. Plots saved to assets/")
    return {"metrics": metrics, "comparison_df": df}


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from preprocess import run_preprocessing
    data = run_preprocessing()

    result = run_evaluation(
        X_test      = data["X_test"],
        y_test      = data["y_test"],
        n_classes   = data["n_classes"],
        class_names = data["class_names"],
    )
    print("\nFinal Comparison Table:")
    print(result["comparison_df"].to_string(index=False))

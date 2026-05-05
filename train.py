"""
train.py
─────────────────────────────────────────────────────────────────────────────
MODULE 2 — MODEL TRAINING & TESTING
MODULE 3 — MODEL FINE-TUNING & OPTIMIZATION

Trains ML & DL models, performs hyperparameter tuning, K-Fold CV,
and saves the best model with performance metrics.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
import warnings
import numpy as np
import joblib

from sklearn.ensemble         import RandomForestClassifier
from sklearn.svm              import SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.naive_bayes      import GaussianNB
from sklearn.tree             import DecisionTreeClassifier
from sklearn.model_selection  import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.metrics          import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing    import label_binarize

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

from preprocess import run_preprocessing

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
os.makedirs(MODEL_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# HELPER — evaluate a fitted estimator
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_model(
    name: str,
    model,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    n_classes: int,
    class_names: list,
) -> dict:
    """Return a metrics dict for a fitted model."""
    y_pred = model.predict(X_test)

    # Probabilities for AUC (if model supports it)
    try:
        y_prob = model.predict_proba(X_test)
        y_bin  = label_binarize(y_test, classes=list(range(n_classes)))
        auc    = round(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"), 4)
    except Exception:
        auc = "N/A"

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test, y_pred),                                4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred,        average="weighted", zero_division=0), 4),
        "roc_auc":   auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    log.info(
        "%-35s  Acc=%.4f  F1=%.4f  AUC=%s",
        name, metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
    )
    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# ML MODEL DEFINITIONS & PARAM GRIDS
# ═════════════════════════════════════════════════════════════════════════════
def get_ml_configs(n_classes: int) -> list[dict]:
    return [
        {
            "name":  "Random Forest",
            "model": RandomForestClassifier(random_state=42, n_jobs=-1),
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth":    [None, 10, 20],
                "min_samples_split": [2, 5],
            },
            "search": "grid",
        },
        {
            "name":  "SVM",
            "model": SVC(probability=True, random_state=42),
            "param_grid": {
                "C":      [0.1, 1, 10],
                "kernel": ["rbf", "linear"],
            },
            "search": "grid",
        },
        {
            "name":  "K-Nearest Neighbors",
            "model": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 11],
                "weights":     ["uniform", "distance"],
                "metric":      ["euclidean", "manhattan"],
            },
            "search": "random",
        },
        {
            "name":  "Naive Bayes",
            "model": GaussianNB(),
            "param_grid": {"var_smoothing": np.logspace(-12, -6, 7)},
            "search": "grid",
        },
        {
            "name":  "Decision Tree",
            "model": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth":        [None, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "criterion":        ["gini", "entropy"],
            },
            "search": "random",
        },
    ]


# ═════════════════════════════════════════════════════════════════════════════
# TRAIN ML MODELS (with hyperparameter tuning + K-Fold CV)
# ═════════════════════════════════════════════════════════════════════════════
def train_ml_models(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    n_classes:   int,
    class_names: list,
) -> tuple[list[dict], dict]:
    """
    Train all ML models with GridSearchCV / RandomizedSearchCV + K-Fold CV.
    Returns (list_of_metrics, dict_of_best_estimators).
    """
    log.info("=" * 60)
    log.info("MODULE 2 — ML MODEL TRAINING")
    log.info("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    configs = get_ml_configs(n_classes)

    all_metrics:  list[dict] = []
    best_models:  dict       = {}

    for cfg in configs:
        name       = cfg["name"]
        base_model = cfg["model"]
        param_grid = cfg["param_grid"]
        search_typ = cfg["search"]

        log.info("── Training: %s ──", name)

        # ── Hyperparameter search ─────────────────────────────────────────
        if search_typ == "grid":
            searcher = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring="f1_weighted", n_jobs=-1, verbose=0,
            )
        else:
            searcher = RandomizedSearchCV(
                base_model, param_grid,
                n_iter=15, cv=cv, scoring="f1_weighted",
                n_jobs=-1, random_state=42, verbose=0,
            )

        searcher.fit(X_train, y_train)
        best_estimator = searcher.best_estimator_
        log.info("   Best params: %s", searcher.best_params_)

        # ── K-Fold cross-validation score ─────────────────────────────────
        cv_scores = cross_val_score(
            best_estimator, X_train, y_train,
            cv=cv, scoring="accuracy", n_jobs=-1,
        )
        log.info("   K-Fold CV Acc: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

        # ── Hold-out test evaluation ──────────────────────────────────────
        metrics = evaluate_model(name, best_estimator, X_test, y_test, n_classes, class_names)
        metrics["cv_accuracy_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_accuracy_std"]  = round(cv_scores.std(),  4)
        metrics["best_params"]      = searcher.best_params_

        all_metrics.append(metrics)
        best_models[name] = best_estimator

        # Save individual model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(best_estimator, os.path.join(MODEL_DIR, f"{safe_name}.pkl"))
        log.info("   Saved  →  models/%s.pkl", safe_name)

    return all_metrics, best_models


# ═════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING — ANN
# ═════════════════════════════════════════════════════════════════════════════
def build_ann(input_dim: int, n_classes: int) -> keras.Model:
    """
    Build a regularised ANN with:
      • Batch Normalisation after each dense layer
      • Dropout for regularisation
    """
    model = keras.Sequential([
        layers.InputLayer(input_shape=(input_dim,)),
        # Block 1
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.4),
        # Block 2
        layers.Dense(128, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),
        # Block 3
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),
        # Output
        layers.Dense(n_classes, activation="softmax"),
    ], name="ANN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_ann(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    n_classes:   int,
    class_names: list,
) -> tuple[dict, keras.Model, dict]:
    """Train the ANN and return metrics, model, and training history."""
    log.info("── Training: ANN (Keras) ──")
    input_dim = X_train.shape[1]

    ann = build_ann(input_dim, n_classes)

    cb_list = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0),
    ]

    history = ann.fit(
        X_train, y_train,
        epochs=100, batch_size=64,
        validation_split=0.15,
        callbacks=cb_list,
        verbose=1,
    )

    # Evaluate
    y_pred_prob = ann.predict(X_test, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    auc   = round(roc_auc_score(y_bin, y_pred_prob, multi_class="ovr", average="macro"), 4)

    metrics = {
        "model":     "ANN (Keras)",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred,        average="weighted", zero_division=0), 4),
        "roc_auc":   auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "cv_accuracy_mean": "N/A (use val_acc)",
        "cv_accuracy_std":  "N/A",
    }
    log.info(
        "%-35s  Acc=%.4f  F1=%.4f  AUC=%.4f",
        "ANN (Keras)", metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
    )

    # Save Keras model
    ann.save(os.path.join(MODEL_DIR, "ann_model.keras"))
    log.info("   Saved  →  models/ann_model.keras")

    history_dict = {
        "loss":        history.history["loss"],
        "val_loss":    history.history["val_loss"],
        "accuracy":    history.history["accuracy"],
        "val_accuracy":history.history["val_accuracy"],
    }
    return metrics, ann, history_dict


# ═════════════════════════════════════════════════════════════════════════════
# DEEP LEARNING — LSTM
# (Sequences formed by treating each symptom group as a time-step)
# ═════════════════════════════════════════════════════════════════════════════
def build_lstm(seq_len: int, feat_per_step: int, n_classes: int) -> keras.Model:
    model = keras.Sequential([
        layers.InputLayer(input_shape=(seq_len, feat_per_step)),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation="softmax"),
    ], name="LSTM")

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_lstm(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    n_classes:   int,
    class_names: list,
) -> tuple[dict, keras.Model]:
    """
    Reshape tabular features into sequences for LSTM.
    Each row is split into fixed-length segments (time-steps).
    """
    log.info("── Training: LSTM (Keras) ──")
    n_features = X_train.shape[1]

    # Choose a step size that divides features cleanly
    step = 5
    # Pad if necessary
    remainder = n_features % step
    if remainder:
        pad = step - remainder
        X_train = np.pad(X_train, ((0, 0), (0, pad)))
        X_test  = np.pad(X_test,  ((0, 0), (0, pad)))
        n_features = X_train.shape[1]

    seq_len       = n_features // step     # number of time-steps
    feat_per_step = step                   # features per step

    X_train_seq = X_train.reshape(-1, seq_len, feat_per_step)
    X_test_seq  = X_test.reshape( -1, seq_len, feat_per_step)

    lstm = build_lstm(seq_len, feat_per_step, n_classes)

    cb_list = [
        callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(patience=8, factor=0.5, verbose=0),
    ]

    lstm.fit(
        X_train_seq, y_train,
        epochs=80, batch_size=64,
        validation_split=0.15,
        callbacks=cb_list,
        verbose=1,
    )

    y_pred_prob = lstm.predict(X_test_seq, verbose=0)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    auc   = round(roc_auc_score(y_bin, y_pred_prob, multi_class="ovr", average="macro"), 4)

    metrics = {
        "model":     "LSTM (Keras)",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred,    average="weighted", zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred,        average="weighted", zero_division=0), 4),
        "roc_auc":   auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "cv_accuracy_mean": "N/A",
        "cv_accuracy_std":  "N/A",
    }
    log.info(
        "%-35s  Acc=%.4f  F1=%.4f  AUC=%.4f",
        "LSTM (Keras)", metrics["accuracy"], metrics["f1_score"], metrics["roc_auc"],
    )

    lstm.save(os.path.join(MODEL_DIR, "lstm_model.keras"))
    log.info("   Saved  →  models/lstm_model.keras")

    return metrics, lstm


# ═════════════════════════════════════════════════════════════════════════════
# SAVE BEST MODEL
# ═════════════════════════════════════════════════════════════════════════════
def save_best_model(all_metrics: list[dict], ml_models: dict) -> str:
    """Pick the highest F1-score model and save it as best_model.pkl."""
    # Exclude DL models from direct pickle (they're saved separately)
    ml_metric_list = [m for m in all_metrics if m["model"] not in ("ANN (Keras)", "LSTM (Keras)")]

    best_metric = max(ml_metric_list, key=lambda m: m["f1_score"])
    best_name   = best_metric["model"]
    best_est    = ml_models[best_name]

    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_est, best_path)

    # Record which model was chosen
    meta = {
        "best_model_name": best_name,
        "accuracy":        best_metric["accuracy"],
        "f1_score":        best_metric["f1_score"],
        "roc_auc":         best_metric["roc_auc"],
    }
    with open(os.path.join(MODEL_DIR, "best_model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log.info("🏆 Best model: %s  (F1=%.4f)  →  saved to models/best_model.pkl", best_name, best_metric["f1_score"])
    return best_name


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def run_training() -> dict:
    """Full training pipeline. Returns dict with all results."""
    data = run_preprocessing()

    X_train      = data["X_train"]
    X_test       = data["X_test"]
    y_train      = data["y_train"]
    y_test       = data["y_test"]
    n_classes    = data["n_classes"]
    class_names  = data["class_names"]

    # ── ML Models ─────────────────────────────────────────────────────────
    ml_metrics, ml_models = train_ml_models(
        X_train, X_test, y_train, y_test, n_classes, class_names
    )

    # ── DL Models ─────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("MODULE 2 — DL MODEL TRAINING")
    log.info("=" * 60)

    ann_metrics, ann_model, ann_history = train_ann(
        X_train, X_test, y_train, y_test, n_classes, class_names
    )
    lstm_metrics, lstm_model = train_lstm(
        X_train, X_test, y_train, y_test, n_classes, class_names
    )

    all_metrics = ml_metrics + [ann_metrics, lstm_metrics]

    # ── Save all metrics ───────────────────────────────────────────────────
    metrics_path = os.path.join(MODEL_DIR, "all_metrics.json")
    # Ensure JSON serialisable
    serialisable = []
    for m in all_metrics:
        entry = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                entry[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                entry[k] = float(v)
            else:
                entry[k] = v
        serialisable.append(entry)

    with open(metrics_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info("All metrics saved  →  %s", metrics_path)

    # ── Save ANN training history ──────────────────────────────────────────
    with open(os.path.join(MODEL_DIR, "ann_history.json"), "w") as f:
        json.dump(ann_history, f, indent=2)

    # ── Pick best ML model ─────────────────────────────────────────────────
    best_name = save_best_model(all_metrics, ml_models)

    return {
        "all_metrics": all_metrics,
        "ml_models":   ml_models,
        "ann_model":   ann_model,
        "lstm_model":  lstm_model,
        "ann_history": ann_history,
        "best_name":   best_name,
        **data,
    }


if __name__ == "__main__":
    results = run_training()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Model Performance Summary")
    print("=" * 60)
    print(f"{'Model':<35} {'Accuracy':>10} {'F1-Score':>10} {'AUC':>10}")
    print("-" * 65)
    for m in results["all_metrics"]:
        auc = str(m["roc_auc"])
        print(f"{m['model']:<35} {m['accuracy']:>10.4f} {m['f1_score']:>10.4f} {auc:>10}")
    print("=" * 60)
    print(f"Best model: {results['best_name']}")

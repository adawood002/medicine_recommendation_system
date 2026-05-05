"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
MODULE 1 — DATA COLLECTION & PRE-PROCESSING
Loads, cleans, encodes, scales, and splits the dataset.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

SYMPTOMS_CSV = os.path.join(DATASET_DIR, "symptoms_diseases.csv")


# ═════════════════════════════════════════════════════════════════════════════
def load_data(path: str = SYMPTOMS_CSV) -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Run  python dataset/generate_dataset.py  first."
        )
    df = pd.read_csv(path)
    log.info("Dataset loaded  →  shape: %s", df.shape)
    log.info("Columns: %s", df.columns.tolist())
    return df


# ═════════════════════════════════════════════════════════════════════════════
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1 — Remove duplicates.
    Step 2 — Handle missing values (numeric → median, categorical → mode).
    Step 3 — Drop constant / near-zero-variance columns.
    """
    original_shape = df.shape

    # 1. Duplicates
    df = df.drop_duplicates()
    log.info("After drop_duplicates  →  %s  (removed %d rows)",
             df.shape, original_shape[0] - df.shape[0])

    # 2. Missing values
    numeric_cols   = df.select_dtypes(include=[np.number]).columns
    categoric_cols = df.select_dtypes(exclude=[np.number]).columns

    df[numeric_cols]   = df[numeric_cols].fillna(df[numeric_cols].median())
    df[categoric_cols] = df[categoric_cols].fillna(df[categoric_cols].mode().iloc[0])
    log.info("Missing values filled  →  remaining nulls: %d", df.isnull().sum().sum())

    # 3. Constant columns
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols)
        log.info("Dropped constant columns: %s", constant_cols)

    return df


# ═════════════════════════════════════════════════════════════════════════════
def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode categorical features:
      - 'gender'  → binary 0/1 via LabelEncoder
      - 'disease' → integer label via LabelEncoder  (saved to models/)
    Returns (encoded_df, label_encoder_for_disease)
    """
    le_gender  = LabelEncoder()
    le_disease = LabelEncoder()

    df = df.copy()
    if "gender" in df.columns:
        df["gender"] = le_gender.fit_transform(df["gender"])
        log.info("Encoded 'gender'  →  classes: %s", le_gender.classes_.tolist())

    df["disease_encoded"] = le_disease.fit_transform(df["disease"])
    log.info("Encoded 'disease'  →  %d unique classes", len(le_disease.classes_))

    # Persist encoder
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
    joblib.dump(le_disease, encoder_path)
    log.info("LabelEncoder saved  →  %s", encoder_path)

    return df, le_disease


# ═════════════════════════════════════════════════════════════════════════════
def scale_features(
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only; transform both splits.
    Saves scaler to models/ for inference time.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    log.info("Scaler saved  →  %s", scaler_path)

    return X_train_sc, X_test_sc, scaler


# ═════════════════════════════════════════════════════════════════════════════
def split_data(
    df: pd.DataFrame,
    target_col:  str   = "disease_encoded",
    test_size:   float = 0.20,
    random_state: int  = 42,
) -> tuple:
    """Split into 80/20 train/test stratified by target."""
    drop_cols = ["disease", "disease_encoded"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    log.info(
        "Train/Test split  →  X_train: %s | X_test: %s",
        X_train.shape, X_test.shape,
    )

    # Save feature column names for inference
    feature_path = os.path.join(MODEL_DIR, "feature_columns.pkl")
    joblib.dump(feature_cols, feature_path)
    log.info("Feature columns saved  →  %s", feature_path)

    return X_train, X_test, y_train, y_test, feature_cols


# ═════════════════════════════════════════════════════════════════════════════
def run_preprocessing() -> dict:
    """
    End-to-end preprocessing pipeline.
    Returns a dict with all splits and metadata for downstream modules.
    """
    log.info("=" * 60)
    log.info("MODULE 1 — DATA PRE-PROCESSING")
    log.info("=" * 60)

    df                    = load_data()
    df_clean              = clean_data(df)
    df_encoded, le        = encode_features(df_clean)
    X_train, X_test, y_train, y_test, feature_cols = split_data(df_encoded)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    log.info("✅ Pre-processing complete.")
    log.info("   Feature count : %d", len(feature_cols))
    log.info("   Classes       : %s", le.classes_.tolist())

    return {
        "X_train":      X_train_sc,
        "X_test":       X_test_sc,
        "X_train_raw":  X_train,
        "X_test_raw":   X_test,
        "y_train":      y_train.values,
        "y_test":       y_test.values,
        "le":           le,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "n_classes":    len(le.classes_),
        "class_names":  le.classes_.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = run_preprocessing()
    print("\nPre-processing Summary")
    print(f"  Train samples : {data['X_train'].shape[0]}")
    print(f"  Test  samples : {data['X_test'].shape[0]}")
    print(f"  Features      : {len(data['feature_cols'])}")
    print(f"  Diseases      : {data['n_classes']}")

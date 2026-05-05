"""
recommendation.py
─────────────────────────────────────────────────────────────────────────────
MODULE 4 — MEDICINE RECOMMENDATION ENGINE
Accepts patient profile (symptoms, age, gender, history),
predicts the disease using the trained model, and returns
a ranked list of medicine recommendations.
─────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import joblib

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
MEDICINE_CSV = os.path.join(DATASET_DIR, "medicine_data.csv")


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class PatientProfile:
    """Represents a patient's input profile."""
    symptoms:        list[str]
    age:             int          = 30
    gender:          str          = "Male"
    medical_history: list[str]    = field(default_factory=list)
    existing_conditions: list[str] = field(default_factory=list)


@dataclass
class MedicineRecommendation:
    """A single medicine recommendation with metadata."""
    rank:               int
    name:               str
    description:        str
    dosage:             str
    warnings:           str
    safety_profile:     str
    effectiveness_score: float
    relevance_score:    float      # computed for this patient


@dataclass
class RecommendationResult:
    """Full result returned by the engine."""
    predicted_disease:    str
    confidence_score:     float
    top_predictions:      list[tuple]          # [(disease, probability), ...]
    recommendations:      list[MedicineRecommendation]
    safety_notes:         list[str]


# ═════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
class MedicineRecommendationEngine:
    """
    Loads the best trained model + artefacts and provides disease
    prediction + medicine recommendations for a given patient profile.
    """

    def __init__(self):
        self._load_artefacts()
        self._load_medicine_data()

    # ── Load saved artefacts ──────────────────────────────────────────────
    def _load_artefacts(self):
        """Load model, label encoder, scaler, and feature columns."""
        required = {
            "model":         os.path.join(MODEL_DIR, "best_model.pkl"),
            "label_encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
            "scaler":        os.path.join(MODEL_DIR, "scaler.pkl"),
            "feature_cols":  os.path.join(MODEL_DIR, "feature_columns.pkl"),
        }
        missing = [name for name, path in required.items() if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing artefacts: {missing}.\n"
                "Run  python train.py  before using the recommendation engine."
            )

        self.model         = joblib.load(required["model"])
        self.label_encoder = joblib.load(required["label_encoder"])
        self.scaler        = joblib.load(required["scaler"])
        self.feature_cols  = joblib.load(required["feature_cols"])

        # Best model name
        meta_path = os.path.join(MODEL_DIR, "best_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            self.model_name = meta.get("best_model_name", "Unknown")
        else:
            self.model_name = "Unknown"

        log.info("Engine loaded  →  model: %s", self.model_name)
        log.info("Feature count  →  %d", len(self.feature_cols))

    # ── Load medicine data ────────────────────────────────────────────────
    def _load_medicine_data(self):
        if not os.path.exists(MEDICINE_CSV):
            raise FileNotFoundError(
                f"Medicine data not found at '{MEDICINE_CSV}'.\n"
                "Run  python dataset/generate_dataset.py  first."
            )
        self.medicine_df = pd.read_csv(MEDICINE_CSV)
        log.info("Medicine DB loaded  →  %d entries", len(self.medicine_df))

    # ── Build feature vector ──────────────────────────────────────────────
    def _build_feature_vector(self, profile: PatientProfile) -> np.ndarray:
        """
        Convert a PatientProfile into a scaled numeric feature vector
        aligned with the training feature columns.
        """
        row = {}

        for col in self.feature_cols:
            if col == "age":
                row[col] = profile.age
            elif col == "gender":
                row[col] = 0 if profile.gender.lower() == "male" else 1
            else:
                # Binary symptom feature
                row[col] = 1 if col in profile.symptoms else 0

        X_raw = pd.DataFrame([row])[self.feature_cols].values
        X_scaled = self.scaler.transform(X_raw)
        return X_scaled

    # ── Compute patient-specific relevance score ──────────────────────────
    def _compute_relevance_score(
        self,
        base_effectiveness: float,
        profile: PatientProfile,
        med_row: pd.Series,
    ) -> float:
        """
        Adjust effectiveness by patient-specific safety factors:
          - Age modifiers for certain medicines
          - Existing conditions
        """
        score = float(base_effectiveness)

        # Age adjustments
        warnings_lower = str(med_row.get("warnings", "")).lower()
        if profile.age < 12 and "children" in warnings_lower:
            score *= 0.6
        if profile.age > 65 and any(kw in warnings_lower for kw in ["renal", "hepatic", "cardiac"]):
            score *= 0.75

        # Condition-based modifiers
        conditions_lower = [c.lower() for c in profile.existing_conditions]
        if "kidney" in conditions_lower or "renal" in conditions_lower:
            if "renal" in warnings_lower or "kidney" in warnings_lower:
                score *= 0.5
        if "liver" in conditions_lower:
            if "liver" in warnings_lower or "hepat" in warnings_lower:
                score *= 0.5
        if "pregnancy" in conditions_lower:
            if "pregnancy" in warnings_lower or "pregnant" in warnings_lower:
                score *= 0.3

        # Clamp to [0, 1]
        return round(min(max(score, 0.0), 1.0), 4)

    # ── Build safety notes ────────────────────────────────────────────────
    def _build_safety_notes(self, profile: PatientProfile, recs: list) -> list[str]:
        notes = []
        if profile.age < 12:
            notes.append("⚠️ Paediatric patient: all dosages must be reviewed by a physician.")
        if profile.age > 65:
            notes.append("⚠️ Elderly patient: watch for drug interactions and renal clearance.")
        if "pregnancy" in [c.lower() for c in profile.existing_conditions]:
            notes.append("⚠️ Pregnancy: several medications may be contraindicated. Consult OB-GYN.")
        if profile.medical_history:
            notes.append(f"ℹ️  Reported history: {', '.join(profile.medical_history)}. Inform prescribing doctor.")
        if not notes:
            notes.append("✅ No critical safety flags detected for this patient profile.")
        return notes

    # ── Main predict & recommend ──────────────────────────────────────────
    def recommend(
        self,
        profile: PatientProfile,
        top_n_medicines: int = 5,
    ) -> RecommendationResult:
        """
        Predicts disease and returns ranked medicine recommendations.

        Parameters
        ----------
        profile          : PatientProfile
        top_n_medicines  : number of medicines to return (default 5)

        Returns
        -------
        RecommendationResult
        """
        if not profile.symptoms:
            raise ValueError("At least one symptom must be provided.")

        # ── 1. Feature vector ─────────────────────────────────────────────
        X = self._build_feature_vector(profile)

        # ── 2. Predict ────────────────────────────────────────────────────
        class_idx   = int(self.model.predict(X)[0])
        disease_name = self.label_encoder.inverse_transform([class_idx])[0]

        # Confidence scores
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[class_idx])
            # Top-3 predictions
            top_idx = np.argsort(proba)[::-1][:3]
            top_predictions = [
                (self.label_encoder.inverse_transform([i])[0], round(float(proba[i]), 4))
                for i in top_idx
            ]
        else:
            confidence      = 1.0
            top_predictions = [(disease_name, 1.0)]

        log.info("Predicted disease: %s  (confidence: %.2f%%)", disease_name, confidence * 100)

        # ── 3. Fetch matching medicines ────────────────────────────────────
        med_rows = self.medicine_df[
            self.medicine_df["disease"] == disease_name
        ].sort_values("rank")

        if med_rows.empty:
            # Fallback: general supportive medicines
            log.warning("No medicines found for '%s'. Using generic fallback.", disease_name)
            recs = [
                MedicineRecommendation(
                    rank=1,
                    name="Paracetamol (Acetaminophen)",
                    description="General fever and pain reliever. Supportive therapy.",
                    dosage="500–1000mg every 4–6 hours",
                    warnings="Do not exceed 4g/day.",
                    safety_profile="High",
                    effectiveness_score=0.70,
                    relevance_score=0.70,
                )
            ]
        else:
            recs = []
            for _, row in med_rows.head(top_n_medicines).iterrows():
                rel_score = self._compute_relevance_score(
                    row["effectiveness_score"], profile, row
                )
                recs.append(MedicineRecommendation(
                    rank               = int(row["rank"]),
                    name               = str(row["medicine_name"]),
                    description        = str(row["description"]),
                    dosage             = str(row["dosage"]),
                    warnings           = str(row["warnings"]),
                    safety_profile     = str(row["safety_profile"]),
                    effectiveness_score= float(row["effectiveness_score"]),
                    relevance_score    = rel_score,
                ))

            # Re-rank by patient-specific relevance score
            recs.sort(key=lambda r: r.relevance_score, reverse=True)
            for i, r in enumerate(recs, 1):
                r.rank = i

        safety_notes = self._build_safety_notes(profile, recs)

        return RecommendationResult(
            predicted_disease = disease_name,
            confidence_score  = round(confidence, 4),
            top_predictions   = top_predictions,
            recommendations   = recs,
            safety_notes      = safety_notes,
        )

    # ── Available symptoms ────────────────────────────────────────────────
    def get_all_symptoms(self) -> list[str]:
        """Return sorted list of symptom column names from training features."""
        exclude = {"age", "gender"}
        return sorted([c for c in self.feature_cols if c not in exclude])

    # ── Available diseases ────────────────────────────────────────────────
    def get_all_diseases(self) -> list[str]:
        return sorted(self.label_encoder.classes_.tolist())


# ═════════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    engine = MedicineRecommendationEngine()

    profile = PatientProfile(
        symptoms=["fever", "cough", "fatigue", "body_ache", "headache"],
        age=35,
        gender="Male",
        medical_history=["seasonal allergies"],
        existing_conditions=[],
    )

    result = engine.recommend(profile, top_n_medicines=5)

    print("\n" + "═" * 60)
    print(f"  Predicted Disease  : {result.predicted_disease}")
    print(f"  Confidence Score   : {result.confidence_score * 100:.2f}%")
    print(f"  Top Predictions    : {result.top_predictions}")
    print("═" * 60)
    print("\n  MEDICINE RECOMMENDATIONS")
    print("-" * 60)
    for med in result.recommendations:
        print(f"\n  [{med.rank}] {med.name}")
        print(f"      Description       : {med.description}")
        print(f"      Dosage            : {med.dosage}")
        print(f"      Warnings          : {med.warnings}")
        print(f"      Safety Profile    : {med.safety_profile}")
        print(f"      Effectiveness     : {med.effectiveness_score:.2f}")
        print(f"      Relevance (patient): {med.relevance_score:.2f}")
    print("\n  SAFETY NOTES")
    for note in result.safety_notes:
        print(f"  {note}")
    print("═" * 60)

"""
algorithms/naive_bayes.py

Gaussian Naïve Bayes risk classifier for symptom intake data.

On first run (no saved model), generates 50 synthetic training samples using
clinical rules and trains a GaussianNB. The fitted model is persisted to
models/nb_risk_model.pkl for subsequent loads.

Risk labels:
  0 → "low"
  1 → "medium"
  2 → "high"

Dependencies: scikit-learn, numpy
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.naive_bayes import GaussianNB

# ── Paths ──────────────────────────────────────────────────────────────────────
_MODEL_PATH = Path(__file__).parent.parent / "models" / "nb_risk_model.pkl"

# ── Feature order (must match intake dict keys) ───────────────────────────────
_FEATURE_KEYS = [
    "age",           # int  0–100
    "skin_type",     # int  1–6 (Fitzpatrick)
    "duration_weeks",# int  0–200+
    "location_int",  # int  0–9 (body region code)
    "itching",       # 0/1
    "bleeding",      # 0/1
    "growing",       # 0/1
    "color_change",  # 0/1
    "painful",       # 0/1
]

_RISK_LABELS = {0: "low", 1: "medium", 2: "high"}


# ── Synthetic training data ───────────────────────────────────────────────────

def _label_sample(row: np.ndarray) -> int:
    """Apply clinical rules to assign a risk class to a synthetic sample."""
    # row order: age, skin_type, duration_weeks, location_int,
    #            itching, bleeding, growing, color_change, painful
    bleeding     = int(row[5])
    growing      = int(row[6])
    color_change = int(row[7])
    duration_wks = float(row[2])
    itching      = int(row[4])

    if bleeding == 1 or (growing == 1 and color_change == 1):
        return 2  # high
    if duration_wks > 12 and itching == 1:
        return 1  # medium
    return 0      # low


def _generate_synthetic_data(n: int = 50, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate n synthetic patient records with rule-based labels."""
    rng = np.random.default_rng(seed)

    X = np.column_stack([
        rng.integers(18, 85, size=n),          # age
        rng.integers(1,   7,  size=n),          # skin_type
        rng.integers(0,  104, size=n),          # duration_weeks
        rng.integers(0,  10,  size=n),          # location_int
        rng.integers(0,   2,  size=n),          # itching
        rng.integers(0,   2,  size=n),          # bleeding
        rng.integers(0,   2,  size=n),          # growing
        rng.integers(0,   2,  size=n),          # color_change
        rng.integers(0,   2,  size=n),          # painful
    ]).astype(float)

    y = np.array([_label_sample(row) for row in X])
    return X, y


# ── Model loading / training ──────────────────────────────────────────────────

def _load_or_train_model() -> GaussianNB:
    """Return a fitted GaussianNB — loaded from disk or freshly trained."""
    if _MODEL_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            model: GaussianNB = pickle.load(f)
        return model

    # Train on synthetic data
    X, y = _generate_synthetic_data()
    model = GaussianNB()
    model.fit(X, y)

    # Persist
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def _intake_to_features(intake: dict) -> np.ndarray:
    """Convert intake dict to ordered feature vector."""
    return np.array(
        [float(intake.get(k, 0)) for k in _FEATURE_KEYS],
        dtype=float,
    ).reshape(1, -1)


# ── Public API ────────────────────────────────────────────────────────────────

def score_risk(intake: dict) -> dict:
    """
    Predict lesion risk level from patient intake data using Gaussian NB.

    Args:
        intake: dict with keys matching _FEATURE_KEYS. Unknown keys are ignored;
                missing keys default to 0.

    Returns:
        {
            "nb_risk_label"       : str   — "low" | "medium" | "high"
            "nb_risk_probability" : float — confidence for the predicted class
            "nb_class"            : int   — 0 | 1 | 2
        }
    """
    model = _load_or_train_model()
    X = _intake_to_features(intake)

    proba = model.predict_proba(X)[0]     # shape: (n_classes,)
    predicted_class = int(np.argmax(proba))
    predicted_proba = float(proba[predicted_class])

    return {
        "nb_risk_label": _RISK_LABELS[predicted_class],
        "nb_risk_probability": round(predicted_proba, 4),
        "nb_class": predicted_class,
    }

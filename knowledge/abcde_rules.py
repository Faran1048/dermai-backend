"""
knowledge/abcde_rules.py

ABCDE rule engine for skin lesion risk scoring.

The ABCDE criteria are the standard clinical framework for evaluating
potentially malignant skin lesions:
  A — Asymmetry
  B — Border irregularity
  C — Colour variation
  D — Diameter / Dermoscopic features
  E — Evolution / Elevation

Each criterion is scored as 0 (absent/low risk) or 1-2 (present/high risk),
giving a total_score in the range 0–10.

Risk tiers:
  0–4   → Low    (likely benign, monitor)
  5–7   → Medium (clinical review recommended)
  8–10  → High   (urgent dermatology referral)

Args to score_abcde():
  lesion_class     (str)   — one of: mel, nv, bcc, akiec, bkl, vasc, df
  colour_flag      (str)   — key from COLOUR_FLAGS (e.g. "black", "red")
  texture_cluster  (int)   — 0 = smooth/uniform, 1 = mildly irregular, 2 = very irregular
  confidence       (float) — model prediction confidence in [0.0, 1.0]
"""

from __future__ import annotations

# ── Per-class ABCDE baseline profiles ─────────────────────────────────────────
# Each entry defines the typical ABCDE presentation for that lesion class.
# Scores are 0, 1, or 2 for each criterion.

_CLASS_PROFILES: dict[str, dict[str, int]] = {
    #          A   B   C   D   E
    "mel":   {"A": 2, "B": 2, "C": 2, "D": 2, "E": 2},  # worst case
    "nv":    {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},  # benign baseline
    "bcc":   {"A": 1, "B": 2, "C": 1, "D": 1, "E": 1},
    "akiec": {"A": 1, "B": 1, "C": 1, "D": 1, "E": 1},
    "bkl":   {"A": 0, "B": 0, "C": 1, "D": 0, "E": 0},
    "vasc":  {"A": 0, "B": 1, "C": 1, "D": 0, "E": 1},
    "df":    {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
}

# ── Colour flag → score contribution ─────────────────────────────────────────
_COLOUR_SCORE: dict[str, int] = {
    "black": 2,
    "dark_brown": 1,
    "red": 1,
    "pink": 0,
    "light_brown": 0,
    "white": 1,
    "blue_gray": 2,
}

# ── Human-readable labels for each criterion ──────────────────────────────────
_ASYMMETRY_LABELS = {0: "Symmetrical", 1: "Mildly asymmetric", 2: "Highly asymmetric"}
_BORDER_LABELS    = {0: "Well-defined", 1: "Slightly irregular", 2: "Poorly defined / ragged"}
_COLOUR_LABELS    = {0: "Uniform colour", 1: "Two colours present", 2: "Multiple colours / regression"}
_DIAMETER_LABELS  = {0: "≤6 mm", 1: "6–10 mm", 2: ">10 mm or dermoscopic atypia"}
_EVOLUTION_LABELS = {0: "Stable", 1: "Slow change", 2: "Rapid growth or bleeding"}

_RISK_TIERS = {
    range(0, 5):  "Low",
    range(5, 8):  "Medium",
    range(8, 11): "High",
}


def _get_risk_tier(score: int) -> str:
    for score_range, label in _RISK_TIERS.items():
        if score in score_range:
            return label
    return "High"


def score_abcde(
    lesion_class: str,
    colour_flag: str,
    texture_cluster: int,
    confidence: float,
) -> dict[str, str | int]:
    """
    Compute an ABCDE risk score for a skin lesion.

    Args:
        lesion_class    : ISIC class label (mel, nv, bcc, akiec, bkl, vasc, df)
        colour_flag     : Dominant colour key from COLOUR_FLAGS
        texture_cluster : Surface irregularity cluster (0=smooth, 1=mild, 2=rough)
        confidence      : Model softmax confidence for the predicted class [0, 1]

    Returns:
        dict with keys A, B, C, D, E (str labels), total_score (int), risk_tier (str)
    """
    # ── Fallback for unknown class ────────────────────────────────────────────
    profile = _CLASS_PROFILES.get(lesion_class, _CLASS_PROFILES["nv"])

    # ── A — Asymmetry ─────────────────────────────────────────────────────────
    # Driven by class profile; penalise high-risk classes with low confidence
    # (uncertainty in a dangerous class → treat conservatively)
    a_score = profile["A"]
    if lesion_class in ("mel", "bcc") and confidence < 0.6:
        a_score = min(a_score + 1, 2)

    # ── B — Border ───────────────────────────────────────────────────────────
    # texture_cluster directly reflects border/surface irregularity
    if texture_cluster == 2:
        b_score = 2
    elif texture_cluster == 1:
        b_score = max(profile["B"], 1)
    else:
        b_score = profile["B"]

    # ── C — Colour ───────────────────────────────────────────────────────────
    colour_contribution = _COLOUR_SCORE.get(colour_flag, 0)
    c_score = min(max(profile["C"], colour_contribution), 2)

    # ── D — Diameter / Dermoscopic features ──────────────────────────────────
    # High-confidence mel/bcc predictions → assume dermoscopic atypia present
    if lesion_class == "mel":
        d_score = 2 if confidence >= 0.7 else 1
    elif lesion_class in ("bcc", "akiec"):
        d_score = 1 if confidence >= 0.5 else profile["D"]
    else:
        d_score = profile["D"]

    # ── E — Evolution / Elevation ────────────────────────────────────────────
    if lesion_class == "mel":
        e_score = 2
    elif lesion_class in ("bcc", "akiec", "vasc"):
        e_score = 1 if confidence >= 0.5 else 0
    else:
        e_score = profile["E"]

    # ── Total ─────────────────────────────────────────────────────────────────
    total = a_score + b_score + c_score + d_score + e_score
    total = max(0, min(total, 10))  # clamp to [0, 10]

    return {
        "A": _ASYMMETRY_LABELS[a_score],
        "B": _BORDER_LABELS[b_score],
        "C": _COLOUR_LABELS[c_score],
        "D": _DIAMETER_LABELS[d_score],
        "E": _EVOLUTION_LABELS[e_score],
        "total_score": total,
        "risk_tier": _get_risk_tier(total),
    }

"""
app/services/symptom_service.py

Rule-based dermatology symptom checker.

Uses a hand-curated knowledge base of skin conditions and their
associated symptoms, severity, and recommendations.

No external APIs — runs entirely locally and instantly.
Scoring is weighted: each matched symptom contributes its weight,
and conditions are ranked by total score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Knowledge base ────────────────────────────────────────────────────────────
# Each condition entry has:
#   symptoms: dict mapping symptom keyword → match weight (higher = more distinctive)
#   severity: default severity level
#   recommendations: list of clinical action strings
#   description: brief clinical description
#   icd10_code: ICD-10 classification code

CONDITION_KB: list[dict] = [
    {
        "condition": "Eczema (Atopic Dermatitis)",
        "icd10_code": "L20.9",
        "description": "Chronic inflammatory condition causing dry, itchy, and inflamed skin.",
        "severity": "moderate",
        "symptoms": {
            "itching": 3,
            "redness": 2,
            "dry skin": 3,
            "scaling": 2,
            "rash": 2,
            "inflammation": 2,
            "flaking": 2,
            "thickened skin": 1,
            "swelling": 1,
            "cracking": 2,
        },
        "recommendations": [
            "Apply fragrance-free moisturizer twice daily.",
            "Avoid known triggers (soaps, detergents, stress).",
            "Consider over-the-counter hydrocortisone cream for mild flares.",
            "Consult a dermatologist if symptoms persist beyond 2 weeks.",
        ],
    },
    {
        "condition": "Psoriasis",
        "icd10_code": "L40.9",
        "description": "Autoimmune condition causing rapid skin cell turnover, resulting in thick, silvery scales.",
        "severity": "moderate",
        "symptoms": {
            "scaling": 4,
            "silver scales": 4,
            "red patches": 3,
            "itching": 2,
            "dry skin": 2,
            "cracking": 2,
            "bleeding": 1,
            "joint pain": 2,
            "nail changes": 2,
            "thickened skin": 3,
        },
        "recommendations": [
            "Use medicated shampoos containing coal tar or salicylic acid for scalp psoriasis.",
            "Keep skin moisturized to reduce scaling.",
            "Avoid triggers such as stress, infections, and certain medications.",
            "Consult a dermatologist for prescription topical or systemic treatments.",
        ],
    },
    {
        "condition": "Acne Vulgaris",
        "icd10_code": "L70.0",
        "description": "Common skin condition caused by clogged hair follicles, leading to pimples, blackheads, and cysts.",
        "severity": "low",
        "symptoms": {
            "pimples": 4,
            "blackheads": 4,
            "whiteheads": 4,
            "oily skin": 3,
            "redness": 2,
            "inflammation": 2,
            "cysts": 3,
            "scarring": 2,
            "nodules": 3,
        },
        "recommendations": [
            "Wash the affected area gently twice daily with a mild cleanser.",
            "Avoid picking or squeezing pimples to prevent scarring.",
            "Use non-comedogenic (oil-free) skincare and makeup products.",
            "Consider benzoyl peroxide or salicylic acid over-the-counter treatments.",
            "Consult a dermatologist for persistent or severe acne.",
        ],
    },
    {
        "condition": "Ringworm (Tinea Corporis)",
        "icd10_code": "B35.4",
        "description": "Fungal skin infection causing ring-shaped, scaly patches.",
        "severity": "moderate",
        "symptoms": {
            "ring-shaped rash": 5,
            "circular rash": 5,
            "itching": 3,
            "scaling": 3,
            "redness": 2,
            "rash": 2,
            "spreading": 2,
            "blisters": 1,
        },
        "recommendations": [
            "Apply over-the-counter antifungal cream (clotrimazole, miconazole) twice daily for 2–4 weeks.",
            "Keep the affected area clean and dry.",
            "Avoid sharing towels, clothing, or bedding.",
            "Consult a doctor if the infection does not improve within 2 weeks.",
        ],
    },
    {
        "condition": "Urticaria (Hives)",
        "icd10_code": "L50.9",
        "description": "Skin reaction causing raised, itchy welts that may appear suddenly and resolve within hours.",
        "severity": "moderate",
        "symptoms": {
            "welts": 5,
            "hives": 5,
            "itching": 4,
            "swelling": 3,
            "redness": 2,
            "burning": 2,
            "rash": 2,
            "allergic reaction": 3,
        },
        "recommendations": [
            "Take over-the-counter antihistamines (cetirizine, loratadine) to relieve itching.",
            "Identify and avoid triggers (foods, medications, stress, insect bites).",
            "Apply cool compresses to the affected area.",
            "Seek emergency care if swelling affects the throat or breathing is difficult.",
        ],
    },
    {
        "condition": "Contact Dermatitis",
        "icd10_code": "L23.9",
        "description": "Skin inflammation caused by contact with an allergen or irritant.",
        "severity": "moderate",
        "symptoms": {
            "rash": 3,
            "redness": 3,
            "itching": 3,
            "blisters": 2,
            "burning": 3,
            "swelling": 2,
            "dry skin": 2,
            "cracking": 2,
            "oozing": 2,
        },
        "recommendations": [
            "Identify and avoid the triggering substance.",
            "Rinse the affected area with mild soap and water.",
            "Apply over-the-counter hydrocortisone cream to reduce inflammation.",
            "Consult a dermatologist for patch testing if the trigger is unknown.",
        ],
    },
    {
        "condition": "Rosacea",
        "icd10_code": "L71.9",
        "description": "Chronic skin condition primarily affecting the face, causing redness, visible blood vessels, and sometimes acne-like breakouts.",
        "severity": "moderate",
        "symptoms": {
            "facial redness": 5,
            "flushing": 4,
            "visible blood vessels": 4,
            "pimples": 2,
            "burning": 2,
            "eye irritation": 2,
            "thickened skin": 2,
        },
        "recommendations": [
            "Use gentle, fragrance-free skincare products.",
            "Apply broad-spectrum sunscreen daily (SPF 30+).",
            "Avoid triggers: spicy food, alcohol, heat, and stress.",
            "Consult a dermatologist for prescription topical or oral treatments.",
        ],
    },
    {
        "condition": "Cellulitis",
        "icd10_code": "L03.90",
        "description": "Bacterial skin infection causing swelling, redness, warmth, and pain. Requires prompt medical treatment.",
        "severity": "high",
        "symptoms": {
            "swelling": 4,
            "redness": 3,
            "warmth": 4,
            "pain": 4,
            "fever": 3,
            "tenderness": 3,
            "skin tightness": 3,
            "streaking": 3,
        },
        "recommendations": [
            "Seek medical attention promptly — cellulitis requires antibiotic treatment.",
            "Elevate the affected limb to reduce swelling.",
            "Do NOT try to drain or puncture the skin.",
            "Go to the emergency room if you develop a high fever or red streaks spreading from the area.",
        ],
    },
    {
        "condition": "Shingles (Herpes Zoster)",
        "icd10_code": "B02.9",
        "description": "Reactivation of the chickenpox virus (VZV) causing painful blisters on one side of the body.",
        "severity": "high",
        "symptoms": {
            "blisters": 4,
            "pain": 4,
            "burning": 3,
            "rash": 3,
            "one-sided": 4,
            "itching": 2,
            "tingling": 3,
            "fever": 2,
            "sensitivity to touch": 3,
        },
        "recommendations": [
            "See a doctor within 72 hours — antiviral medication (acyclovir) is most effective early.",
            "Keep blisters clean and covered to prevent secondary infection.",
            "Do not scratch or break open blisters.",
            "Take pain relievers as directed by your physician.",
            "Consider shingles vaccination to prevent future episodes.",
        ],
    },
    {
        "condition": "Melanoma",
        "icd10_code": "C43.9",
        "description": "The most dangerous form of skin cancer. Early detection is critical for a positive outcome.",
        "severity": "urgent",
        "symptoms": {
            "changing mole": 5,
            "irregular border": 5,
            "multiple colors": 5,
            "asymmetrical mole": 5,
            "growing lesion": 4,
            "bleeding mole": 4,
            "itching mole": 3,
            "dark lesion": 3,
        },
        "recommendations": [
            "SEEK DERMATOLOGIST EVALUATION IMMEDIATELY.",
            "Do not delay — early-stage melanoma is highly treatable.",
            "Avoid sun exposure and use broad-spectrum sunscreen (SPF 50+).",
            "Document any changes in the lesion with photos to share with your doctor.",
        ],
    },
    {
        "condition": "Impetigo",
        "icd10_code": "L01.0",
        "description": "Highly contagious bacterial skin infection common in children, causing red sores and honey-colored crusts.",
        "severity": "moderate",
        "symptoms": {
            "sores": 4,
            "crusts": 4,
            "honey-colored crust": 5,
            "redness": 2,
            "blisters": 3,
            "itching": 2,
            "oozing": 3,
        },
        "recommendations": [
            "Consult a doctor for antibiotic treatment (topical or oral).",
            "Keep the affected area clean and covered.",
            "Avoid touching or scratching the sores.",
            "Wash hands frequently and do not share towels or utensils.",
        ],
    },
]

# Severity ordering for ranking tiebreaking
_SEVERITY_ORDER = {"urgent": 4, "high": 3, "moderate": 2, "low": 1}

# All known symptom keywords (lowercase)
ALL_SYMPTOMS = sorted(
    {symptom for entry in CONDITION_KB for symptom in entry["symptoms"]}
)


@dataclass
class SymptomAnalysisResult:
    condition: str
    icd10_code: str | None
    description: str
    score: float
    matched_symptoms: list[str]
    severity: str
    recommendations: list[str]
    urgency_flag: bool = field(init=False)

    def __post_init__(self) -> None:
        self.urgency_flag = self.severity in ("urgent", "high")


def analyze_symptoms(
    symptoms: list[str],
    body_area: str | None = None,
    duration_days: int | None = None,
    top_k: int = 5,
) -> list[SymptomAnalysisResult]:
    """
    Match a list of patient-reported symptoms against the knowledge base.

    Args:
        symptoms: List of symptom strings (e.g., ["itching", "redness", "scaling"])
        body_area: Optional body area (not currently used in scoring, reserved for future)
        duration_days: Optional symptom duration — increases severity if > 30 days
        top_k: Number of top conditions to return

    Returns:
        Ranked list of SymptomAnalysisResult (by score descending).
    """
    normalised_symptoms = [s.lower().strip() for s in symptoms]

    scored: list[SymptomAnalysisResult] = []

    for entry in CONDITION_KB:
        total_score = 0.0
        matched: list[str] = []

        for symptom_key, weight in entry["symptoms"].items():
            # Match if any symptom string contains the keyword or vice versa
            for patient_symptom in normalised_symptoms:
                if symptom_key in patient_symptom or patient_symptom in symptom_key:
                    total_score += weight
                    if symptom_key not in matched:
                        matched.append(symptom_key)
                    break

        if total_score == 0:
            continue

        # Duration modifier: chronic symptoms (>30 days) increase effective score
        if duration_days and duration_days > 30:
            total_score *= 1.15

        scored.append(
            SymptomAnalysisResult(
                condition=entry["condition"],
                icd10_code=entry["icd10_code"],
                description=entry["description"],
                score=round(total_score, 2),
                matched_symptoms=matched,
                severity=entry["severity"],
                recommendations=entry["recommendations"],
            )
        )

    # Sort by score descending, then by severity as a tiebreaker
    scored.sort(
        key=lambda r: (r.score, _SEVERITY_ORDER.get(r.severity, 0)),
        reverse=True,
    )

    results = scored[:top_k]

    logger.info(
        "Symptom analysis: %d symptoms submitted, %d conditions matched, top=%s",
        len(symptoms),
        len(scored),
        results[0].condition if results else "none",
    )

    return results


def get_all_known_symptoms() -> list[str]:
    """Return the full list of recognised symptom keywords for autocomplete/validation."""
    return ALL_SYMPTOMS

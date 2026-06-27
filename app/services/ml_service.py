"""
app/services/ml_service.py

ML inference service for skin condition analysis from images.

Architecture:
- Runs the ViT model (google/vit-base-patch16-224) on an uploaded image.
- Returns the top-k predicted ImageNet labels.
- Maps those labels through a curated dermatology dictionary to produce
  clinically meaningful skin condition names and descriptions.

Note: The base ViT model is trained on ImageNet, not a dermatology dataset.
For production use, fine-tune on the ISIC 2020 dataset. The mapping layer
provides a best-effort translation of visually similar ImageNet categories
to relevant skin condition terminology.
"""

import io
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError

from app.ml.model_loader import get_model_loader

logger = logging.getLogger(__name__)


# ── Dermatology condition mapping ─────────────────────────────────────────────
# Maps ImageNet label keywords → dermatology condition metadata.
# Keys are lowercase substrings that may appear in ImageNet label names.
# Entries with higher specificity should come first.

DERMATOLOGY_CONDITION_MAP: dict[str, dict] = {
    # Lesion / texture patterns
    "ringworm": {
        "condition": "Ringworm (Tinea Corporis)",
        "description": "Fungal infection causing circular, scaly rashes on the skin.",
        "icd10_code": "B35.4",
        "severity_hint": "moderate",
    },
    "eczema": {
        "condition": "Eczema (Atopic Dermatitis)",
        "description": "Chronic inflammatory skin condition causing itching, redness, and scaling.",
        "icd10_code": "L20.9",
        "severity_hint": "moderate",
    },
    "psoriasis": {
        "condition": "Psoriasis",
        "description": "Autoimmune condition causing rapid skin cell build-up, forming scales and red patches.",
        "icd10_code": "L40.9",
        "severity_hint": "moderate",
    },
    "melanoma": {
        "condition": "Melanoma",
        "description": "Serious form of skin cancer that develops from melanocytes. Requires urgent medical attention.",
        "icd10_code": "C43.9",
        "severity_hint": "urgent",
    },
    "acne": {
        "condition": "Acne Vulgaris",
        "description": "Common skin condition causing pimples, blackheads, and cysts on the face and body.",
        "icd10_code": "L70.0",
        "severity_hint": "low",
    },
    "rosacea": {
        "condition": "Rosacea",
        "description": "Chronic skin condition causing redness, visible blood vessels, and sometimes pimples.",
        "icd10_code": "L71.9",
        "severity_hint": "moderate",
    },
    "wart": {
        "condition": "Warts (Verruca Vulgaris)",
        "description": "Benign skin growths caused by the human papillomavirus (HPV).",
        "icd10_code": "B07.9",
        "severity_hint": "low",
    },
    "mole": {
        "condition": "Melanocytic Nevus (Mole)",
        "description": "Common benign skin growth. Changes in size, shape, or color warrant medical evaluation.",
        "icd10_code": "D22.9",
        "severity_hint": "low",
    },
    "hive": {
        "condition": "Urticaria (Hives)",
        "description": "Skin reaction causing raised, itchy welts. Often triggered by allergens.",
        "icd10_code": "L50.9",
        "severity_hint": "moderate",
    },
    "blister": {
        "condition": "Bullous Skin Condition",
        "description": "Fluid-filled blisters on the skin surface. Causes include burns, infections, and autoimmune disease.",
        "icd10_code": "L13.9",
        "severity_hint": "moderate",
    },
    "impetigo": {
        "condition": "Impetigo",
        "description": "Highly contagious bacterial skin infection causing red sores and honey-colored crusts.",
        "icd10_code": "L01.0",
        "severity_hint": "moderate",
    },
    "lupus": {
        "condition": "Cutaneous Lupus",
        "description": "Autoimmune condition causing butterfly-shaped rash on the face and skin lesions.",
        "icd10_code": "L93.0",
        "severity_hint": "high",
    },
    # Fallback: ImageNet categories that correlate visually with skin conditions
    "bandage": {
        "condition": "Non-specific Skin Lesion",
        "description": "A skin abnormality detected in the image. Recommend clinical examination for accurate diagnosis.",
        "icd10_code": None,
        "severity_hint": "low",
    },
    "skin": {
        "condition": "Possible Skin Condition",
        "description": "Abnormal skin texture or discoloration detected. A dermatologist evaluation is advised.",
        "icd10_code": None,
        "severity_hint": "low",
    },
}

# Default condition for labels that don't match any known pattern
_DEFAULT_CONDITION = {
    "condition": "Unclassified Skin Finding",
    "description": (
        "The AI model detected visual patterns in the image. "
        "Please consult a dermatologist for a proper diagnosis."
    ),
    "icd10_code": None,
    "severity_hint": "low",
}


@dataclass
class ConditionPrediction:
    condition: str
    confidence: float
    description: str
    icd10_code: str | None
    raw_label: str


def _map_label_to_condition(imagenet_label: str) -> dict:
    """Map an ImageNet label string to a dermatology condition dict."""
    label_lower = imagenet_label.lower()
    for keyword, condition_data in DERMATOLOGY_CONDITION_MAP.items():
        if keyword in label_lower:
            return condition_data
    return {**_DEFAULT_CONDITION, "condition": f"Skin Finding ({imagenet_label})"}


def _validate_and_open_image(image_bytes: bytes) -> Image.Image:
    """Open and validate the image bytes. Returns a PIL RGB image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Raises if the image is corrupted
    except (UnidentifiedImageError, Exception) as exc:
        raise ValueError(f"Invalid or corrupted image file: {exc}") from exc

    # Re-open after verify() (which exhausts the file pointer)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img


async def analyze_image(
    image_bytes: bytes,
    top_k: int = 5,
) -> list[ConditionPrediction]:
    """
    Run ViT inference on the given image bytes and return the top-k
    predicted skin conditions.

    Args:
        image_bytes: Raw bytes of the uploaded image (JPEG, PNG, WebP, etc.)
        top_k: Number of top predictions to return (default: 5)

    Returns:
        List of ConditionPrediction dataclasses ordered by confidence (desc)

    Raises:
        ValueError: If the image is invalid or corrupted.
        RuntimeError: If the model has not been loaded.
    """
    image = _validate_and_open_image(image_bytes)
    loader = get_model_loader()

    # Prepare inputs
    inputs = loader.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(loader.device) for k, v in inputs.items()}

    # Run inference
    with torch.no_grad():
        outputs = loader.model(**inputs)
        logits = outputs.logits  # shape: (1, num_labels)

    # Get top-k predictions
    probs = F.softmax(logits, dim=-1)[0]
    top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))

    predictions: list[ConditionPrediction] = []
    id2label = loader.id2label

    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        raw_label = id2label.get(idx, f"label_{idx}")
        condition_data = _map_label_to_condition(raw_label)
        predictions.append(
            ConditionPrediction(
                condition=condition_data["condition"],
                confidence=round(prob, 4),
                description=condition_data["description"],
                icd10_code=condition_data.get("icd10_code"),
                raw_label=raw_label,
            )
        )
        logger.debug(
            "Prediction: %s → %s (confidence=%.4f)",
            raw_label,
            condition_data["condition"],
            prob,
        )

    return predictions

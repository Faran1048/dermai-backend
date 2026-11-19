
"""
algorithms/cnn_classifier.py

EfficientNet-B4 skin lesion classifier.

The classifier uses the local checkpoint:
    models/dermai_best.pt

Class labels are read from the model metadata when available and checked
against the configured output classes.

IMPORTANT:
- No random or mock predictions are generated.
- If the trained model cannot load, classification is unavailable.
- This is an experimental educational tool, not a medical diagnostic device.
"""

import json
import logging
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "dermai_best.pt"
META_PATH = MODEL_DIR / "model_meta.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Class names
# ---------------------------------------------------------------------

CLASS_NAMES = [
    "melanoma",
    "basal_cell_carcinoma",
    "actinic_keratosis",
    "melanocytic_nevi",
    "benign_keratosis",
    "vascular_lesion",
    "petechiae_purpura",
    "leukemia_cutis",
    "ecchymosis",
    "normal_skin",
]

NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------------
# Risk categories
# ---------------------------------------------------------------------

HIGH_RISK_CLASSES = {
    "melanoma",
    "basal_cell_carcinoma",
    "leukemia_cutis",
}

SPECIAL_MESSAGES = {
    "melanoma": (
        "The model flagged a pattern associated with melanoma. "
        "This result is not a diagnosis. Arrange prompt evaluation "
        "by a qualified dermatologist."
    ),
    "basal_cell_carcinoma": (
        "The model flagged a pattern associated with basal cell carcinoma. "
        "Please consult a qualified dermatologist for evaluation."
    ),
    "leukemia_cutis": (
        "The model flagged a pattern associated with leukemia cutis. "
        "Please seek prompt medical evaluation."
    ),
    "actinic_keratosis": (
        "The model flagged a pattern associated with actinic keratosis. "
        "Please consult a qualified dermatologist."
    ),
    "petechiae_purpura": (
        "The model flagged a pattern associated with petechiae or purpura. "
        "Consult a healthcare professional for proper evaluation."
    ),
}

# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

TRANSFORM = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------

# These are application heuristics, not clinically validated thresholds.
HIGH_RISK_THRESHOLD = 0.40
NORMAL_THRESHOLD = 0.55

# ---------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------

_net = None
MODEL_LOADED = False


def _load_model():
    """
    Load the local EfficientNet checkpoint.

    Supports:
    - TorchScript checkpoints
    - Raw PyTorch state dictionaries
    - State dictionaries wrapped in a 'state_dict' key

    Raises RuntimeError if the model cannot be loaded.
    """

    global _net, MODEL_LOADED

    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model checkpoint not found: {MODEL_PATH}"
        )

    logger.info("Loading DermAI model from %s", MODEL_PATH)

    # First attempt: TorchScript
    try:
        model = torch.jit.load(
            str(MODEL_PATH),
            map_location=DEVICE,
        )

        model.eval()
        _net = model
        MODEL_LOADED = True

        logger.info("EfficientNet model loaded as TorchScript.")
        return

    except Exception as jit_error:
        logger.info(
            "TorchScript loading failed; trying state dictionary: %s",
            jit_error,
        )

    # Second attempt: regular PyTorch checkpoint
    try:
        checkpoint = torch.load(
            str(MODEL_PATH),
            map_location="cpu",
            weights_only=True,
        )

        # Support checkpoints wrapped in a state_dict field.
        if (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
            and isinstance(checkpoint["state_dict"], dict)
        ):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if not isinstance(state_dict, dict):
            raise RuntimeError(
                "Checkpoint does not contain a valid state dictionary."
            )

        # Detect classifier output layer.
        if "classifier.bias" not in state_dict:
            raise RuntimeError(
                "Checkpoint is missing classifier.bias. "
                "The model architecture may not match."
            )

        checkpoint_outputs = state_dict["classifier.bias"].shape[0]

        if checkpoint_outputs != NUM_CLASSES:
            raise RuntimeError(
                f"Checkpoint has {checkpoint_outputs} outputs, "
                f"but the application expects {NUM_CLASSES} classes."
            )

        # Read architecture from metadata.
        architecture = "efficientnet_b4"

        if META_PATH.is_file():
            with open(META_PATH, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)

            architecture = metadata.get(
                "model_arch",
                architecture,
            )

            metadata_classes = metadata.get("class_names")

            if metadata_classes and metadata_classes != CLASS_NAMES:
                raise RuntimeError(
                    "Class names in model_meta.json do not match "
                    "the classifier's configured class names."
                )

            metadata_outputs = metadata.get("num_classes")

            if (
                metadata_outputs is not None
                and metadata_outputs != NUM_CLASSES
            ):
                raise RuntimeError(
                    "num_classes in model_meta.json does not match "
                    "the configured class count."
                )

        model = timm.create_model(
            architecture,
            pretrained=False,
            num_classes=checkpoint_outputs,
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(DEVICE)

        _net = model
        MODEL_LOADED = True

        logger.info(
            "EfficientNet model loaded successfully. Architecture=%s, classes=%d",
            architecture,
            checkpoint_outputs,
        )

    except Exception as exc:
        _net = None
        MODEL_LOADED = False

        logger.exception("Could not load the DermAI model.")

        raise RuntimeError(
            "The trained skin classification model could not be loaded. "
            "Image analysis is unavailable."
        ) from exc


# Load model when this module is imported.
try:
    _load_model()

except RuntimeError:
    logger.exception(
        "DermAI classifier is unavailable. "
        "No predictions will be generated."
    )


# ---------------------------------------------------------------------
# Public classification API
# ---------------------------------------------------------------------

def classify_image(
    image_path: str,
    use_tta: bool | None = None,
) -> dict:
    """
    Classify a skin image using the loaded EfficientNet model.

    Args:
        image_path:
            Path to the input image.

        use_tta:
            Whether to use six test-time augmentation passes.
            Defaults to True on CUDA and False on CPU.

    Returns:
        Dictionary containing:
            predicted_class
            confidence
            risk_level
            all_scores
            special_message
            disclaimer

    Raises:
        RuntimeError:
            If the trained model is unavailable or inference fails.

        ValueError:
            If the input image cannot be opened.
    """

    if not MODEL_LOADED or _net is None:
        raise RuntimeError(
            "The trained skin classification model is unavailable. "
            "Please check the backend model checkpoint and startup logs."
        )

    if use_tta is None:
        use_tta = DEVICE == "cuda"

    # -------------------------------------------------------------
    # Open and preprocess image
    # -------------------------------------------------------------

    try:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError(
            "The uploaded image could not be opened or processed. "
            "Please upload a valid image."
        ) from exc

    # -------------------------------------------------------------
    # Model inference
    # -------------------------------------------------------------

    try:
        with torch.inference_mode():

            if use_tta:
                augments = [
                    lambda x: x,
                    lambda x: torch.flip(x, [2]),
                    lambda x: torch.flip(x, [3]),
                    lambda x: torch.rot90(x, 1, [2, 3]),
                    lambda x: torch.rot90(x, 2, [2, 3]),
                    lambda x: torch.rot90(x, 3, [2, 3]),
                ]

                all_logits = []

                for augment in augments:
                    output = _net(augment(tensor))

                    if output.ndim != 2 or output.shape[1] != NUM_CLASSES:
                        raise RuntimeError(
                            "Model output dimensions do not match "
                            "the configured class count."
                        )

                    all_logits.append(output)

                avg_logits = torch.stack(all_logits).mean(dim=0)

            else:
                output = _net(tensor)

                if output.ndim != 2 or output.shape[1] != NUM_CLASSES:
                    raise RuntimeError(
                        "Model output dimensions do not match "
                        "the configured class count."
                    )

                avg_logits = output

            probabilities = F.softmax(
                avg_logits,
                dim=1,
            ).squeeze(0)

            scores = {
                CLASS_NAMES[index]: round(
                    float(probabilities[index].item()),
                    4,
                )
                for index in range(NUM_CLASSES)
            }

        # ---------------------------------------------------------
        # Select prediction
        # ---------------------------------------------------------

        best_high_risk_class = None
        best_high_risk_score = 0.0

        for class_name in HIGH_RISK_CLASSES:
            score = scores[class_name]

            if score > best_high_risk_score:
                best_high_risk_score = score
                best_high_risk_class = class_name

        if best_high_risk_score >= HIGH_RISK_THRESHOLD:
            predicted_class = best_high_risk_class
            confidence = best_high_risk_score
            risk_level = "High"

        else:
            predicted_class = max(
                scores,
                key=scores.get,
            )

            confidence = scores[predicted_class]

            if predicted_class in HIGH_RISK_CLASSES:
                risk_level = "High"

            elif confidence < NORMAL_THRESHOLD:
                risk_level = "Medium"

            else:
                risk_level = "Low"

        # ---------------------------------------------------------
        # Return result
        # ---------------------------------------------------------

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "risk_level": risk_level,
            "all_scores": scores,
            "special_message": SPECIAL_MESSAGES.get(predicted_class),
            "disclaimer": (
                "Experimental educational output only. "
                "Not a medical diagnosis or a substitute for "
                "evaluation by a qualified healthcare professional."
            ),
        }

    except Exception as exc:
        logger.exception("DermAI image classification failed.")

        raise RuntimeError(
            "Image classification failed. Please try again later."
        ) from exc

    finally:
        if "tensor" in locals():
            del tensor

        if "output" in locals():
            del output

        if "avg_logits" in locals():
            del avg_logits

        if "probabilities" in locals():
            del probabilities

        if DEVICE == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# Grad-CAM compatibility
# ---------------------------------------------------------------------

def _load_local_model():
    """
    Return the loaded EfficientNet model instance.

    Used by Grad-CAM to generate attention heatmaps.
    Raises RuntimeError if the model is unavailable.
    """

    if not MODEL_LOADED or _net is None:
        raise RuntimeError(
            "Model is not loaded. Cannot generate Grad-CAM heatmap."
        )

    return _net

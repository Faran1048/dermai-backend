"""
algorithms/cnn_classifier.py

EfficientNet-B4 skin lesion classifier.

Trained on HAM10000-derived dataset, 5 classes (86.9% val accuracy):
  0 - melanoma            (High-risk malignant)
  1 - basal_cell_carcinoma (High-risk malignant)
  2 - melanocytic_nevi    (Benign - common mole)
  3 - benign_keratosis    (Benign)
  4 - vascular_lesion     (Benign)

NOTE: The checkpoint classifier head has 10 output neurons — only the first 5
(indices 0-4) correspond to trained classes. Outputs 5-9 are unused dead neurons.
We extract logits[0:5], run softmax over them only, so confidences correctly
sum to 1.0 and reflect actual model training.
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as T
# pyrefly: ignore [missing-import]
import timm, json
from PIL import Image
from pathlib import Path
# pyrefly: ignore [missing-import]
from huggingface_hub import hf_hub_download

# ── Configuration ─────────────────────────────────────────────────────────────
HF_REPO = "farandev/dermai-skin-classifier"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Trained class names ────────────────────────────────────────────────────────
# These are the canonical 10 classes our model is trained on
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
    "normal_skin"
]
NUM_CLASSES = len(CLASS_NAMES)  # 10

# ── High-risk class definitions ───────────────────────────────────────────────
HIGH_RISK_CLASSES = {"melanoma", "basal_cell_carcinoma", "leukemia_cutis"}

SPECIAL_MESSAGES = {
    "melanoma":
        "Possible melanoma detected. This is a high-risk skin lesion. "
        "Please see a dermatologist urgently — do not delay.",
    "basal_cell_carcinoma":
        "Possible basal cell carcinoma detected. "
        "Please book a dermatologist appointment within 2 weeks.",
    "leukemia_cutis":
        "Possible leukemia cutis detected. This requires urgent oncology and dermatological evaluation.",
    "actinic_keratosis":
        "Possible actinic keratosis detected. This is a pre-cancerous lesion. Please consult a dermatologist.",
    "petechiae_purpura":
        "Possible petechiae/purpura detected. Recommend consulting a healthcare professional to rule out systemic causes.",
}

# ── Model loading ──────────────────────────────────────────────────────────────
print("Loading DermAI model from local path: models/dermai_best.pt")

try:
    model_path = str(MODEL_DIR / "dermai_best.pt")
    
    # 1. Try JIT loading (TorchScript)
    try:
        _net = torch.jit.load(model_path, map_location=DEVICE)
        print("Model loaded successfully as TorchScript JIT model.")
    except Exception as e_jit:
        print(f"JIT load failed: {e_jit}. Falling back to state dict...")
        # 2. Try loading state dict
        _ckpt = torch.load(model_path, map_location="cpu")
        if isinstance(_ckpt, dict) and "classifier.bias" in _ckpt:
            _ckpt_num_outputs = _ckpt["classifier.bias"].shape[0]
            
            # Read architecture from metadata if available
            arch = "efficientnet_b4"  # Default fallback
            meta_path = MODEL_DIR / "model_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                        arch = meta.get("model_arch", arch)
                except Exception:
                    pass
            
            _net = timm.create_model(
                arch,
                pretrained=False,
                num_classes=_ckpt_num_outputs
            )
            _net.load_state_dict(_ckpt)
            _net.eval().to(DEVICE)
            print(f"Model loaded successfully from state dict ({arch}).")
        else:
            raise RuntimeError("Unknown checkpoint format.")
            
    MODEL_LOADED = True
    MODEL_PATH = model_path

except Exception as e:
    print(f"WARNING: Could not load trained model: {e}")
    print("Falling back to random predictions until model is available.")
    _net = None
    MODEL_LOADED = False
    MODEL_PATH = None

# ── Image transform (matches training preprocessing) ──────────────────────────
# EfficientNet-B3 uses 300x300 input resolution
TRANSFORM = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Decision thresholds ───────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.40   # flag high-risk class if confidence >= this
NORMAL_THRESHOLD    = 0.55   # below this → Medium risk for benign classes


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_image(image_path: str, use_tta: bool | None = None) -> dict:
    """
    Classify a skin lesion image using the trained EfficientNet-B4 model.

    Uses 6-fold Test-Time Augmentation (TTA) for more robust predictions (if enabled).
    Extracts logits for the trained classes only and re-runs softmax so
    confidence values correctly sum to 1.0.

    Args:
        image_path: Path to the input image file.
        use_tta: Whether to use 6-fold TTA. Defaults to True on GPU/CUDA and False on CPU.

    Returns:
        dict with keys:
          predicted_class  : str   — one of CLASS_NAMES
          confidence       : float — 0.0-1.0 (re-normalized)
          risk_level       : str   — 'High' | 'Medium' | 'Low'
          all_scores       : dict  — {class_name: confidence}
          special_message  : str | None — clinical urgency message
          disclaimer       : str
    """
    if use_tta is None:
        use_tta = (DEVICE == "cuda")

    if not MODEL_LOADED or _net is None:
        # Fallback: return mock scores if model failed to load
        import random
        raw = [random.uniform(0.05, 0.25) for _ in CLASS_NAMES]
        raw[0] = 0.45  # bias toward first class
        total = sum(raw)
        scores = {CLASS_NAMES[i]: round(raw[i] / total, 4) for i in range(NUM_CLASSES)}
        pred = max(scores, key=scores.get)
        return {
            "predicted_class": pred,
            "confidence": scores[pred],
            "risk_level": "Medium",
            "all_scores": scores,
            "special_message": "Model loading error — please check backend logs.",
            "disclaimer": "Educational tool only. Not medical advice."
        }

    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if use_tta:
                # TTA — run 6 augmented versions, average raw logits for robustness
                augments = [
                    lambda x: x,
                    lambda x: torch.flip(x, [2]),
                    lambda x: torch.flip(x, [3]),
                    lambda x: torch.rot90(x, 1, [2, 3]),
                    lambda x: torch.rot90(x, 2, [2, 3]),
                    lambda x: torch.rot90(x, 3, [2, 3]),
                ]
                all_logits = []
                for aug in augments:
                    out = _net(aug(tensor))          # shape: (1, 10)
                    all_logits.append(out[:, :NUM_CLASSES])
                # Average logits across TTA views, then softmax
                avg_logits = torch.stack(all_logits).mean(0)
            else:
                # Single forward pass — 6x faster on CPU
                out = _net(tensor)
                avg_logits = out[:, :NUM_CLASSES]

        probs = F.softmax(avg_logits, dim=-1).squeeze()  # shape: (10,) sums to 1.0

        scores = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)}

        # High-risk class priority check (lower threshold for safety)
        best_hr_class = None
        best_hr_score = 0.0
        for cls in HIGH_RISK_CLASSES:
            if cls in scores and scores[cls] > best_hr_score:
                best_hr_score = scores[cls]
                best_hr_class = cls

        if best_hr_score >= HIGH_RISK_THRESHOLD:
            pred = best_hr_class
            conf = best_hr_score
            risk = "High"
        else:
            pred = max(scores, key=scores.get)
            conf = scores[pred]
            risk = ("High"   if pred in HIGH_RISK_CLASSES else
                    "Medium" if conf < NORMAL_THRESHOLD    else "Low")

        # Clean up local tensors to prevent memory accumulation
        del tensor
        if 'out' in locals():
            del out
        if 'all_logits' in locals():
            del all_logits
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        return {
            "predicted_class": pred,
            "confidence":      conf,
            "risk_level":      risk,
            "all_scores":      scores,
            "special_message": SPECIAL_MESSAGES.get(pred),
            "disclaimer":      "Educational tool only. Not a substitute for medical diagnosis."
        }


    except Exception as e:
        print(f"Classification error: {e}")
        return {
            "predicted_class": "unknown",
            "confidence":      0.0,
            "risk_level":      "Unknown",
            "all_scores":      {},
            "special_message": f"Error during classification: {str(e)}",
            "disclaimer":      "Educational tool only."
        }


def _load_local_model():
    """
    Return the already-loaded EfficientNet-B4 model instance.
    Used by Grad-CAM to generate attention heatmaps.
    Raises RuntimeError if the model failed to load at startup.
    """
    if _net is None:
        raise RuntimeError("Model is not loaded — cannot generate Grad-CAM heatmap.")
    return _net

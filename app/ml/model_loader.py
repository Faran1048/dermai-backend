"""
app/ml/model_loader.py

Singleton loader for the HuggingFace ViT image classification model.
The model is loaded once at startup and held in memory for the lifetime
of the server process — avoiding costly repeated loads per request.

Model: google/vit-base-patch16-224
  - Pretrained on ImageNet-21k, fine-tuned on ImageNet-1k
  - Input: 224×224 RGB images
  - Output: 1000-class logits (ImageNet labels)

The model labels are post-processed by ml_service.py to map
relevant ImageNet categories to dermatology conditions.

*** STARTUP BEHAVIOUR ***
The model is loaded in a background daemon thread so that the HTTP server
starts accepting requests immediately instead of blocking for minutes while
the weights download. Callers check `ModelLoader().is_ready` before
attempting inference — the /analysis route returns HTTP 503 until ready.
"""

import logging
import threading

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from app.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


class ModelLoader:
    """Thread-safe singleton that loads and caches the ViT model."""

    _instance: "ModelLoader | None" = None
    _lock = threading.Lock()

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._processor: ViTImageProcessor | None = None
        self._model: ViTForImageClassification | None = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True
        self._ready = threading.Event()   # set once load attempt finishes
        self._load_error: str | None = None
        self._thread_started = False

    def load(self) -> None:
        """
        Kick off model download/load in a background thread so the HTTP server
        starts immediately. Safe to call multiple times — only one thread runs.
        """
        if self._thread_started:
            return
        with self._lock:
            if self._thread_started:
                return
            self._thread_started = True

        thread = threading.Thread(target=self._load_in_background, daemon=True, name="vit-model-loader")
        thread.start()
        logger.info("ML model download started in background — server ready immediately.")

    def _load_in_background(self) -> None:
        """Blocking model load — runs inside a daemon thread."""
        cache_dir = str(settings.model_cache_path)
        model_name = settings.model_name

        logger.info("[BG] Loading model '%s' on device '%s' ...", model_name, self._device)
        logger.info(
            "[BG] Model cache dir: %s (first run downloads ~330 MB)",
            cache_dir,
        )

        try:
            self._processor = ViTImageProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            self._model = ViTForImageClassification.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            self._model.to(self._device)
            self._model.eval()
            logger.info("[BG] ViT model loaded successfully — predictions are now enabled.")
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("[BG] Failed to load ML model: %s", exc)
        finally:
            self._ready.set()   # always signal, even on failure

    # ── Status ────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True once the model has finished loading successfully."""
        return self._ready.is_set() and self._model is not None

    @property
    def load_error(self) -> str | None:
        """Non-None if the model failed to load."""
        return self._load_error

    # ── Model accessors (raise if not yet loaded) ─────────────────────────────

    @property
    def processor(self) -> ViTImageProcessor:
        if self._processor is None:
            raise RuntimeError("ViT model not loaded yet. Check is_ready before calling.")
        return self._processor

    @property
    def model(self) -> ViTForImageClassification:
        if self._model is None:
            raise RuntimeError("ViT model not loaded yet. Check is_ready before calling.")
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def id2label(self) -> dict[int, str]:
        """ImageNet label mapping from the model config."""
        return self.model.config.id2label


# ── Module-level convenience accessor ─────────────────────────────────────────

def get_model_loader() -> ModelLoader:
    """Return the singleton ModelLoader instance."""
    return ModelLoader()

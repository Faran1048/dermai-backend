"""
algorithms/kmeans_texture.py

Analyses skin lesion texture using Local Binary Patterns (LBP) extracted
from 16×16 image patches, then clusters patches into 3 texture groups.

Dependencies: OpenCV (opencv-python-headless), scikit-image, scikit-learn, numpy
"""

from __future__ import annotations

import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans

# ── Constants ─────────────────────────────────────────────────────────────────
_LBP_RADIUS = 3
_LBP_N_POINTS = 24       # 8 * radius
_PATCH_SIZE = 16
_N_CLUSTERS = 3
_LBP_BINS = _LBP_N_POINTS + 2   # number of histogram bins for uniform LBP

# ── Cluster → clinical label mapping ─────────────────────────────────────────
_CLUSTER_TEXTURE = {0: "smooth", 1: "irregular", 2: "rough"}

_ABCDE_B_MAP = {
    "smooth":    "Well-defined",
    "irregular": "Slightly irregular",
    "rough":     "Poorly defined / ragged",
}

_ABCDE_A_MAP = {
    "smooth":    "Symmetrical",
    "irregular": "Mildly asymmetric",
    "rough":     "Highly asymmetric",
}


# ── Helper: extract LBP histogram from a single patch ────────────────────────

def _patch_lbp_histogram(patch: np.ndarray) -> np.ndarray:
    """Return a normalised LBP histogram for one greyscale patch."""
    lbp = local_binary_pattern(
        patch,
        P=_LBP_N_POINTS,
        R=_LBP_RADIUS,
        method="uniform",
    )
    hist, _ = np.histogram(lbp.ravel(), bins=_LBP_BINS, range=(0, _LBP_BINS))
    hist = hist.astype(float)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


# ── Main function ─────────────────────────────────────────────────────────────

def analyse_texture(image_path: str) -> dict:
    """
    Compute texture features from a skin lesion image and classify into
    smooth / irregular / rough clusters.

    Args:
        image_path : Path to the image file.

    Returns:
        {
            "texture_cluster" : str — one of "smooth", "irregular", "rough"
            "abcde_b"         : str — ABCDE Border criterion label
            "abcde_a"         : str — ABCDE Asymmetry proxy label
        }
    """
    # 1. Load and resize to 224×224 in grayscale
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (224, 224))

    # 2. Extract LBP histograms from non-overlapping 16×16 patches
    h, w = img_gray.shape
    histograms: list[np.ndarray] = []

    for row in range(0, h - _PATCH_SIZE + 1, _PATCH_SIZE):
        for col in range(0, w - _PATCH_SIZE + 1, _PATCH_SIZE):
            patch = img_gray[row : row + _PATCH_SIZE, col : col + _PATCH_SIZE]
            histograms.append(_patch_lbp_histogram(patch))

    features = np.array(histograms)   # shape: (n_patches, _LBP_BINS)

    # 3. K-Means clustering of patch histograms into 3 groups
    n_samples = features.shape[0]
    n_clusters = min(_N_CLUSTERS, n_samples)   # guard against tiny images

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(features)

    # 4. Dominant cluster = cluster with most patches
    counts = np.bincount(km.labels_, minlength=n_clusters)
    dominant_cluster_idx = int(np.argmax(counts))

    # 5. Rank cluster centres by mean LBP entropy (higher entropy → rougher)
    entropies = []
    for center in km.cluster_centers_:
        # Shannon entropy of the LBP histogram
        p = center[center > 0]
        entropy = float(-np.sum(p * np.log2(p)))
        entropies.append(entropy)

    # Sort cluster indices by entropy to get a consistent smooth→rough ordering
    sorted_by_entropy = sorted(range(n_clusters), key=lambda i: entropies[i])
    entropy_rank = {orig_idx: rank for rank, orig_idx in enumerate(sorted_by_entropy)}

    # Re-map dominant cluster to 0/1/2 (smooth/irregular/rough)
    texture_int = entropy_rank.get(dominant_cluster_idx, 0)
    # Clamp to [0, 2] in case n_clusters < 3
    texture_int = min(texture_int, 2)
    texture_label = _CLUSTER_TEXTURE[texture_int]

    return {
        "texture_cluster": texture_label,
        "abcde_b": _ABCDE_B_MAP[texture_label],
        "abcde_a": _ABCDE_A_MAP[texture_label],
    }

"""
algorithms/kmeans_colour.py

Extracts dominant colour palette from a skin lesion image using K-Means
clustering, then maps each dominant colour to a clinical ABCDE flag.

Dependencies: Pillow, scikit-learn, numpy
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Lazy import to avoid circular dependency at module level
def _get_colour_flags() -> dict[str, str]:
    from knowledge.colour_flags import COLOUR_FLAGS
    return COLOUR_FLAGS


# ── Colour similarity ──────────────────────────────────────────────────────────

def _hex_to_rgb(hex_colour: str) -> np.ndarray:
    """Convert #RRGGBB hex string to numpy RGB array."""
    h = hex_colour.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


# Reference RGB values for each COLOUR_FLAGS key
_REFERENCE_COLOURS: dict[str, np.ndarray] = {
    "black":       np.array([20,  20,  20],  dtype=float),
    "dark_brown":  np.array([80,  40,  20],  dtype=float),
    "red":         np.array([200, 30,  30],  dtype=float),
    "pink":        np.array([230, 150, 160], dtype=float),
    "light_brown": np.array([160, 110, 70],  dtype=float),
    "white":       np.array([240, 240, 235], dtype=float),
    "blue_gray":   np.array([100, 120, 140], dtype=float),
}


def _nearest_colour_flag(rgb: np.ndarray) -> str:
    """Return the COLOUR_FLAGS key whose reference RGB is closest (Euclidean)."""
    best_key = "light_brown"
    best_dist = float("inf")
    for key, ref in _REFERENCE_COLOURS.items():
        dist = float(np.linalg.norm(rgb - ref))
        if dist < best_dist:
            best_dist = dist
            best_key = key
    return best_key


def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = (int(max(0, min(255, v))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


# ── Main function ─────────────────────────────────────────────────────────────

def analyse_colours(image_path: str, n_clusters: int = 5) -> dict:
    """
    Extract dominant colours from a skin lesion image and map them to
    clinical ABCDE colour flags.

    Args:
        image_path : Absolute or relative path to the image file.
        n_clusters : Number of colour clusters (default 5).

    Returns:
        {
            "palette"  : list of 5 hex colour strings (e.g. ["#A05030", ...]),
            "flags"    : list of clinical flag strings from COLOUR_FLAGS,
            "abcde_c"  : ABCDE 'C' (colour) criterion summary string,
        }
    """
    colour_flags = _get_colour_flags()

    # 1. Load and resize
    img = Image.open(image_path).convert("RGB").resize((224, 224))

    # 2. Flatten to (N, 3) pixel array
    pixels = np.array(img, dtype=np.float32).reshape(-1, 3)

    # 3. K-Means clustering
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    km.fit(pixels)
    centers = km.cluster_centers_  # shape: (n_clusters, 3)

    # 4. Sort clusters by frequency (most dominant first)
    counts = np.bincount(km.labels_, minlength=n_clusters)
    sorted_idx = np.argsort(-counts)
    centers = centers[sorted_idx]

    # 5. Build palette and flag list
    palette: list[str] = []
    flags: list[str] = []
    flag_keys: list[str] = []

    for center in centers:
        hex_colour = _rgb_to_hex(center)
        palette.append(hex_colour)
        flag_key = _nearest_colour_flag(center)
        flag_keys.append(flag_key)
        flags.append(colour_flags.get(flag_key, "unknown"))

    # 6. Derive ABCDE 'C' criterion
    unique_keys = list(dict.fromkeys(flag_keys))  # preserve order, deduplicate
    n_unique = len(unique_keys)

    if n_unique >= 3:
        abcde_c = "Multiple colours / regression"
    elif n_unique == 2:
        abcde_c = "Two colours present"
    else:
        abcde_c = "Uniform colour"

    return {
        "palette": palette,
        "flags": flags,
        "abcde_c": abcde_c,
    }

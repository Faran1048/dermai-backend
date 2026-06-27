"""
knowledge/colour_flags.py

Colour-to-clinical-significance mapping for skin lesion analysis.

Used by the ABCDE rule engine and the image analysis pipeline to translate
dominant colour clusters (extracted via K-Means or similar) into clinical flags.

Each key is a colour category name returned by the colour analysis module.
Each value is a human-readable clinical flag string describing its significance.
"""

COLOUR_FLAGS: dict[str, str] = {
    "black": "asymmetric melanin — high risk",
    "dark_brown": "deep pigmentation",
    "red": "vascular or inflammatory",
    "pink": "benign inflammatory",
    "light_brown": "common mole pigment",
    "white": "depigmentation",
    "blue_gray": "regression pattern",
}

# ── Extended metadata for richer reporting ────────────────────────────────────
# Maps each colour flag to additional clinical context and risk contribution.

COLOUR_FLAG_METADATA: dict[str, dict] = {
    "black": {
        "risk_contribution": "high",
        "associated_classes": ["mel"],
        "detail": (
            "Black or jet-dark pigmentation is associated with deep melanin deposits "
            "and is a significant ABCDE 'colour' criterion. Warrants immediate "
            "dermatological evaluation."
        ),
    },
    "dark_brown": {
        "risk_contribution": "medium",
        "associated_classes": ["mel", "nv", "bkl"],
        "detail": (
            "Dark brown colouration reflects deep dermal pigmentation. Common in "
            "dysplastic nevi and early melanoma. Evaluate in conjunction with "
            "asymmetry and border irregularity."
        ),
    },
    "red": {
        "risk_contribution": "medium",
        "associated_classes": ["vasc", "akiec", "bcc"],
        "detail": (
            "Red colouration indicates vascular involvement or inflammation. "
            "May signal a pyogenic granuloma, erythematous BCC, or inflamed AK. "
            "Bleeds easily in pyogenic granuloma."
        ),
    },
    "pink": {
        "risk_contribution": "low",
        "associated_classes": ["bcc", "df", "vasc"],
        "detail": (
            "Pink colouration is common in superficial BCCs (pearly pink nodule) "
            "and benign inflammatory lesions. Less concerning than red but "
            "should be noted alongside border and surface texture."
        ),
    },
    "light_brown": {
        "risk_contribution": "low",
        "associated_classes": ["nv", "bkl", "df"],
        "detail": (
            "Light brown is the typical colouration of benign melanocytic nevi "
            "and seborrheic keratoses. Generally low risk when uniform and "
            "combined with regular borders and symmetry."
        ),
    },
    "white": {
        "risk_contribution": "medium",
        "associated_classes": ["mel", "bkl"],
        "detail": (
            "White or hypopigmented areas within a lesion may indicate regression "
            "in melanoma — a site where the immune system has attacked melanocytes. "
            "Regression is paradoxically a poor prognostic sign."
        ),
    },
    "blue_gray": {
        "risk_contribution": "high",
        "associated_classes": ["mel"],
        "detail": (
            "Blue-grey colouration is a classic dermoscopic regression pattern "
            "in melanoma, representing melanophages in the papillary dermis. "
            "Strongly associated with invasive melanoma."
        ),
    },
}

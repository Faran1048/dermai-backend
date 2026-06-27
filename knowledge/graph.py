"""
knowledge/graph.py

Clinical knowledge graph for the 7 HAM10000 / ISIC skin lesion classes.

Keys use the standard ISIC abbreviations:
  mel    — Melanoma
  nv     — Melanocytic Nevus (common mole)
  bcc    — Basal Cell Carcinoma
  akiec  — Actinic Keratosis / Intraepithelial Carcinoma (Bowen's disease)
  bkl    — Benign Keratosis-like Lesion (seborrheic keratosis, solar lentigo)
  vasc   — Vascular Lesion (angioma, angiokeratoma, pyogenic granuloma)
  df     — Dermatofibroma

Pathway values:
  self_monitor  — patient can observe at home with annual check
  gp            — see a general practitioner within weeks
  dermatologist — refer to dermatology within days-to-weeks
  biopsy        — excisional or punch biopsy required
  oncology      — urgent oncology / multidisciplinary team referral
"""

KNOWLEDGE_GRAPH: dict[str, dict] = {
    "mel": {
        "related": ["nv", "akiec", "bcc"],
        "urgency": 10,
        "pathway": "oncology",
        "description": (
            "Melanoma is the most dangerous form of skin cancer, arising from "
            "melanocytes. Characterised by asymmetry, irregular borders, multiple "
            "colours, and a diameter >6 mm. Early-stage detection is critical — "
            "5-year survival drops sharply with metastatic spread."
        ),
    },
    "nv": {
        "related": ["mel", "bkl", "df"],
        "urgency": 2,
        "pathway": "self_monitor",
        "description": (
            "Melanocytic nevi (common moles) are benign proliferations of melanocytes. "
            "Usually symmetrical, uniformly coloured, and <6 mm. Most require only "
            "annual self-monitoring; any change in size, shape, or colour warrants "
            "professional evaluation to rule out dysplastic change."
        ),
    },
    "bcc": {
        "related": ["akiec", "mel", "bkl"],
        "urgency": 7,
        "pathway": "biopsy",
        "description": (
            "Basal Cell Carcinoma is the most common skin cancer, arising from basal "
            "keratinocytes. Typically presents as a pearly or translucent nodule with "
            "telangiectasia. Locally invasive but rarely metastatic. Requires surgical "
            "excision or Mohs surgery; rarely life-threatening when caught early."
        ),
    },
    "akiec": {
        "related": ["bcc", "mel", "bkl"],
        "urgency": 6,
        "pathway": "dermatologist",
        "description": (
            "Actinic Keratosis (AK) is a pre-malignant lesion caused by cumulative UV "
            "damage. Bowen's disease (squamous cell carcinoma in situ) represents full-"
            "thickness epidermal dysplasia. AKs carry a small but real risk of "
            "progressing to invasive squamous cell carcinoma if untreated."
        ),
    },
    "bkl": {
        "related": ["nv", "mel", "akiec"],
        "urgency": 2,
        "pathway": "gp",
        "description": (
            "Benign keratosis-like lesions include seborrheic keratoses, solar "
            "lentigines, and lichen-planus-like keratoses. They are common, benign, "
            "and non-malignant. However, their variable pigmentation can mimic "
            "melanoma, so uncertain lesions should be reviewed by a GP or dermatologist."
        ),
    },
    "vasc": {
        "related": ["df", "bkl"],
        "urgency": 3,
        "pathway": "gp",
        "description": (
            "Vascular lesions encompass cherry angiomas, angiokeratomas, pyogenic "
            "granulomas, and haemorrhage. Most are benign vascular proliferations. "
            "Pyogenic granulomas bleed easily and may require shave excision. "
            "True malignant vascular tumours (angiosarcoma) are rare but serious."
        ),
    },
    "df": {
        "related": ["nv", "vasc", "bkl"],
        "urgency": 1,
        "pathway": "self_monitor",
        "description": (
            "Dermatofibroma is a common, benign fibrous nodule typically found on the "
            "lower extremities. It dimples inward when pinched (Fitzpatrick sign) and "
            "is firm to the touch. No treatment is required unless symptomatic or "
            "cosmetically bothersome, in which case shave excision is curative."
        ),
    },
}

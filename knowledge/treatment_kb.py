"""
knowledge/treatment_kb.py

Treatment knowledge base for the 7 HAM10000 / ISIC skin lesion classes.

Each class maps to a list of treatment dicts ordered by clinical priority
(highest priority_score first = most urgent / standard of care first).

Fields per treatment entry:
  treatment      — intervention name / description
  priority_score — float 0.0–1.0 (1.0 = first-line, mandatory)
  note           — brief clinical rationale or caveat
"""

TREATMENT_KB: dict[str, list[dict]] = {

    # ── Melanoma ──────────────────────────────────────────────────────────────
    "mel": [
        {
            "treatment": "Wide local excision (WLE)",
            "priority_score": 1.0,
            "note": (
                "Primary treatment for localised melanoma. Excision margins are "
                "determined by Breslow thickness: 1 cm for ≤1 mm, 2 cm for >2 mm."
            ),
        },
        {
            "treatment": "Sentinel lymph node biopsy (SLNB)",
            "priority_score": 0.9,
            "note": (
                "Indicated for melanomas >0.8 mm Breslow thickness or with "
                "ulceration/high mitotic rate. Determines staging and guides "
                "adjuvant therapy decisions."
            ),
        },
        {
            "treatment": "Adjuvant immunotherapy (anti-PD-1: pembrolizumab / nivolumab)",
            "priority_score": 0.8,
            "note": (
                "Standard adjuvant therapy for stage III–IV resected melanoma. "
                "Significantly improves recurrence-free survival. Monitor for "
                "immune-related adverse events."
            ),
        },
        {
            "treatment": "BRAF/MEK targeted therapy (dabrafenib + trametinib)",
            "priority_score": 0.75,
            "note": (
                "Indicated for BRAF V600E/K-mutant melanoma (~50% of cases). "
                "Use when BRAF mutation confirmed on molecular testing. "
                "Not effective in BRAF wild-type disease."
            ),
        },
        {
            "treatment": "Urgent dermatology / oncology referral",
            "priority_score": 1.0,
            "note": (
                "Any suspected melanoma requires same-week urgent referral. "
                "Do not delay — early excision is the single biggest factor "
                "in improving survival."
            ),
        },
    ],

    # ── Melanocytic Nevus ─────────────────────────────────────────────────────
    "nv": [
        {
            "treatment": "Clinical surveillance and dermoscopy",
            "priority_score": 0.9,
            "note": (
                "Annual professional skin check with total body photography "
                "and dermoscopy for patients with multiple nevi or family history "
                "of melanoma."
            ),
        },
        {
            "treatment": "Patient education on self-examination (ABCDE rule)",
            "priority_score": 0.8,
            "note": (
                "Educate patient to monitor for: Asymmetry, Border irregularity, "
                "Colour change, Diameter growth >6 mm, Evolution. "
                "Report changes promptly."
            ),
        },
        {
            "treatment": "Excisional biopsy (if atypical features develop)",
            "priority_score": 0.5,
            "note": (
                "Indicated for dysplastic nevi or nevi with changing features. "
                "Full-thickness excision with 2–3 mm margins preferred over "
                "shave biopsy to preserve histological architecture."
            ),
        },
        {
            "treatment": "Sun protection (SPF 50+ broad-spectrum sunscreen)",
            "priority_score": 0.7,
            "note": (
                "Consistent daily UV protection reduces risk of UV-induced "
                "dysplastic change and new nevus formation. "
                "Avoid tanning beds entirely."
            ),
        },
    ],

    # ── Basal Cell Carcinoma ──────────────────────────────────────────────────
    "bcc": [
        {
            "treatment": "Mohs micrographic surgery",
            "priority_score": 1.0,
            "note": (
                "Gold standard for high-risk BCC (face, ears, nose, periorbital "
                "area; recurrent lesions; morphoeic/infiltrative subtype). "
                "Achieves highest cure rate (99%) with tissue preservation."
            ),
        },
        {
            "treatment": "Standard surgical excision",
            "priority_score": 0.9,
            "note": (
                "First-line for low-risk BCC on trunk/limbs. "
                "4 mm margins for nodular/superficial BCC <2 cm. "
                "Histological confirmation of clear margins mandatory."
            ),
        },
        {
            "treatment": "Photodynamic therapy (PDT)",
            "priority_score": 0.6,
            "note": (
                "Suitable for superficial BCC only. Uses methyl aminolevulinate "
                "(MAL) cream + red light activation. Good cosmetic outcomes; "
                "higher recurrence rate than surgery at 5 years."
            ),
        },
        {
            "treatment": "Imiquimod 5% cream (topical)",
            "priority_score": 0.5,
            "note": (
                "Second-line option for superficial BCC when surgery is not "
                "appropriate. 5×/week application for 6 weeks. "
                "Local inflammatory reaction is expected and indicates response."
            ),
        },
        {
            "treatment": "Hedgehog pathway inhibitor (vismodegib / sonidegib)",
            "priority_score": 0.4,
            "note": (
                "Reserved for locally advanced or metastatic BCC not amenable to "
                "surgery or radiotherapy. Significant side effects including "
                "muscle cramps, alopecia, and dysgeusia."
            ),
        },
    ],

    # ── Actinic Keratosis / Intraepithelial Carcinoma ─────────────────────────
    "akiec": [
        {
            "treatment": "Cryotherapy (liquid nitrogen)",
            "priority_score": 0.9,
            "note": (
                "First-line for discrete AKs. Single freeze-thaw cycle of "
                "10–20 seconds. Effective, rapid, and widely available. "
                "May cause temporary hypopigmentation."
            ),
        },
        {
            "treatment": "Ingenol mebutate / Diclofenac 3% gel (field therapy)",
            "priority_score": 0.75,
            "note": (
                "Field therapy for multiple AKs in a sun-damaged area. "
                "Treats subclinical lesions not visible on examination. "
                "Topical diclofenac for 60–90 days; ingenol mebutate 2–3 days."
            ),
        },
        {
            "treatment": "5-Fluorouracil (5-FU) cream 5%",
            "priority_score": 0.7,
            "note": (
                "Effective field treatment applied twice daily for 2–4 weeks. "
                "Significant local skin reaction (erosion, crusting) is expected "
                "and indicates therapeutic effect."
            ),
        },
        {
            "treatment": "Excision with histology (for Bowen's disease / SCC in situ)",
            "priority_score": 0.85,
            "note": (
                "Full-thickness excision required for Bowen's disease to exclude "
                "invasive squamous cell carcinoma. Margins of 4–5 mm recommended. "
                "PDT is an alternative for large or multiple Bowen's lesions."
            ),
        },
        {
            "treatment": "Annual dermatology surveillance",
            "priority_score": 0.8,
            "note": (
                "Patients with AKs have significantly increased risk of SCC, BCC, "
                "and melanoma. Annual total body skin examination strongly recommended. "
                "Strict daily photoprotection is essential."
            ),
        },
    ],

    # ── Benign Keratosis-like Lesion ──────────────────────────────────────────
    "bkl": [
        {
            "treatment": "Clinical observation / reassurance",
            "priority_score": 0.8,
            "note": (
                "Most BKLs (seborrheic keratoses, solar lentigines) are entirely "
                "benign and require no treatment. Patient reassurance and "
                "explanation of the benign nature is the primary intervention."
            ),
        },
        {
            "treatment": "Cryotherapy (for cosmetic removal or diagnostic uncertainty)",
            "priority_score": 0.5,
            "note": (
                "Light cryotherapy effective for solar lentigines and thin "
                "seborrheic keratoses. If diagnosis is uncertain, histological "
                "confirmation via biopsy should precede cosmetic treatment."
            ),
        },
        {
            "treatment": "Shave excision / curettage",
            "priority_score": 0.4,
            "note": (
                "For symptomatic (itchy, catching on clothing) or cosmetically "
                "bothersome seborrheic keratoses. Quick, effective, and "
                "leaves material for histology if needed."
            ),
        },
        {
            "treatment": "Dermoscopy / biopsy if melanoma cannot be excluded",
            "priority_score": 0.9,
            "note": (
                "Pigmented BKLs can mimic melanoma clinically and dermoscopically. "
                "If any doubt remains after dermoscopy, excision biopsy with "
                "full histology is the safe option."
            ),
        },
    ],

    # ── Vascular Lesion ───────────────────────────────────────────────────────
    "vasc": [
        {
            "treatment": "Observation (for asymptomatic cherry angiomas / angiokeratomas)",
            "priority_score": 0.7,
            "note": (
                "Simple cherry angiomas and angiokeratomas are benign and require "
                "no treatment unless symptomatic or the patient requests removal "
                "for cosmetic reasons."
            ),
        },
        {
            "treatment": "Pulsed dye laser (PDL) or Nd:YAG laser",
            "priority_score": 0.6,
            "note": (
                "Effective and well-tolerated treatment for cherry angiomas and "
                "telangiectasia. Targets oxyhaemoglobin with minimal scarring. "
                "Usually 1–2 sessions required."
            ),
        },
        {
            "treatment": "Shave excision + curettage (pyogenic granuloma)",
            "priority_score": 0.9,
            "note": (
                "Pyogenic granulomas bleed profusely and rarely self-resolve. "
                "Shave excision with curettage and electrocautery of the base "
                "is the standard treatment. Send specimen for histology."
            ),
        },
        {
            "treatment": "Electrocautery / diathermy",
            "priority_score": 0.5,
            "note": (
                "Simple and effective for small vascular lesions. "
                "Useful adjunct after shave excision to reduce recurrence "
                "risk in pyogenic granuloma."
            ),
        },
    ],

    # ── Dermatofibroma ────────────────────────────────────────────────────────
    "df": [
        {
            "treatment": "Observation and reassurance",
            "priority_score": 0.9,
            "note": (
                "The vast majority of dermatofibromas are entirely benign and "
                "require no treatment. Patient education about the benign nature "
                "and the characteristic Fitzpatrick sign is sufficient."
            ),
        },
        {
            "treatment": "Surgical excision (if symptomatic or atypical)",
            "priority_score": 0.5,
            "note": (
                "Excision is appropriate if the lesion is painful, rapidly growing, "
                "or if the diagnosis is uncertain. Ensure excision extends into "
                "subcutaneous fat to reduce recurrence."
            ),
        },
        {
            "treatment": "Shave excision (cosmetic, superficial only)",
            "priority_score": 0.3,
            "note": (
                "Shave excision flattens the lesion for cosmetic purposes but "
                "does not remove the full fibrous component — recurrence is common. "
                "Suitable only when histology is not required."
            ),
        },
        {
            "treatment": "Dermoscopy to exclude DFSP (dermatofibrosarcoma protuberans)",
            "priority_score": 0.7,
            "note": (
                "Large (>3 cm), rapidly growing, or indurated dermatofibromas "
                "should be biopsied to exclude DFSP, a locally aggressive "
                "sarcoma requiring wide excision."
            ),
        },
    ],
}

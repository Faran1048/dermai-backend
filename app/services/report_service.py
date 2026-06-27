"""
app/services/report_service.py

PDF report generation for completed analyses.
Uses ReportLab — a mature, free, fully local PDF library.

The generated PDF includes:
  - DermAI branding header
  - Patient information
  - Analysis date and type
  - Detected conditions table with confidence scores and ICD-10 codes
  - Clinical recommendations
  - Disclaimer footer
"""

from __future__ import annotations

import io
import logging
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.models.analysis import Analysis
from app.models.user import User

logger = logging.getLogger(__name__)

# ── Colour palette ─────────────────────────────────────────────────────────────
BRAND_TEAL = colors.HexColor("#0D9488")
BRAND_DARK = colors.HexColor("#0F172A")
BRAND_LIGHT_GREY = colors.HexColor("#F1F5F9")
SEVERITY_COLORS = {
    "low": colors.HexColor("#16A34A"),
    "moderate": colors.HexColor("#D97706"),
    "high": colors.HexColor("#DC2626"),
    "urgent": colors.HexColor("#7C3AED"),
}


def _get_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=22,
            textColor=BRAND_TEAL,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.grey,
            spaceAfter=12,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            parent=base["Heading2"],
            fontSize=13,
            textColor=BRAND_DARK,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            textColor=BRAND_DARK,
            spaceAfter=4,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            parent=base["Normal"],
            fontSize=8,
            textColor=colors.grey,
            spaceAfter=4,
        ),
    }


def generate_pdf_report(analysis: Analysis, user: User) -> bytes:
    """
    Generate a PDF report for the given analysis and return it as bytes.

    Args:
        analysis: The Analysis ORM record.
        user: The User ORM record (for patient info).

    Returns:
        Raw PDF bytes suitable for streaming in a FastAPI response.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = _get_styles()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("DermAI", styles["title"]))
    story.append(Paragraph("AI-Assisted Dermatology Analysis Report", styles["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_TEAL))
    story.append(Spacer(1, 8 * mm))

    # ── Patient & Report info ─────────────────────────────────────────────────
    report_date = analysis.created_at.strftime("%B %d, %Y at %H:%M UTC")
    info_data = [
        ["Patient Name:", user.full_name, "Report ID:", f"#{analysis.id}"],
        ["Patient Email:", user.email, "Analysis Type:", analysis.analysis_type.upper()],
        ["Date of Analysis:", report_date, "Severity:", (analysis.severity or "N/A").upper()],
    ]

    severity_color = SEVERITY_COLORS.get(analysis.severity or "low", colors.black)

    info_table = Table(info_data, colWidths=[40 * mm, 70 * mm, 35 * mm, 45 * mm])
    info_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (3, 2), (3, 2), severity_color),
                ("FONTNAME", (3, 2), (3, 2), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(info_table)
    story.append(Spacer(1, 6 * mm))

    # ── Summary ───────────────────────────────────────────────────────────────
    if analysis.summary:
        story.append(Paragraph("Summary", styles["section_header"]))
        story.append(Paragraph(analysis.summary, styles["body"]))
        story.append(Spacer(1, 4 * mm))

    # ── Detected Conditions ───────────────────────────────────────────────────
    conditions = analysis.conditions or []
    if conditions:
        story.append(Paragraph("Detected Conditions", styles["section_header"]))

        table_data = [["Condition", "Confidence", "ICD-10", "Description"]]
        for c in conditions:
            confidence_pct = f"{float(c.get('confidence', 0)) * 100:.1f}%"
            table_data.append(
                [
                    c.get("condition", "Unknown"),
                    confidence_pct,
                    c.get("icd10_code") or "N/A",
                    Paragraph(c.get("description", ""), styles["body"]),
                ]
            )

        cond_table = Table(
            table_data,
            colWidths=[50 * mm, 22 * mm, 20 * mm, 78 * mm],
        )
        cond_table.setStyle(
            TableStyle(
                [
                    # Header row
                    ("BACKGROUND", (0, 0), (-1, 0), BRAND_TEAL),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, 0), 9),
                    ("ALIGN", (1, 0), (1, -1), "CENTER"),
                    ("ALIGN", (2, 0), (2, -1), "CENTER"),
                    # Data rows
                    ("FONTSIZE", (0, 1), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BRAND_LIGHT_GREY]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(cond_table)
        story.append(Spacer(1, 6 * mm))

    # ── Submitted Symptoms ────────────────────────────────────────────────────
    symptoms = analysis.symptoms or []
    if symptoms:
        story.append(Paragraph("Reported Symptoms", styles["section_header"]))
        symptom_text = " • ".join(s.title() for s in symptoms)
        story.append(Paragraph(symptom_text, styles["body"]))
        story.append(Spacer(1, 4 * mm))

    # ── Recommendations ───────────────────────────────────────────────────────
    recommendations = analysis.recommendations or []
    if recommendations:
        story.append(Paragraph("Clinical Recommendations", styles["section_header"]))
        for i, rec in enumerate(recommendations, start=1):
            urgency = rec.get("urgency", "routine").upper()
            rec_text = rec.get("recommendation", "")
            urgency_color = "#DC2626" if urgency == "URGENT" else "#0D9488"
            story.append(
                Paragraph(
                    f'<b>{i}. [{urgency}]</b> <font color="{urgency_color}">{rec_text}</font>',
                    styles["body"],
                )
            )
        story.append(Spacer(1, 6 * mm))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            "<b>DISCLAIMER:</b> This report is generated by an AI system and is intended "
            "for informational purposes only. It does not constitute medical advice, diagnosis, "
            "or treatment. Always seek the advice of a qualified healthcare professional for "
            "any medical condition. In case of emergency, contact emergency services immediately.",
            styles["disclaimer"],
        )
    )
    story.append(
        Paragraph(
            f"Generated by DermAI Backend v1.0.0 on {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC",
            styles["disclaimer"],
        )
    )

    doc.build(story)
    return buffer.getvalue()

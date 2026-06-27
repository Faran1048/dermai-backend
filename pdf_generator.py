"""
pdf_generator.py

Generates a premium 4-page PDF report for skin lesion analysis using ReportLab.
Features a clean cover page, side-by-side image analysis (original + Grad-CAM),
K-Means dominant colour palette visualization, CNN probability bar charts,
ABCDE scoring table, A* reasoning chain, Minimax differentials, Greedy BFS treatments,
and Dijkstra referral pathway drawing.
"""

from __future__ import annotations
import io
import os
import requests
from datetime import datetime
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
    Image,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.platypus.flowables import HRFlowable

# ── Styles Setup ──────────────────────────────────────────────────────────────

def _get_pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    
    # Custom styles
    styles = {
        "title_white": ParagraphStyle(
            "title_white",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=26,
            textColor=colors.white,
            spaceAfter=6,
        ),
        "subtitle_white": ParagraphStyle(
            "subtitle_white",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            textColor=colors.HexColor("#93C5FD"),
            spaceAfter=12,
        ),
        "report_title": ParagraphStyle(
            "report_title",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=colors.HexColor("#1B3A6B"),
            spaceBefore=15,
            spaceAfter=15,
        ),
        "section_header": ParagraphStyle(
            "section_header",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.HexColor("#1B3A6B"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
            leading=13,
        ),
        "body_center": ParagraphStyle(
            "body_center",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=colors.HexColor("#334155"),
            alignment=1, # Center
            leading=12,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            textColor=colors.HexColor("#334155"),
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4,
            leading=13,
        ),
        "disclaimer_style": ParagraphStyle(
            "disclaimer_style",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            textColor=colors.HexColor("#991B1B"),
            leading=10,
        ),
        "badge_text": ParagraphStyle(
            "badge_text",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=colors.white,
            alignment=1,
        )
    }
    return styles


# ── Helper: Download and Format Image ─────────────────────────────────────────

def _get_image_flowable(url: str, width: float, height: float) -> Table | Image:
    """Download image from URL and return a ReportLab Image or fallback placeholder (cached)."""
    if not url:
        return Table([["No Image URL"]], colWidths=[width], rowHeights=[height],
                     style=TableStyle([
                         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                         ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                         ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#94A3B8")),
                         ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                         ("FONTSIZE", (0, 0), (-1, -1), 10),
                         ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#CBD5E1")),
                     ]))
                     
    import hashlib
    # Setup cache directory
    cache_dir = os.path.join(os.getcwd(), "tmp", "cache_images")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Hash URL for unique local filename
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{url_hash}.jpg")
    
    # 1. Try reading from local cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                img_data = fh.read()
            # Verify file format
            PILImage.open(io.BytesIO(img_data))
            return Image(io.BytesIO(img_data), width=width, height=height)
        except Exception:
            # Corrupted cache file, clean up and proceed to download
            try:
                os.remove(cache_path)
            except Exception:
                pass
                
    # 2. Download from network if not cached or corrupted
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        img_bytes = resp.content
        
        # Verify image format via PIL before returning or caching
        PILImage.open(io.BytesIO(img_bytes))
        
        # Write to disk cache
        try:
            with open(cache_path, "wb") as f:
                f.write(img_bytes)
        except Exception:
            pass # Non-fatal, proceed with returning downloaded image
            
        return Image(io.BytesIO(img_bytes), width=width, height=height)
    except Exception as e:
        return Table([[f"Failed to Load Image\n({str(e)[:20]}...)"]], colWidths=[width], rowHeights=[height],
                     style=TableStyle([
                         ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
                         ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                         ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                         ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#EF4444")),
                         ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                         ("FONTSIZE", (0, 0), (-1, -1), 8),
                         ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FCA5A5")),
                     ]))



# ── Helper: Canvas Callback for Footer ────────────────────────────────────────

def _draw_page_decorations(canvas, doc) -> None:
    canvas.saveState()
    # Draw thin divider line at the bottom
    canvas.setStrokeColor(colors.HexColor("#E2E8F0"))
    canvas.setLineWidth(0.5)
    canvas.line(54, 45, doc.pagesize[0] - 54, 45)
    
    # Draw footer text
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    canvas.drawString(54, 30, f"DermAI — For Educational Purposes Only | Not Medical Advice | Generated: {date_str}")
    canvas.drawRightString(doc.pagesize[0] - 54, 30, f"Page {doc.page} of 4")
    canvas.restoreState()


# ── Drawing Helper: Probability Bars ──────────────────────────────────────────

def _draw_probability_bar(label: str, score: float) -> Drawing:
    """Draw an ImageNet/lesion class confidence bar graph."""
    d = Drawing(440, 18)
    # Background bar
    d.add(Rect(120, 2, 280, 10, fillColor=colors.HexColor("#F1F5F9"), strokeColor=None))
    # Filled bar
    bar_width = max(0.0, min(1.0, score)) * 280
    bar_color = colors.HexColor("#10B981") if score > 0.5 else colors.HexColor("#3B82F6")
    d.add(Rect(120, 2, bar_width, 10, fillColor=bar_color, strokeColor=None))
    # Text label
    d.add(String(5, 3, f"{label.upper()}", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.HexColor("#0F172A")))
    # Percentage label
    d.add(String(405, 3, f"{score * 100:.1f}%", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#334155")))
    return d


# ── Drawing Helper: Referral Pathway ─────────────────────────────────────────

def _draw_referral_pathway(path_steps: list[str]) -> Drawing:
    """Draw a horizontal layout of nodes connected by arrows."""
    d = Drawing(440, 50)
    n_steps = len(path_steps)
    if n_steps == 0:
        return d
    box_width = 70
    box_height = 24
    gap = 18
    
    for i, step in enumerate(path_steps):
        x = 5 + i * (box_width + gap)
        y = 13
        is_last = (i == n_steps - 1)
        
        bg_color = colors.HexColor("#1B3A6B") if is_last else colors.HexColor("#F8FAFC")
        text_color = colors.white if is_last else colors.HexColor("#334155")
        border_color = colors.HexColor("#1B3A6B") if is_last else colors.HexColor("#CBD5E1")
        
        d.add(Rect(x, y, box_width, box_height, fillColor=bg_color, strokeColor=border_color, strokeWidth=1, rx=2, ry=2))
        
        label = step.replace("_", " ").title()
        if len(label) > 11:
            label1 = label[:9] + "-"
            label2 = label[9:]
            d.add(String(x + box_width / 2, y + 13, label1, fontName="Helvetica-Bold", fontSize=7, fillColor=text_color, textAnchor="middle"))
            d.add(String(x + box_width / 2, y + 5, label2, fontName="Helvetica-Bold", fontSize=7, fillColor=text_color, textAnchor="middle"))
        else:
            d.add(String(x + box_width / 2, y + 8, label, fontName="Helvetica-Bold", fontSize=7.5, fillColor=text_color, textAnchor="middle"))
            
        if i < n_steps - 1:
            arrow_x = x + box_width
            arrow_end_x = arrow_x + gap
            arrow_y = y + box_height / 2
            # Line
            d.add(Line(arrow_x, arrow_y, arrow_end_x, arrow_y, strokeColor=colors.HexColor("#94A3B8"), strokeWidth=1.2))
            # Arrow head
            d.add(Polygon([arrow_end_x, arrow_y, arrow_end_x - 3, arrow_y + 2, arrow_end_x - 3, arrow_y - 2], fillColor=colors.HexColor("#94A3B8"), strokeColor=None))
            
    return d


# ── Main Generator ────────────────────────────────────────────────────────────

def generate_report(analysis_data: dict) -> bytes:
    """
    Generates a 4-page PDF report.
    Returns the PDF content as a raw bytes array.
    """
    buffer = io.BytesIO()
    
    # Standard A4 size is 595.27 x 841.89 points
    # Page margins: left/right=54 (0.75in), top/bottom=54
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=64, # leave room for footer
    )
    
    styles = _get_pdf_styles()
    story = []
    
    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 1 — COVER PAGE
    # ──────────────────────────────────────────────────────────────────────────
    
    # Header Banner Table (Navy block spanning the page width)
    header_content = [
        [Paragraph("DermAI", styles["title_white"])],
        [Paragraph("Automated Clinical Decision Support & AI Diagnostic Assistant", styles["subtitle_white"])]
    ]
    header_banner = Table(header_content, colWidths=[488])
    header_banner.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1B3A6B")),
        ("PADDING", (0, 0), (-1, -1), 22),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 26),
    ]))
    story.append(header_banner)
    story.append(Spacer(1, 25))
    
    story.append(Paragraph("AI Skin Lesion Analysis Report", styles["report_title"]))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1B3A6B"), spaceAfter=20))
    
    # Patient Info Table
    date_str = analysis_data.get("created_at")
    if isinstance(date_str, datetime):
        date_formatted = date_str.strftime("%B %d, %Y at %H:%M")
    elif isinstance(date_str, str):
        try:
            date_formatted = datetime.fromisoformat(date_str.replace("Z", "+00:00")).strftime("%B %d, %Y at %H:%M")
        except Exception:
            date_formatted = date_str
    else:
        date_formatted = datetime.now().strftime("%B %d, %Y")

    intake = analysis_data.get("intake_form") or {}
    info_data = [
        [Paragraph("<b>Analysis ID:</b>", styles["body"]), Paragraph(str(analysis_data.get("analysis_id", "N/A")), styles["body"])],
        [Paragraph("<b>Date of Evaluation:</b>", styles["body"]), Paragraph(date_formatted, styles["body"])],
        [Paragraph("<b>Patient Age:</b>", styles["body"]), Paragraph(str(intake.get("age", "N/A")), styles["body"])],
        [Paragraph("<b>Skin Type (Fitzpatrick):</b>", styles["body"]), Paragraph(f"Type {intake.get('skin_type', 'N/A')}", styles["body"])],
        [Paragraph("<b>Body Location:</b>", styles["body"]), Paragraph(str(intake.get("body_location", "N/A")).title(), styles["body"])],
    ]
    info_table = Table(info_data, colWidths=[160, 328])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 40))
    
    # Risk Badge
    risk_level = str(analysis_data.get("risk_level", "Low")).title()
    badge_colors = {
        "High": colors.HexColor("#DC2626"),   # Red
        "Medium": colors.HexColor("#F59E0B"), # Orange
        "Low": colors.HexColor("#10B981"),    # Green
    }
    badge_bg = badge_colors.get(risk_level, colors.HexColor("#64748B"))
    
    badge_table = Table([[Paragraph(f"RISK LEVEL ASSESSMENT: {risk_level.upper()}", styles["badge_text"])]], colWidths=[350])
    badge_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), badge_bg),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 14),
        ("BOX", (0, 0), (-1, -1), 1.5, colors.white),
    ]))
    story.append(badge_table)
    
    story.append(PageBreak())
    
    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 2 — IMAGE ANALYSIS
    # ──────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Image Analysis & Visualizations", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=12))
    
    # Side-by-side images (Original Lesion & Grad-CAM Heatmap)
    img_orig = _get_image_flowable(analysis_data.get("image_url"), 210, 210)
    img_gradcam = _get_image_flowable(analysis_data.get("heatmap_url"), 210, 210)
    
    image_table = Table([[img_orig, img_gradcam]], colWidths=[244, 244])
    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(image_table)
    
    # Image Captions
    caption_data = [
        [Paragraph("<b>Figure A:</b> Original Lesion Photo", styles["body_center"]),
         Paragraph("<b>Figure B:</b> Grad-CAM Activation Areas", styles["body_center"])]
    ]
    caption_table = Table(caption_data, colWidths=[244, 244])
    caption_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
    ]))
    story.append(caption_table)
    
    # K-Means Dominant Colours
    story.append(Paragraph("K-Means Dominant Colour Palette & Clinical Flags", styles["section_header"]))
    
    palette = analysis_data.get("palette", [])
    flags = analysis_data.get("flags", [])
    while len(palette) < 5:
        palette.append("#CCCCCC")
    while len(flags) < 5:
        flags.append("Unknown flag")
        
    color_row_top = ["", "", "", "", ""]
    color_row_bottom = [
        Paragraph(flags[0], styles["body_center"]),
        Paragraph(flags[1], styles["body_center"]),
        Paragraph(flags[2], styles["body_center"]),
        Paragraph(flags[3], styles["body_center"]),
        Paragraph(flags[4], styles["body_center"]),
    ]
    
    colour_table = Table([color_row_top, color_row_bottom], colWidths=[97, 97, 97, 97, 97])
    colour_table_style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, 0), 28), # Height of colored blocks
    ]
    for i in range(5):
        colour_table_style.append(("BACKGROUND", (i, 0), (i, 0), colors.HexColor(palette[i])))
    colour_table.setStyle(TableStyle(colour_table_style))
    story.append(colour_table)
    story.append(Spacer(1, 15))
    
    # CNN Classification Scores
    story.append(Paragraph("CNN Classifier Probability Distributions", styles["section_header"]))
    all_scores = analysis_data.get("all_scores") or {}
    
    class_long_names = {
        "mel": "Melanoma (mel)",
        "nv": "Melanocytic Nevus (nv)",
        "bcc": "Basal Cell Carcinoma (bcc)",
        "akiec": "Actinic Keratosis / Intraepithelial (akiec)",
        "bkl": "Benign Keratosis-like (bkl)",
        "vasc": "Vascular Lesion (vasc)",
        "df": "Dermatofibroma (df)",
    }
    
    # Sort scores descending
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
    for k, score in sorted_scores:
        name = class_long_names.get(k, k.upper())
        story.append(_draw_probability_bar(name, score))
        story.append(Spacer(1, 3))
        
    story.append(PageBreak())
    
    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 3 — CLINICAL FINDINGS
    # ──────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Clinical Findings & Rule-Based Analytics", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=10))
    
    # ABCDE scoring table
    story.append(Paragraph("ABCDE Scoring Analysis", styles["body_bold"]))
    abcde = analysis_data.get("abcde_results") or {}
    
    abcde_rows = [
        ["Criterion", "Observation / Classification Result"],
        ["Asymmetry (A)", abcde.get("A", "N/A")],
        ["Border Irregularity (B)", abcde.get("B", "N/A")],
        ["Colour Variation (C)", abcde.get("C", "N/A")],
        ["Diameter / Dermoscopic (D)", abcde.get("D", "N/A")],
        ["Evolution (E)", abcde.get("E", "N/A")],
    ]
    abcde_table = Table(abcde_rows, colWidths=[150, 338])
    abcde_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1B3A6B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ]))
    story.append(abcde_table)
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>ABCDE Score:</b> {abcde.get('total_score', 'N/A')}/10 | <b>Risk Tier:</b> {abcde.get('risk_tier', 'N/A')}", styles["body"]))
    story.append(Spacer(1, 10))
    
    # A* Reasoning Path
    story.append(Paragraph("A* Logical Reasoning Chain", styles["body_bold"]))
    reasoning = analysis_data.get("reasoning_path") or []
    if reasoning:
        for idx, step in enumerate(reasoning, start=1):
            story.append(Paragraph(f"<b>{idx}.</b> {step}", styles["bullet"]))
    else:
        story.append(Paragraph("No clear symptoms reasoning path generated.", styles["body"]))
    story.append(Spacer(1, 12))
    
    # Differential Diagnosis Table (Minimax top 3)
    story.append(Paragraph("Top Differential Diagnoses (Minimax Game Tree)", styles["body_bold"]))
    differentials = analysis_data.get("differential") or []
    diff_data = [["Class Name", "Evidence Score", "Assessed Severity"]]
    
    # Mapping helper
    class_map = {
        "mel": "Melanoma", "nv": "Melanocytic Nevus", "bcc": "Basal Cell Carcinoma",
        "akiec": "Actinic Keratosis", "bkl": "Benign Keratosis",
        "vasc": "Vascular Lesion", "df": "Dermatofibroma"
    }
    
    for item in differentials[:3]:
        cls = item.get("class", "N/A")
        diff_data.append([
            class_map.get(cls, cls.upper()),
            f"{item.get('evidence_score', 0.0):.4f}",
            str(item.get("risk_level", "Low")).title()
        ])
    if len(diff_data) == 1:
        diff_data.append(["No matching differential diagnoses met pruning threshold.", "", ""])
        
    diff_table = Table(diff_data, colWidths=[200, 140, 148])
    diff_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(diff_table)
    story.append(Spacer(1, 12))
    
    # Related conditions
    related = analysis_data.get("related_conditions") or []
    related_mapped = [class_map.get(c, c.upper()) for c in related]
    related_text = ", ".join(related_mapped) if related_mapped else "None"
    story.append(Paragraph(f"<b>Related Conditions explore path:</b> {related_text}", styles["body"]))
    story.append(Spacer(1, 10))
    
    # Treatment recommendations
    story.append(Paragraph("Clinical Care Guidelines (Greedy Heap Ranking)", styles["body_bold"]))
    treatments = analysis_data.get("treatments") or []
    if treatments:
        for idx, t in enumerate(treatments, start=1):
            treatment_title = t.get("treatment", "")
            priority = t.get("boosted_priority_score", 0.0)
            note = t.get("note", "")
            story.append(Paragraph(f"<b>{idx}. {treatment_title}</b> (Priority: {priority:.2f})<br/>"
                                   f"<font color='#475569'>{note}</font>", styles["bullet"]))
    else:
        story.append(Paragraph("No treatments registered for this condition.", styles["body"]))
        
    story.append(PageBreak())
    
    # ──────────────────────────────────────────────────────────────────────────
    # PAGE 4 — REFERRAL & DISCLAIMER
    # ──────────────────────────────────────────────────────────────────────────
    story.append(Paragraph("Referral Pathway & Next Steps", styles["section_header"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0"), spaceAfter=10))
    
    # Referral Graph
    referral = analysis_data.get("referral") or {}
    path = referral.get("path") or []
    story.append(Paragraph("<b>Optimized Referral Pathway (Dijkstra Shortest Path):</b>", styles["body"]))
    story.append(Spacer(1, 5))
    story.append(_draw_referral_pathway(path))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"<b>Estimated Action Timeline:</b> {referral.get('estimated_timeline', 'N/A')}", styles["body"]))
    story.append(Paragraph(f"<b>Route Urgency Class:</b> {referral.get('urgency', 'N/A')}", styles["body"]))
    story.append(Spacer(1, 15))
    
    # Intake Form Summary Table
    story.append(Paragraph("Submitted Patient Intake Form Summary", styles["body_bold"]))
    symptom_list_str = []
    symptoms_obj = intake.get("symptoms") or {}
    for sym_name, active in symptoms_obj.items():
        if active:
            symptom_list_str.append(sym_name.replace("_", " ").title())
    if not symptom_list_str:
        symptom_list_str.append("None Reported")
        
    intake_data = [
        ["Patient Age", f"{intake.get('age', 'N/A')} years"],
        ["Fitzpatrick Skin Type", f"Type {intake.get('skin_type', 'N/A')}"],
        ["Reported Location", str(intake.get("body_location", "N/A")).title()],
        ["Estimated Duration", f"{intake.get('duration_weeks', 'N/A')} weeks"],
        ["Symptom Flags", ", ".join(symptom_list_str)],
    ]
    intake_table = Table(intake_data, colWidths=[150, 338])
    intake_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8FAFC")),
    ]))
    story.append(intake_table)
    story.append(Spacer(1, 15))
    
    # Next steps bullet list
    story.append(Paragraph("Recommended Patient Action Steps", styles["body_bold"]))
    story.append(Paragraph("<b>1. Consult medical professional:</b> Book a clinical skin check with your GP or a board-certified dermatologist.", styles["bullet"]))
    story.append(Paragraph("<b>2. Do not self-treat:</b> Avoid applying chemical agents or picking the lesion as it hinders biopsy diagnostics.", styles["bullet"]))
    story.append(Paragraph("<b>3. Monitor evolution:</b> Take weekly high-resolution pictures to track size, boundaries, or colour shifts.", styles["bullet"]))
    story.append(Paragraph("<b>4. Safe sun exposure:</b> Use SPF 50+ broad-spectrum sunscreen and wear hats or UV-shielding clothing.", styles["bullet"]))
    story.append(Spacer(1, 20))
    
    # Medical disclaimer box (Red border, red-background)
    disclaimer_text = (
        "<b>CRITICAL MEDICAL DISCLAIMER:</b> This report is generated by a computer-assisted machine learning system. "
        "It is designed solely for educational, research, and preliminary tracking purposes. This report does NOT "
        "constitute medical advice, diagnosis, or professional treatment. The clinical evaluations here are not a "
        "substitute for clinical examinations. If you have any concerns, consult a physician. In case of emergency, "
        "please visit the nearest emergency medical services center immediately."
    )
    disclaimer_table = Table([[Paragraph(disclaimer_text, styles["disclaimer_style"])]], colWidths=[478])
    disclaimer_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#DC2626")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEF2F2")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(disclaimer_table)
    
    # Build document with footer callback on all pages
    doc.build(story, onFirstPage=_draw_page_decorations, onLaterPages=_draw_page_decorations)
    
    return buffer.getvalue()

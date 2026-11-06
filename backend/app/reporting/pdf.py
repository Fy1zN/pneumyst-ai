"""Screening Analysis Report as a PDF, rebuilt from the saved analysis on every request.

Text stays in plain characters because the built-in PDF fonts cannot draw symbols such as arrows.
Bands are written as words, so the report still reads correctly when printed in grey.
"""
from __future__ import annotations

import io
from datetime import datetime
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from backend.app.agent.facts import pct
from backend.app.db.models import AnalysisRecord
from backend.app.reporting.audit import ordered_stages
from backend.app.services.case_notes import OTHER_SYMPTOMS, case_notes

MAX_HEATMAPS = 3
BAND_ORDER = {"likely": 0, "possible": 1, "unlikely": 2}
BAND_COLOURS = {"likely": colors.HexColor("#b3261e"), "possible": colors.HexColor("#9a5b00"),
                "unlikely": colors.HexColor("#4d4d4d")}
MUTED = colors.HexColor("#666666")
HEATMAP_TEXT = {"shown": "shown", "weak_evidence": "weak evidence", "below_possible": "not needed",
                "no_band": "no band", "not_requested": "not requested", "unavailable": "unavailable"}
LIMITATIONS = [
    "Research prototype. Not a medical device and not validated for clinical use.",
    "Models were trained on public datasets (NIH ChestX-ray14, TBX11K, Shenzhen). NIH labels were mined from "
    "radiology reports by text processing and contain errors.",
    "Tuberculosis detection drops on images from other hospitals: AUROC 0.891 on the Shenzhen test set but 0.744 "
    "on the external Montgomery set.",
    "Some NIH scores partly reflect the image view (AP or PA), for example Edema AUROC 0.850 overall but 0.779 on "
    "AP images.",
    "Hernia has too few validation cases for a reliable band; its score is shown but not interpreted.",
    "Heatmaps show which regions changed a score most when hidden. They are not lesion outlines.",
]


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=17, spaceAfter=4),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13),
        "small": ParagraphStyle("small", parent=base["BodyText"], fontSize=8, leading=10, textColor=MUTED),
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8, leading=10),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=7.5, leading=9, alignment=TA_CENTER),
        "banner": ParagraphStyle("banner", parent=base["BodyText"], fontSize=10, textColor=colors.white,
                                 alignment=TA_CENTER, fontName="Helvetica-Bold"),
    }


def grid(rows: list[list], widths: list[float], header: bool = False) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("FONTSIZE", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ececec")),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    table.setStyle(TableStyle(style))
    return table


def band_word(finding: dict) -> str:
    if finding["score"] is None:
        return "UNAVAILABLE"
    return finding["band"].upper() if finding["band"] else "NOT BANDED"


def ordered(findings: list[dict]) -> list[dict]:
    def key(f: dict) -> tuple:
        rank = BAND_ORDER.get(f["band"], 3) if f["score"] is not None else 4
        return rank, -(f["score"] or 0.0)
    return sorted(findings, key=key)


def chosen_heatmaps(row: AnalysisRecord, images: dict[str, bytes]) -> list[dict]:
    shown = [f for f in ordered(row.findings) if f["finding"] in images and f["heatmap"] == "shown"]
    return shown[:MAX_HEATMAPS]


def image_cell(data: bytes, caption: str, s: dict, size: float) -> list:
    return [Image(io.BytesIO(data), width=size, height=size), Spacer(1, 2), Paragraph(caption, s["caption"])]


def tb_result(row: AnalysisRecord) -> str:
    tb = next((f for f in row.findings if f["finding"] == "Tuberculosis"), None)
    if tb is None or tb["score"] is None:
        return "UNAVAILABLE"
    return f"{band_word(tb)} ({pct(tb['score'])})"


def who_text(screen: dict | None) -> str:
    if screen is None:
        return "Not recorded"
    if screen["result"] == "positive":
        return "POSITIVE: " + ", ".join(screen["positive_symptoms"])
    if screen["result"] == "negative":
        return "NEGATIVE: all four answered no"
    return "INCOMPLETE: none reported; not asked: " + ", ".join(screen["not_asked"])


def other_symptoms_text(case: dict) -> str:
    groups = {"present": [], "absent": [], "not asked": []}
    for field, label in OTHER_SYMPTOMS.items():
        value = case.get(field)
        groups["present" if value == "yes" else "absent" if value == "no" else "not asked"].append(label)
    return "; ".join(f"{name}: {', '.join(items)}" for name, items in groups.items() if items)


def context_section(row: AnalysisRecord, s: dict, width: float) -> list:
    """Case details, cautions and the symptom screen, all side by side with the results and never combined."""
    case = row.case_context or {}
    notes = case_notes(case) or {"who_screen": None, "cautions": []}
    half = width / 2

    def value(field: str) -> Paragraph:
        return Paragraph(case.get(field, "-"), s["cell"])


    rows = [["Case number", value("case_number"), "Age band", value("age_band")],
            ["Sex", value("sex"), "Reason for X-ray", value("reason")],
            ["View", value("view"), "Position", value("position")],
            ["Image source", value("image_source"), "Smoking", value("smoking")],
            ["Close TB contact", value("tb_contact"), "Treated for TB before", value("previous_tb")],
            ["Diabetes", value("diabetes"), "", ""],
            ["Patient name / ID (write by hand)", "", "", ""]]
    details = grid(rows, [36 * mm, half - 36 * mm, 36 * mm, half - 36 * mm])
    details.setStyle(TableStyle([("SPAN", (0, 6), (1, 6)), ("SPAN", (2, 6), (3, 6))]))
    section = [Paragraph("Case and clinical context", s["h2"]), details,
               Paragraph("Provided by the user, not used by the models or the summary. The patient's identity is "
                         "never stored by PNEUMYST AI.", s["small"])]

    if notes["cautions"]:
        box = Table([[Paragraph(f"<b>Caution:</b> {c['text']}", s["cell"])] for c in notes["cautions"]],
                    colWidths=[width], hAlign="LEFT")
        box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff4d6")),
                                 ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9a5b00")),
                                 ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
        section += [Spacer(1, 4), box]

    screen = [["WHO four-symptom screen", Paragraph(who_text(notes["who_screen"]), s["cell"])],
              ["TB model (X-ray only)", tb_result(row)],
              ["Cough duration", value("cough_duration")],
              ["Other symptoms", Paragraph(other_symptoms_text(case), s["cell"])]]
    section += [Paragraph("Symptom screen", s["h2"]), grid(screen, [50 * mm, width - 50 * mm]),
                Paragraph("Shown side by side for the clinician. The AI does not combine symptoms with the X-ray "
                          "scores.", s["small"])]
    return section


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "PNEUMYST AI research prototype. Not a diagnosis; a qualified clinician "
                                        "must review the image and the patient's history.")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(row: AnalysisRecord, report: dict, images: dict[str, bytes] | None, app_version: str,
              generated_at: datetime) -> bytes:
    s = styles()
    width = A4[0] - 36 * mm
    story = [Paragraph("PNEUMYST AI - Research Screening Analysis", s["title"])]

    banner = Table([[Paragraph("RESEARCH PROTOTYPE - NOT A DIAGNOSTIC REPORT", s["banner"])]], colWidths=[width])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#b3261e")),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [banner, Spacer(1, 6)]

    details = [["Analysis ID", row.id],
               ["Analysed (UTC)", row.created_at.strftime("%Y-%m-%d %H:%M:%S")],
               ["Report generated (UTC)", generated_at.strftime("%Y-%m-%d %H:%M:%S")],
               ["Image fingerprint (SHA-256)", row.image_sha256[:16] + "..."],
               ["Image format", row.image_format.upper()],
               ["App version", app_version]]
    story.append(grid(details, [50 * mm, width - 50 * mm]))

    story += context_section(row, s, width)

    story.append(Paragraph("Summary", s["h2"]))
    source = (f"Written by {report['model']} and checked by rule-based guardrails." if report["source"] == "llm"
              else "Template summary written by fixed rules.")
    story += [Paragraph(escape(report["summary"]), s["body"]), Spacer(1, 2), Paragraph(escape(source), s["small"])]

    heading = Paragraph("Images", s["h2"])
    if images and "image" in images:
        size = (width - 9 * mm) / (MAX_HEATMAPS + 1)
        cells = [image_cell(images["image"], "X-ray as analysed (320 px)", s, size)]
        cells += [image_cell(images[f["finding"]], f"{f['display_name']} - {band_word(f)}", s, size)
                  for f in chosen_heatmaps(row, images)]
        row_table = Table([cells], colWidths=[size + 2 * mm] * len(cells), hAlign="LEFT")
        row_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story += [KeepTogether([heading, row_table]), Spacer(1, 3),
                  Paragraph("Red marks the regions that raised a score the most when they were hidden. A heatmap "
                            "shows what influenced the model, not the outline of a lesion.", s["small"])]
    else:
        story += [heading, Paragraph("Images not stored for this analysis.", s["body"])]

    rows = [["Finding", "Area", "Band", "Score", "Possible from", "Likely from", "Heatmap"]]
    for f in ordered(row.findings):
        score = pct(f["score"]) if f["score"] is not None else "-"
        possible = pct(f["possible_threshold"]) if f["possible_threshold"] is not None else "-"
        likely = pct(f["likely_threshold"]) if f["likely_threshold"] is not None else "-"
        word = band_word(f)
        colour = BAND_COLOURS.get(f["band"], MUTED).hexval()[2:]
        rows.append([Paragraph(f["display_name"], s["cell"]), Paragraph(f["group"], s["cell"]),
                     Paragraph(f'<font color="#{colour}"><b>{word}</b></font>', s["cell"]), score, possible, likely,
                     HEATMAP_TEXT.get(f["heatmap"], f["heatmap"])])
    findings_table = grid(rows, [35 * mm, 28 * mm, 26 * mm, 17 * mm, 22 * mm, 20 * mm, width - 148 * mm], header=True)
    story += [KeepTogether([Paragraph("Findings", s["h2"]), findings_table]),
              Paragraph("Scores are calibrated model outputs. Each band compares the score with thresholds fixed on "
                        "validation data.", s["small"])]

    model_rows = [["Model", "Checkpoint (SHA-256)"]]
    model_rows += [[m["name"], (m["checkpoint_sha256"] or "unavailable")[:16]] for m in row.models]
    stages = ", ".join(f"{name}: {status}" for name, status in ordered_stages(row.stages).items())
    errors = ", ".join(f"{name}: {code}" for name, code in row.errors.items()) or "none"
    story += [KeepTogether([Paragraph("Models and processing", s["h2"]),
                            grid(model_rows, [60 * mm, width - 60 * mm], header=True)]),
              Spacer(1, 3), Paragraph(f"<b>Stages:</b> {stages}", s["small"]),
              Paragraph(f"<b>Errors:</b> {errors}", s["small"])]

    story.append(Paragraph("Limitations", s["h2"]))
    story += [Paragraph(f"- {text}", s["cell"]) for text in LIMITATIONS]

    review = [["Reviewer name", ""], ["Decision / comments", ""], ["Signature", ""], ["Date", ""]]
    review_table = Table(review, colWidths=[50 * mm, width - 50 * mm], rowHeights=[8 * mm, 14 * mm, 8 * mm, 8 * mm],
                         hAlign="LEFT")
    review_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8c8c8")),
                                      ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(KeepTogether([Paragraph("Clinician review", s["h2"]),
                               Paragraph("To be completed by a qualified clinician. The AI never fills in this "
                                         "section.", s["small"]), Spacer(1, 3), review_table]))

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=15 * mm,
                            bottomMargin=18 * mm, title=f"PNEUMYST AI screening analysis {row.id}",
                            author="PNEUMYST AI")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buffer.getvalue()
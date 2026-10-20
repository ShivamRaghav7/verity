"""Build the beginner handbook from Markdown without training or model inference.

Run with the bundled document Python (ReportLab/Pillow/pypdf), or requirements-report.txt.
The formal frozen scientific report and all model/scoring sources stay unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Preformatted, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/BEGINNER_GUIDE.md"
OUTPUT = ROOT / "output/pdf/EditGuard-Beginner-Handbook.pdf"
INK = colors.HexColor("#172e38")
TEAL = colors.HexColor("#176965")
PALE = colors.HexColor("#eef5f5")
RULE = colors.HexColor("#d9d9d9")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def architecture() -> Drawing:
    """Small repository-native vector diagram, shared with the Markdown guide."""
    drawing = Drawing(480, 302)

    def box(x, y, width, height, title, subtitle, fill=PALE):
        drawing.add(Rect(x, y, width, height, rx=5, ry=5, fillColor=fill, strokeColor=RULE))
        drawing.add(String(x + width / 2, y + height - 18, title, textAnchor="middle",
                           fontName="Helvetica-Bold", fontSize=10, fillColor=INK))
        drawing.add(String(x + width / 2, y + height - 33, subtitle, textAnchor="middle",
                           fontName="Helvetica", fontSize=8, fillColor=INK))

    def arrow(x1, y1, x2, y2):
        drawing.add(Line(x1, y1, x2, y2, strokeColor=TEAL, strokeWidth=1.1))
        if y2 < y1:
            points = [x2, y2, x2 - 3, y2 + 6, x2 + 3, y2 + 6]
        else:
            points = [x2, y2, x2 - 6, y2 - 3, x2 - 6, y2 + 3]
        drawing.add(Polygon(points, fillColor=TEAL, strokeColor=TEAL))

    box(116, 249, 248, 49, "Question and requested scope", "Product, plan, version and as-of date")
    arrow(240, 249, 240, 232)
    box(75, 179, 330, 53, "Candidate retrieval and deterministic checks",
        "Approved memory, identity, dates and conflict resolution")
    arrow(240, 179, 240, 161)
    box(116, 110, 248, 51, "Trained scope classifier", "Does an eligible correction apply?")
    arrow(169, 110, 78, 80)
    arrow(240, 110, 240, 80)
    arrow(311, 110, 402, 80)
    box(2, 20, 151, 60, "Applicable", "Trained corrected answer")
    box(164, 20, 151, 60, "Not applicable", "Frozen base answer")
    box(326, 20, 151, 60, "Ambiguous or conflicting", "Clarification or resolution")
    drawing.add(String(240, 2, "Sources support an answer; generated prose still requires evaluation.",
                       textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=INK))
    return drawing


def inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"`([^`]+)`", r'<font name="Courier" size="8.4">\1</font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#176965">\1</link>', text)
    return text


class HandbookTemplate(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and getattr(flowable, "chapter_key", None):
            title = flowable.getPlainText()
            key = flowable.chapter_key
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, 0, False)
            self.notify("TOCEntry", (0, title, self.page, key))


def evidence_inventory() -> dict[str, str]:
    names = [
        "README.md", "docs/PROJECT_PLAN.md", "docs/ARCHITECTURE.md", "docs/DATASET_CARD.md",
        "docs/MODEL_CARD.md", "docs/FINAL_EVALUATION.md", "docs/FAILURE_ANALYSIS.md", "docs/DEMO.md",
        "reports/final/summary.json", "reports/final/evaluation/manifest.json",
        "reports/final/evaluation/metrics.json", "reports/final/evaluation/short_qa_metrics.json",
        "reports/final/oracle-runtime-recovery/manifest.json", "reports/final/oracle-runtime-recovery/metrics.json",
        "reports/final/neural-http.json", "reports/final/engineering/github-ci-recovered.json",
        "reports/targeted-training-audit.json", "reports/final/protocol-v2.json",
        "editguard/static/app.js", "editguard/static/index.html",
    ]
    # Bind the guide to the existing experiment instead of silently rescoring it.
    manifest = json.loads((ROOT / "reports/final/evaluation/manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["matched_prediction_records"] == 4080 and manifest["oracle_records"] == 384
    for filename, wanted in manifest["outputs_sha256"].items():
        assert sha(ROOT / "reports/final/evaluation" / filename) == wanted, filename
    recovered = json.loads((ROOT / "reports/final/oracle-runtime-recovery/manifest.json").read_text(encoding="utf-8"))
    assert recovered["status"] == "completed" and recovered["durable_attempt_reservations"] == 319
    for filename, wanted in recovered["outputs_sha256"].items():
        assert sha(ROOT / "reports/final/oracle-runtime-recovery" / filename) == wanted, filename
    return {name: sha(ROOT / name) for name in names}


def build() -> dict:
    sources = evidence_inventory()
    source_text = SOURCE.read_text(encoding="utf-8")
    image_sources = {}
    styles = getSampleStyleSheet()
    body = ParagraphStyle("GuideBody", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.1,
                          leading=14.4, spaceAfter=8, textColor=colors.black,
                          splitLongWords=True, allowWidows=0, allowOrphans=0)
    title_style = ParagraphStyle("GuideTitle", parent=body, fontName="Helvetica-Bold", fontSize=28,
                                leading=33, spaceAfter=16, textColor=colors.black)
    chapter_style = ParagraphStyle("GuideChapter", parent=body, fontName="Helvetica-Bold", fontSize=18,
                                  leading=22, spaceAfter=15, keepWithNext=True, textColor=colors.black)
    sub_style = ParagraphStyle("GuideSub", parent=body, fontName="Helvetica-Bold", fontSize=12,
                              leading=16, spaceBefore=10, spaceAfter=6, keepWithNext=True, textColor=colors.black)
    caption = ParagraphStyle("GuideCaption", parent=body, fontSize=8.8, leading=12, textColor=INK, spaceAfter=12)
    cell = ParagraphStyle("GuideCell", parent=body, fontSize=8.6, leading=11.6, spaceAfter=0)
    header_cell = ParagraphStyle("GuideTableHead", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    code_style = ParagraphStyle("GuideCode", parent=body, fontName="Courier", fontSize=8.6,
                                leading=12.5, leftIndent=0, rightIndent=0, spaceBefore=5, spaceAfter=12)
    width, height = A4
    content_width = width - 112
    document = HandbookTemplate(str(OUTPUT), pagesize=A4, rightMargin=56, leftMargin=56,
                                 topMargin=51, bottomMargin=50,
                                 title="Understanding EditGuard from the Ground Up", author="EditGuard project team")

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(INK)
        canvas.drawString(56, height - 29, "EDITGUARD  |  LEARNING HANDBOOK")
        canvas.drawString(56, 25, "Controlled demonstration dataset  |  20 October 2026")
        canvas.drawRightString(width - 56, 25, str(doc.page))
        canvas.restoreState()

    document.addPageTemplates(PageTemplate(id="guide", frames=[Frame(56, 50, content_width, height - 101,
                                                                          leftPadding=0, rightPadding=0,
                                                                          topPadding=0, bottomPadding=0)], onPage=decorate))
    lines = source_text.splitlines()
    story = []
    index = 0
    toc_inserted = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            story += [Spacer(1, 48), Paragraph(inline(line[2:]), title_style)]
            index += 1
        elif line.startswith("## "):
            if not toc_inserted:
                story += [PageBreak(), Paragraph("Reading map", chapter_style),
                          Paragraph("Read in order for the full explanation. Use the chapter links or PDF bookmarks to return to a topic.", body)]
                toc = TableOfContents()
                toc.levelStyles = [ParagraphStyle("GuideTOC", parent=body, fontSize=10,
                                                 leading=14, spaceBefore=5, leftIndent=0,
                                                 firstLineIndent=0, rightIndent=26)]
                story += [toc]
                toc_inserted = True
                story.append(PageBreak())
            else:
                story += [CondPageBreak(250), Spacer(1, 16)]
            heading = Paragraph(inline(line[3:]), chapter_style)
            heading.chapter_key = "chapter-" + hashlib.sha256(line.encode()).hexdigest()[:12]
            story.append(heading)
            index += 1
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), sub_style))
            index += 1
        elif line.startswith("```"):
            index += 1
            code = []
            while index < len(lines) and not lines[index].startswith("```"):
                code.append(lines[index])
                index += 1
            story.append(Preformatted("\n".join(code), code_style))
            index += 1
        elif line.startswith("|"):
            records = []
            while index < len(lines) and lines[index].lstrip().startswith("|"):
                records.append([x.strip() for x in lines[index].strip().strip("|").split("|")])
                index += 1
            records = [r for r in records if not all(re.fullmatch(r":?-+:?", x) for x in r)]
            count = len(records[0])
            if count == 4:
                fractions = [0.35, 0.14, 0.37, 0.14]
            elif count == 3:
                fractions = [0.28, 0.48, 0.24] if "Outcome" in records[0][0] else [0.48, 0.26, 0.26]
            else:
                fractions = [0.29, 0.71]
            rows = [[Paragraph(inline(value), header_cell if row == 0 else cell)
                     for value in record] for row, record in enumerate(records)]
            table = Table(rows, colWidths=[content_width * f for f in fractions], repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), INK), ("GRID", (0, 0), (-1, -1), .45, RULE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            if len(records) <= 4:
                story.append(KeepTogether([table, Spacer(1, 12)]))
            else:
                story += [table, Spacer(1, 12)]
        elif line.startswith("!["):
            match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if not match:
                raise ValueError(line)
            label, filename = match.groups()
            if filename.endswith("beginner-architecture.svg"):
                story += [KeepTogether([architecture(), Paragraph(inline(label), caption)])]
            else:
                path = SOURCE.parent / filename
                image_sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
                img = Image(str(path))
                factor = content_width / img.imageWidth
                img.drawWidth, img.drawHeight = content_width, img.imageHeight * factor
                story += [KeepTogether([img, Paragraph(inline(label), caption)])]
            index += 1
        elif re.match(r"^(\d+\.|-) ", line):
            marker, text = line.split(" ", 1)
            bullet = "-" if marker == "-" else marker
            story.append(Paragraph(inline(text), ParagraphStyle("GuideList", parent=body, leftIndent=16,
                                                                firstLineIndent=0, bulletIndent=0), bulletText=bullet))
            index += 1
        else:
            text = [line]
            index += 1
            while index < len(lines) and lines[index].strip() and not re.match(r"^(#|\||```|!\[|\d+\. |\- )", lines[index]):
                text.append(lines[index].strip())
                index += 1
            story.append(Paragraph(inline(" ".join(text)), body))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    asset = SOURCE.parent / "assets/beginner-architecture.svg"
    asset.parent.mkdir(parents=True, exist_ok=True)
    renderSVG.drawToFile(architecture(), str(asset))
    asset.write_bytes(asset.read_bytes().replace(b"\r\n", b"\n"))
    document.multiBuild(story)
    from pypdf import PdfReader
    reader = PdfReader(str(OUTPUT))
    metadata = {"schema_version": 1, "purpose": "beginner_learning_handbook", "prepared_date": "2026-10-20",
                "source_markdown": str(SOURCE.relative_to(ROOT)).replace("\\", "/"), "source_markdown_sha256": sha(SOURCE),
                "builder_sha256": sha(Path(__file__)), "pdf_sha256": sha(OUTPUT), "pages": len(reader.pages),
                "source_evidence_sha256": sources, "asset_sha256": {str(asset.relative_to(ROOT)).replace("\\", "/"): sha(asset), **image_sources},
                "training_or_inference_executed": False, "frozen_results_modified": False,
                "limitations": ["Educational explanation; not a new experiment, new scorer or human semantic adjudication.",
                                "Historical frozen reports preserve their original pending and runtime-error records."]}
    OUTPUT.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(OUTPUT), "pages": metadata["pages"], "words": len(source_text.split())}, indent=2))
    return metadata


if __name__ == "__main__":
    build()

"""Build the final visual-editing PDF from completed, saved measurements.

This is a presentation-only script: it never loads a model, runs an experiment,
changes a scorer, or supplies expected results. Run with the bundled PDF Python.
The PDF skill's artifact marker must be run by the authoring workflow first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("base", "prompt_only", "updated_rag", "mend_context", "ft_last_layer")
CATEGORIES = ("direct", "paraphrase", "alternate_image", "text_locality", "visual_locality")
LABEL = {"base": "Base", "prompt_only": "Prompt only", "updated_rag": "Updated RAG",
         "mend_context": "MEND (adapted)", "ft_last_layer": "FT (adapted)",
         "direct": "Direct", "paraphrase": "Paraphrase", "alternate_image": "Other photo",
         "text_locality": "Text locality", "visual_locality": "Visual locality"}
SOURCE_DOCS = ("VLKEB_RESULTS.md", "VLKEB_MODEL_CARD.md", "VLKEB_DATASET_CARD.md",
               "VLKEB_ARCHITECTURE.md", "VLKEB_DEMO.md", "VLKEB_WORKBENCH.md",
               "VLKEB_EVALUATION_PROTOCOL.md", "VLKEB_UPSTREAM_RESULTS.md")
REFERENCES = (
    ("VLKEB paper: benchmark and published results", "https://arxiv.org/abs/2403.07350"),
    ("MEND: learned model editing", "https://arxiv.org/abs/2110.11309"),
    ("Pinned VLKEB source", "https://github.com/VLKEB/VLKEB/tree/10951b7b3788928f578b73f07eac9e1eaa0316f3"),
    ("Pinned benchmark dataset", "https://huggingface.co/datasets/HymanH/VLKEB-data/tree/cf2d0abe73cb638fcf368d8bc5f9ac3caa204d5a"),
    ("BLIP-2 architecture", "https://arxiv.org/abs/2301.12597"),
    ("Pinned MiniLM sentence encoder", "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41"),
    ("Project source and evidence", "https://github.com/ShivamRaghav7/verity"),
)


def read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite measured value, got {value!r}")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Measured value out of range: {value}")
    return value


def validate_inputs(evaluation, figures):
    report = read_json(evaluation)
    if (report.get("execution_status") != "completed" or report.get("worker_return_code") != 0
            or report.get("completed_edit_units") != 20 or report.get("unexecuted_edit_units") != 0):
        raise ValueError("Final PDF requires a completed 20-edit report; incomplete results are never filled in")
    aggregate = report["aggregate"]
    if "own_measured" not in aggregate.get("provenance", ""):
        raise ValueError("Report must identify its own measured provenance")
    if report.get("meta_training_executed") is not False:
        raise ValueError("Unexpected editor meta-training provenance")
    for method in METHODS:
        for category in CATEGORIES:
            cell = aggregate["scores"][method][category]
            if cell["denominator"] != 20:
                raise ValueError("Every category must retain the 20-edit denominator")
            correct = finite(cell["correct"], maximum=20)
            accuracy = finite(cell["accuracy"], maximum=1)
            if int(correct) != correct or abs(accuracy - correct / 20) > 1e-9:
                raise ValueError("Saved accuracy must agree with its integer numerator")
            for key in ("abstentions", "errors_or_unexecuted"):
                if int(finite(cell[key], maximum=20)) != cell[key]:
                    raise ValueError("Saved outcome counts must be integers")
    figure_data = read_json(figures / "figure_data.json")
    if figure_data.get("evidence_kind") != "own measured run evidence":
        raise ValueError("Final PDF cannot use synthetic figure fixtures")
    if figure_data["sources"]["evaluation"]["sha256"] != sha256(evaluation):
        raise ValueError("Figures were not generated from this exact evaluation JSON")
    for name in ("accuracy.png", "query_latency.png"):
        if not (figures / name).is_file():
            raise FileNotFoundError(figures / name)
    return report


def plain(value):
    return str(value).replace("\u2014", " - ").replace("\u2013", "-").replace("\u2011", "-")


def pct(value):
    return "Not defined" if value is None else f"{value * 100:.1f}%"


def sec(value):
    return "Not recorded" if value is None else f"{value:.3f}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def register_report_fonts():
    """Use an installed sans-serif family without requiring font downloads."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.pdfmetrics import Font
    from reportlab.pdfbase.ttfonts import TTFont

    names = ("ReportSans", "ReportSans-Bold", "ReportSans-Italic", "ReportSans-BoldItalic")
    candidates = [
        (Path("C:/Windows/Fonts"), ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf")),
        *[(directory, ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
                       "DejaVuSans-BoldOblique.ttf")) for directory in (
            Path("/usr/share/fonts/truetype/dejavu"), Path("/usr/share/fonts/dejavu"),
            Path("/usr/local/share/fonts/dejavu"), Path.home() / ".local/share/fonts",
        )],
    ]
    family = "Helvetica (built in)"
    for directory, filenames in candidates:
        paths = [directory / filename for filename in filenames]
        if all(path.is_file() for path in paths):
            for name, path in zip(names, paths):
                pdfmetrics.registerFont(TTFont(name, str(path)))
            family = "Arial" if filenames[0] == "arial.ttf" else "DejaVu Sans"
            break
    else:
        for name, face in zip(names, ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique")):
            pdfmetrics.registerFont(Font(name, face, "WinAnsiEncoding"))
    pdfmetrics.registerFontFamily("ReportSans", normal=names[0], bold=names[1],
                                  italic=names[2], boldItalic=names[3])
    return family


def make_pdf(args, report):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (Flowable, Image, KeepTogether, PageBreak,
                                  Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

    register_report_fonts()
    navy, teal, gray = colors.HexColor("#102C43"), colors.HexColor("#007D82"), colors.HexColor("#536475")
    paper, pale = colors.HexColor("#F1F6F9"), colors.HexColor("#E4F4F1")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportBody", fontName="ReportSans", fontSize=10.2, leading=15,
                             textColor=navy, spaceAfter=8, alignment=TA_LEFT))
    styles.add(ParagraphStyle("ReportSmall", parent=styles["ReportBody"], fontSize=8.25, leading=11.5, spaceAfter=5))
    styles.add(ParagraphStyle("ReportTiny", parent=styles["ReportBody"], fontSize=7.5, leading=10, spaceAfter=3))
    styles.add(ParagraphStyle("ReportHeading", parent=styles["ReportBody"], fontName="ReportSans-Bold",
                             fontSize=24, leading=28, spaceAfter=16))
    styles.add(ParagraphStyle("ReportSubhead", parent=styles["ReportBody"], fontName="ReportSans-Bold",
                             fontSize=12, leading=16, spaceBefore=8, spaceAfter=7))
    styles.add(ParagraphStyle("ReportKicker", parent=styles["ReportBody"], fontName="ReportSans-Bold",
                             fontSize=9, leading=12, textColor=teal, spaceAfter=9))
    styles.add(ParagraphStyle("ReportCover", parent=styles["ReportHeading"], fontSize=36, leading=40, spaceAfter=18))
    styles.add(ParagraphStyle("ReportCell", parent=styles["ReportBody"], fontSize=8.2, leading=11, spaceAfter=0))
    styles.add(ParagraphStyle("ReportCellSmall", parent=styles["ReportCell"], fontSize=7.4, leading=10))
    width, height = A4
    margin = 19 * mm
    usable = width - 2 * margin
    story = []
    aggregate = report["aggregate"]
    build_commit = git_commit()
    generated = datetime.now(timezone.utc).strftime("%d %B %Y, %H:%M UTC")

    def p(text, style="ReportBody"):
        return Paragraph(plain(text), styles[style])

    def add(text, style="ReportBody"):
        story.append(p(text, style))

    def title(number, text, subtitle=None):
        if story:
            story.append(PageBreak())
        add(f"EDITGUARD / {number:02d}", "ReportKicker")
        add(text, "ReportHeading")
        if subtitle:
            add(subtitle)

    def sub(text):
        add(text, "ReportSubhead")

    def note(text, bg=pale):
        block = Table([[p(text, "ReportSmall")]], colWidths=[usable])
        block.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                                   ("BOX", (0, 0), (-1, -1), .4, colors.HexColor("#C8DFE4")),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 11),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 11),
                                   ("TOPPADDING", (0, 0), (-1, -1), 9),
                                   ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
        story.extend([block, Spacer(1, 10)])

    def table(headers, rows, widths=None, small=False):
        style = "ReportCellSmall" if small else "ReportCell"
        data = [[p(f"<b>{escape(plain(h))}</b>", style) for h in headers]]
        data += [[p(escape(plain(c)), style) for c in row] for row in rows]
        value = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        value.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), navy), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [paper, colors.white]),
            ("LINEBELOW", (0, 0), (-1, 0), .7, teal), ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#DDE5EC")),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        # Paragraphs carry explicit colors, so style the header paragraphs too.
        for paragraph in data[0]:
            paragraph.style = ParagraphStyle("white-cell", parent=paragraph.style, textColor=colors.white)
        story.extend([value, Spacer(1, 10)])

    def figure(path, max_height, caption):
        img = Image(str(path))
        scale = min(usable / img.imageWidth, max_height / img.imageHeight)
        img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
        img.hAlign = "CENTER"
        story.extend([img, Spacer(1, 5), p(caption, "ReportSmall"), Spacer(1, 7)])

    class Pipeline(Flowable):
        def __init__(self):
            super().__init__(); self.width = usable; self.height = 145

        def draw(self):
            c = self.canv
            boxes = [(0, 89, 112, 47, ["Image + question", "Target-free query"]),
                     (150, 89, 177, 47, ["BLIP-2 visual language model", "EVA > Q-Former > OPT-2.7B"]),
                     (365, 89, usable - 365, 47, ["Short answer", "Up to 24 tokens"]),
                     (0, 6, 153, 48, ["Operator correction", "MEND or 15-step FT"]),
                     (183, 6, 151, 48, ["Inactive candidate", "Preview + probe + restore"]),
                     (364, 6, usable - 364, 48, ["Approve / rollback", "Own saved patch only"])]
            for x, y, w, h, lines in boxes:
                c.setFillColor(paper); c.setStrokeColor(teal); c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
                for i, line in enumerate(lines):
                    c.setFillColor(navy if i == 0 else gray)
                    c.setFont("ReportSans-Bold" if i == 0 else "ReportSans", 8.2 if i == 0 else 7.4)
                    c.drawCentredString(x + w / 2, y + h - 17 - i * 14, line)
            for x1, x2, y in ((112, 150, 112), (327, 365, 112), (153, 183, 29), (334, 364, 29)):
                c.setStrokeColor(teal); c.setLineWidth(1.3); c.line(x1 + 3, y, x2 - 4, y)
                c.line(x2 - 9, y + 3, x2 - 4, y); c.line(x2 - 9, y - 3, x2 - 4, y)

    # 1. A short, concrete opening with actual completion counts.
    add("RESEARCH REPORT / VISUAL MODEL MAINTENANCE", "ReportKicker")
    story.append(Spacer(1, 17))
    add("EditGuard", "ReportCover")
    add("A reversible workbench for visual model corrections", "ReportHeading")
    add("Published model editing, bounded fine-tuning, and retrieval compared using generated answers.")
    story.append(Spacer(1, 14))
    table(["Measured experiment", "Completed scope", "Inference rule"],
          [["20 edit lineages", "5 methods x 5 cases = 500 assigned answers", "Greedy generation, 24-token cap"]],
          [usable * .29, usable * .38, usable * .33])
    note("<b>The result in one sentence.</b> This project measures whether a requested visual association changes, whether that change transfers to new wording or photographs, and what unrelated behavior it disturbs. Review and exact rollback remain necessary even when an edit succeeds.")
    sub("What was built")
    add("A local workbench wraps the published VLKEB/MEND research code and BLIP-2 model. It can propose an actual parameter update, save an inactive candidate, show generated before/after answers and an unrelated probe, require explicit operator approval, and restore original weights exactly.")
    sub("What was measured")
    add(f"The saved final report records <b>{report['completed_edit_units']}/20 completed edit units</b>, {report['unexecuted_edit_units']} unexecuted units and a completed worker. It compares base, prompt only, updated visual RAG, adapted MEND and adapted last-layer FT. All assigned questions remain in denominators.")
    sub("How to read this report")
    add("Pages 2-4 explain the model and the fixed experiment. Pages 5-7 show results, side effects and timing. Pages 8-11 explain failure cases, the application, reproduction and limits. Published paper numbers are not inserted into our result tables.")
    add(f"Prepared {generated}. Measurements are read from a saved JSON report; this document runs no model or training.", "ReportSmall")

    # 2. Model + project boundary.
    title(2, "From an image to a short answer", "The base model combines a visual encoder and a language model. Editing changes selected internal weights; retrieval changes the information supplied to an unchanged model.")
    story.extend([Pipeline(), Spacer(1, 12)])
    table(["Component", "Beginner's explanation", "Who supplied it"], [
        ["EVA image encoder", "Turns the photograph into visual features.", "Published pretrained weights"],
        ["Q-Former", "Condenses visual features into vectors usable by the language model.", "Published BLIP-2 components"],
        ["OPT-2.7B", "Produces the answer one token at a time.", "Published pretrained weights"],
        ["MEND transform", "Uses an edit gradient to propose a targeted weight update.", "Published pretrained editor"],
        ["Our workbench + evaluator", "Adapters, experiment controls, saved evidence, approval and exact restoration.", "This project"],
    ], [usable * .21, usable * .51, usable * .28])
    note("<b>Attribution boundary.</b> We did not pretrain the foundation model or meta-train MEND. FT is real per-correction optimization; using a pretrained editor is a different kind of work.")
    add("The standard query contains the image and question. No expected answer is supplied to the ordinary base/MEND/FT generation call. Prompt and RAG baselines intentionally receive the current correction as a demonstration, which is their declared behavior.")

    # 3. Methods.
    title(3, "Five ways to handle one correction")
    table(["Method", "What changes?", "What the query receives"], [
        ["Base", "Nothing", "Image + question"],
        ["Prompt only", "Input prefix; weights fixed", "Current correction as a QA example, then the actual query"],
        ["Updated visual RAG", "One-correction memory; weights fixed", "QA example when image/question scope score accepts; otherwise base prompt"],
        ["MEND (adapted)", "Six OPT MLP matrices in layers 29-31", "Actual query after one learned-transform parameter update"],
        ["FT (adapted)", "Two OPT layer-31 matrices", "Actual query after 15 AdamW steps"],
    ], [usable * .2, usable * .34, usable * .46])
    sub("Adaptations are part of the comparison")
    add("<b>Question format:</b> every query uses <i>Question: ... Answer:</i>. <b>Context-aligned labels:</b> target labels are taken from the actual joined question/target tokenization. The development diagnosis found six standalone target tokens but four after a separating space; assuming the standalone count can supervise the wrong positions.")
    add("<b>FT:</b> 15 AdamW steps at learning rate 0.0002 update layer 31 fc1/fc2 only. FP32 master optimizer copies retain the released base precision. A supervised newline teaches termination. MEND retains its context-aligned target-only objective, so objectives are not identical.")
    add("<b>RAG:</b> scope combines MiniLM question cosine and BLIP visual-prefix cosine equally. A threshold selected on TRAIN rows 2-9 gates the current correction. The encoders are frozen; the score is not a calibrated probability. Text-only locality is outside the visual memory scope and uses the base model.")
    note("These are documented research adaptations. The experiment does not reproduce every paper setting or isolate editor architecture under identical objectives.")

    # 4. Design and definitions.
    title(4, "A fixed, finite evaluation", "Twenty edit lineages are selected before model inference, with five linked questions per lineage. Each method receives the same assigned cases.")
    table(["Case", "What it tests", "Assigned per method"], [
        ["Direct", "The requested target on the correction image/question", "20"],
        ["Paraphrase", "Transfer to the authors' alternate wording", "20"],
        ["Other photo", "Transfer to another photograph of the source entity", "20"],
        ["Text locality", "An unrelated question with no image", "20"],
        ["Visual locality", "A similar but unrelated image/question", "20"],
    ], [usable * .2, usable * .63, usable * .17])
    add("The source is the authors' benchmark, not a new independent dataset. Source image-directory IDs found in any training edit, alternate-image or visual-locality slot are excluded. Stable lineage hashes determine the first 20 distinct source entities. This is a small nonrandom slice, not a random population sample.")
    table(["Quantity", "Meaning"], [
        ["Strict accuracy", "Entire generated answer matches the target after fixed normalization. A target merely mentioned in repetitive prose is insufficient."],
        ["Abstention / error", "Empty answers, declared nonanswers, runtime errors and missing cases remain failures in assigned denominators."],
        ["Locality correctness", "An unrelated query still receives its factual target answer."],
        ["Preservation", "The answer stays verbatim unchanged. Conditional retention uses only originally correct base answers."],
        ["Paired interval", "95% bootstrap interval over complete editing units, using 10,000 resamples; linked questions stay together."],
    ], [usable * .25, usable * .75])
    note("The source, selection, threshold, code, scorer and settings were sealed before final inference. No post-test answer aliases or scoring changes are introduced by this PDF.")

    # 5. Final results: read only from the report and source-bound figures.
    title(5, "Measured generated-answer results")
    figure(args.figures / "accuracy.png", 233, "Figure 1. Saved final measurements, 20 edits per cell. Locality columns measure factual correctness; empty/error cases remain in denominators.")
    rows = []
    for method in METHODS:
        cells = aggregate["scores"][method]
        rows.append([LABEL[method], *[f"{cells[c]['correct']}/20" for c in CATEGORIES],
                     str(sum(cells[c]["abstentions"] for c in CATEGORIES)),
                     str(sum(cells[c]["errors_or_unexecuted"] for c in CATEGORIES))])
    table(["Method", "Direct", "Para-phrase", "Other photo", "Text local", "Visual local", "Abstain", "Error"],
          rows, [usable * .22] + [usable * .112] * 5 + [usable * .11] * 2, small=True)
    add("Counts are the primary reading: one additional correct case changes a category's accuracy by five percentage points. A high direct score does not establish transfer or safety on unrelated questions. Compare the category pattern before making a general ranking.")
    add("Abstain and Error columns sum the five categories, so each method has 100 assigned outcomes. The fixed whole-answer rule may reject a useful shorter answer; that limitation is disclosed rather than repaired after results are observed.", "ReportSmall")

    # 6. Side effects, routing and uncertainty.
    title(6, "Scope, side effects and uncertainty")
    retention_rows = []
    for method in METHODS:
        for category in ("text_locality", "visual_locality"):
            r = aggregate["locality_preservation"][method][category]
            n = r["baseline_correct_denominator"]
            retention_rows.append([LABEL[method], LABEL[category], f"{r['baseline_correct_factual_retained']}/{n}",
                                   f"{r['baseline_correct_answer_preserved']}/{n}", f"{r['unconditional_answer_preserved']}/20"])
    table(["Method", "Probe family", "Fact retained / base correct", "Verbatim / base correct", "Verbatim / all"],
          retention_rows, [usable * .22, usable * .2, usable * .2, usable * .2, usable * .18], small=True)
    add("A zero base-correct denominator gives no conditional retention estimate. Unchanged wrong answers are not factual success. Normalized factual retention and verbatim preservation can differ.", "ReportSmall")
    retrieval = aggregate["rag_retrieval"]
    sub("Updated RAG: routing is separate from answer quality")
    table(["Accepted in scope", "Missed in scope", "Applied outside scope", "Correctly rejected"],
          [[f"{retrieval['tp']}/60", f"{retrieval['fn']}/60", f"{retrieval['fp']}/40", f"{retrieval['tn']}/40"]],
          [usable / 4] * 4)
    add(f"Routing precision: {pct(retrieval['precision'])}; applicable recall: {pct(retrieval['applicable_recall'])}; false application rate: {pct(retrieval['false_application_rate'])}. Routing errors/unexecuted: {retrieval['routing_errors_or_unexecuted']}. A correct route does not guarantee a correct generated answer.", "ReportSmall")
    sub("Selected paired differences: direct question")
    selected = [r for r in aggregate["paired_differences"] if r["category"] == "direct"
                and ((r["first"] == "ft_last_layer" and r["second"] == "base")
                     or (r["first"] == "updated_rag" and r["second"] == "ft_last_layer"))]
    paired_rows = []
    for row in selected:
        lo, hi = row["paired_edit_bootstrap_95_percent_interval"]
        paired_rows.append([f"{LABEL[row['first']]} minus {LABEL[row['second']]}",
                            f"{row['difference'] * 100:+.1f} pp", f"[{lo * 100:+.1f}, {hi * 100:+.1f}] pp"])
    table(["Comparison", "Difference", "95% paired interval"], paired_rows, [usable * .5, usable * .2, usable * .3])
    add("pp means percentage points. All categories and registered method pairs remain in the JSON report and paired-interval figure. These two rows are fixed descriptive views, not significance-selected claims. With only 20 units, uncertainty and dataset bias remain substantial.", "ReportSmall")

    # 7. Timing.
    title(7, "Time and memory are separate outcomes")
    figure(args.figures / "query_latency.png", 224, "Figure 2. Query latency from saved measurements. Categories are kept separate; sample counts and the exact timing rule are recorded below.")
    latency_rows = []
    for method in ("mend_context", "ft_last_layer"):
        cell = aggregate["edit_latency"][method]
        latency_rows.append([LABEL[method] + " edit", str(cell["sample_count"]), sec(cell["p50_seconds"]), sec(cell["p95_seconds"])])
    cell = aggregate["rag_index_latency"]
    latency_rows.append(["RAG reference index", str(cell["sample_count"]), sec(cell["p50_seconds"]), sec(cell["p95_seconds"])])
    table(["Operation", "Timed n", "p50 seconds", "p95 seconds"], latency_rows,
          [usable * .4, usable * .16, usable * .22, usable * .22])
    add(escape(aggregate["latency_protocol"]), "ReportSmall")
    query_ns = [aggregate["query_latency"][m][c]["sample_count"] for m in METHODS for c in CATEGORIES]
    add(f"Per-category query sample counts range from {min(query_ns)} to {max(query_ns)}. p50 is the median; p95 is an interpolated tail summary, not a worst-case guarantee. Editing and indexing are not hidden inside ordinary query latency.", "ReportSmall")
    resource = report.get("worker_report", {}).get("resources", {})
    if resource:
        gpu = resource.get("gpu", {})
        gpu_memory = (f"{gpu['peak_allocated_bytes'] / 2**30:.3f} / {gpu['peak_reserved_bytes'] / 2**30:.3f} GiB"
                      if "peak_allocated_bytes" in gpu and "peak_reserved_bytes" in gpu else "Not recorded")
        resource_rows = [["Worker elapsed", sec(resource.get("elapsed_seconds")) + " seconds"],
                         ["Sampled process RSS peak", f"{resource['sampled_peak_process_rss_bytes'] / 2**30:.3f} GiB"],
                         ["CUDA allocated / reserved peaks", gpu_memory],
                         ["Recorded device", gpu.get("device", "Not recorded")]]
        table(["Worker resource", "Saved measurement"], resource_rows, [usable * .5, usable * .5])
    add("RSS sampling can miss brief peaks, and CUDA allocator values omit some driver allocations. These measurements do not establish fit on the laptop's 6 GB GPU or predict another host's latency. One GiB is 1,073,741,824 bytes.", "ReportSmall")

    # 8. Development evidence and failures.
    title(8, "Failures that shaped the safeguards", "Development examples explain the engineering decisions. They remain separate from the 20-edit final experiment.")
    table(["TRAIN fixture case", "MEND generated answer", "FT generated answer"], [
        ["Direct; target Albany, New York", "depton, texas", "Albany, New York"],
        ["Paraphrase; same target", "New York, New York", "Albany (strict incomplete answer)"],
        ["Other photograph; same target", "denton, texas", "Albany, New York"],
        ["Unrelated county; target Hidalgo County", "Empty", "Repeated Albany"],
    ], [usable * .35, usable * .3, usable * .35])
    add("The saved context/VQA MEND and FT runs each passed eight mechanical checks, including patch reload in a fresh process and exact restoration. The table still contains failed answers. Persistence is not accuracy, and a successful direct edit can disturb unrelated behavior.")
    sub("Why a perfect token score was not enough")
    add("The loaded MEND patch scored 100% on contextual teacher-forced target tokens for the raw direct question, yet freely generated repeated city text. Teacher forcing places the full target in the model input; free generation must choose its own next tokens and stop. They answer different questions and cannot share an accuracy claim.")
    note("The FT neighbor's baseline answer was already empty and wrong. Changing it to repetition is collateral behavioral change; it does not prove the loss of an originally correct fact. MEND preserving an empty answer is also not factual success.")
    sub("A failure receipt is still evidence")
    add("The first raw/release MEND attempt changed weights and restored them in-process, but its reload worker was killed with return code -9. The complete persistence gate failed. Memory pressure is plausible, but the return code alone does not prove an out-of-memory kill. Missing phase measurements are not replaced with estimates.")
    add("All of these development cases came from the authors' TRAIN annotations and may have been seen during original editor training. They establish concrete feasibility and failure modes, not held-out success rates.")

    # 9. Application and validation boundary.
    title(9, "A reviewable, reversible application")
    table(["Stage", "What the operator sees", "Weight state"], [
        ["Generate", "Actual current answer, selected source and image", "Original or explicitly active state"],
        ["Preview", "Before/candidate answers, target match and one fixed unrelated probe", "Candidate tested, then exact base restoration"],
        ["Review", "Saved question, target, outputs, provenance and hashes", "Inactive candidate"],
        ["Approve + activate", "Explicit operator decision, including any visible warnings", "Own saved patch checked and loaded"],
        ["Rollback", "Restoration result and updated evidence history", "Saved original tensors copied back"],
    ], [usable * .21, usable * .5, usable * .29])
    add("Only one patch may be active. Another preview or activation is rejected until rollback; the UI disables those actions. Clients cannot supply arbitrary patch paths or checkpoint uploads. POST operations require a per-process token. These are local academic controls, not production multi-user authentication.")
    add("The server serializes model operations and binds only to 127.0.0.1. A fresh successful model load starts with original weights and marks formerly active history inactive_after_restart. Historical patches are not silently reapplied. Keep history and exported patches on persistent storage.")
    sub("Evidence boundary")
    add("Eleven application contract tests use a tiny fake backend to check state transitions and failure recovery. They do not establish real-model quality. GitHub CI also built the pinned Linux container, checked offline imports and the CLI, and verified runtime-user history writes. That is CPU packaging evidence, not a full GPU server deployment.")
    if args.app_validation_note:
        note("<b>Recorded application rehearsal.</b> " + escape(args.app_validation_note))
    else:
        note("<b>Separate deployment check.</b> A real-GPU browser/API lifecycle rehearsal is required in addition to experiment receipts. No publicly hosted application or GPU Docker runtime is claimed by this PDF.")
    add("The evaluation dashboard reads a configured local report unchanged. If it is absent, numerical cards are absent. Downloadable evidence makes the displayed claims inspectable; paper results cannot fill a missing local measurement.")

    # 10. Reproduction and evidence.
    title(10, "Reproduce the evidence, not a promise")
    table(["Step", "Concrete artifact or command"], [
        ["Pinned environment", "Python 3.11.9; torch 2.0.1+cu117; exact dependency closure in requirements-vlkeb.txt"],
        ["Verified assets", "25 declared base/editor/fixture inputs; 13,218,483,144 bytes, plus separate pinned MiniLM files"],
        ["Setup", "python scripts/bootstrap_vlkeb_colab.py --fetch-upstream --install"],
        ["Prepare", "Explicit asset download, verification and deterministic selection; fixed threshold-validation receipt"],
        ["Seal then run", "python -m ke_repro.evaluate --seal ; python -m ke_repro.evaluate --run"],
        ["Finite cap", "1,800-second worker cap per attempt; same frozen seal for an interrupted-run resume"],
        ["WorkBench", "artifacts/vlkeb/environment/bin/python -m ke_repro.workbench --port 8003"],
    ], [usable * .27, usable * .73])
    add("The dependency environment is isolated from notebook Python and the Windows tools. Model inference uses verified local files in offline mode. The one-episode planning estimate was approximately 15.5 minutes; it was not a runtime guarantee. No editor meta-training is performed. A complete final attempt requests at most 20 MEND updates and 300 FT optimizer steps.")
    sub("Container and viewer")
    add("On a compatible native Linux NVIDIA host, Docker uses --gpus all and --network host because the application stays on loopback. Mount assets read-only and history writable with the host user ID. GPU container execution remains a separate runtime check. Colab's official notebook port viewer depends on its active runtime and access; it is not public hosting.")
    sub("What to preserve")
    add("Keep the seal, selection checksum, attempt logs, worker report, per-edit rows, evaluation JSON, figure CSV/JSON and source hashes. Model weights and approximately 600 MiB patches remain outside Git. A patch hash identifies a patch but cannot reconstruct missing tensor data.")
    if args.app_evidence:
        add("Application rehearsal receipt: " + escape(str(args.app_evidence.resolve())) + ".", "ReportSmall")
    note("Rights are component-specific: VLKEB/editor, OPT, LAVIS, MiniLM, annotations and photographs have separate terms. OPT's noncommercial research restriction is not removed by another component's permissive code license.")

    # 11. Limits, references and provenance.
    title(11, "Limits, references and provenance")
    add("<b>Scope:</b> 20 nonrandom public benchmark edits and one-edit retrieval memory do not establish arbitrary private-item recognition, open dialogue, lifelong composition or large-memory scalability. Source-directory separation does not rule out foundation-model pretraining or target/locality overlap.", "ReportSmall")
    add("<b>Labels and scoring:</b> authors' labels receive automated structural/checksum review, not independent human semantic review. Whole-answer matching can reject reasonable partial answers. Counterfactual targets are benchmark instructions, not newly verified facts.", "ReportSmall")
    add("<b>Comparisons:</b> contextual labels, prompt formatting, loading precision and FT termination supervision are disclosed adaptations. The prompt-only path lacks a second reference image; MEND and FT objectives differ. Published teacher-forced numbers are not directly comparable with generated-answer scores.", "ReportSmall")
    add("<b>Operations:</b> one fixed preview probe and exact rollback support controlled investigation. They do not guarantee harmless changes, multi-user security or production readiness. GPU memory and timing are host- and workload-specific.", "ReportSmall")
    sub("Primary references")
    for i, (label, url) in enumerate(REFERENCES, 1):
        add(f'{i}. <link href="{escape(url)}" color="#007D82">{escape(label)}</link>', "ReportSmall")
    sub("Auditable inputs")
    add("Final evaluation SHA256: " + sha256(args.evaluation), "ReportTiny")
    add("Sealed experiment SHA256: " + escape(report["seal_sha256"]), "ReportTiny")
    add("Figure data SHA256: " + sha256(args.figures / "figure_data.json"), "ReportTiny")
    add("Presentation builder checkout: " + escape(build_commit), "ReportTiny")
    add("Project source documents: " + ", ".join(SOURCE_DOCS) + ".", "ReportTiny")
    add("The PDF reads the completed report and source-bound figures. It does not recompute model outputs, revise the frozen protocol or introduce missing measurements. Complete per-category latency, paired intervals and raw outcomes remain in the machine-readable evidence.", "ReportSmall")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#DDE5EC")); canvas.setLineWidth(.6)
        canvas.line(margin, height - 12 * mm, width - margin, height - 12 * mm)
        canvas.setFont("ReportSans", 8); canvas.setFillColor(gray)
        canvas.drawString(margin, 11 * mm, "EDITGUARD  |  Own measured evidence; published research core")
        canvas.drawRightString(width - margin, 11 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=margin, leftMargin=margin,
                            topMargin=19 * mm, bottomMargin=20 * mm,
                            title="EditGuard: Visual Model Editing - Final Research Report",
                            author="EditGuard project", subject="Saved generated-answer evaluation and reversible workbench",
                            pageCompression=1)
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--figures", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf/editguard-visual-report.pdf")
    parser.add_argument("--app-evidence", type=Path, help="Optional separately saved real-application rehearsal receipt")
    parser.add_argument("--app-validation-note", help="Concrete measured application validation statement, supplied only after rehearsal")
    parser.add_argument("--check-inputs", action="store_true", help="Validate saved evidence without authoring a PDF")
    args = parser.parse_args(argv)
    for filename in SOURCE_DOCS:
        if not (ROOT / "docs" / filename).is_file():
            raise FileNotFoundError(ROOT / "docs" / filename)
    if args.app_evidence:
        read_json(args.app_evidence)
    if args.app_validation_note and not args.app_evidence:
        parser.error("--app-validation-note requires its saved --app-evidence receipt")
    report = validate_inputs(args.evaluation, args.figures)
    if not args.check_inputs:
        make_pdf(args, report)
    print(json.dumps({"status": "inputs_verified" if args.check_inputs else "created",
                      "evaluation_sha256": sha256(args.evaluation),
                      "output": None if args.check_inputs else str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

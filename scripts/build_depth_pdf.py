"""Typeset the released depth addendum from hash-bound evidence; no model calls."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("R-RAG", "G-RAG", "R-ORACLE", "G-ORACLE")
NAMES = {"R-RAG": "ReasonKE + retrieval", "G-RAG": "Qwen + retrieval",
         "R-ORACLE": "ReasonKE + gold support", "G-ORACLE": "Qwen + gold support"}
OUTCOMES = ("correct", "wrong", "abstain", "execution_error", "unexecuted")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evidence(directory):
    # The shared release gate checks all eight canonical inputs, outcome/depth
    # denominators and each exported figure against the portable host audit.
    from editguard_chat.results import _depth_signature, _depth_verified
    value = json.loads(_depth_verified(str(directory), _depth_signature(directory)))
    report = value["report"]
    if not value["available"] or report["execution_status"] != "completed":
        raise ValueError("Only released, completed depth evidence can be typeset")
    required = {name + ".png" for name in ("depth-accuracy", "category-accuracy",
                 "outcome-composition", "retrieval-and-answer", "latency")}
    if not required.issubset(value["artifacts"]):
        raise ValueError("All five audited scientific PNG figures are required")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("schema") != "editguard-depth-protocol-v1" or protocol.get("status") != "frozen":
        raise ValueError("A frozen depth protocol is required")
    if (protocol.get("final_assigned"), protocol.get("per_main_condition"), protocol.get("oracle_per_model")) != (544, 256, 16):
        raise ValueError("Depth protocol population differs")
    raw = [json.loads((directory / f"final-{model}.json").read_text(encoding="utf-8"))
           for model in ("reasonke", "instruction")]
    return report, protocol, [row for receipt in raw for row in receipt["records"]]


def create(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preserve existing PDF; choose a new output path")
    report, protocol, records = _evidence(directory)
    input_sha = sha(directory / "results.json")
    figures = directory / "derived"
    reasonke = json.loads((ROOT / "configs/reasonke-model.json").read_text(encoding="utf-8"))
    instruction = json.loads((ROOT / "configs/instruction-model.json").read_text(encoding="utf-8"))
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from PIL import Image as PillowImage

    teal, ink, grey = [colors.HexColor(c) for c in ("#167e72", "#1b2d35", "#5c6c76")]
    width = A4[0] - 88
    styles = getSampleStyleSheet()
    for name, size, leading, font, color, after in (
        ("DepthTitle", 29, 34, "Helvetica-Bold", ink, 14),
        ("DepthH1", 20, 25, "Helvetica-Bold", ink, 12),
        ("DepthH2", 12, 16, "Helvetica-Bold", teal, 7),
        ("DepthBody", 9.5, 13.5, "Helvetica", ink, 8),
        ("DepthSmall", 7.7, 10.5, "Helvetica", grey, 7),
        ("DepthCell", 8, 10.5, "Helvetica", ink, 0)):
        styles.add(ParagraphStyle(name=name, fontName=font, fontSize=size, leading=leading,
                                  textColor=color, spaceAfter=after))
    story = []

    def text(value):
        # Body prose and model excerpts remain legible with the standard font.
        return str(value).replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-")

    def para(value, style="DepthBody"):
        story.append(Paragraph(text(value), styles[style]))

    def table(headings, rows, fractions):
        cells = [[Paragraph(escape(text(v)), styles["DepthCell"]) for v in row]
                 for row in [headings, *rows]]
        item = Table(cells, colWidths=[width * f for f in fractions], repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f2ed")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LINEBELOW", (0, 0), (-1, 0), .7, teal),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#dde5e8"))]))
        story.extend((item, Spacer(1, 8)))

    def picture(name, height):
        path = figures / f"{name}.png"
        with PillowImage.open(path) as image:
            w, h = image.size
        scale = min(width / w, height / h)
        story.extend((Image(str(path), width=w * scale, height=h * scale, hAlign="CENTER"), Spacer(1, 7)))

    def metric(name):
        return report["conditions"][name]["overall"]

    def number(value, places=2):
        return "Unavailable" if value is None else f"{value:.{places}f}"

    para("EDITGUARD / SEPARATE DEPTH STUDY", "DepthSmall")
    para("Can an edit travel<br/>through three facts?", "DepthTitle")
    para("A measured addendum to the completed operations study. An ownership correction must follow "
         "service -&gt; team -&gt; on-call person -&gt; office city, while historical-release and other-property answers remain correct.")
    table(["Final requests", "Main questions per model", "Oracle per model", "Lineage units"],
          [[544, 256, 16, 32]], [.23, .29, .24, .24])
    table(["Configuration", "Correct / assigned", "Wrong", "Abstain", "Errors", "Unexecuted"],
          [[NAMES[n], f"{metric(n)['counts']['correct']}/{metric(n)['assigned']}",
            *[metric(n)["counts"][k] for k in OUTCOMES[1:]]] for n in ORDER],
          [.35, .21, .11, .11, .10, .12])
    picture("depth-accuracy", 230)
    para("Primary accuracy is correct / all assigned. Abstentions, errors and unexecuted assignments receive zero primary credit. "
         "These are 256 correlated question variants over 32 graph lineages, not 256 independent graphs. "
         "The original operations-v1 results stay unchanged and are not pooled with this study.", "DepthSmall")
    story.append(PageBreak())

    para("Separate support from reasoning", "DepthH1")
    para("The same bounded lexical selector serves both checkpoints: four facts, two question-based seeds and at most three "
         "answer-frontier expansion rounds. It reads approved text rather than expected answers or gold support IDs. "
         "A development-only comparison selected this policy before final inference.")
    picture("retrieval-and-answer", 250)
    table(["Checkpoint / three-hop main", "Required facts found", "Complete paths", "Correct answers"],
          [[NAMES[n], f"{report['conditions'][n]['retrieval_by_depth']['3']['found']}/192",
            f"{report['conditions'][n]['retrieval_by_depth']['3']['complete']}/64",
            f"{report['conditions'][n]['by_depth']['3']['counts']['correct']}/64"] for n in ORDER[:2]],
          [.37, .23, .20, .20])
    para("Gold-support diagnostics", "DepthH2")
    para("Sixteen three-hop questions per model were selected by a fixed hash before inference. Their gold facts isolate answer generation "
         "from ordinary retrieval. This subset is not targeted failure recovery and does not replace the 256-question primary denominator.")
    diff_rows = []
    for key in ("RAG", "RAG-depth-3", "ORACLE"):
        difference = report["paired_differences"].get(key)
        if difference:
            diff_rows.append([key, f"{100*difference['left_minus_right']:+.2f} pp",
                f"[{100*difference['ci95'][0]:+.2f}, {100*difference['ci95'][1]:+.2f}] pp",
                difference["independent_lineages"]])
    table(["ReasonKE minus Qwen", "Observed difference", "95% paired interval", "Lineages"],
          diff_rows, [.29, .24, .31, .16])
    para("Paired bootstrap resamples whole lineages (10,000 draws, seed 42). A zero-width interval only describes the sampled "
         "lineage differences; it does not establish certainty for arbitrary questions. Reusing one selector with two models is not "
         "two independent retrieval experiments.", "DepthSmall")
    story.append(PageBreak())

    para("Wording, scope and every outcome", "DepthH1")
    picture("category-accuracy", 260)
    picture("outcome-composition", 210)
    para("The packet contains 48 fictional Indian-context lineages and twelve facts per world. Sixteen development lineages and "
         "32 final lineages were split before question augmentation. Wording-template banks, service/team/person identities and aliases "
         "are separated by split; city labels and relation vocabulary repeat. The service and workplace assignments are invented.")
    para("Each final lineage supplies eight questions: direct and paraphrased questions at depths one, two and three, plus two locality "
         "controls. Depth zero labels controls, not a zero-fact task: each still needs one separate property or historical fact. There are "
         "64 assigned questions per depth group and 32 per wording category, per main condition.", "DepthSmall")
    para("Scoring checks the whole endorsed final answer against controlled aliases. Mentioning the target in a rejected option is insufficient. "
         "Strict format scoring can reject an otherwise informative response. A correct final answer does not certify the model's generated explanation. "
         "No sole LLM judge, independent human semantic-review claim or arbitrary-language verification is used.", "DepthSmall")
    story.append(PageBreak())

    para("Measured cost and finite execution", "DepthH1")
    picture("latency", 235)
    table(["Configuration", "Generated n", "Warm p50 / p95 seconds", "Coverage"],
          [[NAMES[n], metric(n)["generated_latency"]["n"],
            f"{number(metric(n)['generated_latency']['p50_seconds'])} / {number(metric(n)['generated_latency']['p95_seconds'])}",
            f"{100*metric(n)['coverage']:.1f}%"] for n in ORDER], [.37, .16, .30, .17])
    para("Warm pipeline latency includes context selection, local IPC and generation, including generated wrong answers and abstentions. "
         "It excludes loading, downloads, browser network and administrator actions. Error latency has its own saved population.", "DepthSmall")
    resource_rows = []
    for label, model in (("ReasonKE", "reasonke"), ("Qwen", "instruction")):
        sessions = report["model_measurements"].get(model, [])
        loads = [s.get("hardware", {}).get("load_peak_allocated_bytes") for s in sessions]
        peaks = [(r.get("response") or {}).get("generation", {}).get("peak_allocated_bytes")
                 for r in records if r["condition"].startswith("R-" if model == "reasonke" else "G-")]
        load_peak = max((v for v in loads if isinstance(v, int)), default=None)
        generation_peak = max((v for v in peaks if isinstance(v, int)), default=None)
        resource_rows.append([label, number(load_peak/1024**3) if load_peak is not None else "Unavailable",
                              number(generation_peak/1024**3) if generation_peak is not None else "Unavailable"])
    table(["Checkpoint", "Peak loading allocation GiB", "Peak generation allocation GiB"], resource_rows, [.34, .33, .33])
    para("Both pinned 7.62B-parameter checkpoints use NF4 double quantization, FP16 computation, greedy decoding, a 4,096-token input limit "
         "and a 512-new-token cap. They were trained upstream. This project performed no checkpoint fine-tuning or per-correction weight update. "
         "The instruction checkpoint is a matched comparison, not a verified causal ablation of ReasonKE training.", "DepthSmall")
    para("Profiles: eight requests per checkpoint. Development: 24 per checkpoint. Final: 272 per checkpoint. Every outcome is written atomically; "
         "exact-plan resume skips recorded outcomes. An earlier development dispatch lost its temporary Colab runtime before export; its unknown outcomes "
         "are excluded from measured evidence. Preserved profiles were restored unchanged before a disclosed replacement development run.", "DepthSmall")
    story.append(PageBreak())

    para("Conclusion and evidence trail", "DepthH1")
    r, g = metric("R-RAG"), metric("G-RAG")
    para(f"On the frozen depth population, ReasonKE with iterative retrieval scored <b>{r['counts']['correct']}/256</b>; "
         f"Qwen with the same retrieval scored <b>{g['counts']['correct']}/256</b>. These measured outcomes and their latency/support trade-offs "
         "are the conclusion of this experiment. No universal edit success, arbitrary paraphrase robustness, faithful reasoning trace or superiority "
         "outside this controlled population is inferred.")
    para("Observed non-correct outputs", "DepthH2")
    failures = sorted((row for row in records if row["outcome"] != "correct"), key=lambda row: row["id"])[:3]
    if failures:
        rows = []
        for row in failures:
            actual = (row.get("response") or {}).get("content", row.get("error", row["outcome"]))
            rows.append([row["id"], row["outcome"], text(row["question"]["expected_answer"])[:80],
                         text(actual).replace("\n", " ")[:120]])
        table(["Record", "Outcome", "Expected", "Final answer / error"], rows, [.27, .16, .22, .35])
        para("Up to three lexicographically first non-correct records are shown, without selection for narrative effect. "
             "The raw ledgers preserve every output and error.", "DepthSmall")
    else:
        para("No non-correct outcomes were recorded in this final population. This is an observed bounded result; no failures are invented "
             "to make the benchmark look harder. Shared templates, automatically audited labels, short answers and small twelve-fact worlds still limit generalization.")
    para("Reproduce and inspect", "DepthH2")
    table(["Artifact / command", "What it establishes"], [
        ["protocol.json + six raw ledgers", "Frozen source/data/configuration identities; original profile, development and all assigned final outputs"],
        ["host-audit.json + derived/figure-manifest.json", "Byte-bound mechanical audit and scientific export provenance; zero new model calls"],
        ["scorecard.csv / retrieval.csv / latency.csv", "Outcome denominators, required support, latency samples, tokens and allocated/reserved memory"],
        ["python -m scripts.build_depth_report", "Audits final receipts before regenerating tables and five figure pairs"],
        ["docs/UNDERSTAND_DEPTH_EXTENSION.md", "Plain-language history, interpretation and India demonstration workflow"]], [.43, .57])
    para("Sources and attribution", "DepthH2")
    for manifest in (reasonke, instruction):
        para(escape(manifest["model_id"]) + "<br/>Pinned revision: " + escape(manifest["revision"]), "DepthSmall")
    para('<link href="https://github.com/ShivamRaghav7/verity" color="#167e72">EditGuard repository</link> | '
         '<link href="' + escape(reasonke["upstream_repository"]) + '" color="#167e72">Reason-KE official code</link> | '
         '<link href="' + escape(reasonke["paper"]) + '" color="#167e72">Reason-KE++ paper</link>. '
         "Published scores are not substituted for these measurements. The released ReasonKE weight-license metadata was not established; "
         "weights are fetched from their pinned upstream source rather than redistributed in the website container.", "DepthSmall")
    para("Generated UTC: " + datetime.now(timezone.utc).isoformat() + "<br/>Results SHA-256: " + input_sha, "DepthSmall")

    def footer(canvas, doc):
        canvas.setFillColor(grey); canvas.setFont("Helvetica", 7.5)
        canvas.drawString(44, 27, "EDITGUARD / DEPTH ADDENDUM / measured outputs, automated audit")
        canvas.drawRightString(A4[0] - 44, 27, str(doc.page))
        canvas.setStrokeColor(colors.HexColor("#dce5e5")); canvas.line(44, 40, A4[0] - 44, 40)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".depth-pdf-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        doc = SimpleDocTemplate(str(staged), pagesize=A4, leftMargin=44, rightMargin=44,
            topMargin=42, bottomMargin=55, title="EditGuard - separate depth study", author="EditGuard project", pageCompression=1)
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
        if output.exists() or sha(directory / "results.json") != input_sha:
            raise ValueError("Inputs or output changed during PDF generation")
        # Exclusive publication also refuses an output created during rendering.
        with output.open("xb") as stream:
            stream.write(staged.read_bytes())
    return {"pdf": str(output), "sha256": sha(output), "result_sha256": input_sha,
            "script_sha256": sha(__file__), "phase": "final", "model_generations": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "reports/reasonke/depth-final")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(create(args.directory, args.output), indent=2))


if __name__ == "__main__":
    main()

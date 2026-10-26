"""Create a typeset measured-study report from audited JSON and scientific figures."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from xml.sax.saxutils import escape


def create(directory, output):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from PIL import Image as PillowImage

    directory, output = Path(directory), Path(output)
    if output.exists():
        raise ValueError("Preserve existing report; choose a new output")
    report = json.loads((directory / "results.json").read_text())
    figures = directory / "derived"
    figure_results = json.loads((figures / "results.json").read_text())
    if (any(report.get(key) != figure_results.get(key) for key in report.keys() | figure_results.keys() if key != "source_receipts")
            or sorted(report["source_receipts"].values()) != sorted(figure_results["source_receipts"].values())):
        raise ValueError("Figures and displayed results identify different studies")
    manifest = json.loads((figures / "figure-manifest.json").read_text())
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name, expected in manifest["figures"].items():
        if sha(figures / name) != expected:
            raise ValueError("Figure integrity mismatch: " + name)
    phase = report["phase"]
    if phase not in ("development", "final"):
        raise ValueError("Only the matched development or final study can be reported")
    if report["execution_status"] != "completed":
        raise ValueError("An incomplete study must not be typeset as completed")
    names = {"R0": "ReasonKE / initial facts", "R-RAG": "ReasonKE / updated RAG", "R-ALL": "ReasonKE / full memory",
             "G-RAG": "Qwen / updated RAG", "G-ALL": "Qwen / full memory"}
    n = report["conditions"]["R-RAG"]["overall"]["assigned"]
    units = report["independent_lineages"]
    teal, ink, grey = colors.HexColor("#167e72"), colors.HexColor("#1b2d35"), colors.HexColor("#5c6c76")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=30, leading=35, textColor=ink, spaceAfter=18))
    styles.add(ParagraphStyle(name="ReportH1", fontName="Helvetica-Bold", fontSize=21, leading=25, textColor=ink, spaceAfter=15))
    styles.add(ParagraphStyle(name="ReportH2", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=teal, spaceBefore=12, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=10, leading=15, textColor=ink, spaceAfter=10))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="Helvetica", fontSize=8, leading=11, textColor=grey, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportCell", fontName="Helvetica", fontSize=8.2, leading=11, textColor=ink))
    story = []
    width = A4[0] - 88

    def para(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))

    def title(text): para(text, "ReportH1")
    def sub(text): para(text, "ReportH2")
    def small(text): para(text, "ReportSmall")
    def page(): story.append(PageBreak())
    def picture(name, limit=290):
        path = figures / (name + ".png")
        with PillowImage.open(path) as image: w, h = image.size
        scale = min(width / w, limit / h)
        story.append(Image(str(path), width=w * scale, height=h * scale, hAlign="CENTER"))
        story.append(Spacer(1, 9))

    def table(headings, rows, widths):
        data = [[Paragraph(escape(str(value)), styles["ReportCell"]) for value in row] for row in [headings, *rows]]
        item = Table(data, colWidths=[width * value for value in widths], repeatRows=1, hAlign="LEFT")
        item.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f2ed")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7), ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LINEBELOW", (0, 0), (-1, 0), .7, teal),
            ("LINEBELOW", (0, 1), (-1, -1), .3, colors.HexColor("#dde5e8"))]))
        story.append(item); story.append(Spacer(1, 12))

    para("EDITGUARD / MEASURED STUDY", "ReportSmall")
    para("Knowledge that can change.<br/>Evidence that stays traceable.", "ReportTitle")
    para("Governed external-memory knowledge editing for connected factual questions.")
    table(["Evidence phase", "Independent lineages", "Assigned generations", "Execution"],
          [[phase.upper(), units, report["assigned_records"], report["execution_status"]]], [.25, .25, .25, .25])
    sub("The problem")
    para("An internal assistant can know yesterday's service owner and therefore name the wrong person to contact today. Updating one fact should change a connected answer while preserving other services, properties and historical releases.")
    sub("The system we built")
    para("An operator approves versioned facts in a transactional memory store. Question-only retrieval supplies bounded current context to an attributed, upstream-trained ReasonKE model. The application records source facts and revisions, compares historical states, handles identity conflicts and supports withdrawal, restoration and portable text workspaces.")
    sub("What is learned; what is deterministic")
    para("The pretrained neural reasoner generates answers and decides how supplied facts answer a question. Approval, canonical identity, version transitions and rollback are software controls. Publishing a fact does not modify weights. This project did not train the released ReasonKE or Qwen checkpoints.")
    small("This report is generated from saved model outputs, not published paper scores. Dataset review was automated, without independent human semantic review. The current study measures short factual answers in fictional handbooks, not arbitrary dialogue accuracy.")
    page()

    title("A controlled, reproducible question")
    para("Can an approved ownership change propagate to direct and two-hop answers without corrupting neighboring facts, and how do retrieval and the model checkpoint affect that behavior?")
    sub("Ground truth and split")
    para("We independently authored 48 fictional service/team graphs with nine facts per world. An ownership edit changes service release 2; old and new teams retain their own contact facts. Each world has eight questions. The dataset contains 384 questions, split by lineage with seed 42 before wording augmentation: 16 development and 32 final worlds. Service, team and person names are disjoint across splits; simple values and templates can repeat.")
    para(("This bounded development run executes three predetermined development worlds, not the entire development partition." if phase == "development" else "This frozen final run executes all 32 held-out lineages. No final answers were used to choose prompts, retrieval, settings or aliases."))
    table(["Question family", "Assigned per main configuration"], [[label, n // 8] for label in
        ("Direct changed fact", "Direct paraphrase", "Two-hop connected answer", "Two-hop paraphrase", "Same service, another property", "Similarly named service", "Older release", "Unrelated fact")], [.76, .24])
    small("Paraphrases within a lineage are correlated. The independent unit is the edit lineage. The older-release category retains its machine key wrong_version_or_tier; this dataset actually tests release versions, not subscription tiers.")
    page()

    title("Five matched configurations")
    table(["ID", "Checkpoint and supplied information"], [[name, names[name]] for name in names], [.18, .82])
    para("R0 receives initial facts and is scored against the new targets: it diagnoses stale knowledge. All updated conditions receive the same approved current information and supporting facts. Updated RAG replaces superseded values; it is not deliberately left stale. Both RAG generators receive identical selected fact IDs. Full-memory conditions expose all nine current facts.")
    sub("Retrieval and generation")
    para("A two-stage lexical retriever first selects two question-relevant facts, expands the query with their answer text, and supplies up to four facts. Main retrieval never sees hidden target answers or gold support IDs. Oracle diagnostics use gold support only in their explicitly labelled smaller subset.")
    para("Both checkpoints have about 7.6 billion parameters and run sequentially on one Tesla T4 with NF4 double-quantized weights, FP16 compute, greedy decoding, a 4,096-token input cap and a 512-token output cap. ReasonKE uses its attributed updated-information prompt; Qwen uses its official chat template with short-answer instructions. Exact checkpoint ancestry is not verified, so this is not a causal ablation of upstream training.")
    sub("Finite execution and source control")
    para("Each request has a hard 240-second parent watchdog and loading has a 1,800-second watchdog. Receipts are saved atomically after every outcome. Resume runs only unexecuted records under identical source, configuration, data and protocol identities. Completed wrong/error answers are never retried for a better score.")
    small("The final protocol was frozen before inference. It assigns 784 ReasonKE and 528 Qwen requests: five 256-question main conditions plus two 16-question oracle-support diagnostics. The separate development study has 120 requests; profiles have 24 requests with different sampled mixes.")
    page()

    title("Measured answers, with all outcomes")
    rows = []
    for name in names:
        m = report["conditions"][name]["overall"]; c = m["counts"]
        rows.append([name, f"{c['correct']}/{m['assigned']}", c["wrong"], c["abstain"], c["execution_error"], c["unexecuted"], f"{100*m['coverage']:.1f}%"])
    table(["Condition", "Correct", "Wrong", "Abstain", "Error", "Unexecuted", "Coverage"], rows, [.18, .14, .12, .13, .11, .17, .15])
    picture("outcome-composition", 250)
    para("Primary accuracy is correct divided by every assigned question. Abstentions, execution errors and unexecuted cases receive zero primary credit. Coverage is the fraction that gives a nonabstaining answer. Answered-only precision is separate; abstentions do not disappear from the success denominator.")
    small("Answers must match the whole endorsed final span against fixed aliases for the requested subject, relation and scope. Mentioning a target as a rejected alternative or inside an explanation is insufficient. The strict matcher can reject a semantically valid longer answer; the raw output remains available, without post-test alias changes.")
    page()

    title("Generalization and connected facts")
    picture("category-accuracy", 310)
    table(["Condition", "Required facts retrieved", "Complete two-hop support"], [[name,
        f"{report['conditions'][name]['retrieval']['fact_recall_numerator']}/{report['conditions'][name]['retrieval']['fact_recall_denominator']}",
        f"{report['conditions'][name]['retrieval']['two_hop_complete_numerator']}/{report['conditions'][name]['retrieval']['two_hop_complete_denominator']}"] for name in ("R-RAG", "G-RAG")], [.28, .36, .36])
    para("Retrieval success is not generation success: the model may receive both correct supporting facts and still connect the new owner to the old contact. The two models share the same retrieval procedure, so the two rows are not independent retrieval experiments.")
    for name in ("R-ORACLE", "G-ORACLE"):
        if name in report["conditions"]:
            m = report["conditions"][name]["overall"]
            para(f"{name}: {m['counts']['correct']}/{m['assigned']} correct with oracle supporting facts on preselected two-hop questions.")
    small("Direct and two-hop paraphrases test unfamiliar wording within shared templates and unseen entities. They do not establish robustness to all natural-language formulations or long conversational chains.")
    page()

    title("Preservation is not correctness")
    picture("efficacy-locality", 315)
    rows = []
    for name in ("R-RAG", "R-ALL"):
        condition = report["conditions"][name]; retained = condition["baseline_correct_retention"]; locality = condition["locality"]
        rows.append([name, f"{locality['counts']['correct']}/{locality['assigned']}", f"{retained['correct']}/{retained['denominator']}", f"{locality['false_correction_applications']}/{locality['assigned']}"])
    table(["Condition", "Locality correct", "Retained R0-correct", "False edited values"], rows, [.22, .26, .26, .26])
    para("Locality accuracy checks truth on every out-of-scope question. Retention checks only those locality answers that R0 already answered correctly. Preserving an incorrect answer cannot count as factual success. False edited-value endorsements detect the corrected value being asserted for a question outside its scope; other wrong-answer types are reported separately.")
    page()

    title("Uncertainty and measured cost")
    picture("paired-differences", 190)
    para(f"Paired differences use the same questions and bootstrap whole lineages (10,000 resamples, seed 42). This study has {units} independent lineages; wording variants are not independent samples. A zero-width interval when two methods answer every sampled question correctly does not imply certainty outside this controlled sample.")
    table(["Condition", "Generated n", "p50 seconds", "p95 seconds"], [[name,
        report["conditions"][name]["overall"]["generated_latency"]["n"],
        f"{report['conditions'][name]['overall']['generated_latency']['p50_seconds']:.2f}",
        f"{report['conditions'][name]['overall']['generated_latency']['p95_seconds']:.2f}"] for name in names], [.34, .22, .22, .22])
    small("Warm pipeline latency includes context selection, local worker communication and generation. It excludes cold loading and the website/browser round trip. The separate operational HTTP workload has a different latency definition.")
    for model, sessions in report["model_measurements"].items():
        for session in sessions:
            para(f"{escape(model)}: loading {session['load_seconds']:.2f} s; peak reserved CUDA during loading {session['hardware']['load_peak_reserved_bytes']/1024**3:.2f} GiB. These are actual session measurements, not an unmeasured hardware guarantee.", "ReportSmall")
    page()

    title("Failures, limits and concrete outcome")
    raw = [json.loads(path.read_text()) for path in sorted(directory.glob("*-reasonke.json")) + sorted(directory.glob("*-instruction.json"))]
    failures = sorted([row for receipt in raw for row in receipt["records"] if row["condition"] != "R0" and not row.get("correct")], key=lambda row: row["id"])
    table(["Case / condition", "Expected", "Actual / outcome"], [[row["question"]["id"] + " / " + row["condition"], row["question"]["expected_answer"],
        str(row.get("response", {}).get("content", row.get("error", row["outcome"])))[:100]] for row in failures[:6]], [.43, .24, .33])
    small("Deterministic first-six failure subset in record-ID order. All failures remain in raw receipts. Absence of a failure in this subset is not evidence that it did not occur.")
    para("A correct final answer does not verify its generated explanation. Prior development traces contained unsupported claims and incorrect intermediate associations. Explanation support must be audited separately; generated traces do not prove faithful internal reasoning.")
    sub("What the product delivers")
    para("A working conversational editing interface with explicit fact approval, canonical scope identities, conflict resolution, version history, historical comparison, withdrawal/restoration, text-workspace export/restore, result exports and scientific graphs. Model output, deterministic state-machine checks and source context remain visibly distinct.")
    sub("What the evidence supports")
    para("The result is the measured scorecard in this report, on this controlled dataset, plus a reproducible reversible editing workflow. It does not require beating every baseline. The current deployment selected retrieved context from development before final inference (ReasonKE 24/24 versus 21/24 with full memory); Qwen retrieved context also scored 24/24. No universal accuracy or superiority is inferred.")
    small("Limits: shared wording templates, a small fictional population, automated labels, strict alias scoring, no calibrated neural confidence, identity-based rather than complete semantic conflict detection, nine-fact evaluated worlds, and no production multi-user authorization. Checkpoint training and research algorithms are upstream contributions.")
    page()

    title("Reproduction and evidence map")
    table(["Artifact", "Purpose"], [["results.json", "All aggregate denominators, retrieval, coverage, latency and paired uncertainty"],
        ["Raw model receipts", "Every assigned question, supplied facts, prompt hash, raw output, model revision and outcome"],
        ["host-audit.json", "Mechanical replay of identities, extraction and scoring; zero new model calls"],
        ["derived/figure-manifest.json", "Figure hashes, plotting source identity and input receipt hashes"],
        ["configs/deployment-policy.json", "Context policy selected from development before final evaluation"],
        ["docs/OPERATIONS_DEMO.md", "Website walkthrough, ownership change, two-hop behavior and rollback"],
        ["docs/CHAT_API.md", "HTTP schemas, scope rules and workspace restoration"],
        ["notebooks/04 and 05", "Pinned chat deployment and explicitly staged sequential-model evaluation"]], [.42, .58])
    para("Source: github.com/ShivamRaghav7/verity. The inference source/data/configuration hashes are recorded in each receipt. Re-run scripts.audit_operations_evidence to recompute integrity and scoring; scripts.build_operations_report and scripts.build_operations_pdf regenerate exports without loading a model.")
    sub("Attributed model checkpoints")
    for model, sessions in report["model_measurements"].items():
        for session in sessions:
            small(escape(session["model_id"]) + "<br/>revision " + escape(session["revision"]))
    small('Reason-KE research: <link href="https://github.com/YukinoshitaKaren/Reason-KE" color="#167e72">official code</link>; <link href="https://arxiv.org/abs/2511.12661" color="#167e72">Reason-KE++ paper</link>. Published scores are not substituted for our measurements. ReasonKE checkpoint weight-license metadata was not established; weights are downloaded from the pinned upstream source, not redistributed in the website container.')
    small("Generated UTC: " + datetime.now(timezone.utc).isoformat() + "<br/>Result SHA-256: " + sha(directory / "results.json"))

    def footer(canvas, doc):
        canvas.setFillColor(grey); canvas.setFont("Helvetica", 8)
        canvas.drawString(44, 27, "EDITGUARD  /  " + phase.upper() + " STUDY  /  measured outputs, automated audit")
        canvas.drawRightString(A4[0] - 44, 27, str(doc.page))
        canvas.setStrokeColor(colors.HexColor("#dce5e5")); canvas.line(44, 40, A4[0] - 44, 40)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=44, rightMargin=44, topMargin=44, bottomMargin=56,
                            title="EditGuard - " + phase + " measured study", author="EditGuard project", pageCompression=1)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return {"pdf": str(output), "sha256": sha(output), "result_sha256": sha(directory / "results.json"),
            "script_sha256": sha(__file__), "model_generations": 0, "phase": phase}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(create(args.directory, args.output), indent=2))


if __name__ == "__main__":
    main()

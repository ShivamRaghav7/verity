"""Generate the release report from completed, hash-bound measured evidence only."""

from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
PAPER = "https://proceedings.mlr.press/v162/mitchell22a/mitchell22a.pdf"
NAMES = {"unchanged_base": "Frozen base", "updated_rag": "Updated RAG", "prompt_only": "Prompt correction",
         "similarity_editor": "Similarity editor", "trained_scope_editor": "Learned editor"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ratio(value):
    n, d = value["numerator"], value["denominator"]
    return f"{n}/{d} ({100 * value['value']:.1f}%)" if value.get("value") is not None else f"not defined ({n}/{d})"


def load_evidence(evaluation, protocol_path, lock_path, figures):
    manifest = read(evaluation / "manifest.json")
    protocol, lock = read(protocol_path), read(lock_path)
    if (manifest.get("status") != "completed" or manifest.get("purpose") != "frozen_final_evaluation"
            or manifest.get("human_reviewed_data") is not False or manifest.get("review_policy") != "automated_final"
            or manifest.get("base_weights_unchanged") is not True or manifest.get("fresh_final_runs") != 1
            or manifest.get("final_test_tuning_performed") is not False
            or manifest.get("protocol_sha256") != sha(protocol_path)
            or lock.get("protocol_sha256") != sha(protocol_path)
            or manifest.get("packet_manifest_sha256") != lock.get("packet_manifest_sha256")):
        raise ValueError("Report requires completed final evidence bound to the active protocol/test lock.")
    for name, expected in manifest["outputs_sha256"].items():
        if Path(name).name != name or sha(evaluation / name) != expected:
            raise ValueError("Final output integrity differs: " + name)
    if (manifest.get("questions") != 816 or manifest.get("oracle_records") != 384
            or manifest.get("matched_prediction_records") != 4080
            or manifest.get("population_counts") != {"scope": 720, "answer": 288, "challenge": 96}
            or manifest.get("scorer_version") != protocol["scoring"]["primary_version"]
            or manifest.get("scorer_sha256") != protocol["scoring"]["primary_sha256"]):
        raise ValueError("Final report populations or primary scoring identity differ.")
    metrics, secondary = read(evaluation / "metrics.json"), read(evaluation / "short_qa_metrics.json")
    if set(metrics["configurations"]) != set(NAMES) or any(m["count"] != manifest["questions"] for m in metrics["configurations"].values()):
        raise ValueError("Five matched configurations with full question denominators are required.")
    figure_manifest = read(figures / "manifest.json")
    # Figure manifests can expose a list or dictionary; verify every referenced
    # local image regardless of descriptive labels used by the plotting tool.
    if figure_manifest.get("protocol_sha256") != sha(protocol_path) or not figure_manifest.get("outputs_sha256"):
        raise ValueError("Scientific figures require the same active protocol and explicit output hashes.")
    source_values = set(figure_manifest.get("source_files_sha256", {}).values())
    if not {sha(evaluation / "manifest.json"), sha(evaluation / "metrics.json"),
            sha(evaluation / "short_qa_metrics.json"), sha(protocol_path)} <= source_values:
        raise ValueError("Scientific figures and report refer to different measured evidence.")
    for key, value in figure_manifest["outputs_sha256"].items():
        expected = value if isinstance(value, str) else value["sha256"]
        if sha(figures / key) != expected:
            raise ValueError("Scientific figure checksum differs: " + key)
    training = read(ROOT / "reports/targeted-training-audit.json")
    sources = {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in
               (evaluation / "manifest.json", evaluation / "metrics.json", evaluation / "short_qa_metrics.json",
                protocol_path, lock_path, figures / "manifest.json", ROOT / "reports/targeted-training-audit.json")}
    extras = {}
    for label, path in (("runtime_failure", ROOT / "reports/final/runtime-failure-disclosure.json"),
                        ("paired_short_qa", ROOT / "reports/final/secondary-paired-uncertainty.json"),
                        ("neural_http", ROOT / "reports/final/neural-http.json")):
        if path.exists():
            extras[label] = read(path)
            sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    recovery_dir = ROOT / "reports/final/oracle-runtime-recovery"
    if (recovery_dir / "interrupted_manifest.json").exists() and not (recovery_dir / "manifest.json").exists():
        path = recovery_dir / "interrupted_manifest.json"
        extras["oracle_recovery_interrupted"] = read(path)
        sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
    if (recovery_dir / "manifest.json").exists():
        recovery = read(recovery_dir / "manifest.json")
        if (recovery.get("status") != "completed" or recovery.get("primary_results_replaced") is not False
                or recovery.get("protocol_sha256") != sha(protocol_path)
                or recovery.get("oracle_questions") != 384 or recovery.get("selected_runtime_errors") != 319):
            raise ValueError("Oracle recovery must be a separate completed supplement of the original execution errors.")
        for filename, checksum in recovery["outputs_sha256"].items():
            if Path(filename).name != filename or sha(recovery_dir / filename) != checksum:
                raise ValueError("Supplemental oracle output differs from sealed evidence.")
            sources[str((recovery_dir / filename).relative_to(ROOT)).replace("\\", "/")] = checksum
        sources[str((recovery_dir / "manifest.json").relative_to(ROOT)).replace("\\", "/")] = sha(recovery_dir / "manifest.json")
        extras["oracle_recovery"] = {"manifest": recovery, "metrics": read(recovery_dir / "metrics.json"),
                                     "secondary": read(recovery_dir / "short_qa_metrics.json")}
    return manifest, protocol, lock, metrics, secondary, training, extras, sources


def build_sections(manifest, protocol, lock, metrics, secondary, extras=None):
    extras = extras or {}
    learned = metrics["configurations"]["trained_scope_editor"]
    oracle = metrics["oracle_answer_diagnostic"]["configurations"]["oracle_answer"]
    scope_qa = secondary["by_population"]["scope"]
    learned_qa = scope_qa["trained_scope_editor"]["all_requested_qa"]
    rag_qa = scope_qa["updated_rag"]["all_requested_qa"]
    rows = []
    for name in NAMES:
        current = metrics["configurations"][name]
        qa = secondary["configurations"][name]["all_requested_qa"]
        rows.append((NAMES[name], ratio(current["accuracy"]), f"{qa['correct']}/{qa['denominator']}",
                     str(current["pending_review"]), ratio(current["coverage"])))
    sections = [
        ("Problem and contribution", [
            "Product limits and policies change while deployed customer-support models remain static. EditGuard addresses short factual questions about a fictional product catalogue. An administrator publishes an approved correction; subsequent questions can use it without retraining for that individual edit. The registered topic is Knowledge Editing.",
            "The delivered platform combines two actually trained auxiliary models, approved versioned memory, a frozen handbook assistant, transactional conflict and rollback controls, five matched comparisons, a leakage-controlled benchmark and a locally deployable application. Generated answers remain fallible. No arbitrary-language accuracy guarantee or production-company claim is made.",
            f"The frozen evaluation contains {manifest['questions']} matched questions and {manifest['oracle_records']} separate oracle-generation records. On the 720 controlled scope questions, request-bound QA credits the learned editor on {learned_qa['correct']}/{learned_qa['denominator']} and updated RAG on {rag_qa['correct']}/{rag_qa['denominator']}. These are supported short-question forms; unfamiliar challenge wording has a separate result.",
            f"The learned editor receives complete-assertion credit on {ratio(learned['accuracy'])}. The original oracle diagnostic was interrupted by a CUDA runtime fault: its {ratio(oracle['accuracy'])} includes {oracle['errors']} execution errors and cannot be read as a complete generator-quality measurement. A separately named operational supplement, when available, preserves the original record. Complete-assertion and request-bound results are separate evidence."]),
        ("Related work and architectural adaptations", [
            "SERAC wraps a fixed base model with edit memory, a scope classifier and a counterfactual generator. Its auxiliary components are trained separately; deployment stores new descriptors and routes relevant requests to conditional generation. Mitchell et al. compare embedding and cross-attention scope architectures and evaluate edit success and preservation. We use these ideas, rather than claim a faithful reproduction of their experiments or algorithm.",
            "Our adaptations are structured product/plan/version/date descriptors, mandatory administrator approval, SQLite transactions, explicit overlapping-conflict resolution, retrieval before pair scoring, calibrated abstention and deterministic compatibility gates. The frozen base retrieves the initial fictional handbook; it does not rely on pretrained knowledge of imaginary products.",
            "The selected DistilBERT classifier uses the versioned entity_neutral_v1 input transform. Only selectors literally referenced by a correction are canonicalized; incompatible selectors and properties remain. Approval, dates, original names and provenance remain in authoritative metadata. This engineered normalization is disclosed separately from learned entity reasoning.",
            "The selected FLAN-T5-small answer model conditions on the question, applicable correction, date and necessary unchanged support context. It produces the original short answer text. A deterministic structured response is a distinct mode and never earns evidence of trained generation. Unknown scope, uncertainty and unresolved conflicts request clarification."]),
        ("Dataset, review and split controls", [
            "The controlled handbook has 80 fictional entity families, 1,280 initial facts and 480 edit scenarios. Families were assigned before paraphrases and counterfactual values. The modest training pilot uses four training and two validation families: 240/120 scope pairs and 96/48 answer examples. Training did not use the entire generated catalogue.",
            "Scope data contains applicable pairs, ordinary negatives, different properties on the same product, the same property on another product, similar names, wrong tiers and incompatible versions. Pilot-v3 balances existing negative question forms without changing facts, labels or family assignments. Targets and example labels received a separately typed automated audit; human review is not claimed.",
            "The final packet preserves all 720 scope, 288 answer and 96 challenge rows. Its 1,104 labels passed independent deterministic consistency checks against structured truth. Every source question, fact, label, split and lineage remains unchanged; current input formatting and explicit oracle metadata are derived separately. Original draft flags are retained and the separate machine registry authorizes only final evaluation.",
            "The twelve final families are disjoint from selected training and development entities and edit lineages. Entity normalization creates nine identical same-label scope pairs shared with training, zero shared with validation and no contradictory pairs. The other 711 scope rows are reported separately. Shared controlled templates still limit linguistic generalization.",
            "The 96 unfamiliar-wording/unseen-value challenges were agent-authored drafts, independently machine-audited. They are not manually authored or human-reviewed questions. The user explicitly authorized this automated final audit. Machine consistency checks do not establish external validity or broad natural-language truth."]),
        ("Training and finite model selection", [
            "Training used one fixed seed and compact backbones after measured four-update CUDA profiles. The selected scope run stopped at 360 updates; the selected answer run stopped at 240. Both were executed by the user. Data, reviews, pretrained revisions, configurations, selected tensors, optimizer/scheduler/sampler/RNG state and exact training source are recorded. The completed exports passed 986 artifact/checkpoint audit checks.",
            "On the common 120-pair development population, raw scope accuracy rose from 98/120 to 119/120. Validation selected temperature 1.0, threshold 0.97 and uncertainty margin 0.025: 48 correct edit applications, zero of 72 false applications and one clarification. This selection-data result is not an independent final-test estimate.",
            "The answer revision gives equal 8x weight to the opening token of both Yes and No comparison targets. Remaining valid tokens have weight one and padding zero. Its weighted selection loss of 0.16824 cannot be compared with the earlier uniform loss. On a common unweighted objective, validation token NLL worsened from 0.04441 to 0.06173.",
            "Generated development negative comparisons improved from 0/12 to 5/12 correct; seven still failed. Oracle complete-answer credit improved from 31/48 to 34/48 with identical revised scope routing. A reviewed complete storage wording alias increased scoring coverage equally for every configuration; original scores are preserved. No threshold, model or scorer was tuned against final outputs.",
            "Recorded training-loop times were 112.55 seconds for scope and 264.54 for answer; complete command startup/export wall times are unavailable. CUDA reserved peaks were 1.410 and 1.656 GiB. Measurements are from an RTX 4050 laptop with 6 GiB VRAM and 23.71 GiB host RAM. Short profiling estimates and actual loop measurements are distinguished."]),
        ("Frozen fair comparison and scoring", [
            "All five configurations answer the same 816 questions: 720 scope questions and 96 challenges. Unchanged base is a diagnostic reference; updated RAG materializes approved corrections, prompt-only receives eligible descriptors, similarity editing uses validation-selected embedding routing, and learned editing uses the trained scope classifier. Both editors use the same trained answer model.",
            "Every updated baseline has the same approved memory and supporting facts. Questions are evaluated in separate mutually exclusive correction scenarios, each with one focal edit and at most eight foreign-product distractors. This is not a dense same-product multi-edit stress test. Deterministic generation uses the same 48-token answer budget and compatible context settings.",
            "Oracle generation uses 288 answer examples plus 96 challenges, separately from the matched end-to-end population. It supplies the applicable correction but not the target answer. Retrieval, raw classifier pairs, actual guarded routes and oracle generation remain separate measurements so failures can be attributed.",
            "Primary scoring checks complete endorsed subject, relation, product/plan/version, value, unit and requested polarity. A target mention, example, rejected possibility or correct database response is insufficient. Unsupported prose stays pending and uncredited; all abstentions/errors remain in fixed-denominator accuracy. Zero complete-assertion credit for a short baseline fragment does not prove that fragment factually wrong.",
            "The frozen secondary rule accepts whole quantities or Yes/No for twelve explicit controlled question forms, after independently checking the requested scope and predicate against truth. It never overrides a primary incorrect verdict. Bare responses do not prove a complete scoped assertion. Unsupported challenge wording remains pending. No LLM judge substitutes for ground truth.",
            "The full-assertion metric favors outputs that restate a scoped fact, a format the trained generator saw in its targets. Baseline fragments can correctly answer a question while receiving no complete-assertion credit. Therefore primary credit differences do not establish superiority in factual QA; use the separately locked request-bound comparison on its supported cohort and disclose unsupported challenge cases.",
            "Paired uncertainty uses 2,000 entity-family bootstrap resamples, seed 42 and 95% intervals. Twelve clusters and one training seed limit precision; these intervals describe family sampling in this controlled benchmark, not uncertainty over retraining or deployed customers."]),
        ("Measured results and their limits", [
            f"On the fair controlled request-bound metric, the learned editor credits {learned_qa['correct']}/{learned_qa['denominator']} (85.14%), updated RAG {rag_qa['correct']}/{rag_qa['denominator']} (90.14%), similarity editing 632/720, prompt correction 533/720 and the frozen base 500/720. Updated RAG is the strongest of these configurations. The learned editor does not beat it overall.",
            "The learned editor credits 72/72 direct corrections, 39/72 held-out paraphrases and 97/144 comparisons on the request-bound metric. Thirty-three paraphrases remain pending; 47 comparison answers are wrong. Out-of-scope factual credit is 405/432, with 23 clarifications and four wrong base answers. Retention on the distinct baseline-correct subset is 405/428. Preserving an incorrect initial answer is not factual correctness.",
            "The 96 challenge questions remain a separate population. The learned editor has zero complete-assertion credits, 48 clarifications and 48 pending outputs. The secondary grammar cannot score these unfamiliar forms, so no secondary semantic correctness claim follows. Similarity editing has 40/96 complete-assertion credits. This challenge exposes limited linguistic generalization.",
            f"Actual false correction application is {ratio(learned['false_correction_application_rate'])}. Classifier precision/recall and calibration below are raw-pair diagnostics; metadata-gated routing is not evidence that every boundary was learned. The oracle diagnostic removes retrieval and routing errors but retains generation failures and incomplete answers.",
            "Outcome tables distinguish credited answers, observed incorrect assertions, pending uncredited prose, clarification, errors and unavailability. Cohort figures distinguish direct questions, held-out paraphrases, comparisons, out-of-scope questions and unfamiliar wording. Updated RAG is assessed honestly; the project does not require beating every baseline.",
            "Serial CLI latency includes retrieval, routing and generation, retaining first-use requests. It excludes model construction, scoring, journal writes, HTTP transport and the application's extra before-answer generation. Fixed ordering and shared caches can bias comparisons. Separate HTTP engineering measurements must not be relabeled as model-route timings."]),
        ("Reliability, engineering and deployment", [
            "The authoritative key is entity, relation, plan, version and a half-open effective interval. Corrections require a source, matching original value and administrator approval. Same-scope contradictory intervals do not use latest-write-wins. Explicit supersession links an approved predecessor; withdrawal restores applicable ancestors, and descendants must be withdrawn first. Mutations and revision increments are transactional; revision keys invalidate caches.",
            "The isolated actual learned workflow passed 31 governance/integration assertions across 33 API requests. These checks validate authority, conflict blocking, historical boundaries and rollback, not representative model accuracy. Development demonstrations still include wrong negative polarity, unsupported paraphrase prose and an unaffected price question that clarifies.",
            "Regression/CI covers the app, scoring, data review, checkpoint recovery, source/model bindings and container persistence. The structured Docker deployment and optional locally configured neural application are documented separately. The minimal stack is Python, FastAPI/Pydantic, SQLite and build-free HTML/CSS/JavaScript, with PyTorch/Transformers and Matplotlib for ML experiments.",
            "The final runner uses an OS-released exclusive session lock, atomic per-observation journal and exact-bound recovery. Completed configurations cannot be silently rerun. A dependency-inventory startup rejection occurred before loading models or predictions; its source and log are archived. The amended protocol preserves byte-identical question/truth files and fixes only the guard mismatch.",
            "Windows access-denied collisions interrupted two mutable progress writes; the lock owner and underlying cause are unknown. A declared finite retry wrapper changes only those writes. A later CUDA unknown error occurred after all 4,080 matched predictions had completed: 319 of the separate 384 oracle attempts were execution errors. All original errors remain uncredited in the original denominator.",
            "The failed GPU instance's final parameter fingerprint was unavailable. Fresh-process closure confirmed unchanged serialized sources and model files and the expected loaded base fingerprint, with zero new predictions. It cannot reconstruct the lost GPU memory state. This distinction limits the original run's end-state integrity measurement and is disclosed rather than inferred away."]),
        ("Panel demonstration and reproducibility", [
            "Demonstrate the initial handbook answer, a sourced administrator correction, direct and paraphrased questions, neighboring property/tier/version questions, explicit ambiguity or conflict, then withdrawal and restored history. Show the answer-kind badge and supporting fact alongside generated text. Include a documented model failure rather than implying every generated response is verified.",
            "The repository supplies immutable data bundles, review registries, pinned run cards, training/evaluation scripts, recorded model metadata, local saved model exports, scientific figure sources, regression tests, CI, API documentation and deployment instructions. Large checkpoint binaries remain local artifacts rather than Git source. Offline result reaggregation and fresh training reproduction are distinct from rerunning the locked final test.",
            "A defensible resume claim describes the SERAC-inspired editor, two trained auxiliary models, transactional correction memory and rollback, and evaluation across five matched configurations on a controlled held-out benchmark. Numerical claims must identify the metric, denominator and dataset. This release does not establish unrestricted dialogue, multi-hop reasoning, large edit memories or guaranteed arbitrary-language accuracy."])
    ]
    measured = sections[5][1]
    if extras.get("paired_short_qa"):
        paired = extras["paired_short_qa"]
        pair = next(p for p in paired["paired_comparisons"]
                    if p["left"] == "updated_rag" and p["right"] == "trained_scope_editor")
        measured.append(f"The existing secondary verdicts were aggregated with the same frozen 2,000-resample paired family bootstrap: learned minus updated RAG is {pair['difference_percentage_points']:.2f} percentage points, 95% interval [{pair['ci95'][0]*100:.2f}, {pair['ci95'][1]*100:.2f}]. Every one of the 720 standard questions remains included. This is a descriptive aggregation of locked verdicts, with no rescoring, inference or test-guided selection.")
    if extras.get("oracle_recovery"):
        recovered = extras["oracle_recovery"]
        r = recovered["metrics"]["configurations"]["oracle_answer"]
        measured.append(f"The separately declared runtime recovery selected only the 319 original CUDA error statuses, permitted at most one new generation per selected ID and checked reconstructed input bytes. The 65 original successful outputs were preserved. Combined oracle complete-assertion credit is {ratio(r['accuracy'])}, with {r['errors']} execution errors and {r['pending_review']} pending outputs. This supplement replaces no original results, changes no model/configuration/scorer and has mixed-session timings; it measures operational recovery, not a model improvement.")
    elif extras.get("oracle_recovery_interrupted"):
        pending = extras["oracle_recovery_interrupted"]
        measured.append(f"A separately declared, status-only recovery of the 319 oracle runtime errors was attempted. It stopped during model loading because CUDA was unavailable, before any of its {pending['maximum_new_generation_calls']} permitted generations: {pending['durable_attempt_reservations']} durable attempts, {pending['not_attempted_selected_count']} not attempted. No recovered answer accuracy or full neural HTTP latency is claimed. The original sealed observations remain unchanged; restoring GPU access is an operational prerequisite for these remaining measurements.")
    measured.append("Available resource snapshots record 933.1 MiB CUDA peak allocated and 1,002 MiB reserved immediately after model construction. The original inference sessions did not retain a successful end-of-run memory snapshot, so these loading measurements are not presented as the full final-inference peak. A fresh closure-only session is likewise not an inference-memory benchmark.")
    if extras.get("neural_http"):
        http = extras["neural_http"]
        measured.append(f"A separate isolated local HTTP workload measured {http['counts']['attempted']} warm learned-mode requests: p50 {http['latency']['p50_ms']:.1f} ms and p95 {http['latency']['p95_ms']:.1f} ms, {http['counts']['errors']} errors. Client timing includes transport, decoding and the API's before-answer generation; cold loading is separate. These repeated demonstration questions measure engineering latency, not held-out accuracy or concurrent load capacity.")
    return sections, rows


def build_report(evaluation, protocol_path, lock_path, figures, pdf_path, markdown_path):
    evidence = load_evidence(evaluation, protocol_path, lock_path, figures)
    manifest, protocol, lock, metrics, secondary, training, extras, sources = evidence
    sections, results = build_sections(manifest, protocol, lock, metrics, secondary, extras)
    # Imports remain optional: report generation does not require the ML runtime.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, NextPageTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyEG", fontName="Helvetica", fontSize=10, leading=14.5, spaceAfter=9))
    styles.add(ParagraphStyle(name="CaptionEG", fontName="Helvetica", fontSize=8, leading=11, textColor=colors.HexColor("#526779"), spaceAfter=12))
    styles.add(ParagraphStyle(name="TableEG", fontName="Helvetica", fontSize=8, leading=10.5))
    styles["Heading1"].textColor = colors.HexColor("#11485b")
    styles["Heading1"].spaceBefore = 16
    styles["Heading1"].spaceAfter = 10
    story = [Spacer(1, 50), Paragraph("EDITGUARD", ParagraphStyle("TitleEG", fontName="Helvetica-Bold", fontSize=35, leading=42, textColor=colors.HexColor("#11485b"))),
             Spacer(1, 10), Paragraph("A Learned Knowledge Editing and Verification Platform for AI Customer-Support Assistants", styles["Heading2"]),
             Spacer(1, 25), Paragraph("Final measured project report", styles["Heading2"]),
             Paragraph("Controlled demonstration dataset | 15 October 2026", styles["BodyEG"]),
             Paragraph("SERAC-inspired | Frozen base | Trained scope and answer models", styles["BodyEG"]),
             Spacer(1, 24), Paragraph("Evidence boundaries", styles["Heading2"]),
             Paragraph("Automated label audit; no independent human review. Agent-authored challenges. Pending prose and abstentions remain uncredited. Source retrieval does not certify generated text.", styles["BodyEG"]),
             Paragraph("The original oracle diagnostic includes CUDA execution failures. Its separately declared recovery is an operational supplement; current recovery status is disclosed in the results section.", styles["BodyEG"]),
             Spacer(1, 20), Paragraph(f"Active protocol: {escape(protocol['protocol_id'])}<br/>Matched questions: {manifest['questions']} | Oracle records: {manifest['oracle_records']}", styles["CaptionEG"]), PageBreak()]
    markdown = ["# EditGuard: final measured project report", "", "Controlled demonstration dataset. Automated final audit; no human-review claim.", ""]
    figure_files = sorted(figures.glob("*.png"))
    for index, (heading, paragraphs) in enumerate(sections):
        story.append(Paragraph(f"{index + 1}. {escape(heading)}", styles["Heading1"]))
        markdown += ["## " + heading, ""]
        for text in paragraphs:
            story.append(Paragraph(escape(text), styles["BodyEG"]))
            markdown += [text, ""]
        if index == 1:
            story.append(Paragraph(f"Primary reference: <link href='{PAPER}'>Mitchell et al., Memory-Based Model Editing at Scale, ICML 2022</link>.", styles["CaptionEG"]))
            markdown += [f"Primary reference: [Mitchell et al., ICML 2022]({PAPER}).", ""]
        if index == 5:
            header = ["Configuration", "Primary credit", "Request QA", "Pending", "Coverage"]
            table = Table([[Paragraph(escape(str(cell)), styles["TableEG"]) for cell in row] for row in [header, *results]],
                          colWidths=[100, 112, 65, 51, 139], repeatRows=1)
            table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5eff1")),
                                       ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")]),
                                       ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                                       ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#9db2bb"))]))
            story += [table, Spacer(1, 12), Paragraph("Primary requires a complete scoped assertion. Request QA is a separate frozen controlled-form metric. Pending counts refer to primary scoring; zero primary credit does not mean zero factual correctness.", styles["CaptionEG"])]
            markdown += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
            markdown += ["| " + " | ".join(row) + " |" for row in results] + [""]
            # Scientific figures use one landscape sheet each. Scaling a wide
            # two-panel chart into a portrait text column made its labels tiny.
            story += [NextPageTemplate("landscape"), PageBreak()]
            for figure_index, figure in enumerate(figure_files):
                image = Image(str(figure))
                scale = min(746 / image.imageWidth, 440 / image.imageHeight)
                image.drawWidth, image.drawHeight = image.imageWidth * scale, image.imageHeight * scale
                caption = figure.stem.replace("-", " ").replace("_", " ") + ". Measured frozen-test data; see raw JSON/CSV for denominators and policy details."
                story += [image, Spacer(1, 5), Paragraph(escape(caption), styles["CaptionEG"])]
                if figure_index == len(figure_files) - 1:
                    story.append(NextPageTemplate("portrait"))
                story.append(PageBreak())
                markdown += [f"![{caption}]({str(figure.relative_to(ROOT / 'docs').as_posix()) if ROOT / 'docs' in figure.parents else '../' + str(figure.relative_to(ROOT)).replace(chr(92), '/')})", ""]
    story.append(Paragraph("Evidence and version record", styles["Heading1"]))
    for task, model in protocol["models"].items():
        text = f"{task.capitalize()}: {model['model_id']} | revision {model['model_revision']} | selected artifact {model['artifact']}"
        story.append(Paragraph(escape(text), styles["CaptionEG"]))
        markdown += [text, ""]
    for name, value in sources.items():
        chunks = " ".join(value[offset:offset + 16] for offset in range(0, len(value), 16))
        story.append(Paragraph(escape(name) + "<br/>SHA-256: " + chunks, styles["CaptionEG"]))
    markdown += ["## Evidence and version record", ""] + [f"- `{name}`: `{value}`" for name, value in sources.items()]
    document = BaseDocTemplate(str(pdf_path), pagesize=A4, rightMargin=64, leftMargin=64, topMargin=48, bottomMargin=48,
                               title="EditGuard - Final Measured Project Report", author="EditGuard project team")
    def decorate(canvas, doc):
        canvas.saveState()
        width, height = doc.pageTemplate.pagesize
        canvas.setStrokeColor(colors.HexColor("#c6d6dc"))
        canvas.line(64, 35, width - 64, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526779"))
        canvas.drawString(64, 23, "EditGuard | Controlled benchmark | Automated audit")
        canvas.drawRightString(width - 64, 23, str(doc.page))
        canvas.restoreState()
    document.addPageTemplates([
        PageTemplate(id="portrait", frames=[Frame(64, 48, A4[0] - 128, A4[1] - 96, id="text")],
                     pagesize=A4, onPage=decorate),
        PageTemplate(id="landscape", frames=[Frame(48, 48, landscape(A4)[0] - 96, landscape(A4)[1] - 96, id="figure")],
                     pagesize=landscape(A4), onPage=decorate)])
    document.build(story)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8", newline="\n")
    metadata = {"schema_version": 1, "artifact_type": "measured_project_report", "source_sha256": sources,
                "pdf_sha256": sha(pdf_path), "markdown_sha256": sha(markdown_path), "report_script_sha256": sha(__file__),
                "human_reviewed_data": False, "training_or_inference_executed": False}
    pdf_path.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "reports/final/evaluation")
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/final/protocol-v2.json")
    parser.add_argument("--test-lock", type=Path, default=ROOT / "reports/final/test_lock-v2.json")
    parser.add_argument("--figures", type=Path, default=ROOT / "reports/final/figures-v2")
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/EditGuard-Report.pdf")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/REPORT.md")
    args = parser.parse_args()
    print(json.dumps(build_report(args.evaluation, args.protocol, args.test_lock, args.figures, args.pdf, args.markdown), indent=2))


if __name__ == "__main__":
    main()

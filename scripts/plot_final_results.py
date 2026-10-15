"""Reproduce scientific figures from completed frozen final exports only.

No model, dataset, GPU inference, scorer extension or threshold selection occurs.
Strict complete assertions and the locked request QA metric remain separate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.metrics import aggregate_predictions, scope_metrics
from scripts.analyze_short_qa import summarize as short_summary

CONFIGS = ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor")
LABELS = ("Frozen base", "Updated RAG", "Prompt correction", "Similarity editor", "Learned editor")
COHORTS = ("direct", "paraphrase", "comparison", "out_of_scope", "challenge")
COHORT_LABELS = ("Direct edit", "Paraphrase", "Comparison", "Out of scope", "Challenge")
NAVY, TEAL, ORANGE, PURPLE, GRAY = "#18304b", "#138b80", "#bd612f", "#8577b5", "#c7d0dc"
COLORS = (NAVY, TEAL, ORANGE, PURPLE, "#416ea0")
ANSWERED = {"answered", "ok", "success", "generated"}
OUTPUT_NAMES = ("predictions.jsonl", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")
OUTCOMES = ("Credited correct", "Explicit incorrect", "Pending / uncredited", "Clarification / nonanswer")
OUTCOME_COLORS = (TEAL, ORANGE, PURPLE, GRAY)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def read_rows(path):
    return [json.loads(line, object_pairs_hook=unique_keys) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def answered(row):
    return row.get("status", "answered") in ANSWERED and bool((row.get("answer") or "").strip())


def outcome(row, metric="primary"):
    if not answered(row):
        return OUTCOMES[3]
    score = row["score"] if metric == "primary" else row["short_qa"]
    if score.get("correct") is True or metric == "qa" and score.get("verdict") == "correct":
        return OUTCOMES[0]
    if score.get("correct") is False or metric == "qa" and score.get("verdict") == "incorrect":
        return OUTCOMES[1]
    return OUTCOMES[2]


def outcome_counts(rows, metric="primary"):
    counts = Counter(outcome(row, metric) for row in rows)
    return {name: counts[name] for name in OUTCOMES}


def cohort(row):
    return "challenge" if row["population"] == "challenge" else "out_of_scope" if not row["applicable"] else row["question_type"]


def descriptive_risk_coverage(rows, threshold, margin):
    """Rank fixed classifier decisions by distance from its frozen threshold.

    Ties enter together. The curve does not select or propose a new threshold.
    Missing probabilities remain in the full coverage denominator.
    """
    groups = {}
    for row in rows:
        probability = row.get("scope_probability")
        if probability is None:
            continue
        if not isinstance(probability, (int, float)) or isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid scope probability")
        distance = abs(probability - threshold)
        group = groups.setdefault(distance, [0, 0])
        group[0] += 1
        group[1] += int((probability >= threshold) != bool(row["scope_label"]))
    covered = errors = 0
    points = []
    fixed = {"covered": 0, "errors": 0, "denominator": len(rows), "coverage": 0., "risk": None}
    for distance in sorted(groups, reverse=True):
        count, mistakes = groups[distance]
        covered += count
        errors += mistakes
        points.append({"coverage": covered / len(rows), "risk": errors / covered, "covered": covered, "errors": errors})
        if distance > margin:
            fixed.update(covered=covered, errors=errors, coverage=covered / len(rows), risk=errors / covered)
    return {"points": points, "fixed": fixed, "threshold": threshold, "margin": margin,
            "policy": "descriptive tied confidence ranking; fixed threshold and margin unchanged"}


def routing_counts(rows):
    positive = [row for row in rows if row["applicable"]]
    negative = [row for row in rows if not row["applicable"]]
    true = sum(row.get("selected_edit_id") == row.get("focal_edit_id") for row in positive)
    selected = sum(bool(row.get("selected_edit_id")) for row in rows)
    false = sum(bool(row.get("selected_edit_id")) for row in negative)
    return {"precision": (true, selected), "recall": (true, len(positive)),
            "false_application": (false, len(negative)), "positive": len(positive), "negative": len(negative)}


def observed_category(row):
    """Observable mutually exclusive categories, without asserting hidden causes."""
    verdict = outcome(row)
    if verdict == OUTCOMES[0]:
        return "Credited correct"
    if verdict == OUTCOMES[3]:
        return "Clarification / nonanswer"
    if verdict == OUTCOMES[2]:
        return "Pending semantics"
    if not row["applicable"] and row.get("selected_edit_id"):
        return "Observed false application"
    if row["applicable"] and row.get("selected_edit_id") != row.get("focal_edit_id"):
        raw = {item["edit_id"] for item in row.get("retrieval_candidates_raw", [])}
        return "Focal edit absent in retrieval" if row.get("focal_edit_id") not in raw else "Retrieved edit not applied"
    return "Incorrect generated answer"


def evidence(evaluation, protocol_path, *, root=ROOT):
    """Reject partial results before reading any prediction content."""
    evaluation, protocol_path, root = Path(evaluation).resolve(), Path(protocol_path).resolve(), Path(root).resolve()
    manifest = read_json(evaluation / "manifest.json")
    if (manifest.get("status") != "completed" or manifest.get("purpose") != "frozen_final_evaluation"
            or manifest.get("review_policy") != "automated_final" or manifest.get("human_reviewed_data") is not False
            or manifest.get("base_weights_unchanged") is not True or manifest.get("final_test_tuning_performed") is not False
            or manifest.get("configurations") != list(CONFIGS)):
        raise ValueError("Only completed, disclosed, frozen final evaluation is accepted")
    if read_json(evaluation / "progress.json").get("status") != "completed":
        raise ValueError("Wait for final command completion before plotting")
    if digest(protocol_path) != manifest["protocol_sha256"]:
        raise ValueError("Result protocol checksum differs")
    protocol = read_json(protocol_path)
    if (protocol.get("status") != "frozen" or protocol.get("review_policy") != "automated_final"
            or protocol.get("human_reviewed_data") is not False or protocol["protocol_id"] != manifest["protocol_id"]
            or protocol["scoring"]["primary_version"] != manifest["scorer_version"]):
        raise ValueError("Frozen final protocol disclosures differ")
    for name in ("editguard/metrics.py", "scripts/analyze_short_qa.py"):
        if digest(ROOT / name) != protocol.get("source_sha256", {}).get(name):
            raise ValueError("Frozen metric derivation source differs: " + name)
    paths = {name: evaluation / name for name in ("manifest.json", "progress.json", *OUTPUT_NAMES)}
    paths["protocol"] = protocol_path
    calibration_record = protocol["calibrations"]["scope"]
    calibration_path = (root / calibration_record["path"]).resolve()
    if root not in calibration_path.parents or digest(calibration_path) != calibration_record["sha256"]:
        raise ValueError("Frozen calibration checksum differs")
    paths["scope_calibration"] = calibration_path
    hashes = {name: digest(path) for name, path in paths.items()}
    for name in OUTPUT_NAMES:
        if manifest["outputs_sha256"].get(name) != hashes[name]:
            raise ValueError("Completed result checksum differs: " + name)
    calibration = read_json(calibration_path)
    threshold, margin = calibration["threshold"], calibration["ambiguity_margin"]
    if (threshold, margin) != (.97, .025):
        raise ValueError("This figure suite describes only the already frozen .97 / .025 policy")
    matched, oracle = read_rows(paths["predictions.jsonl"]), read_rows(paths["oracle_predictions.jsonl"])
    metrics, secondary = read_json(paths["metrics.json"]), read_json(paths["short_qa_metrics.json"])
    if (len(matched) != manifest["matched_prediction_records"] or len(oracle) != manifest["oracle_records"]
            or len(matched) != manifest["questions"] * len(CONFIGS)):
        raise ValueError("Every declared matched/oracle prediction must remain present")
    identities, truths, cohorts = set(), {}, {name: set() for name in CONFIGS}
    for row in [*matched, *oracle]:
        identity = (row["configuration"], row["id"])
        if identity in identities or row.get("review_policy") != "automated_final" or row.get("human_reviewed_data") is not False:
            raise ValueError("Duplicate or undisclosed final prediction")
        identities.add(identity)
        if row["row_sha256"] != manifest["accepted_row_sha256"][row["id"]] or row["final_protocol_sha256"] != manifest["protocol_sha256"]:
            raise ValueError("Final prediction input/protocol binding differs")
        if row["score"]["scorer_version"] != manifest["scorer_version"]:
            raise ValueError("Primary scorer identity differs")
        if row["answer_kind"] not in ("model_generated", "deterministic_clarification"):
            raise ValueError("Structured answers cannot become learned-generation evidence")
        if row["configuration"] != "oracle_answer":
            if row["configuration"] not in cohorts or row["population"] not in ("scope", "challenge"):
                raise ValueError("Unknown matched configuration/population")
            cohorts[row["configuration"]].add(row["id"])
            truth = {key: row.get(key) for key in ("family_id", "question", "request_context", "expected_fact", "expected_polarity", "applicable", "question_type", "population")}
            if row["id"] in truths and truths[row["id"]] != truth:
                raise ValueError("Matched configurations have different questions/truth")
            truths[row["id"]] = truth
    if any(ids != cohorts[CONFIGS[0]] for ids in cohorts.values()) or len(truths) != manifest["questions"]:
        raise ValueError("All five configurations must share every fixed question")
    for name, rows in (("scope", [r for r in matched if r["configuration"] == CONFIGS[0] and r["population"] == "scope"]),
                       ("challenge", [r for r in matched if r["configuration"] == CONFIGS[0] and r["population"] == "challenge"]),
                       ("answer", [r for r in oracle if r["population"] == "answer"])):
        if len(rows) != manifest["population_counts"][name]:
            raise ValueError("Frozen population denominator differs")
    recomputed = aggregate_predictions(matched, bootstrap_seed=42, bootstrap_resamples=2000)
    if any(metrics.get(name) != value for name, value in recomputed.items()):
        raise ValueError("Primary counts or paired intervals cannot be reproduced")
    if metrics.get("protocol_sha256") != manifest["protocol_sha256"] or secondary.get("protocol_sha256") != manifest["protocol_sha256"]:
        raise ValueError("Metric protocol binding differs")
    if secondary["configurations"] != short_summary(matched) or secondary["oracle"] != short_summary(oracle):
        raise ValueError("Locked secondary counts cannot be reproduced")
    for population in ("scope", "challenge"):
        rows = [r for r in matched if r["population"] == population]
        expected = aggregate_predictions(rows, bootstrap_seed=42, bootstrap_resamples=2000)
        if metrics["by_population"][population] != expected or secondary["by_population"][population] != short_summary(rows):
            raise ValueError("Cohort counts cannot be reproduced")
    return {"manifest": manifest, "protocol": protocol, "metrics": metrics, "secondary": secondary,
            "matched": matched, "oracle": oracle, "threshold": threshold, "margin": margin,
            "paths": paths, "hashes": hashes}


def framing(fig, title, subtitle, footer):
    fig.suptitle("EditGuard · " + title, x=.08, ha="left", y=.985, fontsize=19, fontweight="bold", color=NAVY)
    fig.text(.08, .935, subtitle, fontsize=10, color=NAVY)
    fig.text(.08, .035, footer, fontsize=9, color=NAVY, va="bottom")
    fig.subplots_adjust(top=.84, bottom=.22, left=.09, right=.97, hspace=.5, wspace=.33)


def overlap_note(ev):
    shared = ev["metrics"]["scope_normalized_overlap"]
    ids = set(shared["scope_row_ids_sharing_train_pair"]) | set(shared["scope_row_ids_sharing_validation_pair"])
    pairs = shared.get("same_label_pairs_shared_with_train", "unavailable")
    return f"Abstracted overlap: {len(ids)} scope rows; {pairs} input pairs shared with training. Raw families remain separate."


def quality_figure(plt, ev):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    framing(fig, "answer quality with every outcome retained", "Frozen final comparison · identical approved edits · deterministic 48-token generation · all five configurations",
        "Pending complete assertions can be factually meaningful fragments; zero strict credit does not establish zero factual correctness.\n"
        "The request QA diagnostic checks a bounded requested value/predicate separately; it does not prove a complete scoped fact.\n"
        "Automated label audit; agent-authored challenge drafts. " + overlap_note(ev))
    for ax, metric, title in zip(axes, ("primary", "qa"), ("A. Complete scoped assertion (primary)", "B. Request QA (locked secondary)")):
        bottoms = [0.] * len(CONFIGS)
        for name, color in zip(OUTCOMES, OUTCOME_COLORS):
            counts = [outcome_counts([r for r in ev["matched"] if r["configuration"] == c], metric) for c in CONFIGS]
            denoms = [sum(count.values()) for count in counts]
            values = [count[name] / n * 100 for count, n in zip(counts, denoms)]
            bars = ax.bar(range(len(CONFIGS)), values, bottom=bottoms, color=color, label=name, width=.65)
            for bar, value, count in zip(bars, values, counts):
                if value >= 7:
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2, str(count[name]), ha="center", va="center", fontsize=9, color="white" if color != GRAY else NAVY)
            bottoms = [a+b for a, b in zip(bottoms, values)]
        for i, c in enumerate(CONFIGS):
            rows = [r for r in ev["matched"] if r["configuration"] == c]
            ax.text(i, 103, f"{outcome_counts(rows, metric)[OUTCOMES[0]]}/{len(rows)}", ha="center", fontsize=9, fontweight="bold")
        ax.set(title=title, ylabel="Percentage of full answerable population", ylim=(0, 115), xticks=range(5), xticklabels=LABELS)
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.09, .135), ncols=4, frameon=False, fontsize=8.5)
    return fig


def cohort_figure(plt, ev):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    framing(fig, "generalization and locality cohorts", "Every cell shows credited answers / the complete cohort; comparison and out-of-scope questions stay explicit",
        "Primary and request QA use different endorsement requirements; neither drops abstentions or pending outputs.\n"
        "Challenge records preserve original unfamiliar agent-authored wording; they have no independent human authorship/review.\n" + overlap_note(ev))
    for ax, metric, title in zip(axes, ("primary", "qa"), ("A. Complete scoped assertion", "B. Request QA")):
        cells, annotations = [], []
        for config in CONFIGS:
            values, texts = [], []
            for name in COHORTS:
                rows = [r for r in ev["matched"] if r["configuration"] == config and cohort(r) == name]
                correct = outcome_counts(rows, metric)[OUTCOMES[0]]
                values.append(correct / len(rows) * 100 if rows else float("nan"))
                texts.append(f"{correct}/{len(rows)}" if rows else "undefined\nn=0")
            cells.append(values)
            annotations.append(texts)
        image = ax.imshow(cells, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        for y, texts in enumerate(annotations):
            for x, value in enumerate(texts):
                ax.text(x, y, value, ha="center", va="center", fontsize=10, color="white" if cells[y][x] > 55 else NAVY)
        ax.set(title=title, yticks=range(5), yticklabels=LABELS, xticks=range(5), xticklabels=COHORT_LABELS)
        ax.tick_params(axis="x", rotation=25)
        fig.colorbar(image, ax=ax, shrink=.8, label="Credited / full denominator (%)")
    return fig


def ratio_bars(ax, items, title):
    values = [n / d * 100 if d else 0 for _, (n, d), _ in items]
    bars = ax.bar(range(len(items)), values, color=[color for _, _, color in items], width=.6)
    for bar, (_, (n, d), _) in zip(bars, items):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3, f"{n}/{d}" if d else "undefined", ha="center", fontsize=9)
    ax.set(title=title, ylim=(0, 118), ylabel="Percent", xticks=range(len(items)), xticklabels=[name for name, _, _ in items])
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)


def routing_figure(plt, ev):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    framing(fig, "retrieval, pair classification and actual routing", "These stages measure separate events; metadata compatibility and approved-memory checks also affect runtime routing",
        "Raw pair classification includes incompatible hard negatives. Runtime precision/recall use selected edit IDs, not prose influence.\n"
        "Retrieval recall uses raw pre-gate candidates; it is not eligible-candidate recall. False applications concern explicit editors only.\n"
        "All measurements shown concern the scope cohort. " + overlap_note(ev))
    rows = [r for r in ev["matched"] if r["configuration"] == "trained_scope_editor" and r["population"] == "scope"]
    labels, probs = [r["scope_label"] for r in rows], [r.get("scope_probability") for r in rows]
    for ax, threshold, title in ((axes[0, 0], .5, "A. Pair classifier at diagnostic 0.50"), (axes[0, 1], ev["threshold"], "B. Pair classifier at frozen 0.97")):
        summary = scope_metrics(labels, probs, threshold=threshold)
        ratio_bars(ax, [(label, (summary[key]["numerator"], summary[key]["denominator"]), color)
            for label, key, color in (("Precision", "precision", TEAL), ("Recall", "recall", NAVY), ("False positive", "false_positive_rate", ORANGE))], title)
    routed = routing_counts(rows)
    ratio_bars(axes[1, 0], [(name, routed[key], color) for name, key, color in (("Precision", "precision", TEAL), ("Recall", "recall", NAVY), ("False application", "false_application", ORANGE))], "C. Actual gated learned-editor decisions")
    items = []
    for config, label, color in (("similarity_editor", "Similarity", PURPLE), ("trained_scope_editor", "Learned", TEAL)):
        selected = [r for r in ev["matched"] if r["configuration"] == config and r["population"] == "scope"]
        positives = [r for r in selected if r["applicable"]]
        for k in ("1", "3"):
            hits = sum((r.get("retrieval_hits") or {}).get(k) is True for r in positives)
            items.append((f"{label}\nrecall@{k}", (hits, len(positives)), color))
    ratio_bars(axes[1, 1], items, "D. Applicable-edit raw retrieval recall")
    return fig


def calibration_figure(plt, ev):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    framing(fig, "calibration and descriptive coverage", "Frozen validation-selected threshold 0.97, ambiguity margin 0.025; no threshold or model is selected from this plot",
        "Reliability concerns independent question–focal-edit probabilities, before compatibility gates and retrieval selection.\n"
        "Coverage curve ranks fixed 0.97 classifier decisions by distance from that threshold; tied confidences enter together.\n"
        "The marked point uses the pre-existing ±0.025 uncertainty rule; this curve does not prescribe a new operating policy.")
    largest_risk = 0.
    for population, color in (("scope", TEAL), ("challenge", ORANGE)):
        rows = [r for r in ev["matched"] if r["configuration"] == "trained_scope_editor" and r["population"] == population]
        summary = scope_metrics([r["scope_label"] for r in rows], [r.get("scope_probability") for r in rows], threshold=ev["threshold"])
        bins = [b for b in summary["reliability_bins"] if b["count"]]
        axes[0].scatter([b["mean_probability"] for b in bins], [b["empirical_applicability"] for b in bins], s=[35+4*math.sqrt(b["count"]) for b in bins], color=color, label=f"{population}: observed {summary['probability_coverage']['numerator']}/{len(rows)}")
        for b in bins:
            axes[0].annotate(f"{population} n={b['count']}", (b["mean_probability"], b["empirical_applicability"]),
                xytext=(-5, 10 if population == "scope" else -19), ha="right" if b["mean_probability"] > .8 else "left", textcoords="offset points", fontsize=8, color=color)
        curve = descriptive_risk_coverage(rows, ev["threshold"], ev["margin"])
        axes[1].plot([p["coverage"]*100 for p in curve["points"]], [p["risk"]*100 for p in curve["points"]], color=color, label=f"{population} (n={len(rows)})")
        largest_risk = max(largest_risk, max((p["risk"]*100 for p in curve["points"]), default=0.))
        point = curve["fixed"]
        if point["risk"] is not None:
            axes[1].scatter([point["coverage"]*100], [point["risk"]*100], color=color, marker="D" if population == "scope" else "o", s=75, edgecolor=NAVY, zorder=5)
            axes[1].annotate(f"{population} fixed: {point['covered']}/{point['denominator']}\n{point['errors']} pair errors", (point["coverage"]*100, point["risk"]*100), xytext=(-20, 40 if population == "scope" else 12), textcoords="offset points", ha="right", fontsize=8, color=color)
    axes[0].plot([0, 1], [0, 1], color=GRAY, linestyle="--", label="Ideal calibration")
    axes[0].set(title="A. Ten predeclared reliability bins", xlabel="Mean predicted applicability", ylabel="Observed applicability", xlim=(-.04, 1.04), ylim=(-.04, 1.15))
    axes[1].set(title="B. Fixed classifier decision risk versus coverage", xlabel="Covered pairs / entire cohort (%)", ylabel="Pair classification errors / covered pairs (%)", xlim=(0, 105), ylim=(-1, min(105, max(5, largest_risk*1.2+4))))
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.grid(alpha=.15)
    return fig


def latency_figure(plt, ev):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    framing(fig, "measured request latency", "Serial CLI measurements · all requests including first use · model construction recorded separately",
        "Request timing includes retrieval, routing and generation; it excludes model loading, scoring, observation writes and HTTP/browser overhead.\n"
        "Fixed configuration order and shared caches can affect comparisons. Missing timing samples make full-population percentiles unavailable.\n"
        "No extrapolated training duration or unmeasured service latency is shown.")
    for config, label, color in zip(CONFIGS, LABELS, COLORS):
        selected = [r for r in ev["matched"] if r["configuration"] == config]
        values = sorted(r["latency_ms"] for r in selected if isinstance(r.get("latency_ms"), (float, int)) and not isinstance(r.get("latency_ms"), bool) and math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0)
        if values:
            axes[0].step(values, [(i+1)/len(values) for i in range(len(values))], where="post", label=f"{label}: {len(values)}/{len(selected)}", color=color)
    axes[0].set(title="A. Actual latency distributions", xlabel="Recorded latency (ms)", ylabel="Empirical fraction of observed requests")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].grid(alpha=.15)
    table_rows = []
    for config, label in zip(CONFIGS, LABELS):
        data = ev["metrics"]["configurations"][config]["latency"]
        values = [f"{data[key]:.1f}" if data[key] is not None else "unavailable" for key in ("p50_ms", "p95_ms")]
        table_rows.append([label, f"{data['sample_count']}/{data['total_count']}", *values])
    axes[1].axis("off")
    axes[1].set_title("B. Full-population p50 and p95")
    table = axes[1].table(cellText=table_rows, colLabels=["Configuration", "Samples", "p50 ms", "p95 ms"], cellLoc="center", loc="center", colWidths=[.4, .2, .2, .2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.2)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_facecolor(NAVY if row == 0 else "#edf2f7")
        cell.set_text_props(color="white" if row == 0 else NAVY)
    sessions = ev["manifest"].get("sessions", [])
    durations = [s.get("model_construction_seconds") for s in sessions]
    if durations and all(isinstance(value, (int, float)) for value in durations):
        axes[1].text(.5, .13, "Recorded model construction per session: " + ", ".join(f"{v:.2f}s" for v in durations), transform=axes[1].transAxes, ha="center", fontsize=9)
    return fig


def uncertainty_figure(plt, ev):
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    framing(fig, "paired uncertainty and observed failure categories", "Primary complete-assertion differences; predeclared paired family bootstrap, seed 42 and 2,000 resamples",
        "Intervals resample complete entity families; a small family count limits precision and controlled wording limits external validity.\n"
        "Failure categories summarize observable routing/scoring events; they do not infer correctness for pending semantic outputs.\n"
        "Oracle answer accuracy is reported separately in the machine-readable metrics; deterministic answers receive no learned-generation credit.")
    fig.subplots_adjust(left=.11, wspace=.75)
    comparisons = []
    for pair in ev["metrics"]["paired_comparisons"]:
        if "trained_scope_editor" not in (pair["left"], pair["right"]):
            continue
        sign = 1 if pair["right"] == "trained_scope_editor" else -1
        other = pair["left"] if sign == 1 else pair["right"]
        value = pair["difference"]
        interval = pair["ci95"]
        center = sign*value*100 if value is not None else None
        interval = sorted(sign*p*100 for p in interval) if interval else None
        comparisons.append((LABELS[CONFIGS.index(other)], center, interval, pair["family_count"]))
    for y, (label, center, interval, families) in enumerate(comparisons):
        if center is not None:
            axes[0].scatter([center], [y], color=TEAL, zorder=3)
        if interval:
            axes[0].plot(interval, [y, y], color=TEAL, linewidth=2)
        axes[0].annotate(f"{families} families", (center or 0, y), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8)
    axes[0].axvline(0, color=GRAY, linestyle="--")
    axes[0].set(title="A. Learned editor minus comparator", xlabel="Primary accuracy difference (percentage points)", yticks=range(len(comparisons)), yticklabels=[p[0] for p in comparisons])
    axes[0].grid(axis="x", alpha=.15)
    selected = [r for r in ev["matched"] if r["configuration"] == "trained_scope_editor"]
    counts = Counter(observed_category(row) for row in selected)
    labels = list(counts)
    bars = axes[1].barh(range(len(labels)), [counts[label] for label in labels], color=[TEAL if label == "Credited correct" else PURPLE if label == "Pending semantics" else ORANGE for label in labels])
    axes[1].bar_label(bars, labels=[f"{counts[label]}/{len(selected)}" for label in labels], padding=4, fontsize=9)
    axes[1].set(title="B. Observed learned-editor outcomes", xlabel="Questions (full denominator)", yticks=range(len(labels)),
        yticklabels=[textwrap.fill(label, width=19) for label in labels], xlim=(0, max(counts.values())*1.35))
    axes[1].grid(axis="x", alpha=.15)
    axes[1].set_axisbelow(True)
    return fig


def generate(evaluation, protocol_path, output, *, root=ROOT):
    output, evaluation, protocol_path = Path(output).resolve(), Path(evaluation).resolve(), Path(protocol_path).resolve()
    if output.exists():
        raise ValueError("Choose a fresh figure directory; existing results remain immutable")
    if output == evaluation or evaluation in output.parents or output in evaluation.parents or output == protocol_path.parent:
        raise ValueError("Figures must remain separate from immutable evaluation inputs")
    ev = evidence(evaluation, protocol_path, root=root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold", "text.color": NAVY,
        "axes.labelcolor": NAVY, "svg.hashsalt": "editguard-frozen-final-v1"})
    makers = (("final-answer-quality", quality_figure), ("final-cohort-accuracy", cohort_figure),
        ("final-retrieval-and-routing", routing_figure), ("final-calibration-coverage", calibration_figure),
        ("final-latency", latency_figure), ("final-uncertainty-failures", uncertainty_figure))
    description = json.dumps({"purpose": "frozen_final_evaluation_figures", "source_files_sha256": ev["hashes"],
        "script_sha256": digest(__file__), "inference_executed": False}, sort_keys=True)
    output.mkdir(parents=True)
    outputs = {}
    for name, maker in makers:
        fig = maker(plt, ev)
        for suffix in (".png", ".svg"):
            path = output / (name + suffix)
            metadata = {"Description": description}
            if suffix == ".svg":
                metadata["Date"] = None
            fig.savefig(path, dpi=180, metadata=metadata)
            outputs[path.name] = digest(path)
        plt.close(fig)
    if ev["hashes"] != {name: digest(path) for name, path in ev["paths"].items()}:
        raise RuntimeError("Frozen evidence changed while figures were generated")
    metadata = {"schema_version": 1, "purpose": "frozen_final_evaluation_figures", "review_policy": "automated_final",
        "human_reviewed_data": False, "model_or_dataset_loaded": False, "inference_executed": False,
        "source_unchanged": True, "source_files_sha256": ev["hashes"], "script_sha256": digest(__file__),
        "metrics_source_sha256": digest(ROOT / "editguard/metrics.py"),
        "secondary_summary_source_sha256": digest(ROOT / "scripts/analyze_short_qa.py"),
        "matplotlib_version": matplotlib.__version__, "protocol_sha256": ev["manifest"]["protocol_sha256"],
        "fixed_threshold": ev["threshold"], "fixed_ambiguity_margin": ev["margin"],
        "population_counts": ev["manifest"]["population_counts"],
        "scope_normalized_overlap": ev["metrics"]["scope_normalized_overlap"],
        "outputs_sha256": outputs,
        "limitations": ["Controlled fictional products; automated label review and agent-authored challenge drafts.",
            "Complete assertion credit and locked request QA are separate; pending fragments are not declared factually wrong.",
            "All abstentions, unavailable/error outputs and pending outcomes remain in complete denominators.",
            "Descriptive confidence ranking changes no test-time threshold or deployed routing policy.",
            "Scope pair statistics exclude metadata gates; runtime routing and retrieval are separate measurements.",
            "Serial CLI latency excludes loading, scoring, observation writes and HTTP/browser overhead.",
            "Family bootstrap precision and external validity are limited by the controlled test population."]}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, default=ROOT / "reports/final/evaluation")
    parser.add_argument("--protocol", type=Path, default=ROOT / "reports/final/protocol-v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final/figures")
    args = parser.parse_args()
    result = generate(args.evaluation, args.protocol, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "plots": list(result["outputs_sha256"]), "source_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()

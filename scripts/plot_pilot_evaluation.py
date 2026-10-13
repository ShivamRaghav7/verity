"""Plot recorded pilot training/validation evidence without loading ML models.

Only the supplied completed validation report, model logs/manifests and frozen
scope calibration files are read. No dataset, test or challenge file is read.
Outputs require a fresh directory, and every input/output is bound by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGURATIONS = ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor")
LABELS = ("Unchanged base", "Updated RAG", "Prompt only", "Similarity editor", "Trained scope editor")
NAVY, TEAL, ORANGE, GRAY, PALE = "#18304b", "#138b80", "#bd612f", "#65758a", "#ced9e4"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def temperature_scale(probability, temperature):
    probability = max(min(probability, 1 - 1e-8), 1e-8)
    logit = (math.log(probability) - math.log1p(-probability)) / temperature
    return 1 / (1 + math.exp(-max(min(logit, 700), -700)))


def evidence(report, calibration, scope, answer):
    paths = {
        "evaluation_manifest": report / "manifest.json",
        "metrics": report / "metrics.json",
        "predictions": report / "predictions.jsonl",
        "oracle_predictions": report / "oracle_predictions.jsonl",
        "calibration": calibration / "calibration.json",
        "calibration_grid": calibration / "selection_grid.json",
        "raw_scope_probabilities": calibration / "validation_probabilities.jsonl",
        "scope_manifest": scope / "manifest.json", "scope_training_log": scope / "training_log.jsonl",
        "answer_manifest": answer / "manifest.json", "answer_training_log": answer / "training_log.jsonl",
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    manifest, metrics, cal, grid = (read_json(paths[key]) for key in
                                    ("evaluation_manifest", "metrics", "calibration", "calibration_grid"))
    rows, oracle, raw = (read_jsonl(paths[key]) for key in
                         ("predictions", "oracle_predictions", "raw_scope_probabilities"))
    if (manifest.get("purpose") != "validation_development" or manifest.get("status") != "completed"
            or manifest.get("test_rows_read") is not False or manifest.get("base_weights_unchanged") is not True
            or metrics.get("purpose") != "validation_development"
            or metrics.get("record_count") != len(rows) or len({row["id"] for row in rows}) != metrics["question_count"]
            or manifest["predictions_sha256"] != hashes["predictions"]):
        raise ValueError("Complete, unchanged-base validation evidence with matching hashes is required")
    if (cal.get("purpose") != "validation_calibration" or cal.get("split") != "validation"
            or cal.get("status") != "frozen" or cal.get("test_rows_read") is not False
            or cal["probabilities_sha256"] != hashes["raw_scope_probabilities"]
            or cal["selection_grid_sha256"] != hashes["calibration_grid"]
            or cal["accepted_validation_rows"] != len(raw) or grid["rows"] != len(raw)
            or len({row["id"] for row in raw}) != len(raw)
            or any(row.get("split") != "validation" for row in raw)):
        raise ValueError("Frozen validation calibration must match its entire raw pair population")
    if cal["scope_manifest_sha256"] != hashes["scope_manifest"]:
        raise ValueError("Calibration scope model identity mismatch")
    if any(row.get("answer_kind") != "model_generated" for row in rows + oracle):
        raise ValueError("These figures require model-generated outputs only")
    for name in CONFIGURATIONS:
        population = [row for row in rows if row["configuration"] == name]
        summary = metrics["configurations"][name]
        if (len(population) != summary["count"] or len({row["id"] for row in population}) != len(population)
                or summary["accuracy"]["numerator"] != sum(row["score"]["correct"] is True for row in population)
                or summary["pending_review"] != sum(row["score"]["needs_review"] for row in population)):
            raise ValueError("Prediction counts or fixed-denominator metrics disagree")
    oracle_summary = metrics["oracle_answer_diagnostic"]["configurations"]["oracle_answer"]
    if (len(oracle) != oracle_summary["count"]
            or oracle_summary["accuracy"]["numerator"] != sum(row["score"]["correct"] is True for row in oracle)):
        raise ValueError("Oracle counts disagree; its applicable-only population stays separate")
    labels = [row["label"] for row in raw]
    probabilities = [row["scope_probability"] for row in raw]
    if set(labels) != {0, 1} or any(type(label) is not int for label in labels):
        raise ValueError("Scope calibration requires explicit binary labels")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Scope probabilities must be finite in [0,1]")
    calibrated = [temperature_scale(p, cal["temperature"]) for p in probabilities]
    chosen = grid["selected_operating_metrics"]
    if chosen != cal["validation_operating_metrics"]:
        raise ValueError("Chosen calibration policy differs between saved grid and artifact")
    counts = {"true_application": 0, "false_application": 0, "true_rejection": 0,
              "missed_positive": 0, "clarification_positive": 0, "clarification_negative": 0}
    for label, p in zip(labels, calibrated):
        if abs(p - cal["threshold"]) <= cal["ambiguity_margin"]:
            counts["clarification_positive" if label else "clarification_negative"] += 1
        elif p >= cal["threshold"]:
            counts["true_application" if label else "false_application"] += 1
        else:
            counts["missed_positive" if label else "true_rejection"] += 1
    if any(chosen["counts"][name] != value for name, value in counts.items()):
        raise ValueError("Recomputed operating counts differ from raw scope evidence")
    reliability = metrics["configurations"]["trained_scope_editor"]["scope"]
    brier = sum((p - label) ** 2 for p, label in zip(calibrated, labels)) / len(raw)
    ece = 0
    for item in reliability["reliability_bins"]:
        members = [(p, label) for p, label in zip(calibrated, labels)
                   if item["lower"] <= p < item["upper"] or
                   (item["upper_inclusive"] and p == item["upper"])]
        if len(members) != item["count"]:
            raise ValueError("Reliability bin counts differ from calibrated pair evidence")
        if members:
            mean = sum(p for p, _ in members) / len(members)
            empirical = sum(label for _, label in members) / len(members)
            if (not math.isclose(mean, item["mean_probability"], abs_tol=1e-9)
                    or not math.isclose(empirical, item["empirical_applicability"], abs_tol=1e-9)):
                raise ValueError("Reliability bin values differ from calibrated pair evidence")
            ece += len(members) / len(raw) * abs(mean - empirical)
    if (not math.isclose(brier, reliability["brier"]["value"], abs_tol=1e-9)
            or not math.isclose(ece, reliability["ece"]["value"], abs_tol=1e-9)):
        raise ValueError("Brier/ECE differ from calibrated pair evidence")
    training = []
    for task in ("scope", "answer"):
        model = read_json(paths[f"{task}_manifest"])
        log = read_jsonl(paths[f"{task}_training_log"])
        if (model.get("status") != "completed" or model.get("purpose") != "trained"
                or len(log) != model["completed_steps"] or model["completed_steps"] != model["max_steps"]
                or [row["global_step"] for row in log] != list(range(1, len(log) + 1))
                or model["dataset_id"] != manifest["dataset_id"]):
            raise ValueError("Complete capped training logs from the validation dataset are required")
        training.append((task, model, log))
    return paths, hashes, manifest, metrics, rows, oracle, cal, grid, raw, calibrated, training


def title(figure, name, subtitle):
    figure.suptitle(name, y=.98, fontsize=18, fontweight="bold", color=NAVY)
    figure.text(.5, .93, subtitle, ha="center", fontsize=10, color=GRAY)


def footer(figure, text):
    figure.text(.5, .06, text, ha="center", va="center", fontsize=9, linespacing=1.6, color=GRAY)


def outcomes_figure(plt, metrics, rows, oracle):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.1), gridspec_kw={"width_ratios": [2.6, 1]})
    figure.subplots_adjust(left=.155, right=.97, top=.85, bottom=.19, hspace=.53, wspace=.31)
    title(figure, "EditGuard · initial pilot validation", "Strict complete-assertion scorer · two unseen entity families · development data, not final test")
    all_populations = [[row for row in rows if row["configuration"] == name] for name in CONFIGURATIONS]
    categories = (("Automatically credited", TEAL, lambda row: row["status"] == "answered" and row["score"]["correct"] is True),
                  ("Semantic review pending", PALE, lambda row: row["status"] == "answered" and row["score"]["needs_review"]),
                  ("Explicitly incorrect", ORANGE, lambda row: row["status"] == "answered" and row["score"]["correct"] is False),
                  ("Abstention/error/unavailable", GRAY, lambda row: row["status"] != "answered"))
    for axis, populations, names, denominator, panel in (
            (axes[0, 0], all_populations, LABELS, metrics["question_count"], "A. Five matched configurations"),
            (axes[0, 1], [oracle], ["Oracle"], len(oracle), "B. Correct routing supplied")):
        left = [0] * len(populations)
        positions = list(range(len(populations)))
        for label, color, predicate in categories:
            values = [sum(predicate(row) for row in population) for population in populations]
            bars = axis.barh(positions, values, left=left, color=color, height=.6, label=label)
            for bar, value in zip(bars, values):
                if value:
                    axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                              str(value), ha="center", va="center", fontsize=9, color=NAVY)
            left = [start + value for start, value in zip(left, values)]
        axis.set(title=panel, yticks=positions, yticklabels=names, xlim=(0, denominator),
                 xlabel=f"Outputs (fixed denominator = {denominator})")
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=.15)
        axis.set_axisbelow(True)
    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(.56, -.2), ncol=2, frameon=False, fontsize=8.5)
    for axis, names, configs, panel in (
            (axes[1, 0], LABELS, [metrics["configurations"][name] for name in CONFIGURATIONS], "C. Observed CLI inference latency"),
            (axes[1, 1], ["Oracle"], [metrics["oracle_answer_diagnostic"]["configurations"]["oracle_answer"]], "D. Oracle latency")):
        positions = list(range(len(names)))
        p50, p95 = ([item["latency"][key] for item in configs] for key in ("p50_ms", "p95_ms"))
        for y, low, high in zip(positions, p50, p95):
            axis.plot([low, high], [y, y], color=GRAY, lw=1.5)
            axis.text(high + 14, y, f"{low:.0f} / {high:.0f}", va="center", fontsize=8.5)
        axis.scatter(p50, positions, s=35, color=TEAL, label="p50")
        axis.scatter(p95, positions, s=38, marker="s", color=NAVY, label="p95")
        axis.set(title=panel, yticks=positions, yticklabels=names, xlabel="Milliseconds; labels show p50 / p95", xlim=(0, max(p95) * 1.33))
        axis.set_ylim(len(names) - .55, -.55)
        axis.grid(axis="x", alpha=.15)
    axes[1, 0].legend(loc="lower right", frameon=False, fontsize=9)
    footer(figure, "Every configuration answered 120/120; oracle answered 48/48 applicable questions. No deterministic answers or dropped abstentions.\n"
           "Uncredited fragments require semantic review: zero credited baseline assertions does not mean zero factual correctness.\n"
           "CLI timers exclude model construction and HTTP/UI overhead; fixed serial order and shared caches bias latency comparisons.")
    return figure


def calibration_figure(plt, metrics, cal, grid, raw, calibrated):
    from matplotlib.ticker import PercentFormatter
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.1))
    figure.subplots_adjust(left=.09, right=.97, top=.84, bottom=.19, hspace=.55, wspace=.28)
    title(figure, "EditGuard · learned scope calibration", "120 raw labeled pairs · 48 applicable / 72 negative · selected temperature 1.5, threshold 0.11, margin 0")
    scope = metrics["configurations"]["trained_scope_editor"]["scope"]
    axis = axes[0, 0]
    bins = [b for b in scope["reliability_bins"] if b["count"]]
    axis.plot([0, 1], [0, 1], color=GRAY, linestyle="--", linewidth=1, label="Perfect calibration")
    axis.scatter([b["mean_probability"] for b in bins], [b["empirical_applicability"] for b in bins],
                 s=[b["count"] * 4 for b in bins], color=TEAL, alpha=.85)
    for index, b in enumerate(bins):
        axis.annotate(f"n={b['count']}", (b["mean_probability"], b["empirical_applicability"]),
                      xytext=(5, -17 if index == 0 else 8),
                      textcoords="offset points", fontsize=8)
    axis.set(title="A. Calibrated reliability (10 equal-width bins)", xlabel="Mean predicted applicability", ylabel="Empirical applicability", xlim=(-.04, 1.04), ylim=(-.04, 1.12))
    axis.text(.45, .17, f"Brier = {scope['brier']['value']:.4f}\nECE = {scope['ece']['value']:.4f}\nn = 120; empty bins omitted", fontsize=9, color=NAVY)
    axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    axis.grid(alpha=.15)
    axis = axes[0, 1]
    labels = [r["label"] for r in raw]
    for label, name, color in ((0, "Out of scope (72)", ORANGE), (1, "Applicable (48)", TEAL)):
        values = [p for p, y in zip(calibrated, labels) if y == label]
        axis.hist(values, bins=[index / 20 for index in range(21)], histtype="step", color=color, linewidth=2, label=name)
    axis.axvline(cal["threshold"], color=NAVY, linestyle=":", label="Selected threshold")
    axis.set(title="B. Calibrated probability distribution", xlabel="Predicted applicability", ylabel="Pair count", xlim=(0, 1))
    axis.legend(frameon=False, fontsize=8.5, loc="upper right")
    axis.grid(axis="y", alpha=.15)
    axis = axes[1, 0]
    for margin, color in ((0, NAVY), (.025, TEAL), (.05, ORANGE), (.1, GRAY)):
        candidates = [c for c in grid["operating_candidates"] if c["ambiguity_margin"] == margin]
        points = [c for c in candidates if c["selective_accuracy"]["value"] is not None]
        axis.scatter([c["coverage"]["value"] for c in points], [c["selective_accuracy"]["value"] for c in points],
                     s=15, color=color, alpha=.55, label=f"Margin {margin:g}")
    selected = grid["selected_operating_metrics"]
    axis.scatter([selected["coverage"]["value"]], [selected["selective_accuracy"]["value"]], s=100,
                 facecolors="none", edgecolors="black", linewidths=1.5, label="Selected policy", zorder=5)
    axis.set(title="C. Scope decisions: accuracy versus coverage", xlabel="Non-clarification decisions / 120", ylabel="Correct / non-clarification decisions", xlim=(-.04, 1.04), ylim=(0, 1.08))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="lower left")
    axis.grid(alpha=.15)
    axis = axes[1, 1]
    candidates = [c for c in grid["operating_candidates"] if c["ambiguity_margin"] == 0]
    for key, name, color, style in (("precision", "Precision (TP / applications)", TEAL, "-"),
            ("recall_including_clarification_as_missed", "Recall (TP / 48)", NAVY, "-"),
            ("false_application_rate", "False applications / 72", ORANGE, "--")):
        axis.step([c["threshold"] for c in candidates],
                  [c[key]["value"] if c[key]["value"] is not None else math.nan for c in candidates],
                  where="post", color=color, linestyle=style, linewidth=1.8, label=name)
    axis.axvline(cal["threshold"], color=GRAY, linestyle=":")
    axis.set(title="D. Raw pair decisions; zero-margin threshold sweep", xlabel="Calibrated probability threshold", ylabel="Rate", xlim=(0, 1), ylim=(-.04, 1.06))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(frameon=False, fontsize=8, loc="lower left")
    axis.grid(axis="y", alpha=.15)
    footer(figure, "Temperature minimizes validation NLL; operating policy minimizes 4 × false applications + missed positives + 0.25 × clarifications.\n"
           "Selected raw pair counts: TP 36, FP 0, TN 72, FN 12; precision 36/36, recall 36/48. Clarifications 0/120; decision coverage 120/120.\n"
           "This is classifier decision coverage, not answer coverage. Abstentions stay in fixed-denominator accuracy; same-set selection is optimistic.")
    return figure


def training_figure(plt, training):
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9.1))
    figure.subplots_adjust(left=.08, right=.97, top=.84, bottom=.19, hspace=.5, wspace=.25)
    title(figure, "EditGuard · measured auxiliary training", "User-executed capped pilot runs · RTX 4050 Laptop GPU · float32 · batch 2 · seed 42")
    for column, (task, model, log) in enumerate(training):
        axis = axes[0, column]
        points = [row for row in log if row["validation_loss"] is not None]
        color = NAVY if task == "scope" else TEAL
        axis.plot([r["global_step"] for r in points], [r["validation_loss"] for r in points], color=color, marker="o", linewidth=2)
        selected = model["selection"]
        axis.scatter([selected["global_step"]], [selected["value"]], s=120, facecolors="none", edgecolors=ORANGE, linewidths=2, label="Selected checkpoint", zorder=5)
        axis.set(title=f"{'A' if column == 0 else 'B'}. {task.capitalize()} validation loss", xlabel="Completed optimizer steps", ylabel="Example mean loss" if task == "scope" else "Non-padding target-token mean loss", xticks=[r["global_step"] for r in points])
        axis.legend(frameon=False, fontsize=8.5)
        axis.grid(axis="y", alpha=.15)
    names = [task.capitalize() for task, _, _ in training]
    axis = axes[1, 0]
    elapsed = [model["training"]["elapsed_training_seconds"] for _, model, _ in training]
    optimizer = [sum(row["optimizer_seconds"] for row in log) for _, _, log in training]
    validation = [sum(row["seconds"] for row in model["training"]["validation_measurements_this_process"]) for _, model, _ in training]
    checkpoints = [sum(row["seconds"] for row in model["training"]["checkpoint_measurements_this_process"]) for _, model, _ in training]
    other = [e - a - b - c for e, a, b, c in zip(elapsed, optimizer, validation, checkpoints)]
    bottom = [0, 0]
    for values, label, color in ((optimizer, "Optimizer", NAVY), (validation, "Validation", TEAL),
                               (checkpoints, "Checkpoint I/O", ORANGE), (other, "Other loop overhead", PALE)):
        axis.bar(names, values, bottom=bottom, width=.6, color=color, label=label)
        bottom = [start + value for start, value in zip(bottom, values)]
    for index, seconds in enumerate(elapsed):
        axis.text(index, seconds + 7, f"{seconds:.3f} s", ha="center", fontsize=9)
    axis.set(title="C. Recorded loop duration and components", ylabel="Seconds", ylim=(0, max(elapsed) * 1.28))
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    axis = axes[1, 1]
    for offset, key, label, color in ((-.24, "peak_allocated_gpu_bytes", "CUDA allocated", NAVY),
                                     (0, "peak_reserved_gpu_bytes", "CUDA reserved", TEAL),
                                     (.24, "process_rss_bytes", "Maximum sampled RSS", PALE)):
        values = [model["training"]["resources"][key] / 2**30 for _, model, _ in training]
        bars = axis.bar([i + offset for i in range(2)], values, width=.22, color=color, label=label)
        axis.bar_label(bars, labels=[f"{v:.3f}" for v in values], fontsize=8, padding=3)
    axis.set(title="D. Recorded resource peaks and samples", ylabel="GiB (2³⁰ bytes)", xticks=[0, 1], xticklabels=names, ylim=(0, 4.5))
    axis.legend(frameon=False, loc="upper left", fontsize=8)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    footer(figure, "Scope cap 360 steps, selected 120; answer cap and selection 240. Validation loss is not autoregressive factual accuracy.\n"
           "Loop timer includes checkpoint writes but excludes model loading, tokenization and final selected-model export; full command wall time unavailable.\n"
           "CUDA peaks follow the trainer's loop reset; RSS is the largest recorded sample, not a lifetime high-water mark. No test partition was used.")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/pilot-v2-validation-initial")
    parser.add_argument("--calibration", type=Path, default=ROOT / "artifacts/calibration/pilot-v2-initial")
    parser.add_argument("--scope", type=Path, default=ROOT / "artifacts/scope/pilot-v2-initial")
    parser.add_argument("--answer", type=Path, default=ROOT / "artifacts/answer/pilot-v2-initial")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/pilot-v2-figures")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output directory already exists; preserve evidence and choose a fresh directory")
        for source in (args.report, args.calibration, args.scope, args.answer):
            target, source = args.output.resolve(), source.resolve()
            if target == source or target in source.parents or source in target.parents:
                raise ValueError("Output must be separate from source evidence directories")
        paths, hashes, manifest, metrics, rows, oracle, cal, grid, raw, calibrated, training = evidence(
            args.report, args.calibration, args.scope, args.answer)
    except (OSError, KeyError, ValueError, TypeError) as error:
        parser.error(str(error))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlesize": 11,
                         "text.color": NAVY, "axes.labelcolor": NAVY,
                         "svg.hashsalt": "editguard-pilot-validation-figures-v1"})
    figures = [("validation-outcomes", outcomes_figure(plt, metrics, rows, oracle)),
               ("scope-calibration", calibration_figure(plt, metrics, cal, grid, raw, calibrated)),
               ("training-resources", training_figure(plt, training))]
    args.output.mkdir(parents=True)
    outputs = {}
    description = json.dumps({"source_files_sha256": hashes, "script_sha256": digest(Path(__file__)),
                              "purpose": "validation_development", "test_rows_read": False}, sort_keys=True)
    for name, figure in figures:
        for suffix in (".png", ".svg"):
            output = args.output / (name + suffix)
            metadata = {"Description": description}
            if suffix == ".svg":
                metadata["Date"] = None
            figure.savefig(output, dpi=180, metadata=metadata)
            if suffix == ".svg":
                output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
            outputs[output.name] = digest(output)
        plt.close(figure)
    if any(digest(paths[key]) != value for key, value in hashes.items()):
        raise RuntimeError("Input evidence changed during plotting")
    base = {row["id"]: row for row in rows if row["configuration"] == "unchanged_base"}
    summary = {"schema_version": 1, "purpose": "validation_development_figures", "test_rows_read": False,
               "dataset_id": manifest["dataset_id"], "review_policy": manifest["review_policy"],
               "human_reviewed_data": manifest["human_reviewed_data"], "record_count": len(rows),
               "question_count": metrics["question_count"], "family_count": len({row["family_id"] for row in rows}),
               "source_files_sha256": hashes, "source_unchanged": True, "script_sha256": digest(Path(__file__)),
               "matplotlib_version": matplotlib.__version__, "outputs_sha256": outputs,
               "literal_out_of_scope_preservation_diagnostic": {
                   name: {"numerator": sum(row["answer"] == base[row["id"]]["answer"] for row in rows
                                           if row["configuration"] == name and not row["applicable"]),
                          "denominator": sum(row["configuration"] == name and not row["applicable"] for row in rows)}
                   for name in CONFIGURATIONS if name != "unchanged_base"},
               "disclaimer": "Strict credited assertion counts retain pending semantic review and every answerable row. These same-validation-set development figures do not establish baseline factual failure, model superiority, test generalization or HTTP end-to-end latency."}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "source_unchanged": True,
                      "plots": list(outputs), "validation_questions": metrics["question_count"]}, indent=2))


if __name__ == "__main__":
    main()

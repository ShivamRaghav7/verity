"""Render measured VLKEB evidence as CPU-only figures and auditable CSV/JSON.

No model is loaded and no metric is estimated here. Sources must already exist.
--test-fixture marks every figure as synthetic verification, never project results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS = ("base", "prompt_only", "updated_rag", "mend_context", "ft_last_layer")
CATEGORIES = ("direct", "paraphrase", "alternate_image", "text_locality", "visual_locality")
LABELS = {"base": "Base", "prompt_only": "Prompt only", "updated_rag": "Updated RAG", "mend_context": "MEND (adapted)",
          "ft_last_layer": "FT (adapted)", "direct": "Direct", "paraphrase": "Paraphrase",
          "alternate_image": "Other image", "text_locality": "Text locality", "visual_locality": "Visual locality"}
COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00")


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def number(value, *, lower=0, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite measured number: {value!r}")
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"Measured number is outside permitted range: {value}")
    return value


def evaluation_data(report):
    aggregate = report["aggregate"]
    if "own_measured" not in aggregate.get("provenance", ""):
        raise ValueError("Evaluation must declare its own-measured provenance")
    scores, latency = [], []
    for method in METHODS:
        for category in CATEGORIES:
            cell = aggregate["scores"][method][category]
            if cell["denominator"] != 20:
                raise ValueError("This renderer expects the sealed 20-edit denominator")
            correct = number(cell["correct"], upper=20)
            accuracy = number(cell["accuracy"], upper=1)
            if correct != int(correct) or abs(accuracy - correct / 20) > 1e-9:
                raise ValueError("Accuracy disagrees with its integer numerator/denominator")
            scores.append({"method": method, "method_label": LABELS[method], "category": category,
                           "category_label": LABELS[category], **cell})
            latency.append({"kind": "query", "method": method, "category": category,
                            **aggregate["query_latency"][method][category]})
    for method, cell in aggregate["edit_latency"].items():
        latency.append({"kind": "edit", "method": method, "category": None, **cell})
    for cell in latency:
        for key in ("p50_seconds", "p95_seconds"):
            if cell[key] is not None:
                number(cell[key])
    paired = aggregate["paired_differences"]
    for cell in paired:
        interval = cell["paired_edit_bootstrap_95_percent_interval"]
        if cell["unit_count"] != 20 or len(interval) != 2:
            raise ValueError("Paired intervals must use the same 20 edit units")
        number(cell["difference"], lower=-1, upper=1)
        if not (-1 <= interval[0] <= interval[1] <= 1):
            raise ValueError("Invalid paired interval")
    return {"execution_status": report.get("execution_status"), "scores": scores,
            "paired_differences": paired, "latency": latency,
            "latency_protocol": aggregate["latency_protocol"]}


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in rows)


def save(plt, figure, output, name, suffix):
    figure.suptitle(figure._suptitle.get_text() + suffix if figure._suptitle else suffix.strip(), fontsize=12)
    figure.savefig(output / (name + ".png"), dpi=200, bbox_inches="tight")
    figure.savefig(output / (name + ".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_evaluation(plt, data, output, suffix):
    figure, axis = plt.subplots(figsize=(10, 4.8), layout="constrained")
    spacing = .82 / len(METHODS)
    center = (len(METHODS) - 1) / 2
    for i, method in enumerate(METHODS):
        rows = [r for r in data["scores"] if r["method"] == method]
        positions = [j + (i - center) * spacing for j in range(len(CATEGORIES))]
        bars = axis.bar(positions, [r["accuracy"] * 100 for r in rows], width=spacing * .95,
                        color=COLORS[i], label=LABELS[method])
        # Alternate label heights so adjacent equal-count bars remain distinct.
        axis.bar_label(bars, labels=[f'{r["correct"]}/20' for r in rows], fontsize=6.5,
                       padding=2 + 4 * (i % 2))
    axis.set(xticks=range(5), xticklabels=[LABELS[c] for c in CATEGORIES], ylim=(0, 112),
             ylabel="Strict generated-answer accuracy (%)")
    axis.legend(ncols=len(METHODS), loc="upper center", fontsize=8)
    figure.suptitle("Measured held-out results: 20 edit units per category")
    figure.text(.01, -.015, "Abstentions, errors and unexecuted cases remain incorrect in denominators. Locality here is factual correctness.", fontsize=8)
    save(plt, figure, output, "accuracy", suffix)
    comparisons = list(dict.fromkeys((r["first"], r["second"]) for r in data["paired_differences"]))
    if comparisons:
        columns = 2 if len(comparisons) > 4 else 1
        panel_rows = math.ceil(len(comparisons) / columns)
        figure, axes = plt.subplots(panel_rows, columns, figsize=(7 * columns, 2.8 * panel_rows),
                                    squeeze=False, layout="constrained", sharex=True)
        for axis, pair in zip(axes.flat, comparisons):
            rows = [r for r in data["paired_differences"] if (r["first"], r["second"]) == pair]
            for j, row in enumerate(rows):
                lo, hi = row["paired_edit_bootstrap_95_percent_interval"]
                axis.plot([lo * 100, hi * 100], [j, j], color=COLORS[j % 5], lw=2)
                axis.plot(row["difference"] * 100, j, "o", color=COLORS[j % 5])
            axis.axvline(0, color=".5", lw=.8)
            axis.set(yticks=range(len(rows)), yticklabels=[LABELS[r["category"]] for r in rows],
                     xlim=(-105, 105), title=f"{LABELS[pair[0]]} minus {LABELS[pair[1]]}")
            axis.tick_params(axis="x", labelbottom=True)
        for axis in list(axes.flat)[len(comparisons):]:
            axis.set_visible(False)
        figure.supxlabel("Accuracy difference (percentage points); paired edit bootstrap 95% interval")
        figure.suptitle("Paired uncertainty: 20 edit units; intervals copied from evaluation")
        save(plt, figure, output, "paired_intervals", suffix)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for axis, quantile in zip(axes, ("p50_seconds", "p95_seconds")):
        for i, method in enumerate(METHODS):
            rows = [r for r in data["latency"] if r["kind"] == "query" and r["method"] == method]
            valid = [(j, r) for j, r in enumerate(rows) if r[quantile] is not None]
            axis.bar([j + (i - center) * spacing for j, _ in valid], [r[quantile] for _, r in valid],
                     width=spacing * .95, color=COLORS[i], label=LABELS[method])
        axis.set(xticks=range(5), xticklabels=[LABELS[c] for c in CATEGORIES], ylabel="Seconds", title=quantile[:3])
        axis.tick_params(axis="x", labelrotation=25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Measured query latency; no pooled categories or excluded warmups")
    timed_counts = [r.get("sample_count") for r in data["latency"] if r["kind"] == "query"]
    if timed_counts and all(isinstance(n, int) and not isinstance(n, bool) for n in timed_counts):
        counts = sorted(set(timed_counts))
        count_note = (f"Timed n={counts[0]} for every method × question category." if len(counts) == 1 else
                      f"Timed n varies from {counts[0]} to {counts[-1]} by method × question category; exact counts are in latency.csv.")
    else:
        count_note = "Some timed sample counts are unavailable; inspect latency.csv for method × question-category counts."
    figure.text(.01, -.015, count_note, fontsize=8)
    save(plt, figure, output, "query_latency", suffix)
    rows = [r for r in data["latency"] if r["kind"] == "edit"]
    if rows:
        figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
        for i, quantile in enumerate(("p50_seconds", "p95_seconds")):
            valid = [(j, r) for j, r in enumerate(rows) if r[quantile] is not None]
            bars = axis.bar([j + (i - .5) * .3 for j, _ in valid], [r[quantile] for _, r in valid],
                            width=.29, color=COLORS[i], label=quantile[:3])
            axis.bar_label(bars, labels=[f'n={r["sample_count"]}' for _, r in valid], fontsize=8, padding=2)
        axis.set(xticks=range(len(rows)), xticklabels=[LABELS[r["method"]] for r in rows], ylabel="Seconds")
        axis.legend()
        figure.suptitle("Measured edit application latency; model loading excluded")
        save(plt, figure, output, "edit_latency", suffix)


def plot_smoke(plt, report, output, suffix):
    resources = []
    for phase, state in report["phase_execution"].items():
        resource = state.get("resources")
        if resource:
            resources.append({"phase": phase, "execution_status": state["execution_status"], **resource})
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    checks = report["technical_checks"]
    axes[0].bar(["Passed checks", "Recorded checks"], [sum(v is True for v in checks.values()), len(checks)], color=COLORS[:2])
    axes[0].set(ylabel="Check count", title=f'Technical gate: {report["technical_gate_passed"]}')
    for i, row in enumerate(resources):
        values = [row["sampled_peak_process_rss_bytes"], row["gpu"]["peak_allocated_bytes"], row["gpu"]["peak_reserved_bytes"]]
        axes[1].bar([j + (i - .5) * .3 for j in range(3)], [v / 2**30 for v in values], width=.28, color=COLORS[i], label=row["phase"])
    axes[1].set(xticks=range(3), xticklabels=["Host RSS", "CUDA allocated", "CUDA reserved"], ylabel="GiB")
    if resources:
        axes[1].legend()
    figure.suptitle("T4 development smoke: one training-seen edit; mechanical checks and memory")
    save(plt, figure, output, "development_smoke", suffix)
    return {"execution_status": report["execution_status"], "technical_gate_passed": report["technical_gate_passed"],
            "technical_checks": checks, "direct_edit_target_matched": report["direct_edit_target_matched"], "resources": resources}


def plot_diagnosis(plt, report, output, suffix):
    rows = []
    for state in ("baseline", "loaded_patch"):
        for form in ("raw", "vqa"):
            direct = next(r for r in report[state][form] if r["id"] == "direct")
            for alignment in ("release", "context"):
                score = report["teacher_forced"][state][form][alignment]
                rows.append({"state": state, "question_format": form, "metric": "teacher_forced_" + alignment,
                             "value": number(score["token_accuracy"], upper=1), "denominator": len(score["target_ids"]),
                             "conditioning": "full target in input"})
            rows.append({"state": state, "question_format": form, "metric": "whole_generated_answer",
                         "value": int(direct["strict_target_match"]), "denominator": 1,
                         "conditioning": "question/image only", "answer": direct["answer"]})
    figure, axis = plt.subplots(figsize=(10, 4.7), layout="constrained")
    metrics = ("teacher_forced_release", "teacher_forced_context", "whole_generated_answer")
    groups = [(s, f) for s in ("baseline", "loaded_patch") for f in ("raw", "vqa")]
    for i, metric in enumerate(metrics):
        selected = [next(r for r in rows if (r["state"], r["question_format"]) == g and r["metric"] == metric) for g in groups]
        bars = axis.bar([j + (i - 1) * .23 for j in range(4)], [r["value"] * 100 for r in selected], width=.22, color=COLORS[i], label=metric.replace("_", " "))
        axis.bar_label(bars, labels=[f'n={r["denominator"]}' for r in selected], fontsize=8, padding=2)
    axis.set(xticks=range(4), xticklabels=[s.replace("_", " ") + "\n" + f for s, f in groups],
             ylim=(0, 112), yticks=range(0, 101, 20), ylabel="Token correctness / whole-answer match (%)")
    axis.legend(fontsize=8, loc="upper left")
    figure.suptitle("Development diagnosis: one training-seen direct question; formats are paired conditions")
    figure.text(.01, -.015, "Teacher forcing includes the full target. Whole-answer matching scores a target-free continuation; n=1 is not benchmark accuracy.", fontsize=8)
    save(plt, figure, output, "development_teacher_forcing", suffix)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation", "smoke", "diagnosis"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fixture", action="store_true", help="Label synthetic renderer verification, never measured project results")
    args = parser.parse_args(argv)
    sources = {name: getattr(args, name) for name in ("evaluation", "smoke", "diagnosis") if getattr(args, name)}
    if not sources:
        parser.error("Supply at least one existing evidence JSON file")
    loaded = {name: load(path) for name, path in sources.items()}
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory to preserve existing figures")
    data = {"evidence_kind": "SYNTHETIC TEST FIXTURE" if args.test_fixture else "own measured run evidence",
            "sources": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
            "method_labels": LABELS, "limitations": "Charts copy saved metrics; they do not validate scientific provenance or establish faithful upstream reproduction."}
    if "evaluation" in loaded:
        data["evaluation"] = evaluation_data(loaded["evaluation"])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .18, "axes.axisbelow": True,
                         "pdf.fonttype": 42})
    args.output.mkdir(parents=True)
    suffix = " — SYNTHETIC TEST FIXTURE" if args.test_fixture else ""
    if "evaluation" in data:
        plot_evaluation(plt, data["evaluation"], args.output, suffix)
        for key in ("scores", "paired_differences", "latency"):
            write_csv(args.output / (key + ".csv"), data["evaluation"][key])
    if "smoke" in loaded:
        data["smoke"] = plot_smoke(plt, loaded["smoke"], args.output, suffix)
        write_csv(args.output / "smoke_resources.csv", data["smoke"]["resources"])
    if "diagnosis" in loaded:
        data["diagnosis"] = plot_diagnosis(plt, loaded["diagnosis"], args.output, suffix)
        write_csv(args.output / "diagnosis.csv", data["diagnosis"])
    (args.output / "figure_data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "evidence_kind": data["evidence_kind"], "files": sorted(p.name for p in args.output.iterdir())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

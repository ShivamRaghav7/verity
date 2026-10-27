"""Generate a current-backend report and scientific figures from actual frozen receipts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from editguard_benchmark.metrics import aggregate
from editguard_benchmark.data import digest

NAMES = {"R0": "Initial facts · ReasonKE", "G-RAG": "Qwen + updated RAG", "G-ALL": "Qwen + full memory",
         "R-RAG": "ReasonKE + updated RAG", "R-ALL": "ReasonKE + full memory"}
COLORS = {"R0": "#81939b", "G-RAG": "#bf793b", "G-ALL": "#d9b473", "R-RAG": "#5f8fb6", "R-ALL": "#167e72"}


def generate(receipts, output):
    output = Path(output)
    if output.exists():
        raise ValueError("Report directory already exists; preserve it and choose a new derived output")
    report = aggregate(receipts)
    if report["phase"] not in ("development", "final") or set(NAMES) - set(report["conditions"]):
        raise ValueError("A matched development or final comparison with all five configurations is required")
    assigned = 256 if report["phase"] == "final" else 24
    independent = report["independent_lineages"]
    per_category = assigned // 8
    if any(report["conditions"][name]["overall"]["assigned"] != assigned for name in NAMES):
        raise ValueError("Final question denominators do not match the fixed protocol")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "figure.facecolor": "white"})
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raw = [json.loads(Path(path).read_text(encoding="utf-8")) for path in receipts]
    records = [row for receipt in raw for row in receipt["records"]]
    names = list(NAMES)
    categories = list(report["conditions"]["R-ALL"]["by_category"])
    files = {}

    def save(fig, name):
        for extension in ("png", "svg"):
            target = output / f"{name}.{extension}"
            fig.savefig(target, dpi=190, bbox_inches="tight", metadata={"Description": "Derived from measured, frozen EditGuard operations receipts; no model inference performed."})
            files[target.name] = digest(target)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    x = np.arange(len(categories))
    for index, name in enumerate(names):
        values = [100 * report["conditions"][name]["by_category"][category]["accuracy"] for category in categories]
        ax.bar(x + (index - 2) * .15, values, width=.14, color=COLORS[name], label=NAMES[name])
    category_labels = {"same_entity_other_relation": "same entity\nother property", "similar_entity_same_relation": "similar entity\nsame property", "wrong_version_or_tier": "older release"}
    ax.set_xticks(x, [category_labels.get(category, category.replace("_", "\n")) for category in categories], fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("Correct / all assigned questions (%)")
    ax.set_title("Factual accuracy by question category")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False, fontsize=9)
    fig.text(.07, -.04, f"{per_category} assigned questions per category and configuration. Errors, abstentions and unexecuted cases receive zero primary credit.", fontsize=9)
    save(fig, "category-accuracy")

    fig, ax = plt.subplots(figsize=(8.8, 6))
    coordinates = {}
    for name in names:
        condition = report["conditions"][name]
        a, l = 100 * condition["applicable"]["accuracy"], 100 * condition["locality"]["accuracy"]
        ax.scatter(l, a, s=110, color=COLORS[name], label=NAMES[name], edgecolor="white")
        coordinates.setdefault((l, a), []).append(name)
    for (l, a), labels in coordinates.items():
        # The legend identifies distinct points. Label only exact overlaps;
        # individual labels collide when several methods are near 100 percent.
        if len(labels) > 1:
            ax.annotate(" / ".join(labels), (l, a), xytext=(-10, -20 if a > 95 else 10), textcoords="offset points", fontsize=10, ha="right")
    ax.set_xlim(-3, 105); ax.set_ylim(-3, 105); ax.set_xlabel(f"Locality factual accuracy · {assigned // 2} assigned questions (%)")
    ax.set_ylabel(f"Applicable-edit accuracy · {assigned // 2} assigned questions (%)")
    ax.set_title("Correction success and locality are separate outcomes")
    ax.grid(alpha=.15); ax.legend(loc="lower left", frameon=False, fontsize=9)
    save(fig, "efficacy-locality")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(names))
    for outcome, color in (("correct", "#167e72"), ("wrong", "#c36749"), ("abstain", "#c8a353"),
                           ("execution_error", "#78648f"), ("unexecuted", "#d6dce0")):
        values = np.array([report["conditions"][name]["overall"]["counts"][outcome] for name in names])
        ax.barh(np.arange(len(names)), values, left=bottom, color=color, label=outcome.replace("_", " "))
        bottom += values
    ax.set_yticks(np.arange(len(names)), [NAMES[name] for name in names]); ax.invert_yaxis()
    ax.set_xlim(0, assigned); ax.set_xlabel(f"All {assigned} assigned questions per main configuration")
    ax.set_title("Every outcome remains visible")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -.15), frameon=False)
    save(fig, "outcome-composition")

    pairs = report["paired_differences"]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for index, (label, value) in enumerate(pairs.items()):
        middle = 100 * value["left_minus_right"]
        low, high = [100 * limit for limit in value["ci95"]]
        ax.errorbar(middle, index, xerr=[[middle - low], [high - middle]], fmt="o", color="#167e72", capsize=5)
    ax.set_yticks(range(len(pairs)), list(pairs)); ax.axvline(0, color="#8a989c", linewidth=1)
    ax.set_xlabel("Paired accuracy difference (percentage points); 95% lineage-bootstrap interval")
    ax.set_title(f"Matched configuration differences · {independent} independent edit lineages")
    save(fig, "paired-differences")

    fig, ax = plt.subplots(figsize=(10, 5))
    latencies = [[row["pipeline_seconds"] for row in records if row["condition"] == name and row.get("response")]
                 for name in names]
    nonempty = [(name, values) for name, values in zip(names, latencies) if values]
    if nonempty:
        ax.boxplot([values for _, values in nonempty], tick_labels=[name for name, _ in nonempty], showfliers=True)
    ax.set_ylabel("Warm pipeline request seconds")
    ax.set_title("Latency includes context selection, local IPC and generation")
    fig.text(.08, -.02, "Loading and browser network excluded. Execution-error timings and generated n are disclosed separately in the report.", fontsize=9)
    save(fig, "latency")

    with (output / "scorecard.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "category", "correct", "assigned", "wrong", "abstain", "execution_error", "unexecuted", "accuracy", "coverage"])
        for name, condition in report["conditions"].items():
            for category, metric in condition["by_category"].items():
                c = metric["counts"]
                writer.writerow([name, category, c["correct"], metric["assigned"], c["wrong"], c["abstain"], c["execution_error"], c["unexecuted"], metric["accuracy"], metric["coverage"]])

    lines = ["# EditGuard: controlled operations evaluation (" + report["phase"] + ")", "",
        f"This report is generated from actual model receipts. Phase: {report['phase']}; execution: {report['execution_status']}. The independent sample consists of {independent} fictional edit lineages; wording variants are correlated.", "",
        "## Model and context comparison", "", "| Configuration | Correct / assigned | Applicable | Locality | Coverage | p50 / p95 pipeline seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in names:
        condition = report["conditions"][name]; m = condition["overall"]; latency = m["generated_latency"]
        rate = lambda metric: str(metric["counts"]["correct"]) + "/" + str(metric["assigned"])
        times = "unavailable" if not latency["n"] else f"{latency['p50_seconds']:.2f} / {latency['p95_seconds']:.2f} (n={latency['n']})"
        lines.append(f"| {NAMES[name]} | {rate(m)} | {rate(condition['applicable'])} | {rate(condition['locality'])} | {100*m['coverage']:.1f}% | {times} |")
    lines += ["", "![Accuracy by category](category-accuracy.png)", "", "![Correction and locality](efficacy-locality.png)",
              "", "![All assigned outcomes](outcome-composition.png)", "", "## Paired differences", ""]
    for label, metric in pairs.items():
        low, high = metric["ci95"]
        lines.append(f"- {label}: {100*metric['left_minus_right']:.2f} percentage points; 95% grouped interval [{100*low:.2f}, {100*high:.2f}].")
    lines += ["", "![Paired uncertainty](paired-differences.png)", "", "## Retrieval and generation", ""]
    for name in ("G-RAG", "R-RAG"):
        r = report["conditions"][name]["retrieval"]
        lines.append(f"- {name}: {r['fact_recall_numerator']}/{r['fact_recall_denominator']} required facts found; complete two-hop support {r['two_hop_complete_numerator']}/{r['two_hop_complete_denominator']}. Shared retrieval is not two independent retrieval experiments.")
    for name in ("G-ORACLE", "R-ORACLE"):
        if name in report["conditions"]:
            m = report["conditions"][name]["overall"]
            lines.append(f"- {name}: {m['counts']['correct']}/{m['assigned']} correct with gold supporting facts. This isolates a generation diagnostic on preselected questions.")
    lines += ["", "## Preservation and false edited-value endorsements", ""]
    for name in ("R-ALL", "R-RAG"):
        condition = report["conditions"][name]; retention = condition.get("baseline_correct_retention", {})
        lines.append(f"- {name}: preserved {retention.get('correct', 0)}/{retention.get('denominator', 0)} R0-correct locality answers; false edited-value endorsements {condition['locality']['false_correction_applications']}/{condition['locality']['assigned']}. A zero preservation denominator is undefined.")
    lines += ["", "## Runtime and resource use", "", "![Pipeline latency](latency.png)", ""]
    for model, sessions in report["model_measurements"].items():
        for session in sessions:
            hardware = session["hardware"]
            lines.append(f"- {model}: {session['model_id']} at {session['revision']}; {hardware['name']}; loading {session['load_seconds']:.2f} seconds; peak reserved CUDA during loading {hardware['load_peak_reserved_bytes']} bytes. NF4 weights, FP16 compute, greedy decoding.")
    lines += ["", "## Failure examples", "", "Every failure remains in the raw receipts. The following is a deterministic first-12 subset in record-ID order, not a claim to cover all failure modes.", "",
              "| Case | Condition | Expected | Actual / error | Outcome |", "| --- | --- | --- | --- | --- |"]
    failures = sorted((row for row in records if row["condition"] != "R0" and not row["correct"]), key=lambda row: row["id"])[:12]
    def safe(value): return str(value).replace("|", "\\|").replace("\n", " ")[:220]
    for row in failures:
        actual = row.get("response", {}).get("content", row.get("error", row["outcome"]))
        lines.append(f"| {row['question']['id']} | {row['condition']} | {safe(row['question']['expected_answer'])} | {safe(actual)} | {row['outcome']} |")
    lines += ["", "## Interpretation and limits", "",
        "The study measures short factual answers in a controlled fictional handbook, with shared wording templates and automated graph auditing. It is not an estimate of unrestricted dialogue accuracy. Both model checkpoints were trained upstream; this project did not perform their SFT or RL training. Comparing them does not isolate the causal effect of ReasonKE training.", "",
        "R0 is an initial-fact diagnostic scored against post-edit targets, not a measure of its original-world factual accuracy. Whole-final-span alias matching is strict. Correct targets mentioned only as examples, rejections or explanations are not credited. All abstentions, execution errors and unexecuted records stay in the primary denominator. State-machine tests, deterministic answers and neural answer accuracy remain separate evidence.", "",
        "Correct final answers do not establish faithful generated explanations. The earlier development traces contained unsupported claims; explanation auditing must be reported separately. The comparison alone does not prove arbitrary edit safety or semantic conflict detection.", "",
        "The concrete outcome is the reproducible scorecard above plus an implemented reversible maintenance workflow. No victory over updated RAG or universal accuracy is assumed.", "",
        "## Provenance", "", "See `results.json`, `scorecard.csv` and `figure-manifest.json`; each figure is derived from saved records. Raw receipts identify model revisions, fact IDs, prompt digests, tokens, errors and resource measurements."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {"schema": "editguard-operations-figures-v1", "source_receipts": {str(p): digest(p) for p in receipts},
                "script_sha256": digest(__file__), "model_inference_executed": False,
                "matplotlib_version": matplotlib.__version__, "figures": files}
    (output / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.receipts, args.output), indent=2))


if __name__ == "__main__":
    main()

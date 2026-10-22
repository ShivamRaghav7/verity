"""Render measured study-v2 results with counts and grouped uncertainty.

These graphs consume saved reports only. They never load CLIP, fit parameters,
select configurations or fabricate missing observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import file_hash


METHOD_NAMES = {"base": "Original", "object_only": "Object-only edit",
                "protected": "Protected edit", "finetune": "Matched fine-tuning",
                "oracle_mask": "Oracle mask*"}
COLORS = {"base": "#64748b", "object_only": "#e08a38", "protected": "#197b85",
          "finetune": "#7253a4", "oracle_mask": "#a3ad41"}
WORDS = ("camera", "laptop", "watch")
METHODS = tuple(METHOD_NAMES)


def _percent(metric):
    return 100 * metric["numerator"] / metric["denominator"] if metric["denominator"] else float("nan")


def _count(metric):
    return f"{metric['numerator']}/{metric['denominator']}" if metric["denominator"] else "n/a"


def _export(figure, output, name):
    files = []
    for suffix in ("png", "svg"):
        path = output / (name + "." + suffix)
        figure.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        files.append(path)
    return files


def plot_report(report_path: Path, output: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed_frozen_selectivity_evaluation":
        raise ValueError("Graphs require a completed frozen selectivity report")
    if set(report["results"]) != set(WORDS) or report.get("methods") != list(METHODS):
        raise ValueError("Report words or comparison methods differ from the study")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    files = []
    metrics = (("target_distractor_accuracy", "Misleading target word"),
               ("clean_object_accuracy", "Clean object recognition"),
               ("legitimate_text_accuracy", "Legitimate text reading"))
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharey=True)
    for row_index, word in enumerate(WORDS):
        for column, (key, title) in enumerate(metrics):
            axis = axes[row_index, column]
            values = [report["results"][word][method][key] for method in METHODS]
            axis.bar(range(len(METHODS)), [_percent(value) for value in values],
                     color=[COLORS[method] for method in METHODS], width=.7)
            for index, metric in enumerate(values):
                axis.text(index, min(_percent(metric) + 2.2, 105), _count(metric),
                          ha="center", va="bottom", fontsize=9)
            axis.set_ylim(0, 112)
            axis.set_yticks((0, 25, 50, 75, 100))
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
            axis.set_xticks(range(len(METHODS)), [METHOD_NAMES[method] for method in METHODS], rotation=24, ha="right", fontsize=8)
            if row_index == 0:
                axis.set_title(title, fontsize=12, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{word.capitalize()} edit\nAccuracy (%)")
    figure.suptitle("Does repairing one visual association preserve other uses?", fontsize=17, y=.995)
    figure.text(.01, .002,
                "Held-out source photographs; annotations are correct / attempted questions. Two overlay positions share each photo.\n"
                "Text cards are controlled word-reading tasks, not natural-scene OCR. *Oracle masking knows the synthetic overlay location and task.",
                fontsize=9)
    figure.tight_layout(rect=(0, .055, 1, .965))
    files.extend(_export(figure, output, "selectivity"))
    plt.close(figure)

    sequences = report["sequential"]
    if len(sequences) != 6 or any(len(order["stages"]) != 3 for order in sequences):
        raise ValueError("Interference graph requires six complete three-stage edit orders")
    order_labels = [" → ".join(word.capitalize() for word in order["order"]) for order in sequences]
    attacked, preservation, final_values = [], [], []
    for order in sequences:
        attacked.append([stage["results"][word]["target_distractor_accuracy"]
                         for stage in order["stages"] for word in WORDS])
        preservation.append([stage["results"][WORDS[0]][key] for stage in order["stages"]
                             for key in ("clean_object_accuracy", "legitimate_text_accuracy")])
        last = order["stages"][-1]["results"]
        final_values.append([last[word]["target_distractor_accuracy"] for word in WORDS] + [
            last[WORDS[0]]["clean_object_accuracy"], last[WORDS[0]]["legitimate_text_accuracy"]])
    panels = ((attacked, [f"After {stage}\n{word}" for stage in range(1, 4) for word in WORDS],
               "Target-word object accuracy after each sequential edit"),
              (preservation, [f"After {stage}\n{task}" for stage in range(1, 4) for task in ("clean", "text")],
               "Clean-object and legitimate-text preservation after each edit"),
              (final_values, ["Camera target", "Laptop target", "Watch target", "Clean objects", "Legitimate text"],
               "Final state after all three edits"))
    figure, axes = plt.subplots(3, 1, figsize=(15, 13), gridspec_kw={"height_ratios": [1, 1, 1]})
    for axis, (values, labels, title) in zip(axes, panels):
        matrix = np.array([[_percent(metric) for metric in row] for row in values])
        heat = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(6), order_labels)
        axis.set_title(title, fontsize=12, pad=12)
        axis.spines[:].set_visible(False)
        for row, metrics_row in enumerate(values):
            for column, metric in enumerate(metrics_row):
                axis.text(column, row, _count(metric), ha="center", va="center", fontsize=9,
                          color="black" if matrix[row, column] >= 55 else "white")
        figure.colorbar(heat, ax=axis, pad=.02, fraction=.025, label="Accuracy (%)")
    figure.suptitle("Sequential edits: interference and order dependence", fontsize=17, y=.995)
    figure.text(.01, .002,
                "All six orders and 18 parameter edits were fitted on training data before final inference. Same held-out tasks at every stage.\n"
                "Counts describe repeated measurements, not additional independent photos. Exact parameter rollback is recorded separately.", fontsize=9)
    figure.tight_layout(rect=(0, .05, 1, .97), h_pad=2.7)
    files.extend(_export(figure, output, "interference"))
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    comparison_methods = METHODS[1:]
    for row_index, word in enumerate(WORDS):
        for column, key in enumerate(("target_distractor", "legitimate_text_macro")):
            axis = axes[row_index, column]
            for index, method in enumerate(comparison_methods):
                interval = report["paired_uncertainty_vs_base"][word][method][key]
                value, bounds = interval["difference"], interval["interval_95"]
                if value is None:
                    continue
                axis.plot([100 * bounds[0], 100 * bounds[1]], [index, index], color=COLORS[method], linewidth=2)
                axis.scatter([100 * value], [index], color=COLORS[method], s=35, zorder=3)
            axis.axvline(0, color="#64748b", linestyle="--", linewidth=1)
            axis.set_yticks(range(len(comparison_methods)), [METHOD_NAMES[method] for method in comparison_methods])
            axis.invert_yaxis()
            axis.grid(axis="x", alpha=.2)
            axis.set_title(f"{word.capitalize()}: " + ("target-word objects" if column == 0 else "legitimate text"))
            if row_index == 2:
                axis.set_xlabel("Accuracy difference from original (percentage points)")
    figure.suptitle("Paired uncertainty: changes relative to the same original model", fontsize=16, y=.995)
    figure.text(.01, .002,
                "95% percentile bootstrap intervals. Object tasks resampled by source photo; both positions remain grouped.\n"
                "Text tasks resampled by eight word groups; small-group exploratory uncertainty, not a natural-scene OCR guarantee.", fontsize=9)
    figure.tight_layout(rect=(0, .06, 1, .96))
    files.extend(_export(figure, output, "paired_uncertainty"))
    plt.close(figure)
    metadata = {"input_report": str(report_path.resolve()), "report_sha256": file_hash(report_path),
                "plot_code_sha256": file_hash(Path(__file__)), "matplotlib_version": matplotlib.__version__,
                "outputs": {path.name: {"sha256": file_hash(path), "bytes": path.stat().st_size} for path in files},
                "no_model_execution": True}
    (output / "figure-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/visionpatch/study-v2/final.json"))
    parser.add_argument("--output", type=Path, default=Path("output/visionpatch/study-v2"))
    args = parser.parse_args()
    files = plot_report(args.report, args.output)
    print(json.dumps({"figures": [str(path) for path in files]}, indent=2))


if __name__ == "__main__":
    main()

"""Generate scientific figures from recorded observations, never fitted values."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/visionpatch/final-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("output/visionpatch/figures"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(args.report.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "svg.hashsalt": "visionpatch-v1"})
    methods = ["base", "projection_edit", "projection_finetune", "oracle_mask"]
    names = ["Unchanged CLIP", "Projection edit", "Projection fine-tuning", "Oracle masking"]
    colors = ["#9ca9a2", "#275b45", "#789460", "#bc815a"]

    def save(fig, name):
        fig.savefig(args.output / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(args.output / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    values = [report["results"][method]["distractor_accuracy"] for method in methods]
    ax.bar(names, [100 * item["rate"] for item in values], color=colors, width=.62)
    for index, item in enumerate(values):
        ax.text(index, 100 * item["rate"] + 1.5, f"{item['numerator']}/{item['denominator']}", ha="center", fontsize=11)
    ax.set(ylim=(0, 110), ylabel="Correct predictions (%)", title="Fixed holdout: recognition with misleading printed text")
    ax.set_xticks(range(len(names)), [name.replace(" ", "\n", 1) for name in names])
    ax.text(.01, -.22, "80 source photographs · 8 candidates · one word/font/position\nAll methods: clean 80/80. Oracle masking knows the overlay location.", transform=ax.transAxes, fontsize=9, color="#526159")
    save(fig, "accuracy")

    fig, ax = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    comparisons = report["paired_attacked_accuracy_vs_base"]
    for index, (method, name, color) in enumerate(zip(methods[1:], names[1:], colors[1:])):
        item = comparisons[method]
        center = 100 * item["difference"]
        low, high = [100 * v for v in item["interval_95"]]
        ax.errorbar(center, index, xerr=[[center - low], [high - center]], fmt="o", capsize=5, color=color, markersize=7)
    ax.axvline(0, color="#8c9690", linewidth=1, linestyle="--")
    ax.set(yticks=range(3), yticklabels=names[1:], xlabel="Accuracy difference from unchanged CLIP (percentage points)",
           title="Paired uncertainty on the fixed holdout")
    ax.set_xlim(-2, 65)
    ax.text(.01, -.26, "95% percentile bootstrap intervals; 2,000 paired samples grouped by source photo.\nIntervals compare each method with the base, not editor superiority over other methods.", transform=ax.transAxes, fontsize=9, color="#526159")
    save(fig, "paired-gains")

    observations = [r for r in report["observations"] if r["variant"] == "distractor"]
    classes = list(dict.fromkeys(r["class_id"] for r in observations))
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    positions = np.arange(len(classes))
    for offset, method, label, color in [(-.17, "base", "Unchanged CLIP", colors[0]), (.17, "projection_edit", "Projection edit", colors[1])]:
        rates = [100 * sum(r[method] == category for r in observations if r["class_id"] == category)
                 / sum(r["class_id"] == category for r in observations) for category in classes]
        ax.bar(positions + offset, rates, width=.32, label=label, color=color)
    ax.set(xticks=positions, xticklabels=[c.replace("_", " ") for c in classes], ylim=(0, 110),
           ylabel="Correct predictions (%)", title="Per-category recognition: 10 held-out photos per category")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="upper left", frameon=False)
    ax.text(.01, -.25, "Patch support: cup and laptop only. Retention references include all categories.\nGenuine cameras are evaluated as cameras; no ground-truth routing or category suppression.", transform=ax.transAxes, fontsize=9, color="#526159")
    save(fig, "categories")
    print(json.dumps({"source": str(args.report), "figures": ["accuracy", "paired-gains", "categories"],
                      "formats": ["png", "svg"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

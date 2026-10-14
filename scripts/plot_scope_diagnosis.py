"""Plot archived scope diagnostics only; no models, training or test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory; do not overwrite scientific evidence")
    source = args.source / "summary.json"
    probabilities = args.source / "probabilities.jsonl"
    summary = json.loads(source.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in probabilities.read_text(encoding="utf-8").splitlines() if line]
    if (summary.get("status") != "completed" or summary.get("pair_forward_passes") != 720
        or len(records) != 720 or sha(probabilities) != summary.get("probabilities_sha256")
        or summary.get("checks", {}).get("test_rows_read") is not False):
        raise ValueError("Require completed, bound 720-pair development diagnostic")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.hashsalt": "editguard-scope-v3-diagnosis", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#ffffff"})
    names = ("initial_v2", "revision1_v3")
    labels = ("Initial model", "Balanced revision 1")
    colors = ("#c05621", "#126e82")
    figure = plt.figure(figsize=(16, 8))
    grid = GridSpec(2, 2, figure=figure, width_ratios=(1, 2.15), hspace=.40, wspace=.26)
    loss_ax = figure.add_subplot(grid[0, 0])
    roc_ax = figure.add_subplot(grid[1, 0])
    probability_ax = figure.add_subplot(grid[:, 1])
    for index, (name, color) in enumerate(zip(names, colors)):
        values = [summary["models"][name][split]["raw"]["binary_negative_log_likelihood"]
                  for split in ("train", "validation")]
        bars = loss_ax.bar([index - .17, index + .17], values, width=.32,
                          color=color, alpha=.60, edgecolor=color)
        bars[1].set_alpha(1)
        for bar in bars:
            loss_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + .025,
                         f"{bar.get_height():.3f}", ha="center", fontsize=9)
        value = summary["models"][name]["validation"]["raw"]["auroc"]["value"]
        roc_ax.bar(index, value, width=.62, color=color)
        roc_ax.text(index, value + .02, f"{value:.3f}", ha="center")
    loss_ax.axhline(-(.4 * __import__("math").log(.4) + .6 * __import__("math").log(.6)),
                    color="#475569", linestyle="--", linewidth=1)
    loss_ax.set(title="Negative log-likelihood on the SAME v3 pairs",
                ylabel="Binary NLL (lower is better)", ylim=(0, 2.05))
    loss_ax.set_xticks(range(2), labels)
    loss_ax.text(.03, .97, "Light bars: train (240)\nSolid bars: validation (120)\nDashed: constant 40% positive predictor", transform=loss_ax.transAxes,
                 va="top", fontsize=8)
    roc_ax.axhline(.5, color="#475569", linestyle="--", linewidth=1)
    roc_ax.set(title="Validation probability ranking", ylabel="AUROC (higher is better)", ylim=(0, 1.05))
    roc_ax.set_xticks(range(2), labels)
    types = ("applicable_positive", "same_entity_different_property", "different_entity_same_property",
             "similar_name", "wrong_tier", "wrong_version", "ordinary")
    tick_labels = ("Applicable\npositive", "Same entity,\ndifferent property", "Different entity,\nsame property",
                   "Similar\nname", "Wrong\ntier", "Wrong\nversion", "Ordinary\nnegative")
    for offset, (name, color) in zip((-.17, .17), zip(names, colors)):
        values = [[r["raw_probability"] for r in records if r["model"] == name
                   and r["split"] == "validation" and r["negative_type"] == kind] for kind in types]
        if [len(v) for v in values] != [48, 12, 12, 12, 12, 12, 12]:
            raise ValueError("Probability class denominators changed")
        boxes = probability_ax.boxplot(values, positions=[i + offset for i in range(len(types))],
            widths=.26, patch_artist=True, showfliers=True,
            medianprops={"color": "white", "linewidth": 1.5},
            boxprops={"facecolor": color, "edgecolor": color},
            whiskerprops={"color": color}, capprops={"color": color},
            flierprops={"marker": ".", "markersize": 4, "markerfacecolor": color, "markeredgecolor": color})
        probability_ax.plot([], [], color=color, linewidth=8, label=labels[names.index(name)])
    probability_ax.axhline(.5, color="#475569", linestyle="--", linewidth=1)
    probability_ax.set(title="Raw validation probabilities before deterministic compatibility gates",
                       ylabel="Predicted applicability probability", ylim=(-.03, 1.04))
    probability_ax.set_xticks(range(len(types)), tick_labels, fontsize=9)
    probability_ax.legend(loc="upper right", frameon=False)
    probability_ax.grid(axis="y", alpha=.15)
    figure.text(.37, .057, "48 positives; 12 per negative type. All 20 revised raw false positives at 0.5 violate entity/version compatibility.\nApplication guards reject them; this does not establish arbitrary-query reliability.",
                fontsize=9, va="bottom", color="#475569")
    figure.suptitle("EditGuard scope revision: what the 0.486 validation loss means", fontsize=17, fontweight="bold", y=.97)
    figure.text(.055, .025, "Controlled development pilot · 4 train / 2 validation families · automated review · no test evaluation · same v3 inputs for both models", fontsize=10, color="#475569")
    figure.subplots_adjust(left=.055, right=.97, top=.88, bottom=.18)
    args.output.mkdir(parents=True)
    png = args.output / "scope-diagnosis.png"
    svg = args.output / "scope-diagnosis.svg"
    figure.savefig(png, dpi=180, metadata={"Software": "EditGuard scientific diagnostics"})
    figure.savefig(svg, metadata={"Date": None, "Creator": "EditGuard scientific diagnostics"})
    plt.close(figure)
    manifest = {"artifact_type": "editguard_scope_diagnostic_figures", "purpose": "validation_development_diagnostic",
        "test_rows_read": False, "human_reviewed_data": False,
        "inputs_sha256": {str(source): sha(source), str(probabilities): sha(probabilities)},
        "outputs_sha256": {png.name: sha(png), svg.name: sha(svg)}, "script_sha256": sha(Path(__file__))}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "test_rows_read": False}, indent=2))


if __name__ == "__main__":
    main()

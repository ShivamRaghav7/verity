"""Plot immutable validation cosine evidence; no models or test data are loaded."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.calibrate_similarity import operating_metrics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evidence(directory):
    paths = {name: directory / name for name in ("calibration.json", "selection_grid.json", "validation_cosine_scores.jsonl")}
    hashes = {name: digest(path) for name, path in paths.items()}
    calibration = json.loads(paths["calibration.json"].read_text(encoding="utf-8"))
    grid = json.loads(paths["selection_grid.json"].read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in paths["validation_cosine_scores.jsonl"].read_text(encoding="utf-8").splitlines() if line]
    if (calibration.get("purpose") != "similarity_validation_calibration"
            or calibration.get("split") != "validation" or calibration.get("status") != "frozen"
            or calibration.get("test_rows_read") is not False or grid.get("rows") != len(rows)
            or calibration.get("accepted_validation_rows") != len(rows)
            or any(row.get("split") != "validation" for row in rows)):
        raise ValueError("Only complete frozen validation calibration evidence is accepted")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate validation pair identifier")
    labels, scores, gates = ([row[key] for row in rows] for key in ("label", "cosine_score", "compatible"))
    thresholds = calibration["selection"]["threshold_grid"]
    if thresholds != [item["threshold"] for item in grid["candidates"]]:
        raise ValueError("Recorded threshold grid differs from calibration")
    raw = [operating_metrics(labels, scores, threshold) for threshold in thresholds]
    gated = [operating_metrics(labels, scores, threshold, gates) for threshold in thresholds]
    for observed, recorded in zip(gated, grid["candidates"]):
        if observed != recorded:
            raise ValueError("Saved grid differs from recomputed counts and costs")
    selected = calibration["threshold"]
    raw_selected = operating_metrics(labels, scores, selected)
    gated_selected = operating_metrics(labels, scores, selected, gates)
    if (raw_selected != calibration["raw_pair_metrics"]
            or gated_selected != calibration["compatibility_gated_metrics"] or selected != grid["threshold"]):
        raise ValueError("Selected calibration metrics differ from immutable pair evidence")
    return paths, hashes, calibration, rows, raw, gated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "reports/similarity-pilot-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/similarity-pilot-v2/operating-points")
    args = parser.parse_args()
    outputs = [args.output.with_suffix(suffix) for suffix in (".png", ".svg")]
    if any(path.exists() for path in outputs):
        parser.error("Plot output exists; choose a fresh basename instead of overwriting")
    try:
        paths, hashes, calibration, rows, raw, gated = load_evidence(args.input_dir)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    navy, teal, orange, gray = "#18304b", "#138b80", "#bd612f", "#65758a"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "text.color": navy, "axes.labelcolor": navy,
        "svg.hashsalt": "editguard-similarity-validation-v1"})
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 8.3), gridspec_kw={"height_ratios": [1.35, 1]})
    figure.subplots_adjust(left=.075, right=.97, top=.81, bottom=.21, hspace=.58, wspace=.23)
    figure.suptitle("EditGuard · similarity baseline operating points", x=.5, y=.98, fontsize=19, fontweight="bold")
    figure.text(.5, .938, "Gated policy selects τ = 0.46 · raw panel uses the same threshold · validation cost: 4 × FP + FN",
                ha="center", fontsize=11, color=gray)
    selected = calibration["threshold"]
    legend_handles = []
    for column, (title, curve, selected_metrics, cost_color) in enumerate((
            ("A. Cosine alone", raw, calibration["raw_pair_metrics"], navy),
            ("B. Cosine + compatibility gates", gated, calibration["compatibility_gated_metrics"], teal))):
        top, bottom = axes[:, column]
        thresholds = [item["threshold"] for item in curve]
        handles = []
        for name, label, color, style in (("precision", "Precision", teal, "-"),
                ("recall", "Recall", navy, "-"), ("false_application_rate", "False application rate", orange, "--")):
            values = [item[name]["value"] if item[name]["value"] is not None else math.nan for item in curve]
            handle, = top.step(thresholds, values, where="post", color=color, linewidth=2.0,
                               linestyle=style, label=label)
            handles.append(handle)
            top.scatter([selected], [selected_metrics[name]["value"]], s=42, color=color,
                        edgecolor="white", linewidth=.7, zorder=5)
        legend_handles = handles
        top.set(title=title, xlim=(0, 1.01), ylim=(-.03, 1.065), ylabel="Rate (denominators shown below)")
        top.yaxis.set_major_formatter(PercentFormatter(1))
        p, r, f = (selected_metrics[key] for key in ("precision", "recall", "false_application_rate"))
        top.set_xlabel(f"τ = {selected:.2f}: precision {p['numerator']}/{p['denominator']} · recall {r['numerator']}/{r['denominator']} · FAR {f['numerator']}/{f['denominator']}", fontsize=9)
        costs = [item["mean_cost"]["value"] for item in curve]
        bottom.step(thresholds, costs, where="post", color=cost_color, linewidth=2)
        bottom.set(title="Validation cost per pair (separate y scales)", xlim=(0, 1.01),
                   xlabel="Cosine threshold (0–1.01 shown)", ylabel="(4 × FP + FN) / 120")
        bottom.set_ylim(-.03 * max(costs), 1.09 * max(costs))
        chosen_cost = selected_metrics["mean_cost"]
        bottom.scatter([selected], [chosen_cost["value"]], s=55, color=cost_color, edgecolor="white", zorder=5)
        bottom.annotate(f"{int(chosen_cost['numerator'])}/{chosen_cost['denominator']} = {chosen_cost['value']:.3f}",
            xy=(selected, chosen_cost["value"]), xytext=(12, 17), textcoords="offset points", color=cost_color,
            fontsize=10, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9},
            arrowprops={"arrowstyle": "-", "color": gray, "linewidth": .8})
        for axis in (top, bottom):
            axis.axvline(selected, color=gray, linestyle=":", linewidth=1.4, zorder=0)
            axis.grid(axis="y", alpha=.17)
            axis.set_xticks([0, .2, .4, .6, .8, 1])
    figure.legend(legend_handles, [handle.get_label() for handle in legend_handles],
        loc="upper center", bbox_to_anchor=(.5, .907), ncol=3, frameon=False, fontsize=10)
    counts = calibration["compatibility_gated_metrics"]["counts"]
    families = len({row["family_id"] for row in rows})
    footer = (
        f"{len(rows)} validation pairs · {counts['positive']} applicable / {counts['negative']} out of scope · {families} entity families · automated development review; no human label review\n"
        "Selection and reporting use the same validation set: descriptive development results, optimistic for generalization; no held-out test.\n"
        f"Compatibility gates reject {counts['incompatible_pairs']} incompatible pairs. Clarifications: 0/120; routing decision coverage: 120/120. No answers generated.\n"
        "Precision gaps mean zero applications. Cosine is not a calibrated probability; complete source grid −1..1.01, observed scores 0.317..0.718."
    )
    figure.text(.5, .069, footer, ha="center", va="center", fontsize=9, linespacing=1.6, color=gray)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    description = json.dumps({"source_files_sha256": hashes, "threshold": selected,
        "population": "validation_single_focal_pairs", "script_sha256": digest(Path(__file__))}, sort_keys=True)
    for path in outputs:
        if path.exists():
            parser.error("Output appeared during plotting; refusing overwrite")
        metadata = {"Description": description}
        if path.suffix == ".svg":
            metadata["Date"] = None
        figure.savefig(path, dpi=180, metadata=metadata)
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    if any(digest(paths[name]) != expected for name, expected in hashes.items()):
        raise RuntimeError("Input evidence changed during plotting")
    print(json.dumps({"outputs": [str(path.resolve()) for path in outputs],
        "source_files_sha256": hashes, "source_unchanged": True,
        "selected_threshold": selected, "validation_pairs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

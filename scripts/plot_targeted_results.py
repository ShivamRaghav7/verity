"""Reproduce scientific figures from completed, immutable development evidence.

Reads recorded diagnostics and archived training metadata only. No models,
datasets, test/challenge rows, partial evaluation or neural libraries are read.
Every recorded prediction is checked against its summary before plotting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_scope_neutral import summarize as scope_summary
from scripts.diagnose_answer_revision import summaries as answer_summary

NAVY, TEAL, ORANGE, GRAY, PALE = "#18304b", "#138b80", "#bd612f", "#65758a", "#ced9e4"
SCOPE_MODELS = ("revision1_raw", "revision2_neutral")
ANSWER_MODELS = ("initial", "revision")
SCOPE_LABELS = ("Revision 1: raw", "Revision 2: neutral")
ANSWER_LABELS = ("Initial: uniform", "Revision 1: first-token 8×")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evidence(scope_dir, answer_dir, audit_path):
    paths = {"scope_summary": scope_dir / "summary.json", "scope_predictions": scope_dir / "probabilities.jsonl",
             "answer_summary": answer_dir / "summary.json", "answer_records": answer_dir / "teacher_forced_records.jsonl",
             "training_audit": audit_path,
             "scope_summary_source": ROOT / "scripts/diagnose_scope_neutral.py",
             "scope_metrics_source": ROOT / "scripts/diagnose_scope_revision.py",
             "answer_metrics_source": ROOT / "scripts/diagnose_answer_revision.py"}
    hashes = {key: digest(path) for key, path in paths.items()}
    scope, answer, audit = (read_json(paths[key]) for key in ("scope_summary", "answer_summary", "training_audit"))
    for record in (scope, answer):
        if (record.get("status") != "completed" or record.get("purpose") != "validation_development_diagnostic"
                or record.get("review_policy") != "automated_development" or record.get("human_reviewed_data") is not False
                or record.get("test_or_challenge_rows_read") is not False or record.get("source_files_unchanged") is not True
                or record.get("optimizer_updates") != 0):
            raise ValueError("Only completed, frozen, machine-reviewed development diagnostics are accepted")
    if (scope.get("artifact_type") != "editguard_scope_neutral_diagnosis" or scope.get("pair_forward_passes") != 720
            or scope.get("calibration_fits") != 0 or scope.get("probabilities_sha256") != hashes["scope_predictions"]
            or answer.get("artifact_type") != "editguard_answer_revision_diagnosis" or answer.get("teacher_forced_examples") != 288
            or answer.get("generation_calls") != 0
            or answer.get("outputs_sha256", {}).get("teacher_forced_records.jsonl") != hashes["answer_records"]):
        raise ValueError("Diagnostic counts and recorded-forward file hashes must match")
    scope_rows, answer_rows = read_rows(paths["scope_predictions"]), read_rows(paths["answer_records"])
    if len(scope_rows) != 720 or len(answer_rows) != 288:
        raise ValueError("Every recorded forward must remain in its declared population")
    if set(r["model"] for r in scope_rows) != set(SCOPE_MODELS) or set(r["model"] for r in answer_rows) != set(ANSWER_MODELS):
        raise ValueError("Diagnostic model identities differ")
    for name in SCOPE_MODELS:
        for split, expected in (("train", 240), ("validation", 120)):
            rows = [r for r in scope_rows if r["model"] == name and r["split"] == split]
            if len(rows) != expected or len({r["id"] for r in rows}) != expected or scope_summary(rows) != scope["models"][name][split]:
                raise ValueError("Recomputed scope counts and metrics differ from recorded diagnosis")
        state = scope["model_states"][name]
        if state["before"] != state["after"] or state.get("all_parameters_frozen") is not True:
            raise ValueError("Scope model fingerprint is not unchanged")
    for name in ANSWER_MODELS:
        for split, expected in (("train", 96), ("validation", 48)):
            rows = [r for r in answer_rows if r["model"] == name and r["partition"] == split]
            if len(rows) != expected or len({r["id"] for r in rows}) != expected or answer_summary(rows) != answer["models"][name]["partitions"][split]:
                raise ValueError("Recomputed answer counts and both loss objectives differ from recorded diagnosis")
        state = answer["models"][name]
        if state["model_state_sha256_before"] != state["model_state_sha256_after"]:
            raise ValueError("Answer model fingerprint is not unchanged")
    shared = scope["neutral_input_overlap"]["cross_split"]
    if shared["validation_rows_with_shared_pair"] != 8 or shared["contradictory_label_pairs"] != 0:
        raise ValueError("Expected disclosed eight-row same-label abstraction overlap")
    for name in SCOPE_MODELS:
        groups = scope["models"][name]["validation"]["by"]["neutral_overlap_group"]
        if groups["shared_with_train"]["rows"] != 8 or groups["not_shared_with_train"]["rows"] != 112:
            raise ValueError("Overlap group denominators changed")
    if (audit.get("status") != "passed" or audit.get("purpose") != "independent_completed_targeted_training_artifact_audit"
            or audit.get("test_or_challenge_rows_read") is not False or audit.get("training_or_gpu_executed") is not False
            or audit.get("human_reviewed_data") is not False):
        raise ValueError("Completed independent training audit is required")
    for task in ("scope", "answer"):
        run = audit["runs"][task]
        if run.get("status") != "passed" or not all(run["checks"].values()):
            raise ValueError("Every recorded training audit check must pass")
        for relative, expected_hash in run["archives"].items():
            path = (ROOT / relative).resolve()
            if ROOT / "reports" not in path.parents or path.suffix not in (".json", ".jsonl"):
                raise ValueError("Training figures read archived report metadata only")
            if digest(path) != expected_hash:
                raise ValueError("Archived training metadata differs from audited hash")
            paths[relative] = path
            hashes[relative] = expected_hash
        measurements = run["measurements"]
        phases = ("optimizer_phase_sum_seconds", "validation_sum_seconds", "checkpoint_and_pruning_sum_seconds", "remaining_loop_overhead_seconds")
        if not math.isclose(sum(measurements[key] for key in phases), measurements["training_loop_seconds"], abs_tol=1e-7):
            raise ValueError("Recorded training phase times do not sum to training loop time")
        if measurements["full_process_wall_seconds"] is not None or measurements["startup_tokenization_export_seconds"] is not None:
            raise ValueError("Figures explicitly describe loop-only timing with startup/export unavailable")
    scope_manifest_key = "artifacts/scope/pilot-v3-neutral2/manifest.json"
    if scope["sources_sha256"][scope_manifest_key] != audit["runs"]["scope"]["model_manifest_sha256"]:
        raise ValueError("Scope diagnostic and training audit refer to different exports")
    if answer["models"]["revision"]["model_manifest_sha256"] != audit["runs"]["answer"]["model_manifest_sha256"]:
        raise ValueError("Answer diagnostic and training audit refer to different exports")
    return paths, hashes, scope, answer, audit, scope_rows


def framing(figure, heading, subtitle, footnote):
    figure.suptitle(heading, y=.98, fontsize=18, fontweight="bold", color=NAVY)
    figure.text(.5, .935, subtitle, ha="center", fontsize=10, color=GRAY)
    figure.text(.5, .055, footnote, ha="center", va="center", fontsize=9, linespacing=1.6, color=GRAY)
    figure.subplots_adjust(left=.085, right=.97, top=.84, bottom=.21, hspace=.55, wspace=.28)


def paired_bars(axis, populations, labels, names, *, values, annotations=None, percent=False):
    for index, (model, color) in enumerate(zip(names, (NAVY, TEAL))):
        positions = [position + (index - .5) * .34 for position in range(len(populations))]
        bars = axis.bar(positions, [values(model, population) for population in populations], width=.32, color=color, label=labels[index])
        if annotations:
            axis.bar_label(bars, labels=[annotations(model, population) for population in populations], padding=4, fontsize=9)
    axis.set_xticks(range(len(populations)), populations)
    axis.grid(axis="y", alpha=.15)
    axis.set_axisbelow(True)
    if percent:
        from matplotlib.ticker import PercentFormatter
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_ylim(0, 1.16)


def scope_figure(plt, scope, rows):
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    framing(figure, "EditGuard · final scope revision diagnosis", "Same 120 pilot-v3 validation pairs · raw probability threshold 0.5 · no metadata gates or calibration",
        "Automated development review; four train and two validation families. Frozen final evaluation remains pending.\n"
        "Neutral inputs use deterministic literal entity/plan/version normalization; they do not provide learned identity verification.\n"
        "Eight abstracted validation pairs overlap training inputs (all positive); the other 112 are reported separately. No generated-answer accuracy claim.")
    models = scope["models"]
    populations = ("All 120", "Unshared 112", "Shared 8")

    def statistic(model, population):
        validation = models[model]["validation"]
        return validation["raw"] if population == "All 120" else validation["by"]["neutral_overlap_group"][
            "not_shared_with_train" if population == "Unshared 112" else "shared_with_train"]

    paired_bars(axes[0, 0], populations, SCOPE_LABELS, SCOPE_MODELS,
        values=lambda m, p: statistic(m, p)["threshold_0_5"]["accuracy"]["value"],
        annotations=lambda m, p: f"{statistic(m, p)['threshold_0_5']['accuracy']['numerator']}/{statistic(m, p)['rows']}", percent=True)
    axes[0, 0].set(title="A. Raw classifier accuracy; overlap stays visible", ylabel="Correct pair decisions / group population")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(.5, .895))
    paired_bars(axes[0, 1], populations, SCOPE_LABELS, SCOPE_MODELS,
        values=lambda m, p: statistic(m, p)["binary_negative_log_likelihood"],
        annotations=lambda m, p: f"{statistic(m, p)['binary_negative_log_likelihood']:.4f}")
    axes[0, 1].set(title="B. Same-population binary negative log-likelihood", ylabel="Mean pair NLL (lower is better)", ylim=(0, .73))
    counts = ("TP / 48", "FP / 72", "TN / 72", "FN / 48")
    keys = dict(zip(counts, ("true_positive", "false_positive", "true_negative", "false_negative")))
    paired_bars(axes[1, 0], counts, SCOPE_LABELS, SCOPE_MODELS,
        values=lambda m, p: models[m]["validation"]["raw"]["threshold_0_5"][keys[p]],
        annotations=lambda m, p: str(models[m]["validation"]["raw"]["threshold_0_5"][keys[p]]))
    axes[1, 0].set(title="C. Complete raw confusion counts", ylabel="Pair count", ylim=(0, 86))
    for name, color, label in zip(SCOPE_MODELS, (NAVY, TEAL), SCOPE_LABELS):
        for target, linestyle in ((1, "-"), (0, "--")):
            probabilities = [r["raw_probability"] for r in rows if r["model"] == name and r["split"] == "validation" and r["label"] == target]
            axes[1, 1].hist(probabilities, bins=[index / 20 for index in range(21)], histtype="step", linewidth=1.8,
                color=color, linestyle=linestyle, label=label + (" · positive (48)" if target else " · negative (72)"))
    axes[1, 1].axvline(.5, color=GRAY, linestyle=":", linewidth=1)
    axes[1, 1].set(title="D. Raw probability distributions", xlabel="Predicted applicability", ylabel="Pair count", xlim=(0, 1))
    axes[1, 1].legend(frameon=False, fontsize=7.5, loc="upper center")
    axes[1, 1].grid(axis="y", alpha=.15)
    return figure


def answer_figure(plt, answer):
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    framing(figure, "EditGuard · answer polarity revision diagnosis", "Same 48 pilot-v2 validation targets · frozen teacher-forced forwards · complete-answer generation measured separately",
        "Automated development review; two validation families. Frozen final evaluation remains pending.\n"
        "Teacher forcing supplies preceding gold tokens. The first decision uses decoder-start context only; later token losses benefit from gold prefixes.\n"
        "Both loss objectives are recomputed for both exports on identical targets; revised training loss is not comparable to initial unweighted training loss.")
    validations = {name: answer["models"][name]["partitions"]["validation"] for name in ANSWER_MODELS}
    classes = ("Expected No (12)", "Expected Yes (12)")
    subset = dict(zip(classes, ("comparison_no", "comparison_yes")))
    paired_bars(axes[0, 0], classes, ANSWER_LABELS, ANSWER_MODELS,
        values=lambda m, p: validations[m]["subsets"][subset[p]]["first_token_top1_correct"]["value"],
        annotations=lambda m, p: f"{validations[m]['subsets'][subset[p]]['first_token_top1_correct']['numerator']}/12", percent=True)
    axes[0, 0].set(title="A. First-token polarity decision (24 comparisons)", ylabel="Correct first token / declared polarity class")
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
    losses = ("Unweighted\n808 target tokens", "First-token 8×\n976 total weights")
    keys = dict(zip(losses, ("unweighted_token_nll", "comparison_first_token_8x_token_nll")))
    paired_bars(axes[0, 1], losses, ANSWER_LABELS, ANSWER_MODELS,
        values=lambda m, p: validations[m]["all"][keys[p]],
        annotations=lambda m, p: f"{validations[m]['all'][keys[p]]:.4f}")
    axes[0, 1].set(title="B. Common objectives for both models; lower is better", ylabel="Token-weighted NLL", ylim=(0, .23))
    paired_bars(axes[1, 0], classes, ANSWER_LABELS, ANSWER_MODELS,
        values=lambda m, p: validations[m]["subsets"][subset[p]]["mean_first_token_nll"],
        annotations=lambda m, p: f"{validations[m]['subsets'][subset[p]]['mean_first_token_nll']:.3f}")
    axes[1, 0].set(title="C. Polarity first-token uncertainty", ylabel="Mean first-token NLL (12 rows per class)", ylim=(0, 1.3))
    paired_bars(axes[1, 1], classes, ANSWER_LABELS, ANSWER_MODELS,
        values=lambda m, p: validations[m]["subsets"][subset[p]]["other_token_mean_nll"],
        annotations=lambda m, p: f"{validations[m]['subsets'][subset[p]]['other_token_mean_nll']:.4f}")
    axes[1, 1].set(title="D. Continuation loss after gold target prefixes", ylabel="Mean NLL over remaining 202 tokens per class", ylim=(0, .057))
    return figure


def training_figure(plt, audit):
    figure, axes = plt.subplots(2, 2, figsize=(13.6, 9.6))
    framing(figure, "EditGuard · measured targeted training runs", "User-executed finite runs · RTX 4050 Laptop GPU · float32 · batch 2 · seed 42",
        "Scope stopped at 360 updates; answer stopped at 240. Scope and answer losses use different objectives and cannot be compared numerically.\n"
        "Loop timing includes validation/checkpoint writes but excludes model loading, tokenization and final export; complete command wall time was not recorded.\n"
        "CUDA peaks follow the loop's peak-counter reset. RSS is the largest recorded sample, not a lifetime high-water mark. No inference latency claim.")
    for column, task in enumerate(("scope", "answer")):
        run, axis = audit["runs"][task], axes[0, column]
        observations = run["validation_loss_observations"]
        color = NAVY if task == "scope" else TEAL
        axis.plot([o["global_step"] for o in observations], [o["loss"] for o in observations], color=color, linewidth=2, marker="o")
        for point in observations:
            axis.annotate(f"{point['loss']:.4f}", (point["global_step"], point["loss"]), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
        selected = run["selection"]
        axis.scatter([selected["global_step"]], [selected["value"]], s=120, facecolors="none", edgecolors=ORANGE, linewidth=1.6, label="Selected export", zorder=4)
        axis.set(title=f"{'A' if column == 0 else 'B'}. {task.capitalize()} validation objective",
            xlabel="Completed optimizer updates", ylabel="Pair NLL" if task == "scope" else "8× comparison-first-token NLL")
        axis.set_ylim(0, max(o["loss"] for o in observations) * 1.25)
        axis.legend(frameon=False, fontsize=8.5)
        axis.grid(alpha=.15)
    names, positions = ("Scope: 360 steps", "Answer: 240 steps"), (0, 1)
    bottom = [0., 0.]
    for key, label, color in (("optimizer_phase_sum_seconds", "Optimizer phases", NAVY),
        ("validation_sum_seconds", "Validation", TEAL), ("checkpoint_and_pruning_sum_seconds", "Checkpoints / pruning", PALE),
        ("remaining_loop_overhead_seconds", "Remaining overhead", ORANGE)):
        values = [audit["runs"][task]["measurements"][key] for task in ("scope", "answer")]
        bars = axes[1, 0].bar(positions, values, width=.5, bottom=bottom, color=color, label=label)
        for bar, value in zip(bars, values):
            if value > 10:
                axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_y() + bar.get_height()/2,
                    f"{value:.1f}", ha="center", va="center", fontsize=9, color="white" if color == NAVY else NAVY)
        bottom = [offset + value for offset, value in zip(bottom, values)]
    for position, value in zip(positions, bottom):
        axes[1, 0].text(position, value + 5, f"{value:.2f} s", ha="center", fontsize=9, fontweight="bold")
    axes[1, 0].set(title="C. Recorded loop time only", xticks=positions, xticklabels=names, ylabel="Seconds", ylim=(0, max(bottom) * 1.27))
    axes[1, 0].legend(frameon=False, fontsize=7.5, loc="upper left")
    axes[1, 0].grid(axis="y", alpha=.15)
    axes[1, 0].set_axisbelow(True)
    for offset, key, label, color in ((-.24, "peak_allocated_gpu_gib", "CUDA allocated peak", NAVY),
        (0, "peak_reserved_gpu_gib", "CUDA reserved peak", TEAL), (.24, "max_sampled_rss_gib", "Maximum sampled RSS", PALE)):
        values = [audit["runs"][task]["measurements"][key] for task in ("scope", "answer")]
        bars = axes[1, 1].bar([p + offset for p in positions], values, width=.22, color=color, label=label)
        axes[1, 1].bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=8)
    axes[1, 1].set(title="D. Recorded memory measurements", xticks=positions, xticklabels=names, ylabel="GiB (2³⁰ bytes)", ylim=(0, 4.7))
    axes[1, 1].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1, 1].grid(axis="y", alpha=.15)
    axes[1, 1].set_axisbelow(True)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope-diagnosis", type=Path, default=ROOT / "reports/pilot-v3-neutral-scope-diagnosis")
    parser.add_argument("--answer-diagnosis", type=Path, default=ROOT / "reports/pilot-v2-answer-revision-diagnosis")
    parser.add_argument("--training-audit", type=Path, default=ROOT / "reports/targeted-training-audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/targeted-figures")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory; recorded figures remain immutable")
    for source in (args.scope_diagnosis, args.answer_diagnosis, args.training_audit):
        if args.output.resolve() == source.resolve() or source.resolve() in args.output.resolve().parents or args.output.resolve() in source.resolve().parents:
            parser.error("Output directory must be separate from input evidence")
    paths, hashes, scope, answer, audit, scope_rows = evidence(args.scope_diagnosis, args.answer_diagnosis, args.training_audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 10.5,
        "text.color": NAVY, "axes.labelcolor": NAVY, "svg.hashsalt": "editguard-targeted-development-v1"})
    figures = (("scope-neutral-diagnosis", scope_figure(plt, scope, scope_rows)),
               ("answer-polarity-diagnosis", answer_figure(plt, answer)), ("targeted-training-resources", training_figure(plt, audit)))
    args.output.mkdir(parents=True)
    outputs = {}
    description = json.dumps({"purpose": "validation_development_figures", "source_files_sha256": hashes,
        "script_sha256": digest(Path(__file__)), "test_rows_read": False}, sort_keys=True)
    for name, figure in figures:
        for suffix in (".png", ".svg"):
            path = args.output / (name + suffix)
            metadata = {"Description": description}
            if suffix == ".svg":
                metadata["Date"] = None
            figure.savefig(path, dpi=180, metadata=metadata)
            if suffix == ".svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
            outputs[path.name] = digest(path)
        plt.close(figure)
    if {key: digest(path) for key, path in paths.items()} != hashes:
        raise RuntimeError("Recorded input evidence changed during plotting")
    manifest = {"schema_version": 1, "purpose": "validation_development_figures", "review_policy": "automated_development",
        "human_reviewed_data": False, "test_or_challenge_rows_read": False, "source_unchanged": True,
        "scope_validation_pairs": 120, "scope_neutral_shared_inputs": 8, "scope_neutral_unshared_inputs": 112,
        "answer_teacher_forced_validation_targets": 48, "answer_polarity_class_rows": {"yes": 12, "no": 12},
        "source_files_sha256": hashes, "script_sha256": digest(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "outputs_sha256": outputs, "limitations": ["Development selection evidence; frozen final evaluation pending.",
            "Raw scope diagnostics include no metadata gates or calibrated uncertainty policy.",
            "Entity-neutral normalization is a deterministic architectural adaptation, not learned identity verification.",
            "Teacher-forced decisions and loss do not measure free-running complete-answer accuracy.",
            "Training timing covers the loop only; startup/export and full command wall time are unavailable."]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "source_unchanged": True, "plots": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()

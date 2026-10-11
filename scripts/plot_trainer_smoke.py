"""Plot measured synthetic developer-smoke timings and CUDA peaks only."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def measurements(report):
    if report.get("purpose") != "developer_smoke_verification" or report.get("status") != "passed":
        raise ValueError("A passed developer-smoke verification report is required.")
    summaries = []
    expected_models = {"scope": "distilbert/distilbert-base-uncased", "answer": "google/flan-t5-small"}
    for task in ("scope", "answer"):
        details = report["tasks"][task]
        if details["model_id"] != expected_models[task]:
            raise ValueError("Measured backbone differs from the figure's model labels.")
        observations = details["step_observations"]
        if details["completed_steps"] != 4 or details["max_steps"] != 4 or [row["global_step"] for row in observations] != [1, 2, 3, 4]:
            raise ValueError("This figure requires all four recorded optimizer steps per task.")
        times = [row["optimizer_seconds"] for row in observations]
        if any(isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0 for value in times):
            raise ValueError("Optimizer times must be real, positive observed seconds.")
        if any(len(row["batch_indices"]) != 2 for row in observations):
            raise ValueError("The displayed batch size requires two examples in every measured batch.")
        if details["environment"]["precision"] != "float32" or details["environment"]["device"] != "cuda" or details["input_length"] != 192:
            raise ValueError("Measured precision, device or input budget differs from this figure's labels.")
        segments = details["segments"]
        if [segment["stage"] for segment in segments] != ["stop_after_step_2", "resume_to_step_4"] or any(segment["purpose"] != "smoke_test" for segment in segments):
            raise ValueError("Both synthetic interrupted/resumed segments are required.")
        resources = [segment["training"]["resources"] for segment in segments] + observations
        peaks = {}
        for field in ("peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes"):
            values = [item[field] for item in resources]
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
                raise ValueError("CUDA peaks must be positive measured integer byte counts.")
            peaks[field] = max(values) / 2**30
        summaries.append({"task": task, "times": times, "median": statistics.median(times),
                          "allocated_gib": peaks["peak_allocated_gpu_bytes"],
                          "reserved_gib": peaks["peak_reserved_gpu_bytes"],
                          "model_id": details["model_id"]})
    if report["tasks"]["scope"]["environment"]["cuda_devices"] != report["tasks"]["answer"]["environment"]["cuda_devices"]:
        raise ValueError("Both tasks must use the same reported CUDA device inventory.")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/trainer-smoke.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/trainer-smoke-resources"))
    args = parser.parse_args()
    raw = args.input.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    summaries = measurements(report)
    source_hash = hashlib.sha256(raw).hexdigest()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": source_hash})
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 6.0), gridspec_kw={"width_ratios": [1.3, 1]})
    figure.suptitle("EditGuard · auxiliary trainer feasibility", fontsize=17, fontweight="bold", y=.98)
    figure.text(.5, .908, "SYNTHETIC FOUR-STEP DEVELOPER SMOKE · no accuracy or generalization claim",
                ha="center", fontsize=10.5, fontweight="bold", color="#8a3d15")
    colors = ("#0072B2", "#D55E00")
    markers = ("o", "s")
    names = ("Scope · DistilBERT", "Answer · FLAN-T5-small")
    duration_max = max(value for summary in summaries for value in summary["times"]) * 1.12
    for summary, name, color, marker in zip(summaries, names, colors, markers):
        axes[0].plot([1, 2, 3, 4], summary["times"], color=color, marker=marker,
                     linewidth=1.5, markersize=6,
                     label=f"{name}: median {summary['median']:.3f} s (n = 4)")
        axes[0].axhline(summary["median"], color=color, linestyle="--", linewidth=1.1, alpha=.75)
    axes[0].axvline(2.5, color="#6b7280", linestyle=":", linewidth=1.2)
    axes[0].text(2.57, duration_max * .93, "resume", color="#555e68", fontsize=9)
    axes[0].set(title="A. Observed optimizer phase, four points per task",
                xlabel="Global optimizer step", ylabel="Optimizer phase duration (seconds)",
                xticks=[1, 2, 3, 4], ylim=(0, duration_max), xlim=(.8, 4.2))
    axes[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
                   loc="lower left", fontsize=9)
    axes[0].grid(axis="y", alpha=.18)

    positions = [0, 1]
    allocated = [summary["allocated_gib"] for summary in summaries]
    reserved = [summary["reserved_gib"] for summary in summaries]
    bars_a = axes[1].bar([position - .18 for position in positions], allocated, width=.34,
                         color="#24455f", label="Peak allocated")
    bars_r = axes[1].bar([position + .18 for position in positions], reserved, width=.34,
                         color="#86b6c6", label="Peak reserved")
    axes[1].bar_label(bars_a, labels=[f"{value:.3f}" for value in allocated], padding=4, fontsize=9)
    axes[1].bar_label(bars_r, labels=[f"{value:.3f}" for value in reserved], padding=4, fontsize=9)
    axes[1].set(title="B. Measured CUDA memory peaks, two tasks",
                ylabel="CUDA memory (GiB)", xticks=positions,
                xticklabels=["Scope\nDistilBERT", "Answer\nFLAN-T5-small"], ylim=(0, max(reserved + allocated) * 1.2))
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].grid(axis="y", alpha=.18)
    axes[1].set_axisbelow(True)

    device = report["tasks"]["scope"]["environment"]["cuda_devices"][0]
    figure.text(.5, .125, f"{device} · float32 · batch size 2 · input budget 192 tokens\n"
                "Steps 1–2 precede interruption; steps 3–4 follow checkpoint recovery. Dashed lines show medians.",
                ha="center", fontsize=9, color="#46525b")
    figure.text(.5, .062, "Optimizer timings exclude validation, checkpoint writes, model loading and process startup.\n"
                "CUDA peaks are maxima across both recorded segments; 1 GiB = 2³⁰ bytes. Four steps do not forecast full training.",
                ha="center", fontsize=9, color="#46525b")
    figure.text(.5, .02, f"Source: {args.input.as_posix()} · SHA-256 {source_hash[:16]}…",
                ha="center", fontsize=8, color="#6b7280")
    figure.tight_layout(rect=[0, .205, 1, .86])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        output = args.output.with_suffix(suffix)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        if suffix == ".svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    print(json.dumps({"source_sha256": source_hash, "measurements": summaries,
                      "outputs": [str(args.output.with_suffix(suffix).resolve()) for suffix in (".png", ".svg")]}, indent=2))


if __name__ == "__main__":
    main()

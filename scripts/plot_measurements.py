"""Plot recorded engineering HTTP timings; never substitute for ML results."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/structured-http.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/structured-http-latency"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report["answer_kind"] != "deterministic_structured" or report["status"] != "completed":
        raise SystemExit("Plot only a completed, consistent structured benchmark.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = report["latencies_ms"]
    metrics = report["metrics"]
    median, p95 = metrics["http_roundtrip_p50_ms"], metrics["http_roundtrip_p95_ms"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1.4, 1]})
    figure.suptitle("EditGuard · structured application HTTP latency", fontsize=16, fontweight="bold", y=.99)
    axes[0].plot(range(1, len(values) + 1), values, color="#194bbd", linewidth=1.5, marker="o", markersize=3)
    axes[0].set(xlabel="Measured request index", ylabel="HTTP roundtrip latency (ms)", title="Sequential requests, new connection per request")
    axes[1].hist(values, bins=8, color="#438f90", edgecolor="white")
    axes[1].set(xlabel="HTTP roundtrip latency (ms)", ylabel="Request count", title=f"Observed distribution · n = {len(values)}")
    for ax in axes:
        ax.grid(axis="y", alpha=.18)
    axes[0].axhline(median, color="#20776e", linestyle="--", label=f"p50 = {median:.2f} ms")
    axes[0].axhline(p95, color="#a04c24", linestyle=":", label=f"p95 = {p95:.2f} ms")
    axes[0].legend(frameon=False, loc="best")
    figure.text(.5, .025, f"Deterministic answers; no ML accuracy claim. Warmups excluded. Errors: {report['counts']['errors']}; abstentions: {report['counts']['abstentions']}.\nSingle local run on the inspected Windows host; no load/concurrency or neural inference benchmark.", ha="center", fontsize=9, color="#46525b")
    figure.tight_layout(rect=[0, .13, 1, .94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".svg"]:
        output = args.output.with_suffix(suffix)
        figure.savefig(output, dpi=180, bbox_inches="tight")
        if suffix == ".svg":
            output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(figure)
    print(f"Saved {args.output}.png and .svg")


if __name__ == "__main__":
    main()

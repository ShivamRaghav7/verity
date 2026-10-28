"""Export the separately audited depth study; never loads a model or merges studies."""
from __future__ import annotations

import argparse
import csv
from html import escape
import json
import os
from pathlib import Path
import tempfile

from editguard_benchmark.data import digest
from editguard_depth.metrics import audit_and_aggregate

ORDER = ("R-RAG", "G-RAG", "R-ORACLE", "G-ORACLE")
NAMES = {"R-RAG": "ReasonKE + iterative retrieval", "G-RAG": "Qwen + iterative retrieval",
         "R-ORACLE": "ReasonKE + gold support", "G-ORACLE": "Qwen + gold support"}
COLORS = {"R-RAG": "#167e72", "G-RAG": "#bf793b", "R-ORACLE": "#5f8fb6", "G-ORACLE": "#d9b473"}
OUTCOMES = ("correct", "wrong", "abstain", "execution_error", "unexecuted")
OUTCOME_COLORS = ("#167e72", "#c36749", "#c8a353", "#78648f", "#d6dce0")
DEPTHS = ("0", "1", "2", "3")
DEPTH_LABELS = ("Locality controls", "One hop", "Two hops", "Three hops")


def _validate(report):
    if (report.get("schema") != "editguard-depth-results-v1" or report.get("phase") != "final"
            or report.get("assigned_records") != 544 or report.get("independent_lineages") != 32
            or set(report.get("conditions", {})) != set(ORDER)):
        raise ValueError("Report requires both frozen final depth ledgers: 544 assigned outcomes over 32 lineages")
    for name in ORDER:
        condition = report["conditions"][name]
        assigned = 256 if name.endswith("RAG") else 16
        metrics = [condition["overall"], *condition["by_depth"].values(), *condition["by_category"].values()]
        if condition["overall"]["assigned"] != assigned:
            raise ValueError("Depth condition population differs from frozen protocol")
        if name.endswith("RAG") and (set(condition["by_depth"]) != set(DEPTHS)
                or any(condition["by_depth"][d]["assigned"] != 64 for d in DEPTHS)):
            raise ValueError("Keep all 64 assigned questions per main depth")
        for metric in metrics:
            counts = metric["counts"]
            if (any(type(counts.get(key)) is not int or counts[key] < 0 for key in OUTCOMES)
                    or sum(counts[key] for key in OUTCOMES) != metric["assigned"]):
                raise ValueError("All outcomes must remain in each assigned denominator")


def _write_tables(report, records, output):
    with (output / "scorecard.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "population", "group", "assigned", *OUTCOMES, "accuracy", "coverage", "false_correction_applications",
                         "generated_latency_n", "p50_seconds", "p95_seconds", "failed_latency_n"])
        for name in ORDER:
            condition = report["conditions"][name]
            groups = [("overall", "all", condition["overall"])]
            groups += [(kind, key, metric) for kind in ("by_depth", "by_category")
                       for key, metric in condition[kind].items()]
            for kind, key, metric in groups:
                latency = metric["generated_latency"]
                writer.writerow([name, kind, key, metric["assigned"],
                    *[metric["counts"][outcome] for outcome in OUTCOMES], metric["accuracy"], metric["coverage"], metric["false_correction_applications"],
                    latency["n"], latency["p50_seconds"], latency["p95_seconds"], metric["failed_request_latency"]["n"]])
    with (output / "retrieval.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "depth", "found_fact_occurrences", "required_fact_occurrences",
                         "complete_paths", "assigned", "gold_support_diagnostic"])
        for name in ORDER:
            for depth, metric in report["conditions"][name]["retrieval_by_depth"].items():
                writer.writerow([name, depth, metric["found"], metric["required"], metric["complete"],
                                 metric["assigned"], name.endswith("ORACLE")])
    with (output / "latency.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        resource_keys = ("seconds", "input_tokens", "output_tokens", "peak_allocated_bytes", "peak_reserved_bytes",
                         "token_budget_reached", "time_budget_reached")
        writer.writerow(["record_id", "condition", "depth", "outcome", "response_present", "pipeline_seconds",
                         *["generation_" + key for key in resource_keys]])
        for row in records:
            generation = (row.get("response") or {}).get("generation", {})
            writer.writerow([row["id"], row["condition"], row["question"]["depth"], row["outcome"],
                             bool(row.get("response")), row.get("pipeline_seconds", ""),
                             *[generation.get(key, "") for key in resource_keys]])


def _render(report, records, output):
    # Import at execution time: ordinary tests/CLI help need no plotting packages.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "figure.facecolor": "white"})
    files = []

    def save(fig, name):
        try:
            for extension in ("png", "svg"):
                path = output / f"{name}.{extension}"
                fig.savefig(path, dpi=190, bbox_inches="tight", metadata={
                    "Description": "Derived from audited EditGuard depth receipts; no model inference performed."})
                files.append(path.name)
        finally:
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for index, name in enumerate(ORDER[:2]):
        values = [100 * report["conditions"][name]["by_depth"][depth]["accuracy"] for depth in DEPTHS]
        bars = ax.bar([i + (index - .5) * .34 for i in range(4)], values, width=.32,
                      color=COLORS[name], label=NAMES[name])
        for bar, depth in zip(bars, DEPTHS):
            metric = report["conditions"][name]["by_depth"][depth]
            ax.annotate(f"{metric['counts']['correct']}/{metric['assigned']}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(range(4), DEPTH_LABELS); ax.set_ylim(0, 113)
    ax.set_ylabel("Correct / all assigned questions (%)")
    ax.set_title("Accuracy as the edited chain gets deeper")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.20), ncol=2)
    fig.text(.06, -.025, "64 questions per depth and model; locality controls are separate-property/version questions. All outcomes retain denominator credit.", fontsize=8)
    save(fig, "depth-accuracy")

    categories = list(report["conditions"]["R-RAG"]["by_category"])
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for index, name in enumerate(ORDER[:2]):
        ax.bar([i + (index - .5) * .34 for i in range(len(categories))],
            [100*report["conditions"][name]["by_category"][c]["accuracy"] for c in categories],
            width=.32, color=COLORS[name], label=NAMES[name])
    labels = {"wrong_version_or_tier": "Historical\nrelease", "same_entity_other_relation": "Same service\nother property"}
    ax.set_xticks(range(len(categories)), [labels.get(c, c.replace("_", "\n")) for c in categories], fontsize=9)
    ax.set_ylim(0, 105); ax.set_ylabel("Correct / all assigned questions (%)")
    ax.set_title("Direct wording, reserved paraphrases and locality controls")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.19), ncol=2)
    fig.text(.06, -.035, "32 assigned questions per category and model. Wording variants remain correlated within each of 32 held-out lineages.", fontsize=9)
    save(fig, "category-accuracy")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), gridspec_kw={"hspace": .7})
    for ax, names, assigned, title in ((axes[0], ORDER[:2], 256, "Main iterative retrieval"),
                                      (axes[1], ORDER[2:], 16, "Preselected three-hop gold-support diagnostic")):
        left = [0, 0]
        for outcome, color in zip(OUTCOMES, OUTCOME_COLORS):
            values = [report["conditions"][name]["overall"]["counts"][outcome] for name in names]
            ax.barh(range(2), values, left=left, color=color, label=outcome.replace("_", " "))
            left = [a + b for a, b in zip(left, values)]
        ax.set_yticks(range(2), [name for name in names]); ax.invert_yaxis()
        ax.set_xlim(0, assigned); ax.set_xlabel(f"All {assigned} assigned questions per configuration")
        ax.set_title(title, fontsize=11)
    axes[0].legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, 1.55))
    save(fig, "outcome-composition")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ORDER[:2]):
        condition = report["conditions"][name]
        ax.bar([i-.17 for i in range(4)],
               [100*condition["retrieval_by_depth"][d]["complete"]/condition["retrieval_by_depth"][d]["assigned"] for d in DEPTHS],
               width=.32, color="#81939b", label="Complete required support")
        ax.bar([i+.17 for i in range(4)],
               [100*condition["by_depth"][d]["accuracy"] for d in DEPTHS], width=.32,
               color=COLORS[name], label="Correct final answer")
        ax.set_xticks(range(4), ["Controls", "1 hop", "2 hops", "3 hops"])
        ax.set_ylim(0, 105); ax.set_title(NAMES[name], fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12))
    axes[0].set_ylabel("Numerator / all 64 assigned questions (%)")
    fig.suptitle("Support availability and neural answers are different measurements", weight="bold", y=1.04)
    fig.text(.06, -.10, "Both models use the same selector. Repeated use is not two independent retrieval experiments; missing/unexecuted retrieval stays in the denominator.", fontsize=8)
    save(fig, "retrieval-and-answer")

    fig, ax = plt.subplots(figsize=(10, 5))
    samples = [(name, [row["pipeline_seconds"] for row in records if row["condition"] == name and row.get("response")]) for name in ORDER]
    nonempty = [(name, values) for name, values in samples if values]
    if nonempty:
        ax.boxplot([v for _, v in nonempty], tick_labels=[f"{name}\n(n={len(v)})" for name, v in nonempty], showfliers=True)
    else:
        ax.text(.5, .5, "No generated responses recorded", transform=ax.transAxes, ha="center")
    ax.set_ylabel("Warm pipeline seconds")
    ax.set_title("Generated-response latency, including wrong answers and abstentions")
    fig.text(.07, -.03, "Selection + local IPC + generation; excludes model loading, downloads and browser network. Errors/unexecuted disclosed separately.", fontsize=8)
    save(fig, "latency")
    return {"matplotlib_version": matplotlib.__version__, "figure_files": files}


def _markdown(report, records):
    lines = ["# EditGuard: separate depth challenge", "",
        f"Execution: **{report['execution_status']}**. All 544 assigned model requests remain visible: 256 main questions and 16 preselected oracle questions per checkpoint, over 32 held-out fictional graph lineages.", "",
        "This study has its own frozen protocol, entity assignments, wording split and iterative selector. The 256 main questions are correlated variants over 32 independent lineages. Profile and development receipts establish planning/selection evidence and are not additional final-test observations. This study is reported separately from operations-v1; neither score is pooled into a larger headline accuracy. Both checkpoints were trained upstream; no per-edit weight training or causal training comparison is claimed.", "",
        "| Configuration | Correct / assigned | Wrong | Abstained | Execution errors | Unexecuted | Answer coverage | Generated p50 / p95 seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name in ORDER:
        metric = report["conditions"][name]["overall"]; counts = metric["counts"]; latency = metric["generated_latency"]
        times = "Unavailable (n=0)" if not latency["n"] else f"{latency['p50_seconds']:.2f} / {latency['p95_seconds']:.2f} (n={latency['n']})"
        lines.append(f"| {NAMES[name]} | {counts['correct']}/{metric['assigned']} | {counts['wrong']} | {counts['abstain']} | {counts['execution_error']} | {counts['unexecuted']} | {100*metric['coverage']:.1f}% | {times} |")
    lines += ["", "Primary accuracy is correct / all assigned. Abstentions, errors and unexecuted rows receive zero primary credit. Coverage is (correct + wrong) / all assigned, not a filtered success rate. Oracle rows are a distinct, hash-preselected sixteen-question generation diagnostic, not ordinary retrieval accuracy.", "",
        "![Accuracy by depth](depth-accuracy.png)", "", "![Accuracy by category](category-accuracy.png)", "",
        "![Every assigned outcome](outcome-composition.png)", "", "## Retrieval and generation", "",
        "| Configuration / depth | Found / required fact occurrences | Complete paths / assigned | Correct / assigned |",
        "| --- | ---: | ---: | ---: |"]
    for name in ORDER:
        condition = report["conditions"][name]
        for depth, metric in condition["retrieval_by_depth"].items():
            answer = condition["by_depth"][depth]
            lines.append(f"| {name} / {depth} | {metric['found']}/{metric['required']} | {metric['complete']}/{metric['assigned']} | {answer['counts']['correct']}/{answer['assigned']} |")
    lines += ["", "Depth 0 denotes unrelated-property or historical-version controls; each needs an explicit fact. Required support recall is not answer accuracy. Both generators receive the same selector; its duplicated use is not independent retrieval evidence. `results.json` retains complete-support and missing-support generation subgroups as descriptive diagnostics, without replacing assigned primary denominators.", "",
        "![Support and answer quality](retrieval-and-answer.png)", "", "## Paired uncertainty", ""]
    for label, metric in report["paired_differences"].items():
        if metric is not None:
            low, high = metric["ci95"]
            lines.append(f"- {label}: ReasonKE minus Qwen {100*metric['left_minus_right']:.2f} percentage points; 95% lineage-bootstrap interval [{100*low:.2f}, {100*high:.2f}], {metric['paired_questions']} paired questions across {metric['independent_lineages']} lineages, {metric['bootstrap_resamples']} resamples, seed {metric['seed']}.")
    lines += ["", "Related wording variants are correlated; the independent unit is a graph lineage. If a paired bootstrap interval has zero width, it reflects identical sampled lineage differences in this finite dataset, not universal certainty or guaranteed accuracy. A perfect observed result is retained without manufactured failures. These bounded data do not establish arbitrary paraphrase or unrestricted conversation reliability.", "",
        "## Latency and resources", "", "![Generated response latency](latency.png)", "",
        "Warm request time includes context selection, local IPC and generation. Loading, downloads and browser network are excluded. Generated wrong and abstained responses remain in the latency population; failed-request latency and its n are retained separately in `results.json` and `scorecard.csv`.", ""]
    for model, sessions in report["model_measurements"].items():
        for session in sessions:
            hardware = session.get("hardware", {})
            lines.append(f"- {model}: {session.get('model_id', 'unavailable')} at {session.get('revision', 'unavailable')}; GPU {hardware.get('name', 'unavailable')}; load seconds {session.get('load_seconds', 'unavailable')}; peak loading allocated/reserved bytes {hardware.get('load_peak_allocated_bytes', 'unavailable')} / {hardware.get('load_peak_reserved_bytes', 'unavailable')}. See raw receipts for software versions, generation tokens and per-request CUDA usage.")
    lines += ["", "## Retained failure examples", "",
        "The deterministic first twelve non-correct record IDs are shown below. They are illustrative, not a complete failure taxonomy. All outcomes remain in the receipts and exports.", "",
        "| Record | Outcome | Expected | Generated final answer / error |", "| --- | --- | --- | --- |"]
    failures = sorted((r for r in records if r["outcome"] != "correct"), key=lambda r: r["id"])[:12]
    def safe(value):
        return escape(str(value).replace("\n", " ").replace("\r", " ")[:240]).replace("|", "\\|")
    for row in failures:
        actual = (row.get("response") or {}).get("content", row.get("error", row["outcome"]))
        lines.append(f"| {safe(row['id'])} | {row['outcome']} | {safe(row['question']['expected_answer'])} | {safe(actual)} |")
    if not failures:
        lines.append("| No non-correct outcomes recorded | — | — | — |")
    lines += ["", "## Limits and provenance", "",
        "Assignments are controlled fiction, labels and graph paths are automatically audited, and no independent human semantic-review claim is made. Wording banks differ across splits but templates repeat within each split; relation vocabulary and simple city values repeat. Strict whole-final-answer alias matching checks the endorsed answer, not mere target mention. A correct final answer does not verify generated reasoning or surrounding prose. Administrator state/rollback checks and deterministic database answers are not neural answer-quality evidence.", "",
        "The exporter replays the protocol-bound audit before writing, makes zero model calls, and preserves prior report directories. `scorecard.csv`, `retrieval.csv` and `latency.csv` disclose denominators and sample sizes. `figure-manifest.json` binds source receipts, protocol, exporter and exported files by SHA-256. The original operations-v1 study is unchanged."]
    return "\n".join(lines) + "\n"


def generate(receipts, output, *, protocol, config_path=None):
    output, protocol = Path(output), Path(protocol)
    receipts = [Path(path) for path in receipts]
    if output.exists():
        raise ValueError("Preserve existing depth reports; choose a new output directory")
    # Mandatory protocol and full audit precede any output directory or plotting.
    if len(receipts) != 2 or not protocol.is_file():
        raise ValueError("Two final receipts and their frozen protocol are required")
    report = audit_and_aggregate(receipts, config_path=config_path, protocol=protocol)
    _validate(report)
    source_hashes = {str(path): digest(path) for path in receipts}
    audited_hashes = {row["receipt"]: row["sha256"] for row in report["receipt_audit"]}
    if source_hashes != audited_hashes:
        raise ValueError("Depth receipt changed after audit")
    protocol_hash = digest(protocol)
    protocol_value = json.loads(protocol.read_text(encoding="utf-8"))
    bound_inputs = {name: protocol.parent / name for name in protocol_value["input_receipts"]}
    if any(digest(path) != protocol_value["input_receipts"][name] for name, path in bound_inputs.items()):
        raise ValueError("Frozen profile/development receipt changed after audit")
    raw = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    records = [row for receipt in raw for row in receipt["records"]]
    portable_files = {"protocol.json": protocol_hash, **protocol_value["input_receipts"]}
    for path, receipt in zip(receipts, raw):
        model = receipt["plan"]["model"]
        portable_files[f"final-{model}.json"] = source_hashes[str(path)]
    output.parent.mkdir(parents=True, exist_ok=True)
    # All files are staged; an interrupted plot/export never publishes a partial report.
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.staging-", dir=output.parent) as temporary:
        staging = Path(temporary)
        (staging / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _write_tables(report, records, staging)
        plot_metadata = _render(report, records, staging)
        (staging / "REPORT.md").write_text(_markdown(report, records), encoding="utf-8")
        portable_files["results.json"] = digest(staging / "results.json")
        host_audit = {"schema": "editguard-depth-host-audit-v1", "status": "passed", "model_calls": 0,
            "results_sha256": portable_files["results.json"], "protocol_sha256": protocol_hash,
            "exporter_sha256": digest(__file__), "files": portable_files,
            "source_receipts": source_hashes, "assigned": report["assigned_records"],
            "independent_lineages": report["independent_lineages"], "human_reviewed": False,
            "notice": "Protocol-bound mechanical identity/extraction/scoring audit; no independent semantic review or generated-prose verification"}
        manifest = {"schema": "editguard-depth-figures-v1", "source_receipts": source_hashes,
            "protocol_sha256": protocol_hash, "script_sha256": digest(__file__),
            "model_inference_executed": False, "human_reviewed": False,
            **plot_metadata, "files": {p.name: digest(p) for p in sorted(staging.iterdir()) if p.is_file()}}
        (staging / "figure-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Bind manifest -> artifact files, then host audit -> manifest, avoiding
        # a circular byte hash between the two manifests.
        host_audit["figure_manifest_sha256"] = digest(staging / "figure-manifest.json")
        (staging / "host-audit.json").write_text(json.dumps(host_audit, indent=2) + "\n", encoding="utf-8")
        if (output.exists() or digest(protocol) != protocol_hash
                or any(digest(path) != source_hashes[str(path)] for path in receipts)
                or any(digest(path) != protocol_value["input_receipts"][name] for name, path in bound_inputs.items())):
            raise ValueError("Depth inputs or output changed during export; report not published")
        os.rename(staging, output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs=2, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    result = generate(args.receipts, args.output, protocol=args.protocol, config_path=args.config)
    print(json.dumps({"output": str(args.output), "figures": len(result["figure_files"]),
                      "model_inference_executed": False}))


if __name__ == "__main__":
    main()

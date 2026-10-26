"""Audit a completed frozen run, publish its measured exports and launch acceptance.

All postprocessing calls regular versioned Python functions, avoiding nested
notebook command strings. No benchmark reruns or primary-score changes occur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from scripts.audit_operations_explanations import audit_receipt
from scripts.build_operations_report import generate
from scripts.finish_operations_release import launch, require_completed
from editguard_benchmark.metrics import aggregate
from scripts.verify_reasonke_live import atomic_write

ROOT = Path(__file__).resolve().parents[1]


def finalize(benchmark_directory, run_directory, *, trusted_origin=None):
    source, run_directory = Path(benchmark_directory), Path(run_directory)
    audit = require_completed(source)  # Before new output or any model loading.
    target = ROOT / "reports/reasonke/operations-final"
    if (target / "results.json").exists() or run_directory.exists():
        raise ValueError("Preserve existing final results and run directory; finalization is not overwritten")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("final-reasonke.json", "final-instruction.json", "protocol.json"):
        destination = target / name
        if destination.exists() and destination.read_bytes() != (source / name).read_bytes():
            raise ValueError("Existing frozen input differs: " + name)
        if not destination.exists():
            shutil.copyfile(source / name, destination)
    paths = [target / ("final-" + model + ".json") for model in ("reasonke", "instruction")]
    atomic_write(target / "host-audit.json", audit)
    results = aggregate(paths)
    atomic_write(target / "results.json", results)
    explanation = audit_receipt(paths[0], protocol=source / "protocol.json")
    atomic_write(target / "explanation-audit.json", explanation)
    generate(paths, target / "derived")
    state = launch(run_directory, source, trusted_origin=trusted_origin)
    shutil.copyfile(run_directory / "http-acceptance.json", target / "http-acceptance.json")
    shutil.copyfile(run_directory / "history-acceptance.json", target / "history-acceptance.json")
    lines = ["# Completed frozen operations evaluation", "",
        f"Execution: {results['execution_status']}; {results['assigned_records']} assigned generations across {results['independent_lineages']} held-out fictional edit lineages.", "",
        "Settings, source/data identities and scoring were frozen before final inference. All wrong answers, abstentions, execution errors and unexecuted records retain denominator credit. No benchmark question was retried for a better answer.", "",
        "| Condition | Correct / assigned | Wrong | Abstained | Errors | Unexecuted |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, condition in results["conditions"].items():
        metric = condition["overall"]; count = metric["counts"]
        lines.append(f"| {name} | {count['correct']}/{metric['assigned']} | {count['wrong']} | {count['abstain']} | {count['execution_error']} | {count['unexecuted']} |")
    lines += ["", f"The separate explanation audit retains {explanation['assigned_traces']} preselected traces. Only bounded explicit indexed subanswers are assessed; surrounding prose and internal reasoning are not certified.", "",
        "`final-reasonke.json` and `final-instruction.json` preserve all outputs; `protocol.json` identifies the actual frozen inputs. `host-audit.json` checks identities, extraction and scoring without a model call. `derived/` contains the measured report, CSV and scientific PNG/SVG figures.", "",
        "`http-acceptance.json` contains twelve development-world website requests. `history-acceptance.json` contains four further generations for historical comparison and atomic rollback. These operational workloads are not extra independent held-out data and do not alter the primary scorecard. Dataset review and explanation review are automated, without independent human semantic review.", "",
        "Both checkpoints were trained upstream. Current deployment context was selected on development before final inference. No universal accuracy, arbitrary-edit safety, faithful explanation or ReasonKE superiority is assumed."]
    (target / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bundle = run_directory / "completed-project-evidence.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, "final/" + path.relative_to(target).as_posix())
        for name in ("launch.json", "prelaunch-audit.json"):
            archive.write(run_directory / name, "website/" + name)
    completion = {"schema": "editguard-release-finalization-v1", "status": state["status"],
                  "bundle": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                  "assigned_http_generations": 16, "training_steps": 0, "benchmark_retried": False,
                  "primary_scoring_changed": False, "user_chat_database_exported": False}
    atomic_write(run_directory / "completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-directory", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--trusted-origin")
    args = parser.parse_args()
    print(json.dumps(finalize(args.benchmark_directory, args.run_directory, trusted_origin=args.trusted_origin), indent=2))


if __name__ == "__main__":
    main()

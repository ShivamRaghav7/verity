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
    atomic_write(target / "results.json", aggregate(paths))
    atomic_write(target / "explanation-audit.json", audit_receipt(paths[0], protocol=source / "protocol.json"))
    generate(paths, target / "derived")
    state = launch(run_directory, source, trusted_origin=trusted_origin)
    shutil.copyfile(run_directory / "http-acceptance.json", target / "http-acceptance.json")
    bundle = run_directory / "completed-project-evidence.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, "final/" + path.relative_to(target).as_posix())
        for name in ("launch.json", "prelaunch-audit.json"):
            archive.write(run_directory / name, "website/" + name)
    completion = {"schema": "editguard-release-finalization-v1", "status": state["status"],
                  "bundle": str(bundle), "sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
                  "assigned_http_generations": 12, "training_steps": 0, "benchmark_retried": False,
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

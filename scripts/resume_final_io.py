"""Declared IO-only recovery for transient Windows atomic-write collisions.

The frozen evaluator remains byte-identical. This externally hash-bound wrapper
changes only progress/ledger write retries, outside measured model-route latency.
It resumes the same journal and never permits a fresh test run or model tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def retry_writer(original, targets, *, delays=(.05, .1, .2, .4, .4, .4, .4), sleep=time.sleep, events=None):
    """Finite retry of the identical value on two declared mutable files only."""
    targets = {Path(path).resolve() for path in targets}
    def write(path, value):
        if Path(path).resolve() not in targets:
            return original(path, value)
        for attempt in range(len(delays) + 1):
            try:
                return original(path, value)
            except PermissionError as error:
                if attempt == len(delays):
                    raise
                if events is not None:
                    events.append({"target": Path(path).name, "failed_attempt": attempt + 1,
                                   "delay_seconds": delays[attempt], "error_type": type(error).__name__})
                sleep(delays[attempt])
    return write


def recover(spec_path, expected_spec_sha256):
    if sha(spec_path) != expected_spec_sha256:
        raise ValueError("Recovery declaration checksum differs.")
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    if (spec.get("purpose") != "frozen_final_io_recovery_declaration" or spec.get("fresh_run_allowed") is not False
            or spec.get("inference_scoring_or_data_changes") is not False or spec.get("wrapper_sha256") != sha(__file__)
            or spec.get("max_write_attempts") != 8 or spec.get("retry_delays_seconds") != [.05, .1, .2, .4, .4, .4, .4]):
        raise ValueError("Unsupported IO recovery declaration.")
    protocol_path = ROOT / spec["protocol_path"]
    if sha(protocol_path) != spec["protocol_sha256"]:
        raise ValueError("The original protocol changed.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    from scripts import run_final_evaluation as evaluator
    if sha(evaluator.__file__) != spec["frozen_evaluator_sha256"]:
        raise ValueError("The original frozen evaluator changed.")
    targets = [evaluator.workspace_path(ROOT, protocol["execution"]["output_directory"]) / "progress.json",
               evaluator.workspace_path(ROOT, protocol["execution"]["ledger_path"])]
    if [str(path.relative_to(ROOT)).replace("\\", "/") for path in targets] != spec["mutable_write_targets"]:
        raise ValueError("IO recovery can affect only the declared progress and ledger files.")
    original = evaluator.atomic_json
    unchanged = {name: getattr(evaluator, name) for name in
                 ("execution_plan", "score_observation", "final_primary_score", "record_identity", "locked_inputs")}
    events = []
    evaluator.atomic_json = retry_writer(original, targets, events=events)
    try:
        if any(getattr(evaluator, name) is not function for name, function in unchanged.items()):
            raise ValueError("An inference, scoring, input or observation function changed.")
        return evaluator.run_final(protocol_path, spec["protocol_sha256"], ROOT / spec["packet_path"], resume=True)
    finally:
        evaluator.atomic_json = original
        # Append-only disclosure is separate from immutable prediction records.
        destination = ROOT / "runtime/final-io-recovery-events.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({"declaration_sha256": expected_spec_sha256, "write_retries": events,
                                     "frozen_evaluator_sha256": spec["frozen_evaluator_sha256"],
                                     "inference_scoring_or_data_changes": False}, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-sha256", required=True)
    args = parser.parse_args()
    result = recover(args.declaration, args.declaration_sha256)
    print(json.dumps({"status": result["status"], "questions": result["questions"],
                      "oracle_records": result["oracle_records"], "io_recovery_declared": True}))


if __name__ == "__main__":
    main()

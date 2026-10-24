"""Read-only transfer, scoring and restoration audit of the frozen VLKEB run.

Uses only the standard library and the original frozen scorer/aggregator. This
is arithmetic and byte-identity verification, not semantic or human review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ke_repro.evaluate import CATEGORIES, EDIT_COUNT, EDIT_METHODS, METHODS, aggregate, cases_for
from ke_repro.smoke import digest, exact_short_answer

PREREGISTERED_SEAL_SHA256 = "73c303fd260977f624bf39de1d4f58fb8d9f9e2a8ae70f9aeca2fab72292a32f"
PARAMETERS = {f"opt_model.model.decoder.layers.{layer}.fc{matrix}.weight"
              for layer in (29, 30, 31) for matrix in (1, 2)}
STATUSES = {"completed", "abstention", "generation_error", "edit_error", "not_executed"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=reject_constant)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def hash_map(value, label):
    require(isinstance(value, dict) and set(value) == PARAMETERS, f"Wrong editable parameter hashes: {label}")
    require(all(isinstance(checksum, str) and re.fullmatch(r"[0-9a-f]{64}", checksum)
                for checksum in value.values()), f"Invalid SHA256 tensor hash: {label}")


def check_sources(seal):
    paths = sorted([*ROOT.glob("ke_repro/*.py"), ROOT / "scripts/prepare_vlkeb_eval.py",
                    ROOT / "scripts/prepare_vlkeb_assets.py", ROOT / "scripts/prepare_vlkeb_rag.py",
                    ROOT / "docs/VLKEB_EVALUATION_PROTOCOL.md"])
    actual = {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in paths}
    require(seal["fingerprint"]["code_sha256"] == actual,
            "Current scientific code/protocol files differ from the preregistered seal")
    return actual


def image_matches(actual, expected):
    if expected is None:
        return actual is None
    if not isinstance(actual, str):
        return False
    expected_parts = PurePosixPath(expected.replace("\\", "/")).parts
    # cases_for() below uses this fixed dummy root; retain images/<entity>/<file>.
    suffix = expected_parts[expected_parts.index("images"):]
    actual_parts = PurePosixPath(actual.replace("\\", "/")).parts
    return ".." not in actual_parts and actual_parts[-len(suffix):] == suffix


def check_row(row, edit, seal_sha, original_hashes):
    label = edit["edit_id"]
    require(row.get("edit_id") == label and row.get("seal_sha256") == seal_sha,
            f"Row identity/seal mismatch: {label}")
    require(row.get("provenance") == "own_measured", f"Unexpected row provenance: {label}")
    require(set(row.get("methods", {})) == set(METHODS), f"Wrong method set: {label}")
    expected = {case["category"]: case for case in cases_for(edit, Path("__audit_assets__"))}
    scored = 0
    for method in METHODS:
        answers = row["methods"][method]
        require(isinstance(answers, list) and len(answers) == len(CATEGORIES)
                and sorted(answer.get("category") for answer in answers) == sorted(CATEGORIES),
                f"Wrong category set: {label}/{method}")
        for answer in answers:
            category = answer["category"]
            case = expected[category]
            require(answer.get("query") == case["query"] and answer.get("expected") == case["expected"]
                    and image_matches(answer.get("image"), case["image"]),
                    f"Frozen case/query/target/image mismatch: {label}/{method}/{category}")
            require(isinstance(answer.get("answer"), str) and answer.get("status") in STATUSES,
                    f"Invalid answer or execution status: {label}/{method}/{category}")
            status = answer["status"]
            require(status != "completed" or bool(answer["answer"].strip()), "Empty answer marked completed")
            require(status != "abstention" or not answer["answer"].strip(), "Nonempty answer marked abstention")
            matched = status == "completed" and exact_short_answer(answer["answer"], case["expected"])
            require(isinstance(answer.get("strict_target_match"), bool) and answer["strict_target_match"] == matched,
                    f"Saved strict score differs from original scorer: {label}/{method}/{category}")
            seconds = answer.get("seconds")
            require(seconds is None or (isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
                    and math.isfinite(seconds) and seconds >= 0), "Invalid query duration")
            scored += 1
    hash_map(row.get("base_hashes"), label + "/base")
    hash_map(row.get("restored_hashes"), label + "/restored")
    require(row.get("exact_restore") is True and row["base_hashes"] == row["restored_hashes"],
            f"Episode restoration failed: {label}")
    if original_hashes is not None:
        require(row["base_hashes"] == original_hashes, f"Base tensors differ between edit units: {label}")
    require(set(row.get("method_edits", {})) == set(EDIT_METHODS), f"Missing edit-method restoration evidence: {label}")
    for method, operation in row["method_edits"].items():
        require(operation.get("attempted") is True, f"Edit method not attempted: {label}/{method}")
        for name in ("base_hashes", "restored_hashes"):
            hash_map(operation.get(name), label + "/" + method + "/" + name)
            require(operation[name] == row["base_hashes"], f"Method base/rollback hash mismatch: {label}/{method}")
        require(operation.get("exact_restore") is True, f"Method restoration failed: {label}/{method}")
        if "edited_hashes" in operation:
            hash_map(operation["edited_hashes"], label + "/" + method + "/edited")
        else:
            require(bool(operation.get("error")), f"Missing edited tensor hashes without an edit error: {label}/{method}")
        rollback = operation.get("rollback_metadata")
        if rollback is not None:
            require(rollback.get("exact_weight_restore") is True and rollback.get("weight_hashes") == row["base_hashes"],
                    f"Rollback metadata disagrees with hashes: {label}/{method}")
    return scored


def verify(run_dir, preregistered_seal, selection_path):
    seal_path = run_dir / "seal.json"
    summary_path = run_dir / "evaluation.json"
    seal_sha = digest(seal_path)
    require(digest(preregistered_seal) == PREREGISTERED_SEAL_SHA256 == seal_sha,
            "Transported seal and preregistered seal must match the fixed preregistration SHA256")
    require(seal_path.read_bytes() == preregistered_seal.read_bytes(), "Transported/preregistered seal bytes differ")
    seal = read_json(seal_path)
    code_hashes = check_sources(seal)
    require(digest(selection_path) == seal["fingerprint"]["selection_sha256"], "Frozen selection SHA256 mismatch")
    selection = read_json(selection_path)
    edits = selection["edits"]
    require(len(edits) == EDIT_COUNT and len({edit["edit_id"] for edit in edits}) == EDIT_COUNT,
            "Frozen selection must contain exactly20 distinct edit IDs")
    expected_ids = {edit["edit_id"] for edit in edits}
    require(all(re.fullmatch(r"[0-9a-f]{64}", identity) for identity in expected_ids), "Invalid frozen edit ID")
    row_paths = sorted((run_dir / "edits").glob("*.json"))
    require(len(row_paths) == EDIT_COUNT and {path.stem for path in row_paths} == expected_ids,
            "Transported row files do not match the exact20 frozen IDs")
    rows, scored, original_hashes = [], 0, None
    for edit in edits:
        row = read_json(run_dir / "edits" / (edit["edit_id"] + ".json"))
        scored += check_row(row, edit, seal_sha, original_hashes)
        original_hashes = row["base_hashes"]
        rows.append(row)
    require(scored == EDIT_COUNT * len(METHODS) * len(CATEGORIES) == 500, "Expected exactly500 rescored answers")
    summary = read_json(summary_path)
    require(summary.get("seal_sha256") == seal_sha and summary.get("execution_status") == "completed"
            and summary.get("worker_return_code") == 0 and summary.get("completed_edit_units") == EDIT_COUNT
            and summary.get("unexecuted_edit_units") == 0 and summary.get("all20_units_in_denominators") is True,
            "Summary does not represent a completed frozen20-unit run")
    require(summary.get("fine_tuning_attempted_units") == sum(row.get("fine_tuning_attempted") is True for row in rows),
            "Summary fine-tuning attempt count differs from row evidence")
    require(canonical(summary.get("aggregate")) == canonical(aggregate(rows)),
            "Saved aggregate differs from original frozen aggregate recomputation")
    artifacts = [seal_path, summary_path, *row_paths]
    worker_path = run_dir / "worker.json"
    if worker_path.exists():
        require(canonical(read_json(worker_path)) == canonical(summary.get("worker_report")),
                "Transported worker report differs from summary worker evidence")
        artifacts.append(worker_path)
    return {"verification_status": "verified", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Transported byte identity, frozen code/selection identity, scorer arithmetic, aggregate arithmetic and recorded tensor restoration",
            "semantic_review": False, "independent_human_review": False, "external_truth_checked": False,
            "limitations": "Checks recorded evidence and deterministic arithmetic; does not independently rerun the model or certify author labels, image meaning, or factual truth.",
            "preregistered_seal_sha256": seal_sha, "selection_sha256": digest(selection_path),
            "verified_edit_units": len(rows), "verified_generated_answer_records": scored,
            "verified_method_category_pairs": len(METHODS) * len(CATEGORIES),
            "all_recorded_editable_tensor_restorations_exact": True,
            "frozen_code_sha256": code_hashes,
            "artifact_sha256": {str(path.relative_to(run_dir)).replace("\\", "/"): digest(path) for path in artifacts},
            "verifier_sha256": digest(Path(__file__))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--preregistered-seal", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new receipt path; never overwrite")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError("Refusing to replace an existing verification receipt")
        receipt = verify(args.run_dir.resolve(), args.preregistered_seal.resolve(), args.selection.resolve())
        encoded = json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(encoded)
        print(encoded, end="")
        return 0
    except Exception as error:
        print(json.dumps({"verification_status": "failed", "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

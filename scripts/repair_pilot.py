"""Derive one immutable, pending pilot-v2 with deterministic grammar repairs.

Only a complete, unreviewed pilot-v1 development dataset is accepted. Historical
test/challenge artifacts and review registries are never copied or rewritten.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text
from scripts.prepare_data import answer_for, canonical_bytes, content_hash, question_for
from scripts.review_dataset import checked_rows, file_hash, integrity_flags, read_json

TASKS = ("scope", "answer")
SPLITS = ("train", "validation")
ROW_FILES = tuple(f"{task}_{split}.jsonl" for task in TASKS for split in SPLITS)
SOURCE_FILES = (*ROW_FILES, "catalogue.json", "selection.json", "split_assignments.json", "derivation.json")
COUNTS = {"scope": {"train": 240, "validation": 120}, "answer": {"train": 96, "validation": 48}}
TEXT_FIELDS = {"question", "correction_text", "input_text", "target_text"}
NONSEMANTIC_FIELDS = TEXT_FIELDS | {"id", "input_format_version"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _pending(value):
    """Reject accepted/rejected/reviewed records, rather than erase a decision."""
    if isinstance(value, dict):
        if "reviewed" in value and value["reviewed"] is not False:
            raise ValueError("Repair requires every source record to remain unreviewed")
        if "review_status" in value and value["review_status"] != "pending":
            raise ValueError("Repair requires every source record review status to be pending")
        for item in value.values():
            _pending(item)
    elif isinstance(value, list):
        for item in value:
            _pending(item)


def _legacy_fact(fact, value=None, polarity=None):
    value = fact["value"] if value is None else value
    subject = f"{fact['entity_name']} {fact['plan']} version {fact['version']}"
    forms = {"device_limit": f"{subject} supports {value} devices.",
             "price_monthly": f"{subject} costs {value} USD per month.",
             "storage_gb": f"{subject} includes {value} GB of storage.",
             "support_hours": f"{subject} includes {value} support hours per day."}
    return (f"{polarity.capitalize()}. " if polarity else "") + forms[fact["relation"]]


def _legacy_correction(edit):
    return (f"Correction for {edit['entity_name']} (aliases: {', '.join(edit['entity_aliases'])}), "
            f"{edit['plan']}, version {edit['version']}: {edit['relation']} changed from "
            f"{edit['original_value']} to {edit['corrected_value']} {edit['unit']}, "
            f"effective from {edit['effective_from']}. Source: {edit['source']}.")


def _question(row, fact):
    question_type = row["question_type"]
    if question_type in ("direct", "out_of_scope"):
        expected = question_for(fact, 0)
        old = expected
    elif question_type == "paraphrase":
        expected = question_for(fact, 1)
        old = expected
    elif question_type == "comparison":
        patterns = {"device_limit": r"connect (\d+) devices on",
                    "price_monthly": r"cost at most (\d+) USD per month",
                    "storage_gb": r"store (\d+) GB on",
                    "support_hours": r"include at least (\d+) support hours per day"}
        match = re.search(patterns[fact["relation"]], row["question"])
        if not match:
            raise ValueError(f"Source comparison is outside the exact legacy grammar: {row['id']}")
        amount = int(match[1])
        expected = question_for(fact, 2, amount)
        old = expected.replace("connect 1 device on", "connect 1 devices on").replace(
            "at least 1 support hour per day", "at least 1 support hours per day")
    else:
        raise ValueError(f"Unsupported development question type: {question_type}")
    if row["question"] != old:
        raise ValueError(f"Source question differs from the validated legacy template: {row['id']}")
    return expected


def _semantic_record(row):
    return {key: value for key, value in row.items() if key not in NONSEMANTIC_FIELDS}


def _validate_source(source):
    source = Path(source).resolve()
    if not source.is_dir() or (source / "test_lock.json").exists():
        raise ValueError("Source must be an existing, unfrozen pilot-v1 directory")
    manifest = read_json(source / "manifest.json")
    if (manifest.get("purpose") != "development_pilot_not_final_evaluation"
            or manifest.get("pilot_version") != 1 or manifest.get("test_frozen") is not False
            or manifest.get("test_files_included") is not False):
        raise ValueError("Only an unfrozen development pilot-v1 without test files can be repaired")
    if manifest.get("input_format_version") not in (None, "editguard-input-v1"):
        raise ValueError("Source must contain legacy v1 inputs; already repaired or unknown formats are refused")
    if manifest.get("eligible_for_training") is not False or manifest.get("review_status") != "pending_independent_human_decisions":
        raise ValueError("Source must remain pending independent human decisions")
    if set(manifest.get("tasks", [])) != set(TASKS) or manifest.get("counts") != COUNTS:
        raise ValueError("Repair requires the complete finite 504-row scope-and-answer pilot")
    if set(manifest.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Source has partial, test/challenge or unsupported manifest files")
    if {path.name for path in source.iterdir()} != set(SOURCE_FILES) | {"manifest.json"}:
        raise ValueError("Source contains partial, test/challenge, review-registry or unsupported artifacts")
    origin_hashes = {"manifest.json": file_hash(source / "manifest.json")}
    for name in SOURCE_FILES:
        path = source / name
        if not path.is_file() or file_hash(path) != manifest["files"][name]:
            raise ValueError(f"Source checksum mismatch or missing file: {name}")
        origin_hashes[name] = file_hash(path)
    catalogue_rows = read_json(source / "catalogue.json")
    _pending(catalogue_rows)
    catalogue = {fact["fact_id"]: fact for fact in catalogue_rows}
    if len(catalogue) != len(catalogue_rows):
        raise ValueError("Source catalogue contains duplicate fact identifiers")
    assignments = read_json(source / "split_assignments.json")
    if any(split not in SPLITS for split in assignments.values()) or Counter(assignments.values()) != {"train": 4, "validation": 2}:
        raise ValueError("Source family assignments must contain exactly four train and two validation families")
    for fact in catalogue_rows:
        if fact["split"] not in SPLITS or assignments.get(fact["family_id"]) != fact["split"]:
            raise ValueError("Source catalogue contains a test or incompatible family assignment")
    rows_by_file = {}
    all_ids = set()
    for task in TASKS:
        for split in SPLITS:
            rows = checked_rows(source, task, split)
            if len(rows) != COUNTS[task][split]:
                raise ValueError("Source row count differs from the complete finite pilot")
            for row in rows:
                _pending(row)
                if row.get("reviewed") is not False or row.get("review_status") != "pending":
                    raise ValueError("Each source row must explicitly remain pending")
                if row.get("input_format_version") not in (None, "editguard-input-v1"):
                    raise ValueError("Source row has a partial or incompatible input-format migration")
                if not isinstance(row["id"], str) or not row["id"].startswith("pilot-v1:") or row["id"] in all_ids:
                    raise ValueError("Source row identifiers must be unique pilot-v1 identifiers")
                all_ids.add(row["id"])
                if assignments.get(row["family_id"]) != split:
                    raise ValueError("Source row family differs from the assigned development split")
                if row["expected_fact"]["fact_id"] not in catalogue:
                    raise ValueError("Source ground-truth fact is absent from the development catalogue")
                expected_fact = catalogue[row["expected_fact"]["fact_id"]]
                for key in ("entity_id", "relation", "unit", "plan", "version"):
                    if row["expected_fact"][key] != expected_fact[key]:
                        raise ValueError("Source ground-truth scope differs from its structured fact")
                if expected_fact["split"] != split:
                    raise ValueError("Source expected fact leaks across development splits")
                correction = row["correction"]
                original_fact = catalogue.get(correction["fact_id"])
                if not original_fact or original_fact["split"] != split:
                    raise ValueError("Source correction fact is absent or belongs to another split")
                for key in ("entity_id", "relation", "unit", "plan", "version", "entity_name"):
                    if correction[key] != original_fact[key]:
                        raise ValueError("Source correction scope differs from its structured fact")
                if correction["entity_aliases"] != original_fact["aliases"] or correction["original_value"] != original_fact["value"]:
                    raise ValueError("Source correction alias or original value differs from its structured fact")
                for fact in row.get("supporting_context", []):
                    if catalogue.get(fact["fact_id"]) != fact or fact["split"] != split:
                        raise ValueError("Source supporting facts differ from the same-split catalogue")
                flags = integrity_flags(row, catalogue)
                if flags:
                    raise ValueError(f"Source semantic integrity checks failed for {row['id']}: {', '.join(flags)}")
            rows_by_file[f"{task}_{split}.jsonl"] = rows
    previous_derivation = read_json(source / "derivation.json")
    source_rows = {row["id"]: row for rows in rows_by_file.values() for row in rows}
    if (not isinstance(previous_derivation, list) or len(previous_derivation) != len(source_rows)
            or {record.get("id") for record in previous_derivation} != set(source_rows)
            or any(record.get("new_row_sha256") != content_hash(source_rows[record["id"]])
                   for record in previous_derivation)):
        raise ValueError("Source derivation does not bind every current development row")
    return source, manifest, origin_hashes, catalogue, rows_by_file


def repair_pilot(source, output):
    """Publish a new complete dataset; immutable structured semantics are checked."""
    source, source_manifest, origin_hashes, catalogue, rows_by_file = _validate_source(source)
    output = Path(output).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Output must be separate from, and not contain, the source")
    if output.exists():
        raise ValueError("Output must be a new nonexistent directory; partial/existing output is refused")
    files = {name: (source / name).read_bytes() for name in ("catalogue.json", "selection.json", "split_assignments.json")}
    derivations, changes = [], Counter()
    new_ids = set()
    for name, rows in rows_by_file.items():
        output_rows = []
        for original in rows:
            row = copy.deepcopy(original)
            old_correction = _legacy_correction(original["correction"])
            if original["correction_text"] != old_correction:
                raise ValueError(f"Source correction text differs from the legacy formatter: {original['id']}")
            fact = catalogue[original["expected_fact"]["fact_id"]]
            row["question"] = _question(original, fact)
            row["correction_text"] = correction_text(row["correction"])
            row["id"] = "pilot-v2:" + original["id"][len("pilot-v1:"):]
            row["input_format_version"] = INPUT_FORMAT_VERSION
            if row["id"] in new_ids:
                raise ValueError("Derived row IDs are not unique")
            new_ids.add(row["id"])
            if name.startswith("answer_"):
                legacy_input = (f"answer correction: {old_correction} Question: {original['question']} "
                    f"As of: {original['request_context']['as_of']}. Unchanged context: "
                    + " ".join(_legacy_fact(support) for support in original["supporting_context"]))
                legacy_target = _legacy_fact(fact, original["expected_fact"]["value"], original.get("expected_polarity"))
                if original["input_text"] != legacy_input or original["target_text"] != legacy_target:
                    raise ValueError(f"Source answer input/target differs from validated legacy text: {original['id']}")
                row["input_text"] = conditional_input(row["question"], row["correction"],
                    row["supporting_context"], row["request_context"]["as_of"])
                row["target_text"] = answer_for(fact, row["expected_fact"]["value"], row.get("expected_polarity"))
            if _semantic_record(original) != _semantic_record(row):
                raise ValueError("A grammar derivation changed structured facts, labels, scope or polarity")
            changed = sorted(field for field in TEXT_FIELDS if original.get(field) != row.get(field))
            changes.update(changed)
            derivations.append({"id": row["id"], "source_row_id": original["id"],
                "source_row_sha256": content_hash(original), "new_row_sha256": content_hash(row),
                "source_lineage_id": original["lineage_id"], "lineage_id": row["lineage_id"],
                "edit_id": row["edit_id"], "semantic_fields_sha256": content_hash(_semantic_record(row)),
                "changed_text_fields": changed, "repair": "numeric-one noun agreement using shared v2 formatter; no semantic changes"})
            output_rows.append(row)
        files[name] = b"".join(canonical_bytes(row) + b"\n" for row in output_rows)
    files["derivation.json"] = json_bytes(derivations)
    if any(file_hash(source / name) != digest for name, digest in origin_hashes.items()):
        raise ValueError("Source changed during derivation; refusing publication")
    source_id = source_manifest["dataset_id"]
    dataset_id = source_id.replace("-pilot-v1-", "-pilot-v2-") + "-grammar-" + origin_hashes["manifest.json"][:8]
    manifest = {"schema_version": 1, "pilot_version": 2, "repair_version": 1,
        "dataset_id": dataset_id, "input_format_version": INPUT_FORMAT_VERSION,
        "source_input_format_version": "editguard-input-v1", "tasks": list(TASKS),
        "purpose": "development_pilot_not_final_evaluation", "origin_dataset_id": source_id,
        "origin_hashes": origin_hashes, "counts": copy.deepcopy(COUNTS),
        "family_counts": {"train": 4, "validation": 2},
        "review_status": "pending_independent_human_decisions", "eligible_for_training": False,
        "eligible_for_final_evaluation": False, "test_frozen": False, "test_files_included": False,
        "human_review_decisions": {"pending": len(derivations), "accept": 0, "reject": 0},
        "repair_policy": "display grammar only; every review row gets a new ID/hash; structured entity/fact/edit/lineage IDs and labels are retained",
        "semantic_changes": False, "text_change_counts": dict(sorted(changes.items())),
        "source_code_sha256": {name: file_hash(ROOT / name) for name in (
            "editguard/formatting.py", "scripts/prepare_data.py", "scripts/repair_pilot.py")},
        "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    files["manifest.json"] = json_bytes(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.pending-{uuid4().hex}")
    staging.mkdir()
    try:
        for name, content in files.items():
            with (staging / name).open("xb") as target:
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
        if output.exists() or any(file_hash(source / name) != digest for name, digest in origin_hashes.items()):
            raise ValueError("Source or output changed before publication; refusing overwrite")
        os.rename(staging, output)
    finally:
        if staging.exists():
            resolved = staging.resolve()
            if resolved.parent != output.parent.resolve() or resolved.name != staging.name:
                raise ValueError("Staging cleanup target escaped the intended output parent")
            shutil.rmtree(resolved)
    return manifest


def _exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(json_bytes(value))
        target.flush()
        os.fsync(target.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/generated/pilot-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--manifest-report", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    for path in (args.manifest_report, args.report):
        if path is not None and path.exists():
            parser.error("Report output exists; choose a fresh path instead of overwriting")
        if path is not None and any(root.resolve() == path.resolve() or root.resolve() in path.resolve().parents
                                    for root in (args.source, args.output)):
            parser.error("Reports must remain outside source and derived dataset directories")
    if args.manifest_report and args.report and args.manifest_report.resolve() == args.report.resolve():
        parser.error("Manifest and derivation reports need separate output paths")
    try:
        manifest = repair_pilot(args.source, args.output)
        report = {"schema_version": 1, "purpose": "deterministic_pilot_grammar_derivation",
            "source_directory": str(args.source.resolve()), "output_directory": str(args.output.resolve()),
            "dataset_id": manifest["dataset_id"], "origin_dataset_id": manifest["origin_dataset_id"],
            "origin_hashes": manifest["origin_hashes"],
            "manifest_sha256": file_hash(args.output / "manifest.json"),
            "counts": manifest["counts"], "review_decisions": manifest["human_review_decisions"],
            "text_change_counts": manifest["text_change_counts"], "input_format_version": INPUT_FORMAT_VERSION,
            "source_unchanged": all(file_hash(args.source / name) == digest for name, digest in manifest["origin_hashes"].items()),
            "structured_semantics_unchanged": True, "review_registry_copied": False,
            "test_or_challenge_files_read_or_written": False, "model_weights_loaded": False,
            "gpu_used": False, "limitations": ["Deterministic grammar repair is not independent human semantic review.",
                "All 504 review decisions remain pending; this dataset is ineligible for training.",
                "This development derivation neither rewrites nor evaluates historical test/challenge files."]}
        if args.manifest_report:
            _exclusive_json(args.manifest_report, manifest)
        if args.report:
            _exclusive_json(args.report, report)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Derive a finite pilot-v3 that balances question forms across scope labels.

Only the existing pilot-v2 train/validation rows are read. Facts, edit worlds,
splits and answer targets stay fixed; every derived review row has a fresh ID and
remains pending. No models, review decisions or test/challenge files are used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text
from scripts.prepare_data import (NEGATIVE_TYPES, answer_for, canonical_bytes,
                                  content_hash, question_for)
from scripts.review_dataset import checked_rows, file_hash, integrity_flags, read_json

TASKS = ("scope", "answer")
SPLITS = ("train", "validation")
ROW_FILES = tuple(f"{task}_{split}.jsonl" for task in TASKS for split in SPLITS)
SOURCE_FILES = (*ROW_FILES, "catalogue.json", "selection.json", "split_assignments.json", "derivation.json")
COUNTS = {"scope": {"train": 240, "validation": 120}, "answer": {"train": 96, "validation": 48}}
STYLE_NAMES = ("direct", "paraphrase", "comparison_yes", "comparison_no")
MUTABLE_FIELDS = {"id", "question", "question_type", "expected_polarity"}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def question_variant(fact, style, value=None):
    """Use the queried fact's value, including the reversed at-most price test."""
    if type(style) is not int or style not in range(4):
        raise ValueError("Question style must be one of the four bounded forms")
    quantity = int(fact["value"] if value is None else value)
    if style == 3:
        quantity += -1 if fact["relation"] == "price_monthly" else 1
    if quantity < 0:
        raise ValueError("The controlled catalogue must permit a nonnegative comparison quantity")
    polarity = None
    if style >= 2:
        truth = (int(fact["value"] if value is None else value) <= quantity
                 if fact["relation"] == "price_monthly"
                 else quantity <= int(fact["value"] if value is None else value))
        polarity = "yes" if truth else "no"
    return question_for(fact, style, quantity), ("direct", "paraphrase", "comparison", "comparison")[style], polarity


def immutable_fields(row):
    return {key: value for key, value in row.items() if key not in MUTABLE_FIELDS}


def _pending(value):
    if isinstance(value, dict):
        if "reviewed" in value and value["reviewed"] is not False:
            raise ValueError("Embedded source records must remain unreviewed; external registries are never migrated")
        if "review_status" in value and value["review_status"] != "pending":
            raise ValueError("Embedded source records must retain pending review status")
        for item in value.values():
            _pending(item)
    elif isinstance(value, list):
        for item in value:
            _pending(item)


def validate_source(source):
    source = Path(source).resolve()
    if not source.is_dir() or (source / "test_lock.json").exists():
        raise ValueError("Source must be an existing unfrozen pilot-v2 development directory")
    names = {path.name for path in source.iterdir()}
    if names != set(SOURCE_FILES) | {"manifest.json"}:
        raise ValueError("Source contains missing, test/challenge, review-registry or unknown artifacts")
    manifest = read_json(source / "manifest.json")
    if (manifest.get("pilot_version") != 2 or manifest.get("schema_version") != 1
        or manifest.get("input_format_version") != INPUT_FORMAT_VERSION
        or INPUT_FORMAT_VERSION != "editguard-input-v2"
        or manifest.get("purpose") != "development_pilot_not_final_evaluation"
        or manifest.get("test_frozen") is not False or manifest.get("test_files_included") is not False
        or manifest.get("eligible_for_final_evaluation") is not False
        or manifest.get("counts") != COUNTS or set(manifest.get("tasks", [])) != set(TASKS)
        or set(manifest.get("files", {})) != set(SOURCE_FILES)):
        raise ValueError("Source must be the complete supported 504-row pilot-v2 with no test artifacts")
    if (manifest.get("eligible_for_training") is not False
        or manifest.get("review_status") != "pending_independent_human_decisions"):
        raise ValueError("Source rows must retain their original pending state; external decisions are never copied")
    origin_hashes = {"manifest.json": file_hash(source / "manifest.json")}
    for name in SOURCE_FILES:
        if not (source / name).is_file() or file_hash(source / name) != manifest["files"][name]:
            raise ValueError(f"Source checksum mismatch: {name}")
        origin_hashes[name] = file_hash(source / name)
    assignments = read_json(source / "split_assignments.json")
    if Counter(assignments.values()) != {"train": 4, "validation": 2}:
        raise ValueError("Exactly four train and two validation families are required")
    catalogue_rows = read_json(source / "catalogue.json")
    _pending(catalogue_rows)
    catalogue = {fact["fact_id"]: fact for fact in catalogue_rows}
    if len(catalogue) != len(catalogue_rows) or any(
        fact.get("split") not in SPLITS or assignments.get(fact["family_id"]) != fact["split"]
        for fact in catalogue_rows):
        raise ValueError("Source catalogue has duplicate facts or incompatible/test family assignments")
    rows_by_file, all_rows = {}, {}
    scope_groups = {split: defaultdict(list) for split in SPLITS}
    answer_groups = {split: defaultdict(list) for split in SPLITS}
    for task in TASKS:
        for split in SPLITS:
            rows = checked_rows(source, task, split)
            if len(rows) != COUNTS[task][split]:
                raise ValueError("Source row count differs from the complete finite pilot")
            for row in rows:
                _pending(row)
                if (not row["id"].startswith("pilot-v2:") or row["id"] in all_rows
                    or row.get("input_format_version") != INPUT_FORMAT_VERSION
                    or row.get("reviewed") is not False or row.get("review_status") != "pending"
                    or assignments.get(row["family_id"]) != split):
                    raise ValueError("Source row has incompatible identity, format, family or review metadata")
                all_rows[row["id"]] = row
                edit = row["correction"]
                original = catalogue.get(edit["fact_id"])
                expected = catalogue.get(row["expected_fact"]["fact_id"])
                if original is None or expected is None or original["split"] != split or expected["split"] != split:
                    raise ValueError("Each correction and queried fact must exist in the same development split")
                for key in ("entity_id", "relation", "unit", "plan", "version"):
                    if edit[key] != original[key] or row["expected_fact"][key] != expected[key]:
                        raise ValueError("Source fact scopes differ from the controlled catalogue")
                if (edit["original_value"] != original["value"] or edit["entity_name"] != original["entity_name"]
                    or edit["entity_aliases"] != original["aliases"] or row["edit_id"] != edit["edit_id"]
                    or row["lineage_id"] != edit["lineage_id"] or row["correction_text"] != correction_text(edit)):
                    raise ValueError("Source edit identity, original value or correction text differs from its fact")
                if any(catalogue.get(fact["fact_id"]) != fact or fact["split"] != split for fact in row.get("supporting_context", [])):
                    raise ValueError("Source supporting facts must remain exact same-split catalogue records")
                flags = integrity_flags(row, catalogue)
                if flags:
                    raise ValueError(f"Source integrity checks failed for {row['id']}: {', '.join(flags)}")
                if task == "scope" and row["label"] == 0:
                    if (row["negative_type"] not in NEGATIVE_TYPES or row["question_type"] != "out_of_scope"
                        or row.get("expected_polarity") is not None or row["question"] != question_for(expected, 0)
                        or row["expected_fact"]["value"] != expected["value"]):
                        raise ValueError("Source negatives must use the original direct-only catalogue-grounded form")
                else:
                    if not row["id"].endswith(tuple(f":p{style}" for style in range(4))):
                        raise ValueError("Source applicable rows must retain four known positive question forms")
                    style = int(row["id"][-1])
                    question, question_type, polarity = question_variant(expected, style, row["expected_fact"]["value"])
                    if (row["question"], row["question_type"], row.get("expected_polarity")) != (question, question_type, polarity):
                        raise ValueError("Source applicable wording/polarity differs from the controlled form")
                if task == "scope":
                    scope_groups[split][row["edit_id"]].append(row)
                else:
                    answer_groups[split][row["edit_id"]].append(row)
                    if (row["input_text"] != conditional_input(row["question"], edit, row["supporting_context"], row["request_context"]["as_of"])
                        or row["target_text"] != answer_for(expected, row["expected_fact"]["value"], row.get("expected_polarity"))):
                        raise ValueError("Source answer input or target differs from the runtime formatter")
            rows_by_file[f"{task}_{split}.jsonl"] = rows
    for split, groups in scope_groups.items():
        if len(groups) != (24 if split == "train" else 12) or set(groups) != set(answer_groups[split]):
            raise ValueError("Every split must retain its complete paired scope/answer edit worlds")
        for edit_id, rows in groups.items():
            if (len(rows) != 10 or Counter(row["label"] for row in rows) != {0: 6, 1: 4}
                or Counter(row["negative_type"] for row in rows if row["label"] == 0) != Counter(NEGATIVE_TYPES)
                or {row["id"][-1] for row in rows if row["label"] == 1} != set("0123")
                or len(answer_groups[split][edit_id]) != 4
                or {row["id"][-1] for row in answer_groups[split][edit_id]} != set("0123")
                or len({content_hash(row["correction"]) for row in rows + answer_groups[split][edit_id]}) != 1):
                raise ValueError("Each edit world must retain four positives, six negative types and four answer rows")
    derivations = read_json(source / "derivation.json")
    if (not isinstance(derivations, list) or len(derivations) != len(all_rows)
        or {item.get("id") for item in derivations} != set(all_rows)
        or any(item.get("new_row_sha256") != content_hash(all_rows[item["id"]]) for item in derivations)):
        raise ValueError("Source derivation must bind every immutable pilot-v2 row")
    return source, manifest, origin_hashes, catalogue, rows_by_file


def balance_scope_questions(source, output):
    source, source_manifest, origin_hashes, catalogue, rows_by_file = validate_source(source)
    output = Path(output).resolve()
    if output == source or output in source.parents or source in output.parents:
        raise ValueError("Output must be separate from and not contain the immutable source")
    if output.exists():
        raise ValueError("Existing or partial output is refused; choose a new directory")
    files = {name: (source / name).read_bytes() for name in ("catalogue.json", "selection.json", "split_assignments.json")}
    derivations, changes, style_counts = [], Counter(), {}
    for name, source_rows in rows_by_file.items():
        split = source_rows[0]["split"]
        edit_indices = {edit_id: index for index, edit_id in enumerate(sorted({row["edit_id"] for row in source_rows}))}
        output_rows = []
        balance = {style: {"positive": 0, "negative": 0} for style in STYLE_NAMES}
        for original in source_rows:
            row = copy.deepcopy(original)
            row["id"] = "pilot-v3:" + original["id"][len("pilot-v2:"):]
            style = None
            if name.startswith("scope_"):
                if original["label"] == 0:
                    style = (edit_indices[original["edit_id"]] + NEGATIVE_TYPES.index(original["negative_type"])) % 4
                    row["question"], row["question_type"], row["expected_polarity"] = question_variant(
                        catalogue[original["expected_fact"]["fact_id"]], style)
                else:
                    style = int(original["id"][-1])
                balance[STYLE_NAMES[style]]["positive" if original["label"] == 1 else "negative"] += 1
            if immutable_fields(original) != immutable_fields(row):
                raise ValueError("Revision changed fact truth, labels, scope, lineage or answer inputs/targets")
            if integrity_flags(row, catalogue):
                raise ValueError(f"Derived row integrity failed: {row['id']}")
            changed = sorted(key for key in MUTABLE_FIELDS - {"id"} if original.get(key) != row.get(key)
                             or (key in original) != (key in row))
            changes.update(changed)
            derivations.append({"id": row["id"], "source_row_id": original["id"],
                "source_row_sha256": content_hash(original), "new_row_sha256": content_hash(row),
                "source_lineage_id": original["lineage_id"], "lineage_id": row["lineage_id"], "edit_id": row["edit_id"],
                "immutable_fields_sha256": content_hash(immutable_fields(row)),
                "changed_fields": changed, "question_style": STYLE_NAMES[style] if style is not None else None,
                "scope_label_unchanged": True, "ground_truth_fact_unchanged": True,
                "policy": "existing negative question forms rotated by within-split edit rank and negative-type rank; all reviews pending"})
            output_rows.append(row)
        files[name] = b"".join(canonical_bytes(row) + b"\n" for row in output_rows)
        if name.startswith("scope_"):
            wanted = {"positive": 24 if split == "train" else 12, "negative": 36 if split == "train" else 18}
            if any(counts != wanted for counts in balance.values()):
                raise ValueError("Derived scope question forms do not have the required matched 40% positive prior")
            style_counts[split] = {key: {**value, "positive_prior": .4} for key, value in balance.items()}
    files["derivation.json"] = json_bytes(derivations)
    source_id = source_manifest["dataset_id"]
    dataset_id = source_id.replace("-pilot-v2-", "-pilot-v3-") + "-scope-balanced-" + origin_hashes["manifest.json"][:8]
    manifest = {"schema_version": 1, "pilot_version": 3, "revision_version": 1,
        "dataset_id": dataset_id, "input_format_version": INPUT_FORMAT_VERSION,
        "purpose": "development_pilot_not_final_evaluation", "tasks": list(TASKS),
        "origin_dataset_id": source_id, "origin_hashes": origin_hashes,
        "counts": copy.deepcopy(COUNTS), "family_counts": {"train": 4, "validation": 2},
        "review_status": "pending_independent_decisions", "eligible_for_training": False,
        "eligible_for_final_evaluation": False, "test_frozen": False, "test_files_included": False,
        "human_review_decisions": {"pending": len(derivations), "accept": 0, "reject": 0},
        "review_policy": "pending", "human_reviewed_data": False,
        "revision_policy": "bounded wording balance in existing negatives; new row identities; no prior review decisions migrated",
        "rotation": "style=(sorted_edit_index_within_split+negative_type_index)%4",
        "negative_type_order": list(NEGATIVE_TYPES), "question_style_order": list(STYLE_NAMES),
        "scope_style_balance": style_counts, "changed_field_counts": dict(sorted(changes.items())),
        "structured_fact_truth_changed": False, "scope_labels_changed": False,
        "positive_questions_changed": False, "answer_inputs_or_targets_changed": False,
        "source_code_sha256": {name: file_hash(ROOT / name) for name in (
            "scripts/balance_scope_questions.py", "scripts/prepare_data.py", "scripts/review_dataset.py", "editguard/formatting.py")},
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
                raise ValueError("Staging cleanup escaped the intended output parent")
            shutil.rmtree(resolved)
    return manifest


def exclusive_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as target:
        target.write(json_bytes(value))
        target.flush()
        os.fsync(target.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/generated/pilot-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--manifest-report", type=Path, default=ROOT / "reports/pilot-v3-manifest.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/scope-balancing.json")
    args = parser.parse_args()
    try:
        if args.manifest_report.resolve() == args.report.resolve():
            raise ValueError("Manifest and derivation reports require separate paths")
        for path in (args.manifest_report, args.report):
            if path.exists():
                raise ValueError("Existing report output is refused")
            if any(root.resolve() == path.resolve() or root.resolve() in path.resolve().parents for root in (args.source, args.output)):
                raise ValueError("Reports must remain outside source and derived dataset directories")
        manifest = balance_scope_questions(args.source, args.output)
        report = {"artifact_type": "editguard_scope_data_revision", "purpose": "bounded_development_revision",
            "status": "completed", "revision_version": 1, "dataset_id": manifest["dataset_id"],
            "origin_dataset_id": manifest["origin_dataset_id"], "origin_hashes": manifest["origin_hashes"],
            "derived_manifest_sha256": file_hash(args.output / "manifest.json"), "counts": manifest["counts"],
            "scope_style_balance": manifest["scope_style_balance"], "changed_field_counts": manifest["changed_field_counts"],
            "review_decisions": {"pending": 504, "accept": 0, "reject": 0}, "review_registry_copied": False,
            "source_unchanged": all(file_hash(args.source / name) == digest for name, digest in manifest["origin_hashes"].items()),
            "structured_fact_truth_changed": False, "scope_labels_changed": False,
            "positive_questions_changed": False, "answer_inputs_or_targets_changed": False,
            "test_or_challenge_files_read_or_written": False, "model_weights_loaded": False, "gpu_used": False,
            "limitations": ["This is a finite validation-informed development revision, not an accuracy result.",
                "Balancing question forms removes one observed label shortcut, not every possible shortcut.",
                "All 504 new review decisions remain pending; old review hashes cannot authorize these rows.",
                "Catalogue families and templates remain small and controlled; no frozen-test result is reported."]}
        exclusive_json(args.manifest_report, manifest)
        exclusive_json(args.report, report)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

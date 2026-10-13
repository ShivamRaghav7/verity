"""Cross-check explicit independent development decisions; never create human reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.scoring import SCORER_VERSION, score_answer
from scripts.prepare_data import content_hash
from scripts.review_dataset import checked_rows, file_hash, integrity_flags, read_json, validate_decision


ROW_FILES = {f"{task}_{split}.jsonl" for task in ("scope", "answer") for split in ("train", "validation")}
SOURCE_FILES = ROW_FILES | {"catalogue.json", "derivation.json", "selection.json", "split_assignments.json"}
COMMON_CHECKS = {"one_question_form", "one_scoped_catalogue_fact", "question_scope_and_effective_date",
                 "applicability_label_or_answer", "expected_identity_unit_date", "independently_selected_numeric_value",
                 "edit_original_and_scoped_metadata", "all_references_within_split", "source_flags_pending_and_simulated",
                 "input_format_v2", "edit_text_matches_metadata_and_noun_agreement", "independent_comparison_polarity",
                 "question_type"}
ANSWER_CHECKS = {"target_endorsed_subject_relation_value_unit_polarity", "two_unchanged_other_property_support_facts",
                 "input_exact_edit_question_date_support", "positive_scope_counterpart"}


def _crosscheck_accepted_observation(decision, row, catalogue, pilot_version=2):
    checks = decision.get("checks")
    required = set(COMMON_CHECKS)
    required.add(f"v{pilot_version}_row_derivation_hash")
    if pilot_version == 3:
        required.update({"v2_truth_scope_lineage_and_answer_preserved", "derivation_changed_fields_and_style",
                         "scope_style_matches_computed_polarity"})
    if row.get("question_type") == "comparison":
        required.add("comparison_noun_agreement")
    if row.get("label") == 0:
        required.add("negative_type_semantics")
    if "target_text" in row:
        required.update(ANSWER_CHECKS)
    if (not isinstance(checks, dict) or not required <= checks.keys()
            or any(value is not True for value in checks.values()) or decision.get("issues")):
        raise ValueError("Accepted decision lacks the complete passing independent checks.")
    applicable = row.get("label", 1)
    if (type(decision.get("computed_scope_label")) is not int or decision["computed_scope_label"] != applicable
            or type(decision.get("computed_applicable")) is not bool or decision["computed_applicable"] != bool(applicable)
            or decision.get("computed_polarity") != row.get("expected_polarity")
            or decision.get("computed_expected_fact") != row["expected_fact"]):
        raise ValueError("Recorded independent applicability, polarity or fact disagrees with the reviewed row.")
    fact = catalogue[row["expected_fact"]["fact_id"]]
    observed = decision.get("parsed_question_fact")
    if (not isinstance(observed, dict)
            or any(observed.get(key) != fact[key] for key in ("fact_id", "entity_id", "entity_name", "relation", "plan", "version", "value", "unit"))
            or observed.get("as_of") != row["request_context"]["as_of"]):
        raise ValueError("Recorded parsed question fact differs from the unchanged catalogue and request.")
    if pilot_version == 3:
        expected_style = ("comparison_" + row["expected_polarity"]
                          if row["question_type"] == "comparison" else row["question_type"])
        if expected_style not in ("direct", "paraphrase", "comparison_yes", "comparison_no") or decision.get("computed_question_style") != expected_style:
            raise ValueError("Recorded independently parsed question style differs from the revised row.")


def register(dataset, audit_path, output):
    if output.exists():
        raise ValueError("Use a fresh automated registry filename; decisions are immutable.")
    if dataset.resolve() in output.resolve().parents or output.resolve() in dataset.resolve().parents or output.resolve() == dataset.resolve():
        raise ValueError("The automated registry must remain separate from the pending source dataset.")
    audit, manifest = read_json(audit_path), read_json(dataset / "manifest.json")
    version = manifest.get("pilot_version") if isinstance(manifest, dict) else None
    if (not isinstance(audit, dict) or not isinstance(manifest, dict)
            or type(version) is not int or version not in (2, 3) or manifest.get("input_format_version") != "editguard-input-v2"
            or manifest.get("purpose") != "development_pilot_not_final_evaluation"
            or manifest.get("test_files_included") is not False or manifest.get("test_frozen") is not False
            or not isinstance(manifest.get("files"), dict) or set(manifest["files"]) != SOURCE_FILES
            or {path.name for path in dataset.iterdir()} != SOURCE_FILES | {"manifest.json"}):
        raise ValueError("Automated registration is restricted to the complete unfrozen pilot-v2 development dataset or its supported pilot-v3 revision.")
    if (audit.get("reviewer_type") != "automated_development" or audit.get("human_review_performed") is not False
            or audit.get("human_approval_claimed") is not False or audit.get("test_or_challenge_rows_read") is not False
            or audit.get("dataset_id") != manifest["dataset_id"] or type(audit.get("total_rows")) is not int
            or audit["total_rows"] != 504):
        raise ValueError("Expected complete, explicitly automated development review for this pilot.")
    method = audit.get("implementation")
    if (not isinstance(method, dict) or not isinstance(method.get("source"), str) or not method["source"].strip()
            or method.get("sha256") != hashlib.sha256(method["source"].encode()).hexdigest()):
        raise ValueError("Independent review method source hash differs.")
    global_checks = audit.get("global_checks")
    required_global = {"exact504_unique_rows", "partition_counts", f"v{version}_development_only",
                       "disjoint:family", "disjoint:entity", "disjoint:lineage"} | {"manifest_hash:" + name for name in SOURCE_FILES}
    if version == 3:
        required_global.update({"unique_catalogue_and_derivations", "origin_catalogue_selection_and_splits_unchanged",
                                "four_train_two_validation_families", "complete36_edit_worlds",
                                "matched_style_prior:train", "matched_style_prior:validation",
                                "style_manifest_matches:train", "style_manifest_matches:validation"})
        expected_origin_hashes = {"data/generated/pilot-v2/" + name: value
                                  for name, value in manifest.get("origin_hashes", {}).items()}
        if (not expected_origin_hashes or audit.get("origin_source_hashes") != expected_origin_hashes
                or audit.get("sources_unchanged_after_review") is not True):
            raise ValueError("Pilot-v3 review must bind the unchanged complete pilot-v2 origin.")
    if (not isinstance(global_checks, dict) or not required_global <= global_checks.keys()
            or any(value is not True for value in global_checks.values())):
        raise ValueError("Independent dataset/hash/split checks must be complete and passing.")
    for name in (*manifest["files"], "manifest.json"):
        key = f"data/generated/pilot-v{version}/" + name
        if audit.get("source_hashes", {}).get(key) != file_hash(dataset / name):
            raise ValueError(f"Review source hashes differ: {name}")
    catalogue = {fact["fact_id"]: fact for fact in read_json(dataset / "catalogue.json")}
    rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
            for row in checked_rows(dataset, task, split)}
    decisions = audit.get("decisions", [])
    if (not isinstance(decisions, list) or any(not isinstance(item, dict) or not isinstance(item.get("id"), str) for item in decisions)
            or len(rows) != 504 or len(decisions) != len(rows) or {item["id"] for item in decisions} != set(rows)):
        raise ValueError("Review must cover exactly all 504 unique source rows.")
    records, scored_targets = {}, 0
    for decision in decisions:
        row = rows[decision["id"]]
        if (decision.get("row_sha256") != content_hash(row) or decision.get("reviewer_type") != "automated_development"
                or decision.get("decision") not in ("accept", "flagged") or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()):
            raise ValueError("Independent review contains a stale, untyped or unexplained decision.")
        flags = integrity_flags(row, catalogue)
        if decision["decision"] == "accept":
            if flags:
                raise ValueError("Accepted decision disagrees with an independent or integration check.")
            _crosscheck_accepted_observation(decision, row, catalogue, version)
            if "target_text" in row:
                fact = {**catalogue[row["expected_fact"]["fact_id"]], **row["expected_fact"]}
                score = score_answer(row["target_text"], fact, aliases=fact.get("aliases", []),
                                     expected_polarity=row.get("expected_polarity"), answer_kind="dataset_target")
                if score["correct"] is not True:
                    raise ValueError(f"Target assertion requires adjudication: {row['id']}")
                parsed_target = decision.get("parsed_target")
                if (not isinstance(parsed_target, dict) or parsed_target.get("noun_agreement") is not True
                        or any(parsed_target.get(key) != row["expected_fact"][key] for key in ("fact_id", "entity_id", "relation", "plan", "version", "value", "unit"))
                        or parsed_target.get("polarity") != row.get("expected_polarity")):
                    raise ValueError("Recorded independent target assertion differs from the reviewed factual target.")
                scored_targets += 1
        record = {"decision": "accept" if decision["decision"] == "accept" else "reject",
                  "reviewer": decision["reviewer"], "reviewer_type": "automated_development",
                  "reviewed_at": decision["reviewed_at"], "row_sha256": decision["row_sha256"], "reason": decision["reason"]}
        validate_decision(row, record)
        records[row["id"]] = record
    registry = {"schema_version": 1, "dataset_id": manifest["dataset_id"],
                "origin_manifest_sha256": file_hash(dataset / "manifest.json"), "review_policy": "automated_development",
                "human_reviewed_data": False, "audit_path": str(audit_path), "audit_sha256": file_hash(audit_path),
                "authorization_basis": audit["authorization_basis"], "rows": records,
                "integration_crosscheck": {"rows_checked": len(rows), "targets_scored": scored_targets,
                    "scorer_version": SCORER_VERSION, "script_sha256": file_hash(Path(__file__)),
                    "claim": "Dataset-target and label consistency only; no generated-model accuracy or human review."}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return {"rows_checked": len(rows), "accepted": sum(value["decision"] == "accept" for value in records.values()),
            "rejected": sum(value["decision"] == "reject" for value in records.values()), "target_checks": scored_targets,
            "human_reviewed_data": False, "registry": str(output), "registry_sha256": file_hash(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(register(args.dataset, args.audit, args.output), indent=2))
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

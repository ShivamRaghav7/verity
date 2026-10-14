"""Prepare and machine-audit held-out inputs only after immutable protocol freeze.

Original files are preserved. No filtering, new questions, new facts or changed
labels are permitted. Canonical model-input/target display strings are derived
under the source-bound input-v2 formatter. This is a separate final-only workflow;
development training/review gates retain their original restrictions.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text, fact_text
from editguard.scope_inputs import scope_pair
from scripts.audit_pilot_v3 import assertion
from scripts.final_label_audit import audit_rows
from scripts.final_protocol import (canonical_hash, file_hash, read_json, relative_path,
                                    resolve_path, unique_keys, validate_protocol)

ROW_NAMES = {"scope": "scope_test.jsonl", "answer": "answer_test.jsonl", "challenge": "challenge_test.jsonl"}
PACKET_NAMES = {*ROW_NAMES.values(), "catalogue.json", "edits.json", "split_assignments.json",
                "reviews.json", "audit.json", "derivation.json"}
TEXT_FIELDS = {"correction_text", "input_text", "target_text", "input_format_version"}


def read_rows(path):
    return [json.loads(line, object_pairs_hook=unique_keys)
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def derive_row(row, population, catalogue, scenarios, subjects):
    result = dict(row)
    if population == "challenge":
        if "correction" not in result:
            if row["scenario_id"] not in scenarios:
                raise ValueError("Challenge scenario does not refer to an original correction")
            result["correction"] = scenarios[row["scenario_id"]]
        result.update(label=1, question_type="challenge", expected_polarity=None)
    elif population == "answer":
        if "label" in result and result["label"] != 1:
            raise ValueError("Oracle answer source row carries an inapplicable label")
        result["label"] = 1
        # Check the original complete answer semantics before grammar repair.
        observed = assertion(row["target_text"], subjects)
        expected = row["expected_fact"]
        if (observed is None or any(observed[key] != expected[key] for key in (
                "fact_id", "entity_id", "relation", "plan", "version", "unit", "value"))
                or observed["polarity"] != row.get("expected_polarity")):
            raise ValueError("Original answer target disagrees with unchanged structured truth")
        fact = {**catalogue[expected["fact_id"]], "value": expected["value"]}
        prefix = (row["expected_polarity"].capitalize() + ". ") if row.get("expected_polarity") else ""
        result["target_text"] = prefix + fact_text(fact)
        result["input_text"] = conditional_input(row["question"], row["correction"],
            row["supporting_context"], row["request_context"]["as_of"])
    elif population != "scope":
        raise ValueError("Unknown final packet population")
    result["correction_text"] = correction_text(result["correction"])
    result["input_format_version"] = INPUT_FORMAT_VERSION
    # Every original non-text field remains exactly equal; implicit oracle and
    # challenge fields are added only where absent in the original source.
    for key, value in row.items():
        if key not in TEXT_FIELDS and result.get(key) != value:
            raise ValueError("Final packet changed original semantic field: " + key)
    added = sorted(set(result) - set(row))
    allowed = TEXT_FIELDS | ({"label"} if population == "answer" else
        {"label", "question_type", "expected_polarity", "correction"} if population == "challenge" else set())
    if not set(added) <= allowed:
        raise ValueError("Final packet added an unsupported field")
    derivation = {"id": row["id"], "population": population, "source_row_sha256": canonical_hash(row),
        "derived_row_sha256": canonical_hash(result), "changed_fields": sorted(key for key in row if row[key] != result[key]),
        "added_fields": added, "original_question_unchanged": result["question"] == row["question"],
        "original_semantic_fields_sha256": canonical_hash({key: value for key, value in row.items() if key not in TEXT_FIELDS}),
        "source_family_id": row["family_id"], "source_lineage_id": row["lineage_id"]}
    return result, derivation


def references(rows, catalogue):
    sets = {"family": set(), "entity": set(), "lineage": set()}
    for row in rows:
        items = [row, row.get("correction", {}), *row.get("supporting_context", [])]
        if row.get("expected_fact", {}).get("fact_id") in catalogue:
            items.append(catalogue[row["expected_fact"]["fact_id"]])
        for item in items:
            for field, key in (("family_id", "family"), ("entity_id", "entity"), ("lineage_id", "lineage")):
                if item.get(field):
                    sets[key].add(item[field])
        if row.get("question_family_id"):
            sets["family"].add(row["question_family_id"])
    return sets


def leakage_and_overlap(populations, facts, protocol, root):
    dev_dir = resolve_path(root, protocol["development_directory"])
    dev_catalogue = {fact["fact_id"]: fact for fact in read_json(dev_dir / "catalogue.json")}
    final_catalogue = {fact["fact_id"]: fact for fact in facts}
    final_refs = references([row for rows in populations.values() for row in rows], final_catalogue)
    dev_scope = {}
    checks = {}
    for split in ("train", "validation"):
        rows = [row for task in ("scope", "answer") for row in read_rows(dev_dir / f"{task}_{split}.jsonl")]
        dev_refs = references(rows, dev_catalogue)
        for key in final_refs:
            checks[f"raw_disjoint:{split}:{key}"] = not bool(final_refs[key] & dev_refs[key])
        checks[f"raw_disjoint:{split}:declared_family"] = not bool(final_refs["family"] & set(protocol["development_families"][split]))
        dev_scope[split] = [row for row in rows if "scope:" in row["id"]]
    if not all(checks.values()):
        raise ValueError("Final packet leaks raw entity/family/lineage references into development data")
    for task, record in protocol["models"].items():
        training_dir = resolve_path(root, record["training_data_directory"])
        training_catalogue = {fact["fact_id"]: fact for fact in read_json(training_dir / "catalogue.json")}
        for split in ("train", "validation"):
            training_rows = read_rows(training_dir / f"{task}_{split}.jsonl")
            training_refs = references(training_rows, training_catalogue)
            for key in final_refs:
                checks[f"raw_disjoint:selected_{task}_{split}:{key}"] = not bool(final_refs[key] & training_refs[key])
    if not all(checks.values()):
        raise ValueError("Final packet leaks raw entity/family/lineage references into selected model training data")
    policy = protocol["preprocessing_disclosure"]["scope_input_policy"]
    indexed = {}
    for split, rows in [*dev_scope.items(), ("test", populations["scope"])]:
        pairs = defaultdict(list)
        for row in rows:
            pair = scope_pair(row["question"], correction_text(row["correction"]), policy)
            pairs[pair].append(row)
            if pair in indexed and indexed[pair] != row["label"]:
                raise ValueError("Scope preprocessing creates contradictory labels for an identical input pair")
            indexed[pair] = row["label"]
        dev_scope[split] = pairs
    overlap = {"policy": policy, "test_scope_rows": len(populations["scope"]),
               "test_unique_scope_pairs": len(dev_scope["test"]), "contradictory_pairs": 0,
               "disclaimer": "Raw families/entities/lineages are separated; canonicalization and shared controlled logical forms can create identical same-label inputs across splits."}
    for split in ("train", "validation"):
        shared = set(dev_scope["test"]) & set(dev_scope[split])
        ids = sorted(row["id"] for pair in shared for row in dev_scope["test"][pair])
        overlap[f"scope_row_ids_sharing_{split}_pair"] = ids
        overlap[f"same_label_pairs_shared_with_{split}"] = len(shared)
        overlap[f"test_rows_sharing_{split}_pair"] = len(ids)
    shared_ids = set(overlap["scope_row_ids_sharing_train_pair"]) | set(overlap["scope_row_ids_sharing_validation_pair"])
    overlap["scope_row_ids_without_development_pair_overlap"] = sorted(row["id"] for row in populations["scope"] if row["id"] not in shared_ids)
    overlap["test_rows_without_development_pair_overlap"] = len(overlap["scope_row_ids_without_development_pair_overlap"])
    return checks, overlap


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def rows_bytes(rows):
    return b"".join((json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False) + "\n").encode("utf-8") for row in rows)


def prepare_packet(protocol_path, protocol_sha256, output, *, root=ROOT):
    root = root.resolve()
    output = resolve_path(root, output)
    if output.exists():
        raise ValueError("Use a fresh final packet directory; overwriting is forbidden")
    # This validation occurs before any held-out question JSON is deserialized.
    protocol = validate_protocol(protocol_path, protocol_sha256, root=root)
    original = resolve_path(root, protocol["original"]["directory"])
    if output == original or original in output.parents or output in original.parents:
        raise ValueError("Final packet must be separate from the immutable original dataset")
    assignments = read_json(original / "split_assignments.json")
    facts = [fact for fact in read_json(original / "catalogue.json") if fact["split"] == "test"]
    edits = [edit for edit in read_json(original / "edits.json") if edit["split"] == "test"]
    catalogue = {fact["fact_id"]: fact for fact in facts}
    scenarios = {edit["scenario_id"]: edit for edit in edits}
    if len(catalogue) != len(facts) or len(scenarios) != len(edits):
        raise ValueError("Original test catalogue/scenarios contain duplicate identities")
    subjects = defaultdict(list)
    for fact in facts:
        for alias in [fact["entity_name"], *fact.get("aliases", [])]:
            subjects[(f"{alias} {fact['plan']} version {fact['version']}", fact["relation"])].append(fact)
    source_names = {"scope": "scope_test.jsonl", "answer": "answer_test.jsonl", "challenge": "challenge_drafts.jsonl"}
    populations, derivations = {}, []
    for population, name in source_names.items():
        originals = read_rows(original / name)
        if len(originals) != protocol["evaluation"]["question_counts"][population]:
            raise ValueError("Original held-out row count differs from the frozen protocol")
        rows = []
        for source in originals:
            derived, derivation = derive_row(source, population, catalogue, scenarios, subjects)
            rows.append(derived)
            derivations.append(derivation)
        populations[population] = sorted(rows, key=lambda row: row["id"])
    leak_checks, overlap = leakage_and_overlap(populations, facts, protocol, root)
    reviewed_at = datetime.now(timezone.utc).isoformat()
    audit = audit_rows(facts, populations, assignments, reviewed_at)
    audit.update(protocol_sha256=protocol_sha256, original_manifest_sha256=protocol["original"]["manifest_sha256"],
                 global_checks=leak_checks, scope_normalized_overlap=overlap,
                 label_or_question_changes=False, source_dataset_modified=False,
                 implementation_sha256=file_hash(root / "scripts/final_label_audit.py"))
    files = {ROW_NAMES[population]: rows_bytes(rows) for population, rows in populations.items()}
    files.update({"catalogue.json": json_bytes(facts), "edits.json": json_bytes(edits),
                  "split_assignments.json": json_bytes({key: value for key, value in assignments.items() if value == "test"}),
                  "derivation.json": json_bytes(derivations), "audit.json": json_bytes(audit)})
    registry = {"schema_version": 1, "review_policy": "automated_final", "human_reviewed_data": False,
        "human_approval_claimed": False, "protocol_sha256": protocol_sha256,
        "audit_sha256": hashlib.sha256(files["audit.json"]).hexdigest(), "reviewed_at": reviewed_at,
        "rows": {name: {key: item[key] for key in ("row_sha256", "decision", "reviewer", "reviewer_type", "reviewed_at", "reason", "issues")}
                 for name, item in audit["rows"].items()}}
    files["reviews.json"] = json_bytes(registry)
    manifest = {"schema_version": 1, "purpose": "frozen_final_evaluation_packet", "dataset_id": "editguard-controlled-v1-final-" + protocol["protocol_id"],
        "input_format_version": INPUT_FORMAT_VERSION, "protocol_path": relative_path(root, protocol_path),
        "protocol_sha256": protocol_sha256, "original_manifest_sha256": protocol["original"]["manifest_sha256"],
        "files": {name: hashlib.sha256(contents).hexdigest() for name, contents in files.items()},
        "counts": {population: len(rows) for population, rows in populations.items()},
        "review_policy": "automated_final", "human_reviewed_data": False,
        "eligible_for_final_evaluation": audit["flagged"] == 0, "eligible_for_training": False,
        "test_frozen": True, "all_rows_preserved": True, "review_counts": {"accepted": audit["accepted"], "flagged": audit["flagged"]},
        "scope_normalized_overlap": overlap,
        "question_type_mix": {population: dict(Counter(row["question_type"] for row in rows))
                              for population, rows in populations.items()},
        "format_adaptation": "Original questions, facts, labels, plans, versions, dates, splits and lineages unchanged. Current input-v2 correction/answer display strings and explicit oracle/challenge routing metadata only.",
        "challenge_authorship": "original agent_authored_draft_requires_independent_human_revision retained; machine consistency review only",
        "no_model_or_gpu_used": True, "original_files_preserved": True}
    # Reject source/config changes before publishing any derived file.
    validate_protocol(protocol_path, protocol_sha256, root=root)
    output.mkdir(parents=True)
    for name, contents in {**files, "manifest.json": json_bytes(manifest)}.items():
        with (output / name).open("xb") as handle:
            handle.write(contents)
    return manifest


def validate_packet(packet, protocol_path, protocol_sha256, *, root=ROOT):
    root = root.resolve()
    protocol = validate_protocol(protocol_path, protocol_sha256, root=root)
    packet = resolve_path(root, packet)
    if {path.name for path in packet.iterdir()} != PACKET_NAMES | {"manifest.json"}:
        raise ValueError("Final packet inventory differs")
    manifest = read_json(packet / "manifest.json")
    if (manifest.get("purpose") != "frozen_final_evaluation_packet" or manifest.get("protocol_sha256") != protocol_sha256
            or manifest.get("original_manifest_sha256") != protocol["original"]["manifest_sha256"]
            or manifest.get("input_format_version") != "editguard-input-v2" or manifest.get("review_policy") != "automated_final"
            or manifest.get("human_reviewed_data") is not False or manifest.get("test_frozen") is not True
            or manifest.get("eligible_for_training") is not False or manifest.get("all_rows_preserved") is not True
            or set(manifest.get("files", {})) != PACKET_NAMES):
        raise ValueError("Final packet/protocol binding or disclosures differ")
    for name, digest in manifest["files"].items():
        if file_hash(packet / name) != digest:
            raise ValueError("Final packet checksum differs: " + name)
    populations = {population: read_rows(packet / name) for population, name in ROW_NAMES.items()}
    if manifest["counts"] != protocol["evaluation"]["question_counts"] or any(
            len(rows) != manifest["counts"][population] for population, rows in populations.items()):
        raise ValueError("Final packet must preserve every declared population row")
    registry, audit = read_json(packet / "reviews.json"), read_json(packet / "audit.json")
    if (registry.get("protocol_sha256") != protocol_sha256 or registry.get("review_policy") != "automated_final"
            or registry.get("human_reviewed_data") is not False or registry.get("audit_sha256") != file_hash(packet / "audit.json")
            or registry.get("human_approval_claimed") is not False
            or audit.get("protocol_sha256") != protocol_sha256 or audit.get("human_reviewed_data") is not False
            or audit.get("human_approval_claimed") is not False or audit.get("human_semantic_review") is not False
            or audit.get("flagged") != 0 or manifest.get("eligible_for_final_evaluation") is not True
            or manifest.get("review_counts") != {"accepted": audit.get("accepted"), "flagged": audit.get("flagged")}
            or not all(audit.get("global_checks", {}).values())):
        raise ValueError("Final packet has unresolved audit flags or stale automated review evidence")
    all_rows = [row for rows in populations.values() for row in rows]
    if len({row["id"] for row in all_rows}) != len(all_rows) or set(registry["rows"]) != {row["id"] for row in all_rows}:
        raise ValueError("Final registry must cover exactly all unique held-out rows")
    for row in all_rows:
        decision = registry["rows"][row["id"]]
        expected_decision = {key: audit["rows"][row["id"]][key] for key in (
            "row_sha256", "decision", "reviewer", "reviewer_type", "reviewed_at", "reason", "issues")}
        if (decision["decision"] != "accept" or decision["row_sha256"] != canonical_hash(row)
                or decision["reviewer_type"] != "automated_final" or decision["issues"]
                or decision != expected_decision
                or datetime.fromisoformat(decision["reviewed_at"]).utcoffset() is None
                or audit["rows"][row["id"]]["row_sha256"] != canonical_hash(row)):
            raise ValueError("Final review record differs from its exact row")
    derivations = read_json(packet / "derivation.json")
    if len(derivations) != len(all_rows) or {item["id"] for item in derivations} != {row["id"] for row in all_rows}:
        raise ValueError("Every final row requires a unique original derivation")
    for item in derivations:
        row = next(row for row in all_rows if row["id"] == item["id"])
        if item["derived_row_sha256"] != canonical_hash(row) or item["original_question_unchanged"] is not True:
            raise ValueError("Final derivation differs from its exact row")
    facts = read_json(packet / "catalogue.json")
    original = resolve_path(root, protocol["original"]["directory"])
    original_facts = [fact for fact in read_json(original / "catalogue.json") if fact["split"] == "test"]
    original_edits = [edit for edit in read_json(original / "edits.json") if edit["split"] == "test"]
    assignments = read_json(original / "split_assignments.json")
    if (facts != original_facts or read_json(packet / "edits.json") != original_edits
            or read_json(packet / "split_assignments.json") != {key: value for key, value in assignments.items() if value == "test"}):
        raise ValueError("Final packet altered original catalogue, corrections or split assignments")
    catalogue = {fact["fact_id"]: fact for fact in facts}
    scenarios = {edit["scenario_id"]: edit for edit in original_edits}
    subjects = defaultdict(list)
    for fact in facts:
        for alias in [fact["entity_name"], *fact.get("aliases", [])]:
            subjects[(f"{alias} {fact['plan']} version {fact['version']}", fact["relation"])].append(fact)
    expected_derivations = []
    for population, name in {"scope": "scope_test.jsonl", "answer": "answer_test.jsonl", "challenge": "challenge_drafts.jsonl"}.items():
        expected_rows = []
        for original_row in read_rows(original / name):
            expected_row, expected_derivation = derive_row(original_row, population, catalogue, scenarios, subjects)
            expected_rows.append(expected_row)
            expected_derivations.append(expected_derivation)
        if sorted(expected_rows, key=lambda row: row["id"]) != populations[population]:
            raise ValueError("Final packet changed original questions, labels, facts or permitted transformations")
    if expected_derivations != derivations:
        raise ValueError("Final packet original-row derivations differ")
    expected_audit = audit_rows(facts, populations, assignments, registry["reviewed_at"])
    for key, value in expected_audit.items():
        if audit.get(key) != value:
            raise ValueError("Final packet independent audit cannot be reproduced: " + key)
    leak_checks, overlap = leakage_and_overlap(populations, facts, protocol, root)
    if leak_checks != audit["global_checks"] or overlap != manifest["scope_normalized_overlap"] or overlap != audit["scope_normalized_overlap"]:
        raise ValueError("Final split/abstraction disclosure differs from reproducible checks")
    return manifest, facts, populations["scope"], populations["answer"], populations["challenge"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_packet(args.protocol, args.protocol_sha256, args.output)
    print(json.dumps({key: manifest[key] for key in ("counts", "review_counts", "eligible_for_final_evaluation", "human_reviewed_data")}, indent=2))


if __name__ == "__main__":
    main()

"""Freeze model/configuration/scoring choices before reading held-out questions.

Freezing checks original file bytes against the untouched source manifest but
does not deserialize scope/answer test or challenge content. The separate packet
command requires this immutable protocol and its external SHA-256 first.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.scoring import SCORER_VERSION
from editguard.short_qa import SHORT_QA_VERSION

CONTROLLED_ORIGINAL_SHA = "2c5882490ac5f29c45c2bf20f97128d46cc3e0974c84a4838d47bd3ac8aaa42d"
CONFIGURATIONS = ["unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor"]
# Exact inference, scoring, preparation and artifact-validation dependencies.
# The HTTP/UI layer has separate regression evidence and does not execute in
# this benchmark; presentation work must not invalidate unchanged ML evidence.
SOURCE_NAMES = (
    "editguard/__init__.py", "editguard/comparison.py", "editguard/neural.py",
    "editguard/formatting.py", "editguard/scope_inputs.py", "editguard/scoring.py",
    "editguard/short_qa.py", "editguard/metrics.py", "editguard/training.py",
    "scripts/run_comparison.py", "scripts/aggregate_evaluation.py", "scripts/analyze_short_qa.py",
    "scripts/audit_pilot_v3.py", "scripts/prepare_data.py", "scripts/review_dataset.py",
    "scripts/final_protocol.py", "scripts/prepare_final_packet.py",
    "scripts/final_label_audit.py", "scripts/run_final_evaluation.py",
    "scripts/preflight_training.py", "scripts/train_auxiliary.py")
RUNTIME_PACKAGES = ("torch", "transformers", "numpy", "tokenizers", "safetensors",
                    "huggingface-hub", "psutil")
DEFAULT_DEVELOPMENT_EVALUATION = "reports/targeted-candidate-v3-scored/manifest.json"


def runtime_environment():
    """Inspect installed distribution metadata without importing neural libraries."""
    try:
        packages = {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("Freeze and execute final evaluation in the prepared ML environment; missing distribution: " + str(error)) from error
    return {"python_version": sys.version, "runtime_packages": packages}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_keys(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys)


def resolve_path(root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if path == root.resolve() or root.resolve() not in path.parents:
        raise ValueError("Protocol paths must stay within the project workspace")
    return path


def relative_path(root, path):
    return str(resolve_path(root, path).relative_to(root.resolve())).replace("\\", "/")


def record_file(root, path):
    path = resolve_path(root, path)
    return {"path": relative_path(root, path), "sha256": file_hash(path)}


def model_record(root, value, task):
    artifact = resolve_path(root, value)
    manifest_path = artifact / "manifest.json"
    manifest = read_json(manifest_path)
    if (manifest.get("task") != task or manifest.get("status") != "completed" or manifest.get("purpose") != "trained"
            or not manifest.get("files_sha256") or manifest.get("input_format_version") != "editguard-input-v2"):
        raise ValueError("Final evaluation requires completed, trained input-v2 auxiliary artifacts")
    for name, digest in manifest["files_sha256"].items():
        path = resolve_path(root, artifact / name)
        if artifact not in path.parents or file_hash(path) != digest:
            raise ValueError("Model export checksum differs: " + name)
    card_path = artifact / "run_card.json"
    if file_hash(card_path) != manifest["run_card_sha256"]:
        raise ValueError("Model training card checksum differs")
    card = read_json(card_path)
    profile_path = resolve_path(root, card["profile_path"])
    if file_hash(profile_path) != card["profile_sha256"]:
        raise ValueError("Model training profile checksum differs")
    implementation_sources = card["profile_source_sha256"]
    if not {"editguard/training.py", "scripts/train_auxiliary.py", "scripts/prepare_data.py",
            "scripts/preflight_training.py", "editguard/formatting.py", "editguard/scope_inputs.py"} <= implementation_sources.keys():
        raise ValueError("Selected training artifacts require the complete implementation source binding")
    for name, digest in implementation_sources.items():
        if file_hash(resolve_path(root, name)) != digest:
            raise ValueError("Selected training implementation checksum differs: " + name)
    data_dir = resolve_path(root, card["data_dir"])
    data_manifest = read_json(data_dir / "manifest.json")
    if file_hash(data_dir / "manifest.json") != manifest["manifest_sha256"] or data_manifest.get("test_files_included") is not False:
        raise ValueError("Model training data must bind a development-only manifest")
    training_sources = {relative_path(root, card_path): file_hash(card_path),
                        relative_path(root, data_dir / "manifest.json"): manifest["manifest_sha256"]}
    for name, digest in data_manifest["files"].items():
        path = resolve_path(root, data_dir / name)
        if data_dir not in path.parents or file_hash(path) != digest:
            raise ValueError("Model training data checksum differs: " + name)
        training_sources[relative_path(root, path)] = digest
    registry_path = resolve_path(root, card["reviews_path"])
    if file_hash(registry_path) != manifest["reviews_sha256"]:
        raise ValueError("Model training review registry checksum differs")
    training_sources[relative_path(root, registry_path)] = manifest["reviews_sha256"]
    return {"artifact": relative_path(root, artifact), "manifest_sha256": file_hash(manifest_path),
        "files_sha256": manifest["files_sha256"], "selected_model_state_sha256": manifest["selected_model_state_sha256"],
        "model_id": manifest["model_id"], "model_revision": manifest["model_revision"],
        "training_review_policy": manifest["review_policy"], "human_reviewed_data": manifest["human_reviewed_data"],
        "dataset_id": manifest["dataset_id"], "dataset_manifest_sha256": manifest["manifest_sha256"],
        "training_data_directory": relative_path(root, data_dir), "training_sources_sha256": training_sources,
        "run_card_sha256": manifest["run_card_sha256"],
        "training_profile": record_file(root, profile_path),
        "training_implementation_sha256": implementation_sources,
        "scope_input_policy": manifest.get("scope_input_policy", "raw_v1") if task == "scope" else None,
        "scope_input_overlap": manifest.get("scope_input_overlap")}


def verify_original(original, root, *, expected_manifest_sha256, expected_files=None):
    directory = resolve_path(root, original)
    path = directory / "manifest.json"
    if file_hash(path) != expected_manifest_sha256:
        raise ValueError("Untouched original dataset manifest checksum differs")
    manifest = read_json(path)
    if manifest.get("dataset_id") != "editguard-controlled-v1" or not isinstance(manifest.get("files"), dict):
        raise ValueError("Expected the untouched controlled-v1 demonstration dataset")
    if expected_files is not None and manifest["files"] != expected_files:
        raise ValueError("Original dataset file manifest differs from the protocol")
    if {path.name for path in directory.iterdir()} != set(manifest["files"]) | {"manifest.json"}:
        raise ValueError("Original dataset inventory differs; do not mutate it to freeze evaluation")
    for name, digest in manifest["files"].items():
        file_path = resolve_path(root, directory / name)
        if directory not in file_path.parents or file_hash(file_path) != digest:
            raise ValueError("Untouched original dataset checksum differs: " + name)
    return manifest


def freeze_protocol(config_path, original, development, output, execution_output, ledger_path,
                    *, root=ROOT, original_manifest_sha256=CONTROLLED_ORIGINAL_SHA,
                    development_evaluation=DEFAULT_DEVELOPMENT_EVALUATION):
    root = root.resolve()
    output = resolve_path(root, output)
    if output.exists():
        raise ValueError("Use a fresh immutable protocol filename; overwriting is forbidden")
    config_path = resolve_path(root, config_path)
    config = read_json(config_path)
    environment = runtime_environment()
    development_evaluation = resolve_path(root, development_evaluation)
    development_result = read_json(development_evaluation)
    expected_base = development_result.get("base_parameter_sha256_before")
    if (development_result.get("purpose") != "validation_development" or development_result.get("status") != "completed"
            or development_result.get("base_weights_unchanged") is not True
            or not re.fullmatch(r"[0-9a-f]{64}", expected_base or "")
            or development_result.get("base_parameter_sha256_after") != expected_base
            or development_result.get("inference_config") != config
            or development_result.get("scorer_version") != SCORER_VERSION):
        raise ValueError("Final configuration requires completed, matching validation evidence and an unchanged frozen-base fingerprint")
    original_manifest = verify_original(original, root, expected_manifest_sha256=original_manifest_sha256)
    if output.parent == resolve_path(root, original) or resolve_path(root, original) in output.parents:
        raise ValueError("The final protocol must remain separate from original dataset files")
    models = {task: model_record(root, config[task + "_artifact"], task) for task in ("scope", "answer")}
    for field in ("base_revision", "embedding_revision"):
        if not re.fullmatch(r"[0-9a-f]{40}", config.get(field, "")):
            raise ValueError("Frozen base and embedding model revisions must be full pinned commit IDs")
    if (config.get("input_format_version") != "editguard-input-v2" or config.get("max_new_tokens") != 48
            or config.get("local_files_only") is not True):
        raise ValueError("Expected locked input-v2, offline models and matched 48-token answer budgets")
    calibrations = {"scope": record_file(root, config["calibration_artifact"]),
                    "similarity": record_file(root, config["similarity_calibration_artifact"])}
    for task, record in calibrations.items():
        calibration = read_json(root / record["path"])
        if calibration.get("status") != "frozen" or calibration.get("split") != "validation":
            raise ValueError("Operating thresholds must already be frozen from validation")
        if task == "scope" and calibration.get("scope_manifest_sha256") != models["scope"]["manifest_sha256"]:
            raise ValueError("Scope calibration does not bind the selected scope artifact")
    dev_dir = resolve_path(root, development)
    dev_manifest = read_json(dev_dir / "manifest.json")
    selection = read_json(dev_dir / "selection.json")
    development_families = selection["selected_families"]
    if (set(development_families) != {"train", "validation"}
            or set(development_families["train"]) & set(development_families["validation"])
            or dev_manifest.get("test_files_included") is not False):
        raise ValueError("Development references must contain disjoint train/validation families only")
    development_names = ("manifest.json", "selection.json", "split_assignments.json", "catalogue.json",
                         "scope_train.jsonl", "scope_validation.jsonl", "answer_train.jsonl", "answer_validation.jsonl")
    development_sources = {relative_path(root, dev_dir / name): file_hash(dev_dir / name)
                           for name in development_names}
    for name in development_names[1:]:
        if dev_manifest["files"].get(name) != development_sources[relative_path(root, dev_dir / name)]:
            raise ValueError("Development evidence checksum differs: " + name)
    if models["scope"]["dataset_manifest_sha256"] != file_hash(dev_dir / "manifest.json"):
        raise ValueError("Development scope references must bind the selected model's training dataset")
    source_hashes = {name: file_hash(root / name) for name in SOURCE_NAMES}
    # Bind calibration score/grid evidence too, without interpreting held-out data.
    calibration_sources = {relative_path(root, item): file_hash(item)
        for record in calibrations.values() for item in sorted((root / record["path"]).parent.iterdir())
        if item.is_file() and item.suffix in (".json", ".jsonl", ".csv")}
    counts = {"scope": original_manifest["counts"]["scope"]["test"],
              "answer": original_manifest["counts"]["answer"]["test"],
              "challenge": original_manifest["challenge_drafts"]}
    execution = {"output_directory": relative_path(root, execution_output),
                 "ledger_path": relative_path(root, ledger_path)}
    if any((root / path).exists() for path in execution.values()):
        raise ValueError("Final output and single-attempt ledger must both be fresh before protocol freezing")
    protocol = {"schema_version": 1, "purpose": "frozen_final_evaluation_protocol", "status": "frozen",
        "frozen_at": datetime.now(timezone.utc).isoformat(), "configuration": record_file(root, config_path),
        "models": models, "calibrations": calibrations, "source_sha256": source_hashes,
        "calibration_source_sha256": calibration_sources, "development_source_sha256": development_sources,
        "development_evaluation": record_file(root, development_evaluation),
        **environment,
        "original": {"directory": relative_path(root, original), "dataset_id": original_manifest["dataset_id"],
                     "manifest_sha256": original_manifest_sha256, "files_sha256": original_manifest["files"]},
        "development_families": development_families, "development_directory": relative_path(root, dev_dir),
        "preprocessing_disclosure": {"scope_input_policy": models["scope"]["scope_input_policy"],
            "development_overlap": models["scope"]["scope_input_overlap"],
            "policy": "Raw entity/family/lineage separation remains required. Shared logical forms and same-label transformed inputs are disclosed; contradictory transformed pairs are forbidden. Test transformed overlap is reported after packet preparation, without model tuning."},
        "review_policy": "automated_final", "human_reviewed_data": False,
        "review_authorization": "User explicitly authorized a fully disclosed automated final audit; no human review or independent human challenge authorship is claimed.",
        "scoring": {"primary_version": SCORER_VERSION, "primary_sha256": source_hashes["editguard/scoring.py"],
                    "short_qa_version": SHORT_QA_VERSION, "short_qa_sha256": source_hashes["editguard/short_qa.py"],
                    "short_qa_role": "separate request-bound controlled-form metric; bare answers do not prove complete fact endorsement",
                    "abstentions": "uncredited in full denominator on answerable rows",
                    "pending_semantic_review": "uncredited in full denominator; no model judge or invented human decisions",
                    "structured_answers": "never count as learned model generation evidence"},
        "evaluation": {"configurations": CONFIGURATIONS, "question_counts": counts,
            "matched_populations": ["scope", "challenge"], "oracle_populations": ["answer", "challenge"],
            "matched_questions": counts["scope"] + counts["challenge"],
            "oracle_questions": counts["answer"] + counts["challenge"], "max_final_runs": 1,
            "order": "population_then_row_id_ascending", "answer_budget_tokens": 48,
            "generation": {"do_sample": False, "num_beams": 1},
            "expected_base_parameter_fingerprint_sha256": expected_base,
            "memory_policy": "single_focal_edit_plus_at_most_8_foreign_product_distractors",
            "bootstrap": {"paired": True, "cluster_unit": "entity_family", "seed": 42, "resamples": 2000, "confidence": .95},
            "latency": {"warmup_requests": 0, "policy": "complete per-request retrieval/scope/generation; first-use latency retained and labeled; load time recorded separately"},
            "policy": "One frozen final evaluation; no threshold/model/scorer changes or test-guided tuning after question access."},
        "execution": execution,
        "original_question_content_deserialized_during_freeze": False,
        "limitations": ["Controlled fictional catalogue and finite question forms do not establish unrestricted-language reliability.",
                        "Automated final consistency review is not independent human semantic review.",
                        "Original challenge records remain agent-authored drafts; machine audit does not make them human-authored.",
                        "Entity-neutral classifier adaptation can create same-label abstracted overlap; report this separately from raw entity separation."]}
    protocol["protocol_id"] = "editguard-final-" + canonical_hash(protocol)[:16]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(protocol, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    validate_protocol(output, file_hash(output), root=root)
    return protocol


def validate_protocol(path, expected_sha256, *, root=ROOT, verify_original_files=True):
    root = root.resolve()
    path = resolve_path(root, path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256 or "") or file_hash(path) != expected_sha256:
        raise ValueError("Frozen protocol checksum differs or its external checksum is missing")
    protocol = read_json(path)
    identity = protocol.get("protocol_id")
    if identity != "editguard-final-" + canonical_hash({key: value for key, value in protocol.items() if key != "protocol_id"})[:16]:
        raise ValueError("Frozen protocol identity differs")
    if (protocol.get("schema_version") != 1 or protocol.get("status") != "frozen"
            or protocol.get("purpose") != "frozen_final_evaluation_protocol"
            or protocol.get("review_policy") != "automated_final" or protocol.get("human_reviewed_data") is not False
            or protocol["evaluation"]["max_final_runs"] != 1 or protocol["evaluation"]["configurations"] != CONFIGURATIONS):
        raise ValueError("Unsupported or nonfrozen final evaluation protocol")
    environment = runtime_environment()
    if any(protocol.get(name) != value for name, value in environment.items()):
        raise ValueError("Frozen Python or neural distribution versions differ; resume requires the same runtime")
    if set(protocol["source_sha256"]) != set(SOURCE_NAMES):
        raise ValueError("Frozen repository source inventory differs")
    for group in (protocol["source_sha256"], protocol["calibration_source_sha256"], protocol["development_source_sha256"]):
        for name, digest in group.items():
            if file_hash(resolve_path(root, name)) != digest:
                raise ValueError("Frozen source/evidence checksum differs: " + name)
    for record in (protocol["configuration"], protocol["development_evaluation"], *protocol["calibrations"].values()):
        if file_hash(resolve_path(root, record["path"])) != record["sha256"]:
            raise ValueError("Frozen configuration/calibration checksum differs: " + record["path"])
    for task, record in protocol["models"].items():
        if model_record(root, record["artifact"], task) != record:
            raise ValueError("Frozen auxiliary model differs: " + task)
    if verify_original_files:
        original = protocol["original"]
        verify_original(original["directory"], root, expected_manifest_sha256=original["manifest_sha256"],
                        expected_files=original["files_sha256"])
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=ROOT / "data/generated/controlled-v1")
    parser.add_argument("--development", type=Path, default=ROOT / "data/generated/pilot-v3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--development-evaluation", type=Path, default=ROOT / DEFAULT_DEVELOPMENT_EVALUATION)
    args = parser.parse_args()
    protocol = freeze_protocol(args.config, args.original, args.development, args.output, args.execution_output, args.ledger,
                               development_evaluation=args.development_evaluation)
    print(json.dumps({"status": "frozen", "protocol_id": protocol["protocol_id"], "protocol_sha256": file_hash(args.output),
                      "question_counts": protocol["evaluation"]["question_counts"], "question_content_deserialized": False}, indent=2))


if __name__ == "__main__":
    main()

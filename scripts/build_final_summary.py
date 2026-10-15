"""Export completed, hash-bound final evidence; no inference or training.

The external test-lock checksum anchors the protocol and packet frozen before
inference. Primary complete-assertion scores and the separately locked short-QA
diagnostic retain their original cohorts and denominators.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.comparison import CONFIGURATIONS
from editguard.metrics import aggregate_predictions
from editguard.scoring import SCORER_VERSION, score_answer
from editguard.short_qa import SHORT_QA_VERSION, score_short_qa
from scripts.analyze_short_qa import summarize as short_summary
from scripts.final_protocol import SOURCE_NAMES, canonical_hash

COUNTS = {"scope": 720, "answer": 288, "challenge": 96}
OUTPUT_NAMES = ("predictions.jsonl", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
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
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))


def read_records(path):
    return [json.loads(line, object_pairs_hook=unique_keys,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))
            for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def within(root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Evidence path escapes the declared workspace")
    return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_file(root, record, watched):
    path = within(root, record["path"])
    require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None and file_hash(path) == record["sha256"],
            "Evidence checksum differs: " + str(record["path"]))
    watched[path] = record["sha256"]
    return path


def rate(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else None}


def check_rate(value, denominator=None):
    require(isinstance(value, dict) and type(value.get("denominator")) is int and value["denominator"] >= 0,
            "Every rate requires an explicit nonnegative denominator")
    require(type(value.get("numerator")) in (int, float) and math.isfinite(value["numerator"])
            and 0 <= value["numerator"] <= value["denominator"], "Rate numerator is invalid")
    if denominator is not None:
        require(value["denominator"] == denominator, "A rate discarded rows from its frozen denominator")
    expected = value["numerator"] / value["denominator"] if value["denominator"] else None
    # Missing-calibration and configurations without a routing decision can be
    # explicitly undefined, with a reason, even when their denominator is nonzero.
    require(value.get("value") == expected or value.get("value") is None and bool(value.get("reason")),
            "Rate value differs from its numerator and denominator")


def check_secondary_counts(value, denominator):
    require(type(value.get("correct")) is int and type(value.get("incorrect")) is int
            and type(value.get("pending_uncredited")) is int
            and value.get("denominator") == denominator
            and value["correct"] + value["incorrect"] + value["pending_uncredited"] == denominator,
            "Secondary counts lost pending or nonanswer outcomes")
    check_rate({"numerator": value["correct"], "denominator": value["denominator"],
                "value": value["accuracy_including_all_nonanswers_and_pending"]}, denominator)


def primary_row(name, current, count, applicable):
    require(current["count"] == count and current["answerable_count"] == count
            and current["counts"]["total"] == count and current["counts"]["input_records"] == count
            and current["counts"]["missing_records"] == 0, "Primary metrics lost matched observations")
    for key in ("accuracy", "coverage", "model_generated_answer_accuracy", "model_generated_answer_coverage"):
        check_rate(current[key], count)
    out_of_scope = count - applicable
    check_rate(current["subsets"]["out_of_scope"]["accuracy"], out_of_scope)
    check_rate(current["false_correction_application_rate"], out_of_scope)
    check_rate(current["baseline_correct_retention"])
    require(current["baseline_correct_retention"]["denominator"] <= out_of_scope,
            "Retention includes questions outside its baseline-correct population")
    check_rate(current["output_preservation"], out_of_scope)
    for key in ("pending_review", "abstentions", "errors", "unavailable"):
        require(type(current[key]) is int and 0 <= current[key] <= count, "Missing-outcome counts must remain explicit")
    latency = current["latency"]
    require(latency["total_count"] == count and latency["sample_count"] + latency["missing_count"] == count,
            "Latency population differs from the full matched population")
    if latency["complete"]:
        require(latency["sample_count"] == count and all(type(latency[key]) in (float, int)
                and math.isfinite(latency[key]) and latency[key] >= 0 for key in ("p50_ms", "p95_ms"))
                and latency["p50_ms"] <= latency["p95_ms"], "Complete latency quantiles are invalid")
    else:
        require(latency["p50_ms"] is None and latency["p95_ms"] is None, "Incomplete latency cannot claim full-population quantiles")
    return {"configuration": name, "questions": count, "applicable_questions": applicable,
        **{key: current[key] for key in ("accuracy", "coverage", "model_generated_answer_accuracy", "model_generated_answer_coverage",
            "baseline_correct_retention", "output_preservation", "retrieval_recall", "pending_review", "abstentions", "errors", "unavailable", "status_counts")},
        "out_of_scope_accuracy": current["subsets"]["out_of_scope"]["accuracy"],
        "false_application_rate": current["false_correction_application_rate"],
        "p50_ms": latency["p50_ms"], "p95_ms": latency["p95_ms"], "latency": latency,
        "pending_rate": rate(current["pending_review"], count), "abstention_rate": rate(current["abstentions"], count),
        "error_rate": rate(current["errors"], count), "unavailable_rate": rate(current["unavailable"], count),
        "by_question_type": {key: item for key, item in current["subsets"].items()
                             if key not in ("out_of_scope", "direct_corrected")}}


def population_summary(metrics, count, applicable):
    require(set(metrics["configurations"]) == set(CONFIGURATIONS) and metrics["question_count"] == count
            and metrics["record_count"] == count * len(CONFIGURATIONS), "All five configurations must retain their frozen cohort")
    return {"questions": count, "applicable_questions": applicable,
            "configurations": [primary_row(name, metrics["configurations"][name], count, applicable) for name in CONFIGURATIONS],
            "scope": metrics["configurations"]["trained_scope_editor"]["scope"],
            "paired_comparisons": metrics["paired_comparisons"]}


def oracle_summary(metrics, count):
    require(set(metrics["configurations"]) == {"oracle_answer"} and metrics["question_count"] == count
            and metrics["record_count"] == count, "Oracle diagnostics must keep their separate frozen cohort")
    current = metrics["configurations"]["oracle_answer"]
    result = primary_row("oracle_answer", current, count, count)
    for style in ("direct", "paraphrase", "comparison", "challenge"):
        subset = current["subsets"].get(style)
        result[style + "_accuracy"] = subset["accuracy"] if subset else rate(0, 0)
        check_rate(result[style + "_accuracy"])
    require(sum(result[style + "_accuracy"]["denominator"] for style in ("direct", "paraphrase", "comparison", "challenge")) == count,
            "Oracle question-type cohorts do not cover its separate denominator")
    return result


def check_records(records, populations, names, *, kind, accepted, protocol_sha):
    by_population = {}
    for population, expected_count in populations.items():
        cohort = [row for row in records if row.get("population") == population]
        require(len(cohort) == expected_count * len(names), "A frozen cohort is incomplete")
        indexed = {}
        for row in cohort:
            require(row.get("configuration") in names and row.get("evaluation_kind") == kind
                    and row.get("review_policy") == "automated_final" and row.get("human_reviewed_data") is False
                    and row.get("final_protocol_sha256") == protocol_sha and type(row.get("applicable")) is bool
                    and row.get("row_sha256") == accepted.get(row.get("id")), "An observation differs from its final contract")
            require(row.get("answer_kind") == "model_generated" or row.get("answer_kind") == "deterministic_clarification"
                    and row.get("status") in ("clarification", "conflict"), "Structured fact answers cannot count as final ML observations")
            expected = row["expected_fact"]
            text = row.get("answer") or ""
            if row.get("status") not in ("answered", "ok", "success", "generated") or row["answer_kind"] != "model_generated":
                text = ""
            scored = score_answer(text, expected, aliases=expected.get("aliases", []),
                                  expected_polarity=row.get("expected_polarity"), answer_kind=row["answer_kind"])
            require(row.get("score") == scored and row.get("short_qa") == score_short_qa(row), "An observation score differs from its frozen scorer")
            identity = (row["id"], row["configuration"])
            require(identity not in indexed, "Duplicate final question/configuration observation")
            indexed[identity] = row
        ids = {identity for identity, _ in indexed}
        require(len(ids) == expected_count and set(indexed) == {(identity, name) for identity in ids for name in names},
                "Configurations have incompatible matched question identities")
        applicable = set()
        for identity in ids:
            selected = [indexed[(identity, name)] for name in names]
            truth = {key: selected[0].get(key) for key in ("family_id", "question", "request_context", "question_type", "expected_fact", "expected_polarity", "applicable")}
            require(all({key: row.get(key) for key in truth} == truth for row in selected), "Matched ground truth differs across configurations")
            if truth["applicable"]:
                applicable.add(identity)
        if kind == "oracle" or population == "challenge":
            require(len(applicable) == expected_count, "Oracle/challenge populations must contain applicable edits")
        by_population[population] = {"questions": expected_count, "applicable_questions": len(applicable), "ids": ids}
    require(len(records) == sum(populations.values()) * len(names), "Unexpected records inflate the frozen population")
    return by_population


def io_recovery_summary(root, declaration, declaration_sha256, events, *, protocol_path,
                        protocol, packet, run, ledger_path, watched):
    """Bind a completed run's optional IO log without inferring session success.

    The declaration is externally anchored before recovery. The wrapper emits
    one event line in its finally block, including failed sessions; the log does
    not record their outcomes. Its final bytes are frozen only at summary export.
    """
    options = (declaration, declaration_sha256, events)
    if all(value is None for value in options):
        return None
    require(all(value is not None for value in options),
            "IO recovery declaration, external checksum and events must be supplied together")
    declaration_path = bound_file(root, {"path": str(declaration), "sha256": declaration_sha256}, watched)
    spec = read_json(declaration_path)
    expected_targets = [run / "progress.json", ledger_path]
    declared_targets = [str(path.relative_to(root)).replace("\\", "/") for path in expected_targets]
    delays = [.05, .1, .2, .4, .4, .4, .4]
    require(spec.get("schema_version") == 1
            and spec.get("purpose") == "frozen_final_io_recovery_declaration"
            and spec.get("fresh_run_allowed") is False
            and spec.get("inference_scoring_or_data_changes") is False
            and spec.get("observations_rewritten") is False
            and within(root, spec["protocol_path"]) == protocol_path
            and spec.get("protocol_sha256") == file_hash(protocol_path)
            and within(root, spec["packet_path"]) == packet
            and spec.get("frozen_evaluator_sha256") == protocol["source_sha256"]["scripts/run_final_evaluation.py"]
            and spec.get("mutable_write_targets") == declared_targets
            and type(spec.get("max_write_attempts")) is int and spec["max_write_attempts"] == 8
            and spec.get("retry_delays_seconds") == delays
            and type(spec.get("maximum_retry_delay_seconds")) in (int, float)
            and math.isfinite(spec["maximum_retry_delay_seconds"])
            and math.isclose(spec["maximum_retry_delay_seconds"], math.fsum(delays), rel_tol=0, abs_tol=1e-12)
            and isinstance(spec.get("reason"), str) and bool(spec["reason"])
            and isinstance(spec.get("performance_scope"), str) and bool(spec["performance_scope"]),
            "IO recovery declaration differs from the frozen run or permitted write-only policy")
    wrapper_path = bound_file(root, {"path": spec["wrapper_path"], "sha256": spec["wrapper_sha256"]}, watched)
    events_path = within(root, events)
    require(events_path not in watched, "IO events must be separate from other immutable evidence")
    events_sha = file_hash(events_path)
    watched[events_path] = events_sha
    entries = read_records(events_path)
    require(bool(entries), "A completed IO recovery log must include at least one wrapper-session record")
    target_names = {path.name for path in expected_targets}
    require(len(target_names) == len(expected_targets), "IO targets must have distinct logged filenames")
    counts = {name: 0 for name in sorted(target_names)}
    retry_delays = []
    for entry in entries:
        require(isinstance(entry, dict) and entry.get("declaration_sha256") == declaration_sha256
                and entry.get("frozen_evaluator_sha256") == spec["frozen_evaluator_sha256"]
                and entry.get("inference_scoring_or_data_changes") is False
                and isinstance(entry.get("write_retries"), list),
                "IO event differs from its declared wrapper and frozen evaluator")
        previous = None
        for retry in entry["write_retries"]:
            require(isinstance(retry, dict) and retry.get("target") in target_names
                    and type(retry.get("failed_attempt")) is int and 1 <= retry["failed_attempt"] < spec["max_write_attempts"]
                    and type(retry.get("delay_seconds")) in (int, float)
                    and retry["delay_seconds"] == delays[retry["failed_attempt"] - 1]
                    and retry.get("error_type") == "PermissionError",
                    "IO retry is outside the declared targets, error or finite delay schedule")
            require(retry["failed_attempt"] == 1 or previous is not None
                    and previous["target"] == retry["target"]
                    and previous["failed_attempt"] + 1 == retry["failed_attempt"],
                    "IO retry attempt sequence is incomplete or inconsistent")
            counts[retry["target"]] += 1
            retry_delays.append(retry["delay_seconds"])
            previous = retry
    return {"declaration_path": str(declaration_path), "declaration_sha256": declaration_sha256,
        "wrapper_path": str(wrapper_path), "wrapper_sha256": spec["wrapper_sha256"],
        "frozen_evaluator_sha256": spec["frozen_evaluator_sha256"],
        "protocol_sha256": spec["protocol_sha256"], "events_path": str(events_path), "events_sha256": events_sha,
        "event_records": len(entries), "recorded_write_retries": sum(counts.values()),
        "retry_counts_by_target": counts, "recorded_retry_delay_seconds": math.fsum(retry_delays),
        "max_write_attempts": spec["max_write_attempts"], "retry_delays_seconds": delays,
        "maximum_retry_delay_seconds_per_write": spec["maximum_retry_delay_seconds"],
        "mutable_write_targets": spec["mutable_write_targets"], "fresh_run_allowed": False,
        "inference_scoring_or_data_changes": False, "observations_rewritten": False,
        "declared_reason": spec["reason"], "cause_status": "not_established",
        "cause_disclosure": "Windows access-denied atomic-write collisions were observed. The lock owner and underlying cause were not established; OneDrive involvement was not verified.",
        "performance_scope": spec["performance_scope"],
        "events_frozen_at_summary_export": True,
        "limitations": ["Retry counts cover recorded write retries, not successful writes, unretried failures or every process session. Each event line can describe a successful or failed wrapper invocation; the log does not record its outcome.",
            "Recorded retry delays are configured sleep durations, not measured elapsed time. IO recovery occurs outside reported model-route request latency.",
            "Final event-file bytes are bound only after the result and single-run ledger are completed; scores, cohorts and model inputs remain unchanged."]}


def runtime_failure_summary(root, path, checksum, *, protocol_sha, records, oracle_records, watched):
    if path is None and checksum is None:
        return None
    require(path is not None and checksum is not None, "Runtime disclosure and external checksum must be supplied together")
    path = bound_file(root, {"path": str(path), "sha256": checksum}, watched)
    value = read_json(path)
    counts = {}
    for kind, rows in (("matched", records), ("oracle", oracle_records)):
        counts[kind] = {status: sum(row["status"] == status for row in rows)
                        for status in sorted({row["status"] for row in rows})}
    # The frozen plan commits every matched row before the oracle cohort; the
    # export deliberately contains prediction fields rather than journal indices.
    errors = [len(records) + index for index, row in enumerate(oracle_records) if row["status"] == "error"]
    require(value.get("schema_version") == 1 and value.get("purpose") == "final_runtime_failure_disclosure"
            and value.get("protocol_sha256") == protocol_sha
            and value.get("observations_committed") == len(records) + len(oracle_records)
            and value.get("status_by_kind") == counts
            and value.get("first_error_task_index") == min(errors, default=None)
            and value.get("last_error_task_index") == max(errors, default=None)
            and value.get("original_gpu_instance_post_inference_fingerprint") == "unavailable_cuda_unknown_error"
            and value.get("persistent_model_or_data_changes") is False
            and isinstance(value.get("closure_recovery_policy"), str)
            and "do not reconstruct" in value["closure_recovery_policy"]
            and isinstance(value.get("oracle_measurement_policy"), str),
            "Runtime disclosure differs from the preserved original observations or GPU-state limitation")
    return {**value, "disclosure_path": str(path), "disclosure_sha256": checksum,
            "original_results_replaced": False,
            "limitation": "The original GPU process lost its CUDA context after the matched comparison. Its end-of-run parameter fingerprint was unavailable. Fresh closure/source verification does not establish the lost GPU memory state; committed oracle runtime errors remain in all original denominators."}


def build(run, test_lock, test_lock_sha256, output, *, root=ROOT, csv_output=None, config_path=None,
          io_recovery_declaration=None, io_recovery_declaration_sha256=None, io_recovery_events=None,
          runtime_failure_disclosure=None, runtime_failure_disclosure_sha256=None):
    root = Path(root).resolve()
    run, test_lock, output = (within(root, value) for value in (run, test_lock, output))
    if csv_output is not None:
        csv_output = within(root, csv_output)
    require(not output.exists() and (csv_output is None or not csv_output.exists()), "Use fresh export paths; final evidence is never overwritten")
    watched = {}
    bound_file(root, {"path": str(test_lock), "sha256": test_lock_sha256}, watched)
    lock = read_json(test_lock)
    require(lock.get("purpose") == "final_packet_lock_before_inference" and lock.get("status") == "frozen"
            and lock.get("review_policy") == "automated_final" and lock.get("human_reviewed_data") is False
            and lock.get("inference_started") is False and lock.get("counts") == COUNTS
            and lock.get("review_counts") == {"accepted": sum(COUNTS.values()), "flagged": 0}, "An intact pre-inference automated final test lock is required")
    protocol_path = bound_file(root, {"path": lock["protocol_path"], "sha256": lock["protocol_sha256"]}, watched)
    protocol = read_json(protocol_path)
    bootstrap = protocol.get("evaluation", {}).get("bootstrap", {})
    require(protocol.get("purpose") == "frozen_final_evaluation_protocol" and protocol.get("status") == "frozen"
            and protocol.get("review_policy") == "automated_final" and protocol.get("human_reviewed_data") is False
            and protocol.get("protocol_id") == "editguard-final-" + canonical_hash({key: value for key, value in protocol.items() if key != "protocol_id"})[:16]
            and protocol["evaluation"]["question_counts"] == COUNTS
            and protocol["evaluation"]["configurations"] == list(CONFIGURATIONS)
            and protocol["evaluation"]["max_final_runs"] == 1
            and protocol["evaluation"].get("matched_populations") == ["scope", "challenge"]
            and protocol["evaluation"].get("oracle_populations") == ["answer", "challenge"]
            and bootstrap == {"paired": True, "cluster_unit": "entity_family", "seed": 42, "resamples": 2000, "confidence": .95},
            "Frozen protocol identity, populations, configurations or uncertainty settings differ")
    require(set(protocol["source_sha256"]) == set(SOURCE_NAMES), "Frozen inference/scoring source inventory differs")
    for name, digest in protocol["source_sha256"].items():
        bound_file(root, {"path": name, "sha256": digest}, watched)
    scoring = protocol["scoring"]
    require(scoring["primary_version"] == SCORER_VERSION and scoring["short_qa_version"] == SHORT_QA_VERSION
            and scoring["primary_sha256"] == protocol["source_sha256"]["editguard/scoring.py"]
            and scoring["short_qa_sha256"] == protocol["source_sha256"]["editguard/short_qa.py"], "Frozen scorers differ from the evidence exporters")
    config_file = bound_file(root, protocol["configuration"], watched)
    if config_path is not None:
        require(within(root, config_path) == config_file, "Requested configuration differs from the frozen model comparison")
    config = read_json(config_file)
    calibrations = {}
    for task, field in (("scope", "calibration_artifact"), ("similarity", "similarity_calibration_artifact")):
        record = protocol["calibrations"][task]
        path = bound_file(root, record, watched)
        require(within(root, config[field]) == path, "Configured calibration differs from the frozen operating point")
        card = read_json(path)
        require(card.get("status") == "frozen" and card.get("split") == "validation",
                "Operating points must remain frozen validation selections")
        if task == "scope":
            require(card.get("scope_manifest_sha256") == protocol["models"]["scope"]["manifest_sha256"],
                    "Scope threshold belongs to a different trained classifier")
        calibrations[task] = {"path": str(path), "sha256": record["sha256"], "card": card}
    for name, digest in protocol.get("calibration_source_sha256", {}).items():
        bound_file(root, {"path": name, "sha256": digest}, watched)
    artifacts = {}
    for task in ("scope", "answer"):
        expected = protocol["models"][task]
        artifact = within(root, expected["artifact"])
        require(within(root, config[task + "_artifact"]) == artifact, "Configured auxiliary model differs from frozen protocol")
        path = bound_file(root, {"path": str(artifact / "manifest.json"), "sha256": expected["manifest_sha256"]}, watched)
        model = read_json(path)
        require(model.get("task") == task and model.get("purpose") == "trained" and model.get("status") == "completed"
                and model.get("input_format_version") == "editguard-input-v2" and model.get("human_reviewed_data") is False
                and bool(model.get("files_sha256")) and model.get("files_sha256") == expected["files_sha256"]
                and model.get("selected_model_state_sha256") == expected["selected_model_state_sha256"], "Auxiliary export is incomplete or differs from selected trained weights")
        for name, digest in model["files_sha256"].items():
            path = within(root, artifact / name)
            require(artifact in path.parents, "Auxiliary file path escapes its export")
            bound_file(root, {"path": str(path), "sha256": digest}, watched)
        require(type(model.get("completed_steps")) is int and model["completed_steps"] > 0
                and type(model.get("selection", {}).get("global_step")) is int
                and 1 <= model["selection"]["global_step"] <= model["completed_steps"], "Trained model steps must be real and selected")
        artifacts[task] = {"artifact_path": str(artifact), "manifest_sha256": expected["manifest_sha256"],
            "selected_model_state_sha256": model["selected_model_state_sha256"], "completed_steps": model["completed_steps"],
            "selected_step": model["selection"]["global_step"], "model_id": model["model_id"], "model_revision": model["model_revision"],
            "training_review_policy": model["review_policy"], "human_reviewed_data": False,
            "dataset_id": model.get("dataset_id"), "dataset_manifest_sha256": model.get("manifest_sha256"),
            "training_reviews_sha256": model.get("reviews_sha256"), "selection": model["selection"],
            "scope_input_policy": model.get("scope_input_policy"), "answer_loss_policy": model.get("answer_loss_policy")}
    packet = within(root, lock["packet_path"])
    require(all(packet not in path.parents and run not in path.parents for path in (output, csv_output) if path is not None),
            "Export paths must remain separate from immutable packet and raw run directories")
    require(csv_output != output, "CSV and JSON exports require distinct fresh paths")
    packet_manifest_path = bound_file(root, {"path": str(packet / "manifest.json"), "sha256": lock["packet_manifest_sha256"]}, watched)
    packet_manifest = read_json(packet_manifest_path)
    require(packet_manifest.get("purpose") == "frozen_final_evaluation_packet" and packet_manifest.get("protocol_sha256") == lock["protocol_sha256"]
            and packet_manifest.get("counts") == COUNTS and packet_manifest.get("files") == lock["packet_files_sha256"]
            and packet_manifest.get("review_policy") == "automated_final" and packet_manifest.get("human_reviewed_data") is False
            and packet_manifest.get("test_frozen") is True and packet_manifest.get("eligible_for_final_evaluation") is True
            and packet_manifest.get("all_rows_preserved") is True and packet_manifest.get("review_counts") == lock["review_counts"], "Packet differs from its pre-inference freeze and audit")
    for name, digest in packet_manifest["files"].items():
        require(Path(name).name == name, "Packet inventory contains an escaping filename")
        bound_file(root, {"path": str(packet / name), "sha256": digest}, watched)
    manifest_path = run / "manifest.json"
    manifest = read_json(manifest_path)
    manifest_sha = file_hash(manifest_path)
    watched[manifest_path] = manifest_sha
    expected_base = protocol["evaluation"]["expected_base_parameter_fingerprint_sha256"]
    require(manifest.get("purpose") == "frozen_final_evaluation" and manifest.get("status") == "completed"
            and manifest.get("protocol_sha256") == lock["protocol_sha256"] and manifest.get("protocol_id") == protocol["protocol_id"]
            and manifest.get("packet_manifest_sha256") == lock["packet_manifest_sha256"]
            and manifest.get("configuration_sha256") == protocol["configuration"]["sha256"]
            and manifest.get("review_policy") == "automated_final" and manifest.get("human_reviewed_data") is False
            and manifest.get("test_rows_read") is True and manifest.get("challenge_rows_read") is True
            and manifest.get("reviewed_data") is True and manifest.get("dataset_id") == packet_manifest.get("dataset_id")
            and manifest.get("base_weights_unchanged") is True
            and manifest.get("base_parameter_sha256_before") == expected_base == manifest.get("base_parameter_sha256_after")
            and manifest.get("runtime_packages") == protocol["runtime_packages"] and manifest.get("python_version") == protocol["python_version"]
            and manifest.get("fresh_final_runs") == 1 and manifest.get("final_test_tuning_performed") is False
            and manifest.get("population_counts") == COUNTS and manifest.get("configurations") == list(CONFIGURATIONS)
            and manifest.get("scorer_version") == SCORER_VERSION and manifest.get("scorer_sha256") == scoring["primary_sha256"]
            and manifest.get("secondary_scorer_version") == SHORT_QA_VERSION
            and set(manifest.get("outputs_sha256", {})) == set(OUTPUT_NAMES), "Only completed, unchanged, frozen final evidence is eligible")
    require(within(root, protocol["execution"]["output_directory"]) == run, "Final run path differs from its protocol")
    ledger_path = within(root, protocol["execution"]["ledger_path"])
    ledger = read_json(ledger_path)
    watched[ledger_path] = file_hash(ledger_path)
    require(ledger.get("status") == "completed" and ledger.get("fresh_final_runs") == 1
            and ledger.get("result_manifest_sha256") == manifest_sha and ledger.get("output_directory") == str(run)
            and all(ledger.get("binding", {}).get(key) == manifest.get(key) for key in
                    ("protocol_sha256", "protocol_id", "packet_manifest_sha256", "configuration_sha256", "execution_plan_sha256")),
            "The single-run ledger does not seal this final result")
    io_recovery = io_recovery_summary(root, io_recovery_declaration, io_recovery_declaration_sha256,
        io_recovery_events, protocol_path=protocol_path, protocol=protocol, packet=packet, run=run,
        ledger_path=ledger_path, watched=watched)
    for name in OUTPUT_NAMES:
        bound_file(root, {"path": str(run / name), "sha256": manifest["outputs_sha256"][name]}, watched)
    metrics, secondary = read_json(run / "metrics.json"), read_json(run / "short_qa_metrics.json")
    require(metrics.get("purpose") == "frozen_final_evaluation" and metrics.get("review_policy") == "automated_final"
            and metrics.get("human_reviewed_data") is False and metrics.get("protocol_sha256") == lock["protocol_sha256"]
            and metrics.get("final_test_tuning_performed") is False, "Primary metrics are not frozen final evidence")
    require(secondary.get("purpose") == "locked_final_secondary_short_qa" and secondary.get("diagnostic_version") == SHORT_QA_VERSION
            and secondary.get("protocol_sha256") == lock["protocol_sha256"] and secondary.get("review_policy") == "automated_final"
            and secondary.get("human_reviewed_data") is False and secondary.get("primary_results_replaced") is False
            and secondary.get("post_hoc_policy") is False, "Short-QA must remain a separately locked secondary metric")
    records, oracle_records = read_records(run / "predictions.jsonl"), read_records(run / "oracle_predictions.jsonl")
    runtime_failure = runtime_failure_summary(root, runtime_failure_disclosure, runtime_failure_disclosure_sha256,
        protocol_sha=lock["protocol_sha256"], records=records, oracle_records=oracle_records, watched=watched)
    matched_counts = {name: COUNTS[name] for name in ("scope", "challenge")}
    oracle_counts = {name: COUNTS[name] for name in ("answer", "challenge")}
    matched = check_records(records, matched_counts, CONFIGURATIONS, kind="matched", accepted=manifest["accepted_row_sha256"], protocol_sha=lock["protocol_sha256"])
    oracle = check_records(oracle_records, oracle_counts, ("oracle_answer",), kind="oracle", accepted=manifest["accepted_row_sha256"], protocol_sha=lock["protocol_sha256"])
    require(not (matched["scope"]["ids"] & matched["challenge"]["ids"])
            and not (oracle["answer"]["ids"] & oracle["challenge"]["ids"])
            and not (matched["scope"]["ids"] & oracle["answer"]["ids"])
            and oracle["challenge"]["ids"] == matched["challenge"]["ids"]
            and set(manifest["accepted_row_sha256"]) == matched["scope"]["ids"] | matched["challenge"]["ids"] | oracle["answer"]["ids"],
            "Final question identities collide across distinct cohorts or the accepted row inventory differs")
    require(len(records) == manifest["matched_prediction_records"] == sum(matched_counts.values()) * len(CONFIGURATIONS)
            and len(oracle_records) == manifest["oracle_records"] == sum(oracle_counts.values())
            and manifest["questions"] == sum(matched_counts.values()), "Raw result counts differ from frozen denominators")
    bootstrap = protocol["evaluation"]["bootstrap"]
    kwargs = {"bootstrap_seed": bootstrap["seed"], "bootstrap_resamples": bootstrap["resamples"]}
    computed = aggregate_predictions(records, **kwargs)
    computed["by_population"] = {name: aggregate_predictions([row for row in records if row["population"] == name], **kwargs)
                                 for name in matched_counts}
    overlap = packet_manifest["scope_normalized_overlap"]
    for row in records:
        if row["population"] == "scope":
            require(row.get("normalized_scope_input_seen_in_train") is (row["id"] in overlap["scope_row_ids_sharing_train_pair"])
                    and row.get("normalized_scope_input_seen_in_validation") is (row["id"] in overlap["scope_row_ids_sharing_validation_pair"]),
                    "Normalized scope overlap differs from the frozen packet")
    unshared = [row for row in records if row["population"] == "scope" and not row["normalized_scope_input_seen_in_train"]
                and not row["normalized_scope_input_seen_in_validation"]]
    computed["scope_unshared_with_development_inputs"] = aggregate_predictions(unshared, **kwargs)
    computed["scope_normalized_overlap"] = overlap
    computed["oracle_answer_diagnostic"] = aggregate_predictions(oracle_records, **kwargs)
    computed["oracle_answer_by_population"] = {name: aggregate_predictions([row for row in oracle_records if row["population"] == name], **kwargs)
                                             for name in oracle_counts}
    computed.update(purpose="frozen_final_evaluation", review_policy="automated_final", human_reviewed_data=False,
                    protocol_sha256=lock["protocol_sha256"], final_test_tuning_performed=False)
    require(computed == metrics, "Saved primary metrics differ from recomputation of immutable observations")
    expected_secondary = {"configurations": short_summary(records),
        "by_population": {name: short_summary([row for row in records if row["population"] == name]) for name in matched_counts},
        "oracle": short_summary(oracle_records),
        "oracle_by_population": {name: short_summary([row for row in oracle_records if row["population"] == name]) for name in oracle_counts},
        "scope_unshared_with_development_inputs": short_summary(unshared)}
    require(all(secondary.get(key) == value for key, value in expected_secondary.items()),
            "Saved secondary metrics differ from recomputation of immutable observations")
    matched_total = population_summary(metrics, sum(matched_counts.values()), sum(item["applicable_questions"] for item in matched.values()))
    by_population = {name: population_summary(metrics["by_population"][name], item["questions"], item["applicable_questions"])
                     for name, item in matched.items()}
    oracle_total = oracle_summary(metrics["oracle_answer_diagnostic"], sum(oracle_counts.values()))
    oracle_total["by_population"] = {name: oracle_summary(metrics["oracle_answer_by_population"][name], item["questions"])
                                     for name, item in oracle.items()}
    for name, count in (("configurations", sum(matched_counts.values())),):
        require(set(secondary[name]) == set(CONFIGURATIONS), "Secondary configurations differ")
        for current in secondary[name].values():
            check_secondary_counts(current["all_requested_qa"], count)
    for population, count in matched_counts.items():
        require(set(secondary["by_population"][population]) == set(CONFIGURATIONS), "Secondary cohort configurations differ")
        for current in secondary["by_population"][population].values():
            check_secondary_counts(current["all_requested_qa"], count)
    check_secondary_counts(secondary["oracle"]["oracle_answer"]["all_requested_qa"], sum(oracle_counts.values()))
    for population, count in oracle_counts.items():
        check_secondary_counts(secondary["oracle_by_population"][population]["oracle_answer"]["all_requested_qa"], count)
    for group in (metrics, *metrics["by_population"].values()):
        for pair in group["paired_comparisons"]:
            require(all(pair.get(key) == bootstrap[key] for key in ("seed", "resamples"))
                    and pair.get("confidence_level") == bootstrap["confidence"], "Paired uncertainty differs from frozen settings")
    scope = matched_total["scope"]
    scope_summary = {key: scope[key] for key in ("precision", "recall", "false_positive_rate", "brier", "ece")}
    scope_card = calibrations["scope"]["card"]
    scope_policy = {key: scope_card[key] for key in ("temperature", "threshold", "ambiguity_margin", "routing_policy", "population",
        "raw_pair_metrics", "compatibility_gated_metrics", "scope_manifest_sha256") if key in scope_card}
    scope_policy.update(calibration_path=calibrations["scope"]["path"], calibration_sha256=calibrations["scope"]["sha256"],
        calibration_dataset_id=scope_card.get("dataset_id"), selected_on_split="validation",
        metrics_role="Development selection evidence; final held-out raw pair measurements are separately reported.")
    summary = {"schema_version": 1, "artifact_type": "editguard_final_evaluation", "purpose": "frozen_final_evaluation",
        "status": "completed", "review_policy": "automated_final", "human_reviewed_data": False,
        "test_rows_read": True, "challenge_rows_read": True, "test_frozen": True, "final_test_tuning_performed": False,
        "dataset_id": manifest["dataset_id"], "questions": matched_total["questions"],
        "matched_applicable_questions": matched_total["applicable_questions"], "oracle_questions": oracle_total["questions"],
        "population_counts": COUNTS, "model_artifacts": artifacts, "scope": scope_summary,
        "scope_full_metrics": scope, "configurations": matched_total["configurations"], "by_population": by_population,
        "scope_policy": scope_policy,
        "scope_metric_role": "Independent raw labeled question-correction pairs; metadata-gated routing and generated-answer accuracy are different measurements.",
        "paired_comparisons": matched_total["paired_comparisons"], "oracle": oracle_total,
        "scope_unshared_with_development_inputs": metrics["scope_unshared_with_development_inputs"],
        "scope_normalized_overlap": metrics["scope_normalized_overlap"], "secondary_short_qa": secondary,
        "primary_scoring": {"scorer_version": SCORER_VERSION, "definition": "Complete scoped fact endorsement; pending prose, nonanswers and errors remain uncredited in full denominators."},
        "runtime_packages": manifest["runtime_packages"], "python_version": manifest["python_version"],
        "sessions": manifest["sessions"], "resumed": manifest["resumed"],
        "provenance": {"config_sha256": protocol["configuration"]["sha256"], "config_path": str(config_file),
            "scorer_version": SCORER_VERSION, "scorer_sha256": scoring["primary_sha256"],
            "secondary_scorer_version": SHORT_QA_VERSION, "secondary_scorer_sha256": scoring["short_qa_sha256"],
            "protocol_path": str(protocol_path), "protocol_sha256": lock["protocol_sha256"], "protocol_id": protocol["protocol_id"],
            "calibrations": {name: {key: value for key, value in record.items() if key != "card"} for name, record in calibrations.items()},
            "test_lock_path": str(test_lock), "test_lock_sha256": test_lock_sha256,
            "packet_manifest_path": str(packet_manifest_path), "packet_manifest_sha256": lock["packet_manifest_sha256"],
            "packet_files_sha256": packet_manifest["files"], "raw_manifest_path": str(manifest_path), "raw_manifest_sha256": manifest_sha,
            "ledger_path": str(ledger_path), "ledger_sha256": watched[ledger_path],
            **{name.removesuffix(".jsonl").removesuffix(".json") + "_sha256": manifest["outputs_sha256"][name] for name in OUTPUT_NAMES},
            "base_weights_unchanged": True, "base_parameter_sha256": expected_base,
            "builder_sha256": file_hash(Path(__file__)), "evidence_files_sha256": {str(path): digest for path, digest in watched.items()}},
        "limitations": [*manifest["limitations"],
            "Primary complete-assertion accuracy and locked short-QA answer accuracy have separate definitions; neither replaces the other.",
            "Matched applicability is counted from the matched observations. Separate oracle rows are never added to the end-to-end denominator.",
            "A zero automatically credited primary score does not establish zero factual correctness; unsupported output forms remain pending and uncredited."]}
    if io_recovery is not None:
        summary["execution_recovery"] = {"io": io_recovery}
        summary["limitations"].extend([io_recovery["cause_disclosure"], *io_recovery["limitations"]])
    if runtime_failure is not None:
        summary.setdefault("execution_recovery", {})["cuda_runtime_failure"] = runtime_failure
        summary["limitations"].append(runtime_failure["limitation"])
    for path, digest in watched.items():
        require(file_hash(path) == digest, "Evidence changed while the final summary was being built")
    output.parent.mkdir(parents=True, exist_ok=True)
    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with csv_output.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("population", "configuration", "primary_correct", "denominator", "primary_accuracy", "coverage", "pending", "abstentions", "errors", "unavailable", "short_qa_correct", "short_qa_denominator", "p50_ms", "p95_ms"))
            for population, collection in (("matched_all", matched_total), *by_population.items()):
                secondary_rows = secondary["configurations"] if population == "matched_all" else secondary["by_population"][population]
                for item in collection["configurations"]:
                    diagnostic = secondary_rows[item["configuration"]]["all_requested_qa"]
                    writer.writerow((population, item["configuration"], item["accuracy"]["numerator"], item["accuracy"]["denominator"],
                        item["accuracy"]["value"], item["coverage"]["value"], item["pending_review"], item["abstentions"], item["errors"], item["unavailable"],
                        diagnostic["correct"], diagnostic["denominator"], item["p50_ms"], item["p95_ms"]))
            for population, item in (("oracle_all", oracle_total), *[("oracle_" + name, item) for name, item in oracle_total["by_population"].items()]):
                cohort = population.removeprefix("oracle_")
                diagnostic = (secondary["oracle"] if cohort == "all" else secondary["oracle_by_population"][cohort])["oracle_answer"]["all_requested_qa"]
                writer.writerow((population, "oracle_answer", item["accuracy"]["numerator"], item["accuracy"]["denominator"],
                    item["accuracy"]["value"], item["coverage"]["value"], item["pending_review"], item["abstentions"], item["errors"], item["unavailable"],
                    diagnostic["correct"], diagnostic["denominator"], item["p50_ms"], item["p95_ms"]))
        summary["csv_export"] = {"path": str(csv_output), "sha256": file_hash(csv_output)}
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--test-lock", type=Path, required=True)
    parser.add_argument("--test-lock-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional additional assertion of the selected runtime configuration")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--io-recovery-declaration", type=Path,
                        help="Optional immutable IO-only recovery declaration; requires its external checksum and events")
    parser.add_argument("--io-recovery-declaration-sha256")
    parser.add_argument("--io-recovery-events", type=Path,
                        help="Append-only wrapper log; its final completed-run bytes are bound at summary export")
    parser.add_argument("--runtime-failure-disclosure", type=Path)
    parser.add_argument("--runtime-failure-disclosure-sha256")
    args = parser.parse_args()
    try:
        result = build(args.run, args.test_lock, args.test_lock_sha256, args.output, csv_output=args.csv,
            config_path=args.config, io_recovery_declaration=args.io_recovery_declaration,
            io_recovery_declaration_sha256=args.io_recovery_declaration_sha256, io_recovery_events=args.io_recovery_events,
            runtime_failure_disclosure=args.runtime_failure_disclosure,
            runtime_failure_disclosure_sha256=args.runtime_failure_disclosure_sha256)
        print(json.dumps({"output": str(args.output), "questions": result["questions"], "oracle_questions": result["oracle_questions"],
                          "purpose": result["purpose"], "human_reviewed_data": False}, indent=2))
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

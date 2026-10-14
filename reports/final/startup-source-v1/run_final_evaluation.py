"""Execute one hash-locked final comparison, with exact-bound interruption recovery.

No training, selection or parameter search. The protocol fixes the models,
configuration, scorers, populations, order, output and single execution ledger.
Final packet creation/review is a separate explicit action after freezing.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.comparison import CONFIGURATIONS, ComparisonRunner
from editguard.metrics import aggregate_predictions
from editguard.neural import file_sha256
from editguard.scoring import SCORER_VERSION, score_answer
from editguard.short_qa import SHORT_QA_VERSION, score_short_qa
from scripts.analyze_short_qa import summarize as short_summary
from scripts.prepare_data import content_hash
from scripts.review_dataset import read_json
from scripts.run_comparison import oracle_answer_result, scenario_memory, score_record

REQUIRED_SOURCES = {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "editguard").glob("*.py")} | {
    "scripts/run_comparison.py", "scripts/analyze_short_qa.py", "scripts/aggregate_evaluation.py",
    "scripts/prepare_data.py", "scripts/review_dataset.py", "scripts/audit_pilot_v3.py",
    "scripts/final_protocol.py", "scripts/prepare_final_packet.py", "scripts/final_label_audit.py",
    "scripts/run_final_evaluation.py"}
ANSWERED_STATUSES = {"answered", "ok", "success", "generated"}
OUTPUT_NAMES = ("predictions.jsonl", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")


@contextmanager
def session_lock(ledger):
    """Hold an exclusive OS lock; process termination releases it automatically.

    The persistent file is only a lock target, never a stale PID or approval to
    rerun. Windows byte-range locking and POSIX flock both reject a concurrent
    fresh/resume session before it constructs or loads any neural component.
    """
    target = ledger.with_name(ledger.name + ".session.lock")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("Another final evaluation session already holds the execution lock") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def workspace_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Protocol paths must be explicit relative workspace paths")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if root not in target.parents:
        raise ValueError("Protocol path escapes the declared workspace")
    return target


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def locked_inputs(protocol_path, protocol_sha256, packet, *, root=ROOT):
    """Validate every binding before constructing neural components."""
    from scripts.final_protocol import validate_protocol
    from scripts.prepare_final_packet import validate_packet
    protocol = validate_protocol(protocol_path, protocol_sha256, root=root)
    if (protocol.get("review_policy") != "automated_final" or protocol.get("human_reviewed_data") is not False
            or not REQUIRED_SOURCES <= protocol.get("source_sha256", {}).keys()
            or protocol.get("scoring", {}).get("primary_version") != SCORER_VERSION
            or protocol.get("scoring", {}).get("short_qa_version") != SHORT_QA_VERSION):
        raise ValueError("Final protocol must bind all runner/scorer sources and explicitly disclose automated review")
    evaluation = protocol["evaluation"]
    bootstrap = evaluation.get("bootstrap", {})
    if (evaluation.get("configurations") != list(CONFIGURATIONS)
            or evaluation.get("matched_populations") != ["scope", "challenge"]
            or evaluation.get("oracle_populations") != ["answer", "challenge"]
            or evaluation.get("max_final_runs") != 1
            or bootstrap.get("seed") != 42 or bootstrap.get("resamples") != 2000
            or bootstrap.get("paired") is not True or bootstrap.get("cluster_unit") != "entity_family"
            or bootstrap.get("confidence") != .95):
        raise ValueError("All five matched configurations, separate locked cohorts and finite uncertainty settings required")
    if (not re.fullmatch(r"[0-9a-f]{64}", evaluation.get("expected_base_parameter_fingerprint_sha256", ""))
            or not isinstance(protocol.get("runtime_packages"), dict) or not protocol["runtime_packages"]
            or not isinstance(protocol.get("python_version"), str)):
        raise ValueError("Final protocol must bind the measured frozen base and actual runtime package versions")
    manifest, facts, scope, answer, challenge = validate_packet(packet, protocol_path, protocol_sha256, root=root)
    populations = {"scope": scope, "answer": answer, "challenge": challenge}
    expected = evaluation["question_counts"]
    if any(len(populations[name]) != expected[name] for name in populations):
        raise ValueError("Packet populations differ from their locked denominators")
    for name, rows in populations.items():
        if (not rows or len({row["id"] for row in rows}) != len(rows)
                or any(type(row.get("label")) is not int or row["label"] not in (0, 1) for row in rows)):
            raise ValueError("Every final row needs unique identity and explicit applicability")
        if name in ("answer", "challenge") and any(row["label"] != 1 for row in rows):
            raise ValueError("Oracle cohorts must contain only applicable edits")
    all_ids = [row["id"] for rows in populations.values() for row in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Distinct final cohorts must have distinct recorded question IDs")
    overlap = manifest.get("scope_normalized_overlap", {})
    for name in ("scope_row_ids_sharing_train_pair", "scope_row_ids_sharing_validation_pair"):
        ids = overlap.get(name)
        if (not isinstance(ids, list) or any(not isinstance(identity, str) for identity in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= {row["id"] for row in scope}):
            raise ValueError("Final scope abstraction overlap must be explicitly disclosed and row-bound")
    return protocol, manifest, facts, populations


def execution_plan(populations):
    """The complete finite order is fixed independently of model outputs."""
    plan = []
    for population in ("scope", "challenge"):
        for row in sorted(populations[population], key=lambda item: item["id"]):
            for configuration in CONFIGURATIONS:
                plan.append({"kind": "matched", "population": population, "row_id": row["id"], "configuration": configuration})
    for population in ("answer", "challenge"):
        for row in sorted(populations[population], key=lambda item: item["id"]):
            plan.append({"kind": "oracle", "population": population, "row_id": row["id"], "configuration": "oracle_answer"})
    return plan


def resource_snapshot(runner, *, reset=False):
    """Optional loop observations; never mislabel them as process lifetime peaks."""
    torch = sys.modules.get("torch")
    cuda = runner.base.device == "cuda" and torch is not None
    if reset and cuda:
        torch.cuda.reset_peak_memory_stats()
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 2**20
    except ImportError:
        rss = None
    return {"device": runner.base.device, "process_rss_mib": rss,
        "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if cuda else None,
        "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if cuda else None,
        "torch_version": str(torch.__version__) if torch else None}


def record_identity(record, task, row, facts):
    """Recovery may reuse only a complete record for the exact locked input."""
    fact = next(item for item in facts if item["fact_id"] == row["expected_fact"]["fact_id"])
    expected = {**row["expected_fact"], "entity_name": fact["entity_name"], "aliases": fact.get("aliases", [])}
    if (record.get("id") != task["row_id"] or record.get("family_id") != row["family_id"]
            or record.get("focal_edit_id") != row["correction"]["edit_id"] or record.get("configuration") != task["configuration"]
            or record.get("population") != task["population"] or record.get("evaluation_kind") != task["kind"]
            or record.get("row_sha256") != content_hash(row) or record.get("question") != row["question"]
            or record.get("request_context") != row["request_context"]
            or record.get("question_type") != row["question_type"] or record.get("applicable") is not bool(row["label"])
            or record.get("expected_polarity") != row.get("expected_polarity")
            or record.get("expected_fact") != expected
            or record.get("review_policy") != "automated_final" or record.get("human_reviewed_data") is not False
            or not permitted_answer_kind(record)
            or record.get("score", {}).get("scorer_version") != SCORER_VERSION
            or record.get("short_qa", {}).get("diagnostic_version") != SHORT_QA_VERSION):
        raise ValueError("Saved observation differs from the locked question, cohort or scoring contract")
    recomputed = final_primary_score(record)
    if recomputed != record["score"] or score_short_qa(record) != record["short_qa"]:
        raise ValueError("Saved scores differ from the frozen scoring rules")


def permitted_answer_kind(result):
    """Explicit clarification remains an uncredited outcome in every denominator."""
    return (result.get("answer_kind") == "model_generated" or
            result.get("answer_kind") == "deterministic_clarification" and result.get("status") in ("clarification", "conflict"))


def final_primary_score(record):
    """A clarification's text is preserved, but never treated as an answer.

    Every error/unavailable/abstention also receives an explicit False verdict,
    even if the preserved diagnostic text happens to contain the target fact.
    """
    expected = record["expected_fact"]
    text = record.get("answer") or ""
    if record.get("status") not in ANSWERED_STATUSES or record.get("answer_kind") != "model_generated":
        text = ""
    return score_answer(text, expected, aliases=expected.get("aliases", []),
                        expected_polarity=record.get("expected_polarity"), answer_kind=record["answer_kind"])


def score_observation(result, row, facts, runner):
    """Retain a completed request when its separate raw pair diagnostic fails.

    The pair forward is outside request latency and actual retrieved routing.
    Missing probabilities stay explicit in calibration/coverage denominators.
    Invalid question/truth metadata still raises through the original scorer.
    """
    if result["configuration"] != "trained_scope_editor":
        return score_record(result, row, facts, runner)
    proxy = SimpleNamespace(scope=None, calibration=runner.calibration, retrieval_k=runner.retrieval_k)
    record = score_record(result, row, facts, proxy)
    from editguard.formatting import correction_text
    from editguard.neural import calibrated_probability
    try:
        record["scope_probability"] = calibrated_probability(
            runner.scope.score(row["question"], correction_text(row["correction"])), runner.calibration["temperature"])
    except Exception as exc:
        record["scope_probability"] = None
        record["scope_diagnostic_error"] = {"type": type(exc).__name__, "reason": str(exc)}
    return record


def recover_journal(output, progress, binding, plan, rows_by_id, facts):
    """Validate the committed hash chain; recover at most one atomic orphan.

    A crash between an observation rename and cursor rename leaves one complete
    orphan. Partial temporary files are never used, and no observation is edited.
    """
    journal = output / "observations"
    files = sorted(journal.glob("*.json"))
    expected_names = [f"{index:08d}.json" for index in range(len(files))]
    if [path.name for path in files] != expected_names or len(files) > len(plan):
        raise ValueError("Observation journal is not a contiguous finite prefix")
    cursor = progress["next_task_index"]
    if type(cursor) is not int or not 0 <= cursor <= len(plan) or len(files) not in (cursor, cursor + 1):
        raise ValueError("Recovery requires the exact committed prefix and at most one atomic orphan")
    previous, records = None, []
    for index, path in enumerate(files):
        entry = read_json(path)
        task = plan[index]
        if (entry.get("binding") != binding or entry.get("task_index") != index or entry.get("task") != task
                or entry.get("previous_observation_sha256") != previous):
            raise ValueError("Saved observation binding/order/hash chain differs from this run")
        record_identity(entry["record"], task, rows_by_id[task["row_id"]], facts)
        records.append(entry["record"])
        previous = file_sha256(path)
        if index + 1 == cursor and previous != progress.get("last_observation_sha256"):
            raise ValueError("Committed observation hash differs from its saved cursor")
    if cursor == 0 and progress.get("last_observation_sha256") is not None:
        raise ValueError("Empty progress must not claim a journal hash")
    if len(files) == cursor + 1:
        progress.update(next_task_index=len(files), last_observation_sha256=previous,
                        recovered_atomic_orphans=progress.get("recovered_atomic_orphans", 0) + 1)
        atomic_json(output / "progress.json", progress)
    return records


def write_jsonl(path, records):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def initialize_progress(output, binding):
    """Recover only the pre-observation creation window of this bound ledger.

    A ledger is committed before output creation. If the process exits before
    progress is committed, the same ledger can reconstruct an empty output;
    existing observations or unknown files prevent this zero-observation repair.
    """
    if output.exists():
        if not output.is_dir():
            raise ValueError("Initialization recovery requires a directory")
        unexpected = [path for path in output.iterdir()
                      if path.name != "observations" and not path.name.startswith("progress.json.tmp-")]
        if unexpected:
            raise ValueError("Missing initial progress cannot adopt unknown existing outputs")
    else:
        output.mkdir(parents=True, exist_ok=False)
    journal = output / "observations"
    journal.mkdir(exist_ok=True)
    if not journal.is_dir() or any(not path.name.startswith("00000000.json.tmp-") for path in journal.iterdir()):
        raise ValueError("Missing progress cannot adopt already committed observations")
    progress = {"schema_version": 1, "purpose": "locked_final_evaluation_progress", "status": "running", "binding": binding,
        "next_task_index": 0, "last_observation_sha256": None, "base_parameter_sha256_before": None,
        "sessions": [], "recovered_atomic_orphans": 0, "inference_completed": False}
    atomic_json(output / "progress.json", progress)
    return progress


def run_final(protocol_path, protocol_sha256, packet, *, resume=False, export_only=False, root=ROOT, runner_factory=None):
    # Only read the externally hash-bound protocol to locate its OS lock. The
    # complete protocol/packet/package validation occurs while holding the lock.
    protocol_path = Path(protocol_path)
    if file_sha256(protocol_path) != protocol_sha256:
        raise ValueError("Frozen protocol checksum differs")
    preliminary = read_json(protocol_path)
    ledger = workspace_path(root, preliminary["execution"]["ledger_path"])
    output = workspace_path(root, preliminary["execution"]["output_directory"])
    packet = Path(packet).resolve()
    if (ledger == output or output in ledger.parents or ledger in output.parents
            or any(path == packet or packet in path.parents or path in packet.parents for path in (output, ledger))):
        raise ValueError("Output, immutable packet and execution ledger must be separate")
    if export_only and not resume:
        raise ValueError("Export-only recovery requires --resume of the same final execution")
    with session_lock(ledger):
        return _run_final_locked(protocol_path, protocol_sha256, packet, resume=resume, export_only=export_only,
                                 root=root, runner_factory=runner_factory)


def _run_final_locked(protocol_path, protocol_sha256, packet, *, resume=False, export_only=False, root=ROOT, runner_factory=None):
    protocol_path, packet = Path(protocol_path), Path(packet)
    protocol, packet_manifest, facts, populations = locked_inputs(protocol_path, protocol_sha256, packet, root=root)
    output = workspace_path(root, protocol["execution"]["output_directory"])
    ledger = workspace_path(root, protocol["execution"]["ledger_path"])
    config = workspace_path(root, protocol["configuration"]["path"])
    if ledger == output or output in ledger.parents or ledger in output.parents or packet.resolve() == output or packet.resolve() in output.parents:
        raise ValueError("Output, immutable packet and execution ledger must be separate")
    plan = execution_plan(populations)
    rows_by_id = {row["id"]: row for rows in populations.values() for row in rows}
    binding = {"protocol_sha256": protocol_sha256, "protocol_id": protocol["protocol_id"],
        "packet_manifest_sha256": file_sha256(packet / "manifest.json"),
        "configuration_sha256": file_sha256(config), "execution_plan_sha256": content_hash(plan)}
    if resume:
        if not ledger.is_file():
            raise ValueError("Resume requires the existing locked output and single-run ledger")
        saved = read_json(ledger)
        if (saved.get("binding") != binding or saved.get("output_directory") != str(output)
                or saved.get("fresh_final_runs") != 1 or saved.get("status") not in ("initializing", "running", "interrupted", "failed")):
            raise ValueError("Resume must bind the same unfinished final execution; completed runs cannot be rerun")
        if not (output / "progress.json").is_file():
            if saved["status"] != "initializing":
                raise ValueError("Missing progress may be recovered only from its bound initialization ledger")
            initialize_progress(output, binding)
        progress = read_json(output / "progress.json")
        if progress.get("binding") != binding or progress.get("status") not in ("running", "failed", "completed"):
            raise ValueError("Saved progress is not an unfinished exact-bound run")
        records = recover_journal(output, progress, binding, plan, rows_by_id, facts)
        # A crash after final exports but before the ledger rename requires
        # sealing the existing result, never replaying a completed test run.
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            completed = read_json(manifest_path)
            if (completed.get("status") != "completed" or completed.get("purpose") != "frozen_final_evaluation"
                    or len(records) != len(plan) or any(completed.get(key) != value for key, value in binding.items())
                    or completed.get("base_weights_unchanged") is not True
                    or completed.get("base_parameter_sha256_before") != progress["base_parameter_sha256_before"]
                    or completed.get("base_parameter_sha256_after") != progress["base_parameter_sha256_before"]
                    or completed.get("base_parameter_sha256_before") != protocol["evaluation"]["expected_base_parameter_fingerprint_sha256"]
                    or progress.get("inference_completed") is not True
                    or progress.get("base_parameter_sha256_after") != completed.get("base_parameter_sha256_after")
                    or completed.get("runtime_packages") != protocol["runtime_packages"]
                    or completed.get("python_version") != protocol["python_version"]
                    or completed.get("review_policy") != "automated_final" or completed.get("human_reviewed_data") is not False
                    or completed.get("scorer_version") != SCORER_VERSION or completed.get("secondary_scorer_version") != SHORT_QA_VERSION
                    or completed.get("fresh_final_runs") != 1
                    or set(completed.get("outputs_sha256", {})) != set(OUTPUT_NAMES)):
                raise ValueError("Recovery cannot seal an incomplete or mismatched final manifest")
            for name, expected_hash in completed["outputs_sha256"].items():
                if Path(name).name != name or file_sha256(output / name) != expected_hash:
                    raise ValueError("Completed output hash changed before ledger sealing")
            progress.update(status="completed")
            atomic_json(output / "progress.json", progress)
            atomic_json(ledger, {**saved, "status": "completed", "completed_at": timestamp(),
                                "result_manifest_sha256": file_sha256(manifest_path)})
            return completed
        if progress["status"] == "completed":
            raise ValueError("Completed progress is missing its immutable result manifest")
    else:
        if output.exists() or ledger.exists():
            raise ValueError("Final execution is single-use: existing output/ledger refuses a fresh run; use exact-bound --resume")
        ledger.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(ledger, {"schema_version": 1, "binding": binding, "output_directory": str(output),
            "status": "initializing", "started_at": timestamp(), "fresh_final_runs": 1})
        progress = initialize_progress(output, binding)
        records = []
    expected_base = protocol["evaluation"]["expected_base_parameter_fingerprint_sha256"]
    if (len(records) == len(plan) and progress.get("inference_completed") is True
            and progress.get("base_parameter_sha256_before") == expected_base
            and progress.get("base_parameter_sha256_after") == expected_base):
        return export_results(output, ledger, protocol, packet_manifest, populations, binding, progress, records)
    if export_only:
        raise ValueError("Export-only recovery requires the complete journal and recorded final frozen-base check")
    try:
        load_started = perf_counter()
        runner = (runner_factory or ComparisonRunner.from_config)(config)
        construction_seconds = perf_counter() - load_started
        if runner.scope is None or runner.answer_model is None:
            raise ValueError("Final comparison requires both completed trained auxiliary models")
        before = runner.base.parameter_fingerprint()
        if before != expected_base:
            raise ValueError("Frozen-base fingerprint differs from the pre-test protocol")
        if progress["base_parameter_sha256_before"] not in (None, before):
            raise ValueError("Resumed frozen-base fingerprint differs from original execution")
        if records and progress["base_parameter_sha256_before"] is None:
            raise ValueError("Saved observations require a recorded frozen-base fingerprint")
        progress.update(status="running", base_parameter_sha256_before=before)
        progress["sessions"].append({"started_at": timestamp(), "resume": resume, "starting_task_index": len(records),
                                    "model_construction_seconds": construction_seconds,
                                    "resources_before": resource_snapshot(runner, reset=True)})
        atomic_json(output / "progress.json", progress)
        atomic_json(ledger, {**read_json(ledger), "status": "running", "last_session_started_at": timestamp()})
        matched_rows = [row for name in ("scope", "challenge") for row in populations[name]]
        overlap = packet_manifest["scope_normalized_overlap"]
        train_shared = set(overlap["scope_row_ids_sharing_train_pair"])
        validation_shared = set(overlap["scope_row_ids_sharing_validation_pair"])
        original_facts = content_hash(facts)
        for index in range(len(records), len(plan)):
            task = plan[index]
            row = rows_by_id[task["row_id"]]
            if task["kind"] == "oracle":
                result = oracle_answer_result(runner, row, copy.deepcopy(facts))
            else:
                memory = scenario_memory(row, matched_rows)
                result = runner.answer(row["question"], copy.deepcopy(row["request_context"]), copy.deepcopy(facts),
                    copy.deepcopy(memory), configuration=task["configuration"])
            if result.get("configuration") != task["configuration"] or not permitted_answer_kind(result):
                raise ValueError("Final ML evaluation cannot credit a structured or mismatched configuration result")
            record = score_observation(result, row, facts, runner)
            record.update(population=task["population"], evaluation_kind=task["kind"], review_policy="automated_final",
                          human_reviewed_data=False, final_protocol_sha256=protocol_sha256)
            record.update(normalized_scope_input_seen_in_train=task["row_id"] in train_shared if task["population"] == "scope" else None,
                          normalized_scope_input_seen_in_validation=task["row_id"] in validation_shared if task["population"] == "scope" else None)
            record["score"] = final_primary_score(record)
            if task["kind"] == "oracle":
                record["retrieval_hits"] = {}
            record["short_qa"] = score_short_qa(record)
            record_identity(record, task, row, facts)
            if content_hash(facts) != original_facts:
                raise ValueError("Fixed initial fact snapshot changed during inference")
            entry = {"schema_version": 1, "binding": binding, "task_index": index, "task": task,
                "previous_observation_sha256": progress["last_observation_sha256"], "record": record}
            target = output / "observations" / f"{index:08d}.json"
            if target.exists():
                raise ValueError("Observation is immutable and cannot be overwritten")
            atomic_json(target, entry)
            progress.update(next_task_index=index + 1, last_observation_sha256=file_sha256(target))
            atomic_json(output / "progress.json", progress)
            records.append(record)
            if (index + 1) % 25 == 0 or index + 1 == len(plan):
                print(json.dumps({"completed_tasks": index + 1, "finite_tasks": len(plan), "test_tuning": False}), flush=True)
        after = runner.base.parameter_fingerprint()
        if after != before:
            progress["invalidated"] = True
            raise RuntimeError("Frozen base weights changed during final evaluation")
        # Revalidate all files after inference, before publishing complete status.
        locked_inputs(protocol_path, protocol_sha256, packet, root=root)
        if file_sha256(packet / "manifest.json") != binding["packet_manifest_sha256"]:
            raise ValueError("Final packet manifest changed during execution")
        progress["sessions"][-1].update(completed_at=timestamp(), resources_after=resource_snapshot(runner))
        progress.update(inference_completed=True, base_parameter_sha256_after=after)
        atomic_json(output / "progress.json", progress)
        return export_results(output, ledger, protocol, packet_manifest, populations, binding, progress, records)
    except BaseException as exc:
        # Preserve committed observations. Only the same exact-bound run can
        # continue; failures never authorize a new configuration or fresh output.
        progress.update(status="invalidated" if progress.get("invalidated") else "failed",
                        failure={"type": type(exc).__name__, "reason": str(exc), "at": timestamp()})
        atomic_json(output / "progress.json", progress)
        atomic_json(ledger, {**read_json(ledger), "status": "invalidated" if progress.get("invalidated") else "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            "failure": progress["failure"]})
        raise


def export_results(output, ledger, protocol, packet_manifest, populations, binding, progress, records):
    """Derive exports from committed observations without constructing models."""
    protocol_sha256 = binding["protocol_sha256"]
    before, after = progress["base_parameter_sha256_before"], progress["base_parameter_sha256_after"]
    expected_base = protocol["evaluation"]["expected_base_parameter_fingerprint_sha256"]
    if progress.get("inference_completed") is not True or before != expected_base or after != expected_base:
        raise ValueError("Export requires the recorded protocol-bound frozen-base check")
    matched_rows = [row for name in ("scope", "challenge") for row in populations[name]]
    overlap = packet_manifest["scope_normalized_overlap"]
    matched = [r for r in records if r["evaluation_kind"] == "matched"]
    oracle = [r for r in records if r["evaluation_kind"] == "oracle"]
    metrics = aggregate_predictions(matched, bootstrap_seed=42, bootstrap_resamples=2000)
    metrics["by_population"] = {name: aggregate_predictions([r for r in matched if r["population"] == name], bootstrap_seed=42, bootstrap_resamples=2000)
                                for name in ("scope", "challenge")}
    unshared = [r for r in matched if r["population"] == "scope" and not r["normalized_scope_input_seen_in_train"] and not r["normalized_scope_input_seen_in_validation"]]
    metrics["scope_unshared_with_development_inputs"] = aggregate_predictions(unshared, bootstrap_seed=42, bootstrap_resamples=2000)
    metrics["scope_normalized_overlap"] = overlap
    metrics["oracle_answer_diagnostic"] = aggregate_predictions(oracle, bootstrap_seed=42, bootstrap_resamples=2000)
    metrics["oracle_answer_by_population"] = {name: aggregate_predictions([r for r in oracle if r["population"] == name], bootstrap_seed=42, bootstrap_resamples=2000)
                                              for name in ("answer", "challenge")}
    metrics.update(purpose="frozen_final_evaluation", review_policy="automated_final", human_reviewed_data=False,
                   protocol_sha256=protocol_sha256, final_test_tuning_performed=False)
    secondary = {"purpose": "locked_final_secondary_short_qa", "diagnostic_version": SHORT_QA_VERSION,
        "primary_results_replaced": False, "post_hoc_policy": False, "protocol_sha256": protocol_sha256,
        "review_policy": "automated_final", "human_reviewed_data": False,
        "configurations": short_summary(matched), "by_population": {name: short_summary([r for r in matched if r["population"] == name]) for name in ("scope", "challenge")},
        "scope_unshared_with_development_inputs": short_summary(unshared),
        "oracle": short_summary(oracle), "oracle_by_population": {name: short_summary([r for r in oracle if r["population"] == name]) for name in ("answer", "challenge")},
        "disclaimer": "Locked secondary request-bound diagnostic only; unsupported challenge wording remains pending/uncredited. Bare values/polarities do not prove a complete scoped fact endorsement."}
    write_jsonl(output / "predictions.jsonl", matched)
    write_jsonl(output / "oracle_predictions.jsonl", oracle)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "short_qa_metrics.json", secondary)
    metadata = {"schema_version": 1, "purpose": "frozen_final_evaluation", "status": "completed", **binding,
        "completed_at": timestamp(), "configurations": list(CONFIGURATIONS), "scorer_version": SCORER_VERSION,
        "scorer_sha256": protocol["scoring"]["primary_sha256"], "secondary_scorer_version": SHORT_QA_VERSION,
        "review_policy": "automated_final", "human_reviewed_data": False, "reviewed_data": True,
        "dataset_id": packet_manifest["dataset_id"], "test_rows_read": True, "challenge_rows_read": True,
        "questions": len(matched_rows), "matched_prediction_records": len(matched), "oracle_records": len(oracle),
        "population_counts": {name: len(rows) for name, rows in populations.items()},
        "accepted_row_sha256": {row["id"]: content_hash(row) for rows in populations.values() for row in rows},
        "base_parameter_sha256_before": before, "base_parameter_sha256_after": after, "base_weights_unchanged": True,
        "expected_base_parameter_fingerprint_sha256": expected_base,
        "runtime_packages": protocol["runtime_packages"], "python_version": protocol["python_version"],
        "nonanswer_scoring_policy": "All nonanswered statuses and deterministic clarifications receive primary False; diagnostic text remains preserved.",
        "exclusive_session_lock": "Windows byte-range lock or POSIX flock; OS releases on process termination",
        "configuration_order": "fixed five configurations per question; scope then challenge; each cohort sorted IDs",
        "model_load_time_recorded_separately": True, "warmup_requests": 0,
        "memory_policy": "single_focal_edit_plus_at_most_8_foreign_product_distractors",
        "oracle_answer_diagnostic": True, "final_test_tuning_performed": False, "fresh_final_runs": 1,
        "sessions": progress["sessions"], "resumed": len(progress["sessions"]) > 1,
        "recovered_atomic_orphans": progress["recovered_atomic_orphans"],
        "outputs_sha256": {name: file_sha256(output / name) for name in OUTPUT_NAMES},
        "limitations": ["Controlled fictional products with automated final label audit; no independent human-review claim.",
            "All abstentions, errors, unavailable outputs and pending semantic cases remain in fixed denominators.",
            "Scope and challenge cohorts are reported separately; answer-cohort duplicates do not inflate the matched population.",
            "Raw scope pair diagnostics are separate from metadata-gated retrieved routing and generated-answer accuracy.",
            "All updated baselines receive identical approved corrections and the initial supporting facts; RAG materializes an updated snapshot.",
            "Serial CLI request latency includes retrieval/routing/generation but excludes model loading, scoring, checkpoint IO and HTTP/browser overhead; order and shared caches can bias comparisons.",
            "Resource measurements cover individual resumed sessions, not a single uninterrupted command or lifetime peak.",
            "Exact-bound interruption recovery may replay an inference that had not reached its atomic observation commit; each committed question/configuration appears once.",
            "Unsupported secondary question forms remain pending/uncredited; no scorer is extended after the final protocol freeze."]}
    metadata["predictions_sha256"] = metadata["outputs_sha256"]["predictions.jsonl"]
    atomic_json(output / "manifest.json", metadata)
    progress.update(status="completed", completed_at=timestamp())
    atomic_json(output / "progress.json", progress)
    atomic_json(ledger, {**read_json(ledger), "status": "completed", "completed_at": timestamp(),
                        "result_manifest_sha256": file_sha256(output / "manifest.json")})
    return metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continue only the same unfinished hash-locked execution")
    parser.add_argument("--export-only", action="store_true", help="With --resume, derive missing exports only after complete committed observations and the final base check; load no models")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256):
        parser.error("An explicit frozen protocol SHA-256 is required")
    result = run_final(args.protocol, args.protocol_sha256, args.packet, resume=args.resume, export_only=args.export_only)
    print(json.dumps({"status": result["status"], "questions": result["questions"],
                      "oracle_records": result["oracle_records"], "base_weights_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()

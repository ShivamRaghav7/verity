"""Supplement a sealed oracle diagnostic after a declared CUDA runtime fault.

Selection uses execution status only. Original final records are never replaced.
Each selected generation is durably reserved once; an interrupted reservation
becomes an explicit uncredited outcome, rather than silently repeating a call.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from time import sleep
from types import SimpleNamespace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from editguard.comparison import ComparisonRunner
from editguard.metrics import aggregate_predictions
from editguard.neural import file_sha256
from editguard.short_qa import score_short_qa
from scripts.analyze_short_qa import summarize as short_summary
from scripts.final_protocol import canonical_hash, read_json
from scripts.run_comparison import oracle_answer_result, score_record
from scripts.run_final_evaluation import final_primary_score, locked_inputs, record_identity, session_lock, workspace_path

IO_DELAYS = (.05, .1, .2, .4, .4, .4, .4)


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_bytes(path, raw):
    """Finite retries apply only to file replacement, never model generation."""
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        for attempt in range(8):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                sleep(IO_DELAYS[attempt])
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))


def write_records(path, records):
    atomic_bytes(path, "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for row in records).encode("utf-8"))


def cuda_unknown(record):
    return record.get("status") == "error" and isinstance(record.get("error"), str) and re.search(
        r"cuda(?:\s+error)?\s*:\s*unknown error", record["error"], re.I) is not None


def fatal_cuda(record):
    return record.get("status") == "error" and isinstance(record.get("error"), str) and "cuda" in record["error"].lower()


def original_inputs(protocol_path, protocol_sha, packet, run, *, root=ROOT):
    """Only sealed, externally protocol-bound evidence can be supplemented."""
    protocol, packet_manifest, facts, populations = locked_inputs(protocol_path, protocol_sha, packet, root=root)
    requested_run = Path(run).resolve() if Path(run).is_absolute() else (Path(root) / run).resolve()
    run = workspace_path(root, protocol["execution"]["output_directory"])
    require(requested_run == run, "Requested original run differs from the frozen protocol")
    manifest = read_json(run / "manifest.json")
    require(manifest.get("purpose") == "frozen_final_evaluation" and manifest.get("status") == "completed"
            and manifest.get("protocol_sha256") == protocol_sha
            and manifest.get("configuration_sha256") == protocol["configuration"]["sha256"]
            and manifest.get("packet_manifest_sha256") == file_sha256(Path(packet) / "manifest.json")
            and manifest.get("review_policy") == "automated_final" and manifest.get("human_reviewed_data") is False
            and manifest.get("final_test_tuning_performed") is False,
            "Supplement requires the sealed original frozen final evaluation")
    ledger = read_json(workspace_path(root, protocol["execution"]["ledger_path"]))
    require(ledger.get("status") == "completed" and ledger.get("result_manifest_sha256") == file_sha256(run / "manifest.json"),
            "Original final execution ledger is not sealed")
    for name, digest in manifest["outputs_sha256"].items():
        require(Path(name).name == name and file_sha256(run / name) == digest, "Original final export checksum differs")
    records = read_records(run / "oracle_predictions.jsonl")
    expected_count = sum(protocol["evaluation"]["question_counts"][name] for name in ("answer", "challenge"))
    rows = {row["id"]: row for name in ("answer", "challenge") for row in populations[name]}
    require(len(records) == expected_count and len({row["id"] for row in records}) == expected_count
            and {row["id"] for row in records} == set(rows), "Original oracle cohort must retain every frozen question")
    for record in records:
        row = rows[record["id"]]
        task = {"kind": "oracle", "population": record["population"], "row_id": row["id"], "configuration": "oracle_answer"}
        record_identity(record, task, row, facts)
        require(record.get("final_protocol_sha256") == protocol_sha, "Original oracle record protocol differs")
    return protocol, packet_manifest, facts, rows, manifest, records, run


def prepare(protocol_path, protocol_sha, packet, run, output, declaration_path, *, root=ROOT):
    protocol, packet_manifest, facts, rows, manifest, records, run = original_inputs(protocol_path, protocol_sha, packet, run, root=root)
    output = workspace_path(root, str(Path(output).relative_to(root)) if Path(output).is_absolute() else str(output))
    declaration_path = Path(declaration_path)
    ledger = output.with_name(output.name + ".execution.json")
    require(not output.exists() and not declaration_path.exists() and not ledger.exists(), "Supplement paths are single-use and must be fresh")
    require(run != output and run not in output.parents and output not in run.parents
            and Path(packet).resolve() not in output.parents, "Supplement output must be separate from original evidence")
    selected = [record for record in records if cuda_unknown(record)]
    require(bool(selected), "No original oracle CUDA unknown-error records qualify; nothing may be replayed")
    require(all(isinstance(record.get("input_text"), str) and record["input_text"] for record in selected),
            "Every selected runtime error must preserve the exact original generator input")
    selected.sort(key=lambda record: (record["population"], record["id"]))
    original_files = {name: file_sha256(run / name) for name in ("manifest.json", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")}
    declaration = {"schema_version": 1, "purpose": "frozen_oracle_runtime_recovery_declaration", "status": "frozen", "declared_at": now(),
        "protocol_path": str(Path(protocol_path).resolve().relative_to(root)).replace("\\", "/"), "protocol_sha256": protocol_sha,
        "packet_path": str(Path(packet).resolve().relative_to(root)).replace("\\", "/"), "packet_manifest_sha256": file_sha256(Path(packet) / "manifest.json"),
        "original_run": str(run.relative_to(root)).replace("\\", "/"), "original_files_sha256": original_files,
        "configuration_sha256": protocol["configuration"]["sha256"], "expected_base_parameter_fingerprint_sha256": protocol["evaluation"]["expected_base_parameter_fingerprint_sha256"],
        "supplement_source_path": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"), "supplement_source_sha256": file_sha256(Path(__file__)),
        "output_directory": str(output.relative_to(root)).replace("\\", "/"), "ledger_path": str(ledger.relative_to(root)).replace("\\", "/"),
        "oracle_questions": len(records), "selected_questions": len(selected), "maximum_new_generation_calls": len(selected), "passes": 1,
        "selected": [{"id": record["id"], "population": record["population"], "record_sha256": canonical_hash(record),
                       "row_sha256": record["row_sha256"], "input_utf8_sha256": hashlib.sha256(record["input_text"].encode("utf-8")).hexdigest()} for record in selected],
        "selection_rule": "Original oracle status=error and error text matches CUDA error: unknown error; never select using answer correctness.",
        "review_policy": "automated_final", "human_reviewed_data": False, "primary_results_replaced": False,
        "training_or_parameter_selection": False, "successful_original_records_reexecuted": False,
        "io_policy": {"replacement_attempts": 8, "delays_seconds": list(IO_DELAYS), "maximum_delay_seconds": sum(IO_DELAYS)},
        "limitations": ["Separate operational supplement; original final scores and all original records remain unchanged.",
            "A resumed process cannot rerun a durably reserved generation; interrupted reservations remain explicitly uncredited.",
            "On a new CUDA error, stop immediately; later selected rows retain original uncredited errors and explicit not-attempted counts.",
            "Serialized model/source/configuration hashes remain fixed; a later healthy instance cannot establish an unavailable fingerprint of an earlier failed GPU instance.",
            "Combined latency includes retained original timings and new supplemental timings; it is not a fresh uninterrupted benchmark."]}
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    with declaration_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(declaration, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
    return declaration


class CheckedGenerator:
    def __init__(self, generator, expected):
        self.generator, self.expected = generator, expected
        self.input_verified = False
        self.last_trace = {}

    def generate(self, text):
        require(text.encode("utf-8") == self.expected.encode("utf-8"), "Reconstructed oracle input differs byte-for-byte from the original runtime-error input")
        self.input_verified = True
        result = self.generator.generate(text)
        self.last_trace = copy.deepcopy(getattr(self.generator, "last_trace", {}))
        return result


def missing_attempt(original, declaration_sha, reason):
    record = copy.deepcopy(original)
    record.update(status="error", answer=None, answer_kind="model_generated", generation_trace=None, error=reason, latency_ms=None)
    record["runtime_recovery"] = {"declaration_sha256": declaration_sha, "original_record_sha256": canonical_hash(original),
        "original_status": original["status"], "original_error": original["error"], "original_score": original["score"],
        "generation_result_durably_recorded": False, "interrupted_reservation": True, "selection_used_correctness": False}
    record["score"] = final_primary_score(record)
    record["short_qa"] = score_short_qa(record)
    return record


def export(output, declaration, declaration_sha, originals, recovered, progress, *, complete):
    by_id = {record["id"]: record for record in recovered}
    combined = [by_id.get(record["id"], record) for record in originals]
    metrics = aggregate_predictions(combined, bootstrap_seed=42, bootstrap_resamples=2000)
    selected_ids = {record["id"] for record in declaration["selected"]}
    untouched = [record for record in originals if record["id"] not in selected_ids]
    metrics.update(purpose="supplemental_oracle_runtime_recovery", primary_results_replaced=False,
        original_denominator=len(originals), selected_runtime_error_count=len(selected_ids),
        supplemental_records=len(recovered), not_attempted_selected_count=len(selected_ids - by_id.keys()),
        original_successful_or_other_unselected_records_preserved=len(untouched),
        interrupted_reservation_count=sum(record["runtime_recovery"].get("interrupted_reservation", False) for record in recovered),
        new_runtime_error_count=sum(record["status"] == "error" and record["runtime_recovery"]["generation_result_durably_recorded"] for record in recovered),
        by_population={name: aggregate_predictions([record for record in combined if record["population"] == name], bootstrap_seed=42, bootstrap_resamples=2000)
                       for name in ("answer", "challenge")})
    secondary = {"purpose": "supplemental_oracle_runtime_recovery_secondary", "primary_results_replaced": False,
        "configurations": short_summary(combined), "by_population": {name: short_summary([record for record in combined if record["population"] == name]) for name in ("answer", "challenge")}}
    write_records(output / "recovery_predictions.jsonl", recovered)
    write_records(output / "combined_oracle_predictions.jsonl", combined)
    atomic_json(output / "metrics.json", metrics)
    atomic_json(output / "short_qa_metrics.json", secondary)
    metadata = {"schema_version": 1, "purpose": "supplemental_oracle_runtime_recovery", "status": "completed" if complete else "interrupted",
        "primary_results_replaced": False, "review_policy": "automated_final", "human_reviewed_data": False,
        "declaration_sha256": declaration_sha, "protocol_sha256": declaration["protocol_sha256"],
        "configuration_sha256": declaration["configuration_sha256"], "supplement_source_sha256": declaration["supplement_source_sha256"],
        "original_files_sha256": declaration["original_files_sha256"], "oracle_questions": len(originals),
        "selected_runtime_errors": declaration["selected_questions"], "durable_attempt_reservations": progress["next_index"],
        "interrupted_reservation_count": metrics["interrupted_reservation_count"],
        "new_runtime_error_count": metrics["new_runtime_error_count"],
        "maximum_new_generation_calls": declaration["maximum_new_generation_calls"], "not_attempted_selected_count": declaration["selected_questions"] - len(recovered),
        "sessions": progress["sessions"], "original_records_unchanged": True, "selection_used_correctness": False,
        "finite_stopping_condition": "One reservation per selected runtime-error ID; no generation after a new CUDA error.",
        "outputs_sha256": {name: file_sha256(output / name) for name in ("recovery_predictions.jsonl", "combined_oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json")},
        "limitations": declaration["limitations"]}
    atomic_json(output / ("manifest.json" if complete else "interrupted_manifest.json"), metadata)
    return metadata


def run(declaration_path, declaration_sha, *, resume=False, root=ROOT, runner_factory=None):
    root = Path(root).resolve()
    declaration_path = Path(declaration_path)
    require(file_sha256(declaration_path) == declaration_sha, "Supplement declaration external checksum differs")
    declaration = read_json(declaration_path)
    require(declaration.get("purpose") == "frozen_oracle_runtime_recovery_declaration" and declaration.get("status") == "frozen"
            and declaration.get("primary_results_replaced") is False and declaration.get("passes") == 1
            and declaration.get("supplement_source_sha256") == file_sha256(Path(__file__)), "Supplement source or immutable declaration differs")
    output = workspace_path(root, declaration["output_directory"])
    ledger = workspace_path(root, declaration["ledger_path"])
    with session_lock(ledger):
        return run_locked(declaration_path, declaration_sha, declaration, output, ledger, resume=resume, root=root, runner_factory=runner_factory)


def run_locked(declaration_path, declaration_sha, declaration, output, ledger, *, resume, root, runner_factory):
    protocol_path, packet = workspace_path(root, declaration["protocol_path"]), workspace_path(root, declaration["packet_path"])
    original_run = workspace_path(root, declaration["original_run"])
    protocol, packet_manifest, facts, rows, manifest, originals, original_run = original_inputs(protocol_path, declaration["protocol_sha256"], packet, original_run, root=root)
    require(file_sha256(packet / "manifest.json") == declaration["packet_manifest_sha256"]
            and all(file_sha256(original_run / name) == digest for name, digest in declaration["original_files_sha256"].items()), "Original evidence differs from supplemental declaration")
    selected = sorted([record for record in originals if cuda_unknown(record)], key=lambda record: (record["population"], record["id"]))
    require(len(selected) == declaration["selected_questions"] == declaration["maximum_new_generation_calls"]
            and len(originals) == declaration["oracle_questions"]
            and [{"id": record["id"], "population": record["population"], "record_sha256": canonical_hash(record), "row_sha256": record["row_sha256"],
                  "input_utf8_sha256": hashlib.sha256(record["input_text"].encode("utf-8")).hexdigest()} for record in selected] == declaration["selected"],
            "Recovery selection differs from the declared execution-status-only subset")
    binding = {"declaration_sha256": declaration_sha, "protocol_sha256": declaration["protocol_sha256"], "original_oracle_sha256": declaration["original_files_sha256"]["oracle_predictions.jsonl"]}
    if resume:
        require(output.is_dir() and ledger.exists(), "Resume requires this existing unfinished supplemental execution")
        saved = read_json(ledger)
        require(saved.get("binding") == binding and saved.get("status") in ("running", "interrupted", "failed"), "Completed or differently bound supplements cannot rerun")
        progress = read_json(output / "progress.json")
        require(progress["binding"] == binding, "Supplement progress differs from declaration")
    else:
        require(not output.exists() and not ledger.exists(), "Supplement is single-use; exact-bound --resume is required after interruption")
        output.mkdir(parents=True)
        (output / "observations").mkdir()
        (output / "attempts").mkdir()
        (output / "source").mkdir()
        for name in declaration["original_files_sha256"]:
            (output / "source" / name).write_bytes((original_run / name).read_bytes())
        (output / "source/declaration.json").write_bytes(declaration_path.read_bytes())
        progress = {"binding": binding, "next_index": 0, "committed_observation_count": 0,
                    "last_observation_sha256": None, "sessions": [], "status": "running"}
        atomic_json(output / "progress.json", progress)
        atomic_json(ledger, {"binding": binding, "status": "running", "started_at": now()})
    recovered, previous = [], None
    committed_cursor = progress["committed_observation_count"]
    committed_hash = progress.get("last_observation_sha256")
    for name, digest in declaration["original_files_sha256"].items():
        require(file_sha256(output / "source" / name) == digest, "Immutable source snapshot differs from original final evidence")
    require(file_sha256(output / "source/declaration.json") == declaration_sha, "Immutable supplemental declaration snapshot differs")
    reservations = sorted((output / "attempts").glob("*.json"))
    require([path.name for path in reservations] == [f"{index:08d}.json" for index in range(len(reservations))]
            and len(reservations) <= len(selected), "Supplement reservations are not a finite contiguous prefix")
    for index, path in enumerate(reservations):
        attempt = read_json(path)
        require(attempt["binding"] == binding and attempt["id"] == selected[index]["id"] and attempt["index"] == index,
                "Durable reservation differs from declared selected question")
        observation = output / "observations" / path.name
        if not observation.exists():
            record = missing_attempt(selected[index], declaration_sha, "Supplement interrupted after durable reservation; no generated result was durably recorded. This case is not replayed.")
            atomic_json(observation, {"binding": binding, "index": index, "previous_sha256": previous, "record": record})
        entry = read_json(observation)
        require(entry["binding"] == binding and entry["index"] == index and entry["previous_sha256"] == previous
                and entry["record"]["runtime_recovery"]["original_record_sha256"] == canonical_hash(selected[index]), "Supplement observation hash chain differs")
        record = entry["record"]
        task = {"kind": "oracle", "population": record["population"], "row_id": record["id"], "configuration": "oracle_answer"}
        record_identity(record, task, rows[record["id"]], facts)
        recovered.append(record)
        previous = file_sha256(observation)
        if index + 1 == committed_cursor and committed_hash is not None:
            require(previous == committed_hash, "Committed supplemental observation checksum differs from progress")
    require(len(list((output / "observations").glob("*.json"))) == len(reservations), "Unreserved supplemental observation is forbidden")
    progress["next_index"] = len(reservations)
    progress["committed_observation_count"] = len(reservations)
    progress["last_observation_sha256"] = previous
    atomic_json(output / "progress.json", progress)
    runner = None
    session = {"started_at": now(), "resume": resume, "starting_index": len(reservations),
               "base_parameter_sha256_before": None, "base_parameter_sha256_after": None}
    progress["sessions"].append(session)
    atomic_json(output / "progress.json", progress)
    try:
        runner = (runner_factory or ComparisonRunner.from_config)(workspace_path(root, protocol["configuration"]["path"]))
        before = runner.base.parameter_fingerprint()
        require(before == declaration["expected_base_parameter_fingerprint_sha256"], "New-process base fingerprint differs from the frozen protocol")
        require(runner.answer_model is not None, "Supplement requires the same completed trained answer artifact")
        session["base_parameter_sha256_before"] = before
        atomic_json(output / "progress.json", progress)
        for index in range(len(reservations), len(selected)):
            original = selected[index]
            attempt = {"binding": binding, "index": index, "id": original["id"], "reserved_at": now(), "maximum_calls_for_this_id": 1}
            atomic_json(output / "attempts" / f"{index:08d}.json", attempt)
            progress["next_index"] = index + 1
            atomic_json(output / "progress.json", progress)
            checked = CheckedGenerator(runner.answer_model, original["input_text"])
            proxy = SimpleNamespace(answer_model=checked, retriever=runner.retriever)
            result = oracle_answer_result(proxy, rows[original["id"]], copy.deepcopy(facts))
            require(checked.input_verified, "Reconstructed input did not pass exact-byte verification; no generation is authorized")
            record = {**copy.deepcopy(original), **score_record(result, rows[original["id"]], facts, runner)}
            record["runtime_recovery"] = {"declaration_sha256": declaration_sha, "original_record_sha256": canonical_hash(original),
                "original_status": original["status"], "original_error": original["error"], "original_score": original["score"],
                "generation_result_durably_recorded": True, "input_utf8_exactly_reconstructed": True, "selection_used_correctness": False}
            record["score"] = final_primary_score(record)
            record["short_qa"] = score_short_qa(record)
            task = {"kind": "oracle", "population": original["population"], "row_id": original["id"], "configuration": "oracle_answer"}
            record_identity(record, task, rows[original["id"]], facts)
            target = output / "observations" / f"{index:08d}.json"
            atomic_json(target, {"binding": binding, "index": index, "previous_sha256": previous, "record": record})
            previous = file_sha256(target)
            progress["last_observation_sha256"] = previous
            progress["committed_observation_count"] = index + 1
            atomic_json(output / "progress.json", progress)
            recovered.append(record)
            print(json.dumps({"reserved_calls": index + 1, "finite_selected_calls": len(selected), "primary_results_replaced": False}), flush=True)
            if fatal_cuda(record):
                raise RuntimeError("New CUDA error; stop supplemental generation immediately to prevent an error cascade")
        after = runner.base.parameter_fingerprint()
        if after != before:
            progress["invalidated"] = True
            raise ValueError("New-process frozen base fingerprint changed during supplement; this supplement is invalidated")
        session.update(completed_at=now(), base_parameter_sha256_after=after)
        original_inputs(protocol_path, declaration["protocol_sha256"], packet, original_run, root=root)
        require(file_sha256(Path(__file__)) == declaration["supplement_source_sha256"], "Supplement source changed during execution")
        progress.update(status="completed")
        atomic_json(output / "progress.json", progress)
        metadata = export(output, declaration, declaration_sha, originals, recovered, progress, complete=True)
        atomic_json(ledger, {"binding": binding, "status": "completed", "result_manifest_sha256": file_sha256(output / "manifest.json"), "completed_at": now()})
        return metadata
    except BaseException as exc:
        progress.update(status="invalidated" if progress.get("invalidated") else "interrupted",
                        failure={"type": type(exc).__name__, "reason": str(exc), "at": now()})
        if progress["sessions"]:
            progress["sessions"][-1]["failure"] = progress["failure"]
            progress["sessions"][-1].update(base_parameter_sha256_after=None,
                fingerprint_after_status="unavailable_for_this_interrupted_session")
        atomic_json(output / "progress.json", progress)
        atomic_json(ledger, {"binding": binding, "status": progress["status"], "failure": progress["failure"]})
        export(output, declaration, declaration_sha, originals, recovered, progress, complete=False)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    declaration = commands.add_parser("prepare", help="Freeze execution-status-only replay selection after the original run is sealed")
    declaration.add_argument("--protocol", type=Path, required=True)
    declaration.add_argument("--protocol-sha256", required=True)
    declaration.add_argument("--packet", type=Path, required=True)
    declaration.add_argument("--run", type=Path, required=True)
    declaration.add_argument("--output", type=Path, required=True)
    declaration.add_argument("--declaration", type=Path, required=True)
    execution = commands.add_parser("run")
    execution.add_argument("--declaration", type=Path, required=True)
    execution.add_argument("--declaration-sha256", required=True)
    execution.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.protocol, args.protocol_sha256, args.packet, args.run, args.output, args.declaration)
        print(json.dumps({"declaration_sha256": file_sha256(args.declaration), "selected_runtime_errors": result["selected_questions"], "maximum_new_generation_calls": result["maximum_new_generation_calls"]}, indent=2))
    else:
        result = run(args.declaration, args.declaration_sha256, resume=args.resume)
        print(json.dumps({"status": result["status"], "oracle_questions": result["oracle_questions"], "primary_results_replaced": False}, indent=2))


if __name__ == "__main__":
    main()

"""Finite isolated-model inference with a hard watchdog and immutable outcome records."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

from editguard_reasonke.__main__ import write_receipt
from .data import CATEGORIES, digest, encoded, load_dataset
from .retrieval import retrieve
from .scoring import score

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"reasonke": ("R0", "R-RAG", "R-ALL"), "instruction": ("G-RAG", "G-ALL")}


def now():
    return datetime.now(timezone.utc).isoformat()


def source_identity(config_path):
    sources = sorted((ROOT / "editguard_benchmark").glob("*.py"))
    sources += [ROOT / "editguard_reasonke/runtime.py", Path(config_path)]
    config = json.loads(Path(config_path).read_text())
    sources += [ROOT / config[key] for key in ("reasonke_manifest", "instruction_manifest")]
    sources += [ROOT / config["dataset"] / "manifest.json"]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in sources}


def _worker(connection, model_name, config):
    try:
        if model_name == "reasonke":
            from editguard_reasonke.runtime import ReasonKERuntime
            runtime = ReasonKERuntime(ROOT / config["reasonke_manifest"])
        else:
            from .models import InstructionRuntime
            runtime = InstructionRuntime(ROOT / config["instruction_manifest"], ROOT / config["reasonke_manifest"])
        runtime.load()
        connection.send({"ready": True, "runtime": runtime.status()})
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                connection.send({"response": runtime.generate(request)})
            except Exception as exc:
                connection.send({"error": {"type": type(exc).__name__, "message": str(exc)}})
    except BaseException as exc:
        try:
            connection.send({"error": {"type": type(exc).__name__, "message": str(exc)}})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def balanced(rows, count):
    groups = {category: sorted((row for row in rows if row["category"] == category), key=lambda row: row["id"])
              for category in CATEGORIES}
    result = []
    for index in range(max(map(len, groups.values()))):
        for category in CATEGORIES:
            if index < len(groups[category]):
                result.append(groups[category][index])
    return result[:count]


def tasks_for(config, model_name, phase):
    split = "final" if phase == "final" else "development"
    manifest, lineages, questions = load_dataset(ROOT / config["dataset"], split)
    tasks = []
    conditions = CONDITIONS[model_name]
    count = (config["development_requests_per_condition"] if phase == "development" else
             config["profile_requests_per_model"] // len(conditions) if phase == "profile" else len(questions))
    ordered = balanced(questions, len(questions))
    for condition_index, condition in enumerate(conditions):
        selected = ([ordered[((condition_index * count if phase == "profile" else 0) + offset) % len(ordered)]
                     for offset in range(count)] if phase != "final" else sorted(questions, key=lambda row: row["id"]))
        for question in selected:
            tasks.append({"id": condition + ":" + question["id"], "condition": condition,
                          "question": question, "outcome": "unexecuted"})
    if phase == "final":
        oracle = sorted((row for row in questions if row["category"] in ("two_hop", "two_hop_paraphrase")),
                        key=lambda row: hashlib.sha256(("oracle-42:" + row["id"]).encode()).hexdigest())[:config["oracle_questions_per_model"]]
        for question in oracle:
            condition = "R-ORACLE" if model_name == "reasonke" else "G-ORACLE"
            tasks.append({"id": condition + ":" + question["id"], "condition": condition,
                          "question": question, "outcome": "unexecuted"})
    return manifest, lineages, tasks


def request_for(task, lineages, config):
    question = task["question"]
    lineage = lineages[question["lineage_id"]]
    facts = lineage["initial_facts"] if task["condition"] == "R0" else lineage["current_facts"]
    if task["condition"].endswith("RAG"):
        facts = retrieve(question["question"], facts, config["retrieval_k"])
    elif task["condition"].endswith("ORACLE"):
        facts = [row for row in facts if row["id"] in question["required_fact_ids"]]
    support = set(question["required_fact_ids"])
    found = support.intersection(row["id"] for row in facts)
    payload = {"question": question["question"], "facts": [{key: row[key] for key in
                ("id", "question", "answer", "scope", "source")} for row in facts],
               "history": [], "max_new_tokens": config["max_new_tokens"]}
    return payload, {"required": len(support), "found": len(found), "complete": found == support,
                     "selected_fact_ids": [row["id"] for row in facts],
                     "gold_support_used": task["condition"].endswith("ORACLE")}


def verify_protocol(path, config_path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("status") != "frozen" or protocol.get("sources") != source_identity(config_path):
        raise ValueError("Frozen source/data/configuration identities do not match; final inference refused")
    profiles = protocol.get("profiles", {})
    if set(profiles) != {"reasonke", "instruction"}:
        raise ValueError("Both measured model profiles are required")
    return protocol


def run(config_path, model_name, phase, output, protocol=None, resume=False):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if phase == "final":
        if protocol is None:
            raise ValueError("Final inference requires a profile-grounded frozen protocol")
        verify_protocol(protocol, config_path)  # Must occur before accessing final questions.
    manifest, lineages, tasks = tasks_for(config, model_name, phase)
    plan = {"model": model_name, "phase": phase, "sources": source_identity(config_path),
            "task_ids_sha256": hashlib.sha256(encoded([row["id"] for row in tasks])).hexdigest(),
            "protocol_sha256": digest(protocol) if protocol else None}
    output = Path(output)
    if output.exists():
        if not resume:
            raise ValueError("Output already exists; use --resume with the identical plan")
        receipt = json.loads(output.read_text())
        if receipt.get("plan") != plan or [row["id"] for row in receipt["records"]] != [row["id"] for row in tasks]:
            raise ValueError("Resume plan differs from saved source, input, task or protocol identities")
    else:
        receipt = {"schema": "editguard-operations-inference-v1", "status": "running", "started_utc": now(),
                   "plan": plan, "dataset": manifest, "records": tasks, "sessions": [],
                   "disclosures": ["Controlled fictional data; automated graph audit, no independent human review",
                       "Both model checkpoints trained upstream; no training or weight updates here",
                       "Same-size model comparison does not isolate causal effects of upstream training",
                       "All assigned questions retain denominator credit, including unexecuted/error/abstain",
                       "Latency covers context selection, local IPC and inference; not a browser network round trip"]}
    remaining = [row for row in receipt["records"] if row["outcome"] == "unexecuted"]
    if not remaining:
        print("All assigned outcomes already recorded; no model loaded.")
        return receipt
    session = {"started_utc": now(), "planned_remaining_requests": len(remaining), "model_loaded": False}
    receipt["sessions"].append(session)
    write_receipt(output, receipt)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_worker, args=(child, model_name, config), daemon=True)
    started = time.perf_counter()
    process.start()
    child.close()
    try:
        if not parent.poll(config["model_load_watchdog_seconds"]):
            raise TimeoutError("Model loading exceeded the hard load watchdog")
        ready = parent.recv()
        if not ready.get("ready"):
            raise RuntimeError("Model loading failed: " + json.dumps(ready))
        session.update(model_loaded=True, runtime=ready["runtime"], load_wall_seconds=time.perf_counter() - started)
        write_receipt(output, receipt)
        for row in remaining:
            if time.perf_counter() - started > config["final_wall_budget_seconds_per_model"]:
                raise TimeoutError("Finite per-model wall budget reached; unfinished records remain unexecuted")
            began = time.perf_counter()
            payload, retrieval = request_for(row, lineages, config)
            row.update(started_utc=now(), request_sha256=hashlib.sha256(encoded(payload)).hexdigest(),
                       supplied_facts=payload["facts"], retrieval=retrieval)
            parent.send(payload)
            if not parent.poll(config["request_watchdog_seconds"]):
                row.update(error={"type": "TimeoutError", "message": "Hard request watchdog exceeded"},
                           pipeline_seconds=time.perf_counter() - began)
                row.update(score(row["question"], error=row["error"]))
                write_receipt(output, receipt)
                raise TimeoutError("Request exceeded watchdog; worker terminated without answer-based retry")
            generated = parent.recv()
            row.update(generated, pipeline_seconds=time.perf_counter() - began, completed_utc=now())
            row.update(score(row["question"], row.get("response"), row.get("error")))
            write_receipt(output, receipt)
            print(json.dumps({"id": row["id"], "outcome": row["outcome"],
                              "seconds": round(row["pipeline_seconds"], 3)}), flush=True)
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt.update(status="interrupted", last_error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if process.is_alive():
            try:
                parent.send(None)
                process.join(timeout=3)
            except (OSError, EOFError):
                pass
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
        parent.close()
        session.update(completed_utc=now(), seconds=time.perf_counter() - started)
        receipt.update(last_updated_utc=now(), assigned=len(tasks),
                       recorded=sum(row["outcome"] != "unexecuted" for row in receipt["records"]),
                       unexecuted=sum(row["outcome"] == "unexecuted" for row in receipt["records"]))
        write_receipt(output, receipt)
    return receipt


def freeze(config_path, profile_paths, output):
    config_path = Path(config_path).resolve()
    if Path(output).exists():
        raise ValueError("Protocol already exists; it cannot be overwritten")
    profiles, sources = {}, source_identity(config_path)
    for path in profile_paths:
        receipt = json.loads(Path(path).read_text())
        if receipt["plan"]["phase"] != "profile" or receipt["status"] != "completed" or receipt["plan"]["sources"] != sources:
            raise ValueError("Only completed profiles from identical sources can ground the freeze")
        rows = receipt["records"]
        if len(rows) != 12 or any(row["outcome"] in ("execution_error", "unexecuted") for row in rows):
            raise ValueError("Profile must contain 12 executed, resource-measured requests")
        model = receipt["plan"]["model"]
        if model in profiles:
            raise ValueError("Duplicate model profile supplied")
        config = json.loads(config_path.read_text())
        manifest = json.loads((ROOT / config["reasonke_manifest" if model == "reasonke" else "instruction_manifest"]).read_text())
        for row in rows:
            response = row.get("response", {})
            generated = response.get("generation", {})
            if (generated.get("model") != manifest["model_id"] or generated.get("revision") != manifest["revision"]
                    or not isinstance(response.get("raw_output"), str) or not response["raw_output"].strip()
                    or generated.get("input_tokens", 0) <= 0 or generated.get("output_tokens", 0) <= 0
                    or generated.get("peak_reserved_bytes", 0) <= 0):
                raise ValueError("Profiles require real pinned-checkpoint raw outputs, tokens and CUDA measurements")
        seconds = sorted(row["pipeline_seconds"] for row in rows)
        final_count = 784 if model == "reasonke" else 528
        profiles[model] = {"receipt_sha256": digest(path), "profile_requests": len(rows),
            "measured_seconds": sum(seconds), "slowest_request_seconds": max(seconds),
            "optimizer_or_training_steps": 0, "final_assigned_requests": final_count,
            "forecast_seconds_using_profile_mean": sum(seconds) / len(seconds) * final_count,
            "forecast_seconds_using_profile_max": max(seconds) * final_count,
            "limitation": "12 development requests; extrapolation is uncertain and excludes fresh downloads/loading"}
    if set(profiles) != {"reasonke", "instruction"}:
        raise ValueError("Provide exactly one measured profile per model")
    protocol = {"schema": "editguard-operations-protocol-v1", "status": "frozen", "created_utc": now(),
        "sources": sources, "profiles": profiles, "final_main_requests": 1280, "oracle_requests": 32,
        "planned_outcome_denominators": {"per_main_condition": 256, "applicable": 128, "locality": 128},
        "selection_note": "Profiles/dev only; no final answers used for settings or scoring"}
    write_receipt(output, protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/operations-evaluation.json")
    parser.add_argument("--model", choices=CONDITIONS)
    parser.add_argument("--phase", choices=("profile", "development", "final"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--freeze-profiles", nargs=2)
    args = parser.parse_args()
    if args.freeze_profiles:
        print(json.dumps(freeze(args.config, args.freeze_profiles, args.output), indent=2))
    else:
        if not args.model or not args.phase:
            parser.error("Choose --model and --phase")
        receipt = run(args.config, args.model, args.phase, args.output, args.protocol, args.resume)
        print(json.dumps({key: receipt.get(key) for key in ("status", "assigned", "recorded", "unexecuted")}))
        if receipt["status"] != "completed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()

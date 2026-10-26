"""Recompute receipt integrity and frozen scoring without loading a model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from editguard_benchmark.data import digest, encoded
from editguard_benchmark.metrics import aggregate
from editguard_benchmark.run import request_for, source_identity, tasks_for, verify_protocol
from editguard_benchmark.scoring import score
from editguard_reasonke.runtime import extract_answer

ROOT = Path(__file__).resolve().parents[1]


def audit_receipts(paths, *, protocol=None, config_path=None):
    config_path = Path(config_path or ROOT / "configs/operations-evaluation.json")
    config = json.loads(config_path.read_text())
    sources = source_identity(config_path)
    checks = []
    if protocol:
        verify_protocol(protocol, config_path)
    for path in paths:
        path = Path(path)
        receipt = json.loads(path.read_text())
        plan = receipt["plan"]
        if plan["sources"] != sources:
            raise ValueError("Receipt source/config/data identity differs: " + str(path))
        phase, model = plan["phase"], plan["model"]
        if phase == "final" and (protocol is None or plan["protocol_sha256"] != digest(protocol)):
            raise ValueError("Final receipt requires its identical frozen protocol")
        manifest, worlds, tasks = tasks_for(config, model, phase)
        ids = [row["id"] for row in tasks]
        if (receipt["dataset"] != manifest or [row["id"] for row in receipt["records"]] != ids
                or plan["task_ids_sha256"] != hashlib.sha256(encoded(ids)).hexdigest()):
            raise ValueError("Assigned tasks or dataset identity changed: " + str(path))
        model_manifest = json.loads((ROOT / config[model + "_manifest"]).read_text())
        generated = attempted = 0
        for row, task in zip(receipt["records"], tasks):
            if row["question"] != task["question"] or row["condition"] != task["condition"]:
                raise ValueError("Question/condition changed: " + row["id"])
            if row["outcome"] == "unexecuted":
                if row.get("response") or row.get("error"):
                    raise ValueError("Unexecuted row contains an outcome: " + row["id"])
                continue
            attempted += 1
            request, routing = request_for(task, worlds, config)
            if (row["request_sha256"] != hashlib.sha256(encoded(request)).hexdigest()
                    or row["supplied_facts"] != request["facts"] or row["retrieval"] != routing):
                raise ValueError("Supplied context or prompt identity changed: " + row["id"])
            response = row.get("response")
            if response:
                generated += 1
                content, format_info = extract_answer(response["raw_output"])
                generation = response["generation"]
                if (content != response["content"] or format_info != response["format"]
                        or generation["model"] != model_manifest["model_id"]
                        or generation["revision"] != model_manifest["revision"]
                        or generation["fact_ids"] != routing["selected_fact_ids"]):
                    raise ValueError("Generated provenance or answer extraction differs: " + row["id"])
            recomputed = score(task["question"], response, row.get("error"))
            if any(row.get(key) != value for key, value in recomputed.items()):
                raise ValueError("Frozen scorer disagrees with recorded outcome: " + row["id"])
        checks.append({"receipt": str(path), "sha256": digest(path), "phase": phase, "model": model,
                       "assigned": len(tasks), "attempted": attempted, "generated": generated,
                       "unexecuted": len(tasks) - attempted, "status": receipt["status"]})
    report = aggregate(paths)
    return {"schema": "editguard-operations-host-audit-v1", "status": "passed",
            "created_utc": datetime.now(timezone.utc).isoformat(), "receipts": checks,
            "phase": report["phase"], "assigned": report["assigned_records"],
            "independent_lineages": report["independent_lineages"],
            "verified": ["source/config/data hashes", "complete assigned task identities", "question labels",
                         "request digest and supplied facts", "retrieval without gold support except oracle",
                         "pinned generation provenance", "raw-output extraction", "frozen answer scoring",
                         "aggregate denominators and paired comparisons"],
            "protocol_sha256": digest(protocol) if protocol else None,
            "audit_script_sha256": digest(__file__), "new_model_generations": 0,
            "human_reviewed": False,
            "notice": "Mechanical integrity/scoring audit; does not independently establish semantic labels or faithful explanations."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipts", nargs=2)
    parser.add_argument("--protocol")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Audit artifact exists; preserve it and choose a new output")
    result = audit_receipts(args.receipts, protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

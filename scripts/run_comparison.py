"""Finite neural smoke or reviewed validation comparison; never reads test rows."""

import argparse
import hashlib
import json
import sys
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from editguard.comparison import CONFIGURATIONS, ComparisonRunner
from editguard.formatting import conditional_input
from editguard.neural import file_sha256
from editguard.scoring import SCORER_VERSION, score_answer
from scripts.prepare_data import content_hash, load_reviewed_rows


def smoke_scenarios():
    """Curated engineering fixtures, kept separate from train/validation/test."""
    payload = json.loads((ROOT / "data/catalogue.json").read_text(encoding="utf-8"))
    facts = payload["facts"] if isinstance(payload, dict) else payload
    edited = next(fact for fact in facts if fact["fact_id"] == "nimbus-basic-v2-devices")
    edit = {**edited, "edit_id": "neural-smoke-correction", "original_value": "5",
            "corrected_value": "3", "status": "approved", "effective_from": "2026-10-01",
            "source": "controlled://engineering-smoke/approved-device-limit",
            "entity_aliases": edited.get("aliases", [])}
    cases = [
        ("How many devices does Nimbus Basic version v2 support?", "nimbus-basic-v2-devices", True, "direct", None),
        ("What is the device allowance on Nimbus Basic version v2?", "nimbus-basic-v2-devices", True, "paraphrase", None),
        ("Can I connect four devices on Nimbus Basic version v2?", "nimbus-basic-v2-devices", True, "comparison", "no"),
        ("What is the monthly price of Nimbus Basic version v2?", "nimbus-basic-v2-price", False, "out_of_scope", None),
        ("How many devices does Nimbus Pro version v2 support?", "nimbus-pro-v2-devices", False, "out_of_scope", None),
        ("How many devices does Nimbus Basic version v1 support?", "nimbus-basic-v1-devices", False, "out_of_scope", None),
    ]
    by_id = {fact["fact_id"]: fact for fact in facts}
    rows = []
    for index, (question, fact_id, applicable, kind, polarity) in enumerate(cases):
        fact = by_id[fact_id]
        rows.append({"id": f"neural-smoke-{index}", "family_id": "engineering-nimbus",
                     "question": question, "question_type": kind, "label": int(applicable),
                     "expected_polarity": polarity, "correction": edit,
                     "request_context": {key: fact[key] for key in ("entity_id", "plan", "version")}
                     | {"as_of": "2026-10-11"},
                     "expected_fact": {**fact, "value": "3" if applicable else fact["value"]}})
    return facts, rows


def reviewed_validation(dataset, reviews):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    for name in ("catalogue.json", "scope_validation.jsonl"):
        if file_sha256(dataset / name) != manifest["files"][name]:
            raise ValueError(f"Dataset manifest integrity failed: {name}")
    rows = load_reviewed_rows(dataset / "scope_validation.jsonl", reviews)
    if any(row["split"] != "validation" for row in rows):
        raise ValueError("Only validation rows are permitted in this development runner")
    facts = json.loads((dataset / "catalogue.json").read_text(encoding="utf-8"))
    return [fact for fact in facts if fact["split"] == "validation"], rows, manifest


def scenario_memory(row, rows, distractor_count=8):
    """One focal edit plus foreign-product distractors; no scoring truth accessed.

    Distractors exclude the explicit question product, so each row's original
    single-edit ground truth remains valid. This is a bounded initial benchmark,
    not a dense same-product multi-edit stress test.
    """
    focal = row["correction"]
    memory = [focal]
    excluded = {focal["entity_id"], row["request_context"]["entity_id"]}
    seen = {focal["lineage_id"]}
    for other in sorted(rows, key=lambda item: item["id"]):
        candidate = other["correction"]
        if candidate["entity_id"] not in excluded and candidate["lineage_id"] not in seen:
            memory.append(candidate)
            seen.add(candidate["lineage_id"])
            if len(memory) >= distractor_count + 1:
                break
    return memory


def bounded_selection(rows, cap):
    """Alternate applicable/out-of-scope rows for an informative pilot smoke."""
    buckets = [[row for row in sorted(rows, key=lambda item: item["id"]) if bool(row["label"]) == value]
               for value in (True, False)]
    selected = []
    for index in range(max(map(len, buckets), default=0)):
        for bucket in buckets:
            if index < len(bucket):
                selected.append(bucket[index])
                if len(selected) == cap:
                    return selected
    return selected


def score_record(result, row, facts, runner):
    # Scoring data is attached only after the inference call has finished.
    expected = dict(row["expected_fact"])
    fact = next(f for f in facts if f["fact_id"] == expected["fact_id"])
    expected.update(entity_name=fact["entity_name"], aliases=fact.get("aliases", []))
    record = {**result, "id": row["id"], "family_id": row["family_id"],
              "request_context": dict(row["request_context"]),
              "question_type": row["question_type"], "applicable": bool(row["label"]),
              "expected_fact": expected, "expected_polarity": row.get("expected_polarity"),
              "row_sha256": content_hash(row), "focal_edit_id": row["correction"]["edit_id"]}
    record["score"] = score_answer(result.get("answer") or "", expected,
                                   aliases=fact.get("aliases", []), expected_polarity=row.get("expected_polarity"),
                                   answer_kind=result["answer_kind"])
    # Base is a diagnostic reference. RAG recalls corrected facts, editors recall
    # corrections. Recall is reported only for genuinely applicable rows.
    record["retrieval_hits"] = {}
    if row["label"] and result["status"] != "unavailable":
        if result["configuration"] == "updated_rag":
            hits = [hit["fact_id"] for hit in result["retrieved_facts"]]
            target = expected["fact_id"]
        elif result["configuration"] != "unchanged_base":
            hits = [hit["edit_id"] for hit in result["retrieval_candidates_raw"]]
            target = row["correction"]["edit_id"]
        else:
            hits, target = [], None
        if target:
            record["retrieval_hits"] = {str(k): target in hits[:k] for k in (1, runner.retrieval_k)}
    if result["configuration"] == "trained_scope_editor":
        from editguard.formatting import correction_text
        from editguard.neural import calibrated_probability
        # Independent pair assessment includes hard negatives rejected by gates.
        record["scope_label"] = row["label"]
        record["scope_probability"] = calibrated_probability(
            runner.scope.score(row["question"], correction_text(row["correction"])), runner.calibration["temperature"]) if runner.scope and runner.calibration else None
        record["scope_threshold"] = runner.calibration["threshold"] if runner.calibration else .5
    return record


def oracle_answer_result(runner, row, facts):
    """Isolated generator diagnostic with the known applicable correction.

    Ground truth routing is intentional and recorded as oracle; it is never an
    end-to-end editor route and receives no endorsed target answer as input.
    """
    started = perf_counter()
    result = {"configuration": "oracle_answer", "question": row["question"], "status": "unavailable",
              "answer": None, "answer_kind": "model_generated", "route": "oracle_correct_routing",
              "selected_edit_id": None, "retrieved_facts": [], "retrieval_candidates_raw": [],
              "retrieval_candidates_eligible": [], "scope_scores": [], "input_text": None,
              "generation_trace": None, "error": "Trained answer artifact is missing."}
    if not row["label"]:
        raise ValueError("Oracle answer diagnostic requires an applicable correction")
    if runner.answer_model:
        correction = row["correction"]
        from editguard.comparison import compatible
        from editguard.formatting import fact_text
        supporting = [fact for fact in facts if compatible(fact, row["request_context"])
                      and fact["relation"] != correction["relation"]]
        supporting = [hit["record"] for hit in runner.retriever.rank(row["question"], supporting, fact_text, k=2)]
        text = conditional_input(row["question"], correction, supporting, row["request_context"]["as_of"])
        result.update(selected_edit_id=correction["edit_id"], supporting_context=supporting, input_text=text)
        try:
            answer = runner.answer_model.generate(text)
            result.update(answer=answer, status="answered" if answer else "abstention", error=None,
                          generation_trace=dict(runner.answer_model.last_trace))
        except Exception as exc:
            result.update(status="error", error=str(exc))
    result["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/inference.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--max-questions", type=int, default=6)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    parser.add_argument("--oracle-answer", action="store_true", help="Separate generator diagnostic on applicable questions only")
    args = parser.parse_args()
    if not 1 <= args.max_questions <= (20 if args.smoke else 2000):
        parser.error("Question cap is 1..20 for smoke, 1..2000 for validation")
    if args.output.exists():
        parser.error("Use a fresh output directory; evaluation records are not overwritten")
    if args.smoke:
        if args.dataset or args.reviews:
            parser.error("Smoke fixtures cannot be mixed with dataset/review inputs")
        facts, rows = smoke_scenarios()
        manifest = {"dataset_id": "engineering-nimbus-smoke", "purpose": "engineering_smoke_not_performance_evaluation"}
    else:
        if not args.dataset or not args.reviews:
            parser.error("Reviewed validation requires --dataset and --reviews; test runs are unavailable here")
        facts, rows, manifest = reviewed_validation(args.dataset, args.reviews)
    selected = rows[:args.max_questions] if args.smoke else bounded_selection(rows, args.max_questions)
    runner = ComparisonRunner.from_config(args.config)
    before = runner.base.parameter_fingerprint()
    import torch
    import psutil
    if runner.base.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True)
    metadata = {"schema_version": 1, "purpose": "engineering_smoke" if args.smoke else "validation_development",
                "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
                "dataset_id": manifest["dataset_id"], "configuration_sha256": file_sha256(args.config),
                "scorer_version": SCORER_VERSION, "scorer_sha256": file_sha256(ROOT / "editguard/scoring.py"),
                "questions": len(selected), "configurations": args.configurations,
                "memory_policy": "single_focal_edit_plus_at_most_8_foreign_product_distractors",
                "oracle_answer_diagnostic": args.oracle_answer,
                "selection": "curated_fixture_order" if args.smoke else "applicable_negative_alternation_sorted_ids_bounded_development", "test_rows_read": False,
                "base_parameter_sha256_before": before, "torch_version": torch.__version__,
                "device": runner.base.device, "missing_auxiliaries": not (runner.scope and runner.answer_model),
                "inference_config": json.loads(args.config.read_text(encoding="utf-8"))}
    if args.reviews:
        metadata["review_registry_sha256"] = file_sha256(args.reviews)
        metadata["dataset_manifest_sha256"] = file_sha256(args.dataset / "manifest.json")
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    records, oracle_records = [], []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in selected:
            memory = scenario_memory(row, rows)
            for configuration in args.configurations:
                result = runner.answer(row["question"], row["request_context"], facts, memory, configuration=configuration)
                record = score_record(result, row, facts, runner)
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({"id": row["id"], "configuration": configuration, "status": result["status"],
                                  "latency_ms": result["latency_ms"], "answer": result["answer"]}), flush=True)
            if args.oracle_answer and row["label"]:
                result = oracle_answer_result(runner, row, facts)
                record = score_record(result, row, facts, runner)
                record["retrieval_hits"] = {}  # Oracle routing is not retrieval evidence.
                oracle_records.append(record)
    from editguard.metrics import aggregate_predictions
    metrics = aggregate_predictions(records)
    if args.oracle_answer:
        # Oracle has a different population (applicable questions only); mixing
        # it into matched end-to-end configs would pad negatives as missing.
        metrics["oracle_answer_diagnostic"] = aggregate_predictions(oracle_records)
        (args.output / "oracle_predictions.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in oracle_records), encoding="utf-8")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metadata.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat(),
                    base_parameter_sha256_after=runner.base.parameter_fingerprint(),
                    process_rss_mib=psutil.Process().memory_info().rss / 2**20,
                    cuda_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20 if runner.base.device == "cuda" else None,
                    cuda_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20 if runner.base.device == "cuda" else None,
                    predictions_sha256=file_sha256(args.output / "predictions.jsonl"))
    metadata["base_weights_unchanged"] = metadata["base_parameter_sha256_before"] == metadata["base_parameter_sha256_after"]
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not metadata["base_weights_unchanged"]:
        raise RuntimeError("Frozen base parameter fingerprint changed")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Preserve partial observations while making failure explicit. Argparse
        # exits (including refusal to overwrite) are not caught here.
        if "--output" in sys.argv:
            index = sys.argv.index("--output")
            if index + 1 < len(sys.argv):
                manifest_path = Path(sys.argv[index + 1]) / "manifest.json"
                if manifest_path.is_file():
                    failed = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if failed.get("status") == "running":
                        failed.update(status="failed", failed_at=datetime.now(timezone.utc).isoformat(),
                                      failure={"type": type(exc).__name__, "reason": str(exc)})
                        manifest_path.write_text(json.dumps(failed, indent=2) + "\n", encoding="utf-8")
        raise

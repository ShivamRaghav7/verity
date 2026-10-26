"""Serve current-backend measured evidence separately from earlier visual experiments."""
import json
import hashlib
from pathlib import Path

from .reasonke_backend import MODEL_ID, MODEL_REVISION

ARTIFACTS = {name + extension for name in ("category-accuracy", "efficacy-locality", "outcome-composition",
              "paired-differences", "latency") for extension in (".png", ".svg")} | {"REPORT.md", "scorecard.csv", "figure-manifest.json", "study.pdf", "study-manifest.json"}


def benchmark_location(root):
    root = Path(root)
    for phase, folder in (("final", "operations-final"), ("development", "operations-development-v1")):
        path = root / "reports/reasonke" / folder / "results.json"
        if path.is_file():
            return phase, path
    return None, None


def experiment_artifact(root, name):
    if name not in ARTIFACTS:
        raise ValueError("Unknown experiment artifact")
    _, report = benchmark_location(root)
    if report is None or not (report.parent / "derived" / name).is_file():
        raise FileNotFoundError("Experiment artifact has not been recorded")
    return report.parent / "derived" / name


def current_results(root):
    root = Path(root)
    phase, path = benchmark_location(root)
    if path:
        report = json.loads(path.read_text(encoding="utf-8"))
        assigned = 256 if phase == "final" else 24
        if (report.get("schema") != "editguard-operations-results-v1" or report.get("phase") != phase
                or report.get("execution_status") != "completed"
                or any(report.get("conditions", {}).get(name, {}).get("overall", {}).get("assigned") != assigned
                       for name in ("R0", "R-RAG", "R-ALL", "G-RAG", "G-ALL"))):
            raise ValueError("Current benchmark report does not match the declared experiment")
        return {"kind": "controlled_" + phase + "_benchmark", "model": MODEL_ID, "model_revision": MODEL_REVISION,
                "report": report, "artifacts": sorted(name for name in ARTIFACTS if (path.parent / "derived" / name).is_file()),
                "notice": "Controlled fictional operations study (" + phase + "). All assigned outcomes retain their denominator. "
                          + ("Development results guided context-policy selection; they are not held-out final accuracy. " if phase == "development" else "Settings were fixed before held-out inference. ")
                          + "Labels were audited automatically, without independent human review."}
    directory = root / "reports/reasonke/t4-restart-1"
    profile = json.loads((directory / "acceptance.json").read_text(encoding="utf-8"))
    live = json.loads((directory / "live-acceptance.json").read_text(encoding="utf-8"))
    if (profile["runtime"]["model_id"] != MODEL_ID or profile["runtime"]["revision"] != MODEL_REVISION
            or live["model_revision"] != MODEL_REVISION):
        raise ValueError("Development evidence identifies another model")
    return {"kind": "development_acceptance", "model": MODEL_ID, "model_revision": MODEL_REVISION,
        "notice": "Real model generations from a finite development check. These selected examples are not a general accuracy benchmark.",
        "profile": {"assigned": len(profile["cases"]), "matching_final_answers": sum(row.get("exact_alias_match") is True for row in profile["cases"]),
                    "errors": profile["error_count"], "load_seconds": profile["runtime"]["load_seconds"],
                    "peak_reserved_bytes": profile["runtime"]["hardware"]["load_peak_reserved_bytes"],
                    "cases": [{"id": row["id"], "expected": row["expected_answer"],
                               "actual": row.get("response", {}).get("content"), "status": row["status"],
                               "seconds": row.get("response", {}).get("generation", {}).get("seconds")}
                              for row in profile["cases"]]},
        "workflow": {"assigned": len(live["cases"]), "factual_matches": live["matched_factual_cases"],
                     "factual_assigned": live["assigned_factual_cases"], "unexecuted": live["unexecuted_count"],
                     "owned_fact_inactive_at_end": live["owned_fact_inactive_at_end"], "seconds": live["total_seconds"],
                     "cases": [{key: row[key] for key in ("id", "expected", "actual", "passed", "factual_quality_evaluated")}
                               for row in live["cases"]]},
        "limitations": ["Five repeated startup fixtures do not become new independent test facts",
                        "Ordinary conversation was checked for nonempty generation, not general factual accuracy",
                        "Raw explanations contained unsupported claims despite correct final answers",
                        "An earlier contradictory-source note defeated the intended override",
                        "The exact checkpoint-to-paper-score association is unverified",
                        "Current exact/canonical-key conflict controls do not certify arbitrary semantic contradictions"],
        "final_benchmark": {"status": "not_recorded", "notice": "The broader operations benchmark is separate from these development checks."}}


def experiment_cases(root, filters=None):
    """Bounded read-only raw-answer explorer; never runs a model or rescoring."""
    filters = filters or {}
    if set(filters) - {"condition", "category", "outcome", "offset"}:
        raise ValueError("Unknown case filter")
    evidence = current_results(root)
    if not evidence["kind"].startswith("controlled_"):
        raise FileNotFoundError("Controlled experiment cases are not recorded")
    report = evidence["report"]
    conditions = report["conditions"]
    categories = sorted({key for value in conditions.values() for key in value["by_category"]})
    outcomes = ("correct", "wrong", "abstain", "execution_error", "unexecuted")
    for key, valid in (("condition", conditions), ("category", categories), ("outcome", outcomes)):
        if filters.get(key) and filters[key] not in valid:
            raise ValueError("Unknown " + key + " filter")
    try:
        offset = int(filters.get("offset", "0"))
        if offset < 0 or offset > report["assigned_records"]:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("Choose a valid case offset") from None
    _, result_path = benchmark_location(root)
    records, hashes = [], sorted(report["source_receipts"].values())
    for model in ("reasonke", "instruction"):
        path = result_path.parent / (report["phase"] + "-" + model + ".json")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() not in hashes:
            raise ValueError("Raw answer receipt does not match the displayed experiment")
        records.extend(json.loads(raw)["records"])
    selected = [row for row in records if (not filters.get("condition") or row["condition"] == filters["condition"])
                and (not filters.get("category") or row["question"]["category"] == filters["category"])
                and (not filters.get("outcome") or row["outcome"] == filters["outcome"])]
    selected.sort(key=lambda row: row["id"])
    rows = []
    for row in selected[offset:offset + 20]:
        response = row.get("response") or {}
        rows.append({key: value for key, value in {
            "id": row["id"], "condition": row["condition"], "question": row["question"],
            "outcome": row["outcome"], "content": response.get("content"), "raw_output": response.get("raw_output"),
            "generation": response.get("generation"), "supplied_facts": row.get("supplied_facts", []),
            "retrieval": row.get("retrieval"), "pipeline_seconds": row.get("pipeline_seconds"),
            "request_sha256": row.get("request_sha256"), "error": row.get("error")}.items() if value is not None})
    return {"schema": "editguard-experiment-cases-v1", "phase": report["phase"], "total": len(selected),
            "offset": offset, "page_size": 20, "cases": rows,
            "next_offset": offset + 20 if offset + 20 < len(selected) else None,
            "conditions": list(conditions), "categories": categories, "outcomes": list(outcomes),
            "notice": "Saved model outputs, not a new generation. Raw explanations are unverified. Filters change the view, never the scoring denominator."}

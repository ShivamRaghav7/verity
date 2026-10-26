"""Serve current-backend measured evidence separately from earlier visual experiments."""
import json
from pathlib import Path

from .reasonke_backend import MODEL_ID, MODEL_REVISION


def current_results(root):
    root = Path(root)
    final = root / "reports/reasonke/operations-final/results.json"
    if final.is_file():
        report = json.loads(final.read_text(encoding="utf-8"))
        if (report.get("schema") != "editguard-operations-results-v1" or report.get("phase") != "final"
                or any(report.get("conditions", {}).get(name, {}).get("overall", {}).get("assigned") != 256
                       for name in ("R0", "R-RAG", "R-ALL", "G-RAG", "G-ALL"))):
            raise ValueError("Current benchmark report does not match the declared experiment")
        return {"kind": "controlled_final_benchmark", "model": MODEL_ID, "model_revision": MODEL_REVISION,
                "report": report, "notice": "Controlled fictional operations study; all assigned outcomes retain their denominator."}
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

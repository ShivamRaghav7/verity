"""Serve current-backend measured evidence separately from earlier visual experiments."""
import json
import hashlib
import math
from functools import lru_cache
from pathlib import Path

from .reasonke_backend import MODEL_ID, MODEL_REVISION

ARTIFACTS = {name + extension for name in ("category-accuracy", "efficacy-locality", "outcome-composition",
              "paired-differences", "latency") for extension in (".png", ".svg")} | {"REPORT.md", "scorecard.csv", "figure-manifest.json", "study.pdf", "study-manifest.json"}
AUDIT_ARTIFACTS = {"explanation-audit.json", "host-audit.json"}
DEPTH_INPUTS = {f"{phase}-{model}.json" for phase in ("profile", "development", "final")
                for model in ("reasonke", "instruction")} | {"protocol.json", "results.json"}
DEPTH_FIGURES = {name + extension for name in ("depth-accuracy", "category-accuracy", "outcome-composition",
                "retrieval-and-answer", "latency") for extension in (".png", ".svg")}
DEPTH_EXPORTS = DEPTH_FIGURES | {"REPORT.md", "scorecard.csv", "retrieval.csv", "latency.csv", "figure-manifest.json"}
DEPTH_NOTICE = ("Separate controlled depth challenge with a new fictional Indian-context population and iterative retrieval. "
                "Its scores are not pooled with the original operations study. Labels were audited automatically; "
                "no independent human semantic review or verification of generated explanatory prose is claimed.")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _depth_signature(directory):
    paths = [directory / name for name in sorted(DEPTH_INPUTS | {"host-audit.json"})]
    paths += [directory / "derived" / name for name in sorted(DEPTH_EXPORTS)]
    return tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) if path.is_file()
                 else (str(path), None, None) for path in paths)


def _validate_depth_counts(metric, assigned):
    outcomes = ("correct", "wrong", "abstain", "execution_error", "unexecuted")
    counts = metric.get("counts", {})
    if (metric.get("assigned") != assigned or set(counts) != set(outcomes)
            or any(type(counts[key]) is not int or counts[key] < 0 for key in outcomes)
            or sum(counts.values()) != assigned
            or metric.get("accuracy") != counts["correct"] / assigned
            or metric.get("coverage") != (counts["correct"] + counts["wrong"]) / assigned):
        raise ValueError("Depth scorecard denominator or outcome counts differ")
    latency = metric.get("generated_latency", {})
    n = latency.get("n")
    if type(n) is not int or not 0 <= n <= assigned:
        raise ValueError("Depth generated-latency population differs")
    if n and any(not isinstance(latency.get(key), (int, float)) or not math.isfinite(latency[key])
                 or latency[key] < 0 for key in ("p50_seconds", "p95_seconds")):
        raise ValueError("Depth latency must be finite and nonnegative")


@lru_cache(maxsize=4)
def _depth_verified(directory_text, signature):
    """Check saved audit/digests once per observed file state, without a model."""
    directory = Path(directory_text)
    audit = json.loads((directory / "host-audit.json").read_text(encoding="utf-8"))
    if (audit.get("schema") != "editguard-depth-host-audit-v1" or audit.get("status") != "passed"
            or audit.get("model_calls") != 0 or set(audit.get("files", {})) != DEPTH_INPUTS):
        raise ValueError("A complete saved depth host audit is required")
    hashes = audit["files"]
    if any(_sha(directory / name) != checksum for name, checksum in hashes.items()):
        raise ValueError("Depth evidence differs from the saved host audit")
    if (audit.get("results_sha256") != hashes["results.json"]
            or audit.get("protocol_sha256") != hashes["protocol.json"]):
        raise ValueError("Depth audit does not bind the displayed results and protocol")
    report = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if (report.get("schema") != "editguard-depth-results-v1" or report.get("phase") != "final"
            or report.get("execution_status") != "completed" or report.get("assigned_records") != 544
            or report.get("independent_lineages") != 32
            or set(report.get("conditions", {})) != {"R-RAG", "G-RAG", "R-ORACLE", "G-ORACLE"}):
        raise ValueError("Depth scorecard does not match the declared final population")
    for name, condition in report["conditions"].items():
        oracle = name.endswith("ORACLE")
        _validate_depth_counts(condition["overall"], 16 if oracle else 256)
        expected_depths = {"3": 16} if oracle else {str(depth): 64 for depth in (0, 1, 2, 3)}
        if set(condition["by_depth"]) != set(expected_depths):
            raise ValueError("Depth groups differ from the declared population")
        for depth, assigned in expected_depths.items():
            _validate_depth_counts(condition["by_depth"][depth], assigned)
            retrieved = condition["retrieval_by_depth"][depth]
            required = assigned * max(1, int(depth))
            if (retrieved.get("assigned") != assigned or retrieved.get("required") != required
                    or type(retrieved.get("found")) is not int or not 0 <= retrieved["found"] <= required
                    or type(retrieved.get("complete")) is not int or not 0 <= retrieved["complete"] <= assigned):
                raise ValueError("Depth retrieval-support denominator differs")
        if any(sum(metric["counts"][key] for metric in condition["by_depth"].values()) != count
               for key, count in condition["overall"]["counts"].items()):
            raise ValueError("Depth groups do not reproduce their overall outcomes")
    exports = []
    manifest_path = directory / "derived/figure-manifest.json"
    if manifest_path.is_file():
        if audit.get("figure_manifest_sha256") != _sha(manifest_path):
            raise ValueError("Depth figure manifest differs from the saved audit")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("schema") != "editguard-depth-figures-v1"
                or manifest.get("protocol_sha256") != hashes["protocol.json"]
                or manifest.get("files", {}).get("results.json") != hashes["results.json"]):
            raise ValueError("Depth figures identify a different scorecard")
        for name in sorted(DEPTH_EXPORTS - {"figure-manifest.json"}):
            path = directory / "derived" / name
            if path.is_file():
                if manifest.get("files", {}).get(name) != _sha(path):
                    raise ValueError("Depth artifact differs from the figure manifest")
                exports.append(name)
        exports.append("figure-manifest.json")
    return json.dumps({"kind": "separate_depth_study", "status": "completed", "available": True,
                       "report": report, "artifacts": exports, "notice": DEPTH_NOTICE})


def depth_results(root):
    """Optional independent scorecard; absent/unaudited data never becomes a score."""
    directory = Path(root).resolve() / "reports/reasonke/depth-final"
    if not (directory / "results.json").is_file():
        return {"kind": "separate_depth_study", "status": "unmeasured", "available": False,
                "notice": "The separate one-, two- and three-hop challenge has no released final measurements yet. "
                          "The original operations study remains unchanged. No accuracy is inferred from retrieval-only checks."}
    try:
        return json.loads(_depth_verified(str(directory), _depth_signature(directory)))
    except (ValueError, KeyError, OSError, TypeError):
        return {"kind": "separate_depth_study", "status": "unavailable", "available": False,
                "notice": "The saved depth evidence is incomplete or differs from its audit. No numerical depth results are displayed."}


def depth_artifact(root, name):
    evidence = depth_results(root)
    if not evidence["available"]:
        raise FileNotFoundError("Audited final depth evidence is unavailable")
    directory = Path(root).resolve() / "reports/reasonke/depth-final"
    if name in {"results.json", "protocol.json", "host-audit.json"}:
        return directory / name
    if name in evidence["artifacts"]:
        return directory / "derived" / name
    raise ValueError("Unknown depth artifact")


def benchmark_location(root):
    root = Path(root)
    for phase, folder in (("final", "operations-final"), ("development", "operations-development-v1")):
        path = root / "reports/reasonke" / folder / "results.json"
        if path.is_file():
            return phase, path
    return None, None


def experiment_artifact(root, name):
    if name not in ARTIFACTS | AUDIT_ARTIFACTS:
        raise ValueError("Unknown experiment artifact")
    _, report = benchmark_location(root)
    path = (report.parent / name if name in AUDIT_ARTIFACTS else report.parent / "derived" / name) if report else None
    if path is None or not path.is_file():
        raise FileNotFoundError("Experiment artifact has not been recorded")
    return path


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
        audit_path = path.parent / "explanation-audit.json"
        explanation = None
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if (audit.get("phase") != phase or audit.get("schema") != "editguard-explanation-support-audit-v1"
                    or audit.get("receipt_sha256") not in report["source_receipts"].values()):
                raise ValueError("Explanation audit does not identify the displayed experiment")
            explanation = {key: audit[key] for key in ("phase", "assigned_traces", "by_condition", "denominators", "limitations", "human_reviewed", "model_generations")}
        return {"kind": "controlled_" + phase + "_benchmark", "model": MODEL_ID, "model_revision": MODEL_REVISION,
                "report": report, "artifacts": sorted(name for name in ARTIFACTS if (path.parent / "derived" / name).is_file()),
                "audit_artifacts": sorted(name for name in AUDIT_ARTIFACTS if (path.parent / name).is_file()), "explanation_audit": explanation,
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

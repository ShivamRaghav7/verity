"""Local FastAPI application and explicit administrator boundary."""

import copy
import hashlib
import json
import math
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from threading import RLock
from time import perf_counter

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .models import AskRequest, CorrectionCreate
from .service import StructuredAssistant
from .store import DomainError, Store

STATIC = Path(__file__).parent / "static"


class ArtifactStatusCache:
    """Validate configured artifact files once; recheck when watched bytes change.

    Health reads metadata and file hashes, never loads a neural model. Subsequent
    calls use cheap stat signatures, including artifact directories for new files.
    """

    def __init__(self, config_path):
        self.path = Path(config_path) if config_path else None
        self.watched = [self.path] if self.path else []
        self.signature = None
        self.cached = None
        self.lock = RLock()

    def _signature(self):
        result = []
        for path in self.watched:
            try:
                info = path.stat()
                result.append((str(path), info.st_mtime_ns, info.st_size))
            except OSError:
                result.append((str(path), None, None))
        return tuple(result)

    @staticmethod
    def provenance(manifest, manifest_hash=None):
        return {"artifact_purpose": manifest.get("purpose"), "dataset_id": manifest.get("dataset_id"),
                "model_id": manifest.get("model_id"), "model_revision": manifest.get("model_revision"),
                "review_policy": manifest.get("review_policy", "unspecified"),
                "human_reviewed_data": manifest.get("human_reviewed_data"),
                "completed_steps": manifest.get("completed_steps"),
                "selected_step": manifest.get("selection", {}).get("global_step"),
                "input_format_version": manifest.get("input_format_version"),
                "artifact_manifest_sha256": manifest_hash}

    @staticmethod
    def scope_runtime_evidence(path, calibration, watched):
        """Optional new policy evidence is file-validated without loading models."""
        if "routing_policy" not in calibration:
            return  # Preserve the archived legacy calibration contract.
        from .neural import file_sha256
        if (calibration.get("routing_policy") != "approved_compatible_then_scope_uncertainty_v1"
            or calibration.get("population") != "single_focal_question_correction_pairs"
            or calibration.get("test_rows_read") is not False):
            raise ValueError("Scope routing policy must describe validation-only approved compatible pairs.")
        source_root = Path(__file__).resolve().parents[1]
        source_names = {"editguard/comparison.py", "editguard/neural.py", "editguard/formatting.py",
                        "scripts/calibrate_scope.py", "scripts/calibrate_similarity.py"}
        source_hashes = calibration.get("policy_source_sha256")
        if not isinstance(source_hashes, dict) or set(source_hashes) != source_names:
            raise ValueError("Scope routing evidence lacks its exact policy source identities.")
        sources = [(source_root / name, expected) for name, expected in source_hashes.items()]
        sources.append((source_root / "scripts/calibrate_scope_runtime.py", calibration.get("script_sha256")))
        evidence = [(path.parent / "validation_probabilities.jsonl", calibration.get("probabilities_sha256")),
                    (path.parent / "selection_grid.json", calibration.get("selection_grid_sha256")), *sources]
        watched.extend(file for file, _ in evidence)
        for file, expected in evidence:
            if (not re.fullmatch(r"[0-9a-f]{64}", str(expected)) or not file.is_file()
                or file_sha256(file) != expected):
                raise ValueError(f"Scope routing evidence integrity failed: {file.name}.")
        total = calibration.get("accepted_validation_rows")
        if type(total) is not int or not 2 <= total <= 2000:
            raise ValueError("Scope policy must retain a finite accepted pair population.")
        for key in ("raw_pair_metrics", "compatibility_gated_metrics"):
            DevelopmentReportCache.scope_policy_metrics(calibration.get(key), total)
        if calibration.get("validation_operating_metrics") != calibration["compatibility_gated_metrics"]:
            raise ValueError("Scope validation metrics must disclose the combined deterministic and learned policy.")
        grid = json.loads((path.parent / "selection_grid.json").read_text(encoding="utf-8"))
        if (grid.get("rows") != total or grid.get("routing_policy") != calibration["routing_policy"]
            or any(grid.get(key) != calibration.get(key) for key in ("temperature", "threshold", "ambiguity_margin",
                "declared_costs", "raw_pair_metrics", "compatibility_gated_metrics"))):
            raise ValueError("Scope policy metrics disagree with the hashed selection grid.")
        records = [json.loads(line) for line in (path.parent / "validation_probabilities.jsonl").read_text(encoding="utf-8").splitlines() if line]
        hashes = calibration.get("accepted_row_sha256")
        if not isinstance(hashes, dict) or len(hashes) != total:
            raise ValueError("Scope policy lacks complete reviewed validation row identities.")
        seen = set()
        for record in records:
            row_id, probability = record.get("id"), record.get("scope_probability")
            if (row_id in seen or row_id not in hashes or record.get("row_sha256") != hashes[row_id]
                or not re.fullmatch(r"[0-9a-f]{64}", str(hashes[row_id])) or record.get("split") != "validation"
                or type(record.get("label")) is not int or record["label"] not in (0, 1)
                or isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("Scope policy probability records differ from reviewed validation evidence.")
            seen.add(row_id)
        if len(records) != total or seen != set(hashes):
            raise ValueError("Scope policy omitted or duplicated validation pairs.")
        if sum(record["label"] for record in records) != calibration["raw_pair_metrics"]["counts"]["positive"]:
            raise ValueError("Scope policy class counts differ from the saved probability population.")

    def snapshot(self):
        with self.lock:
            signature = self._signature()
            if self.cached is not None and signature == self.signature:
                return copy.deepcopy(self.cached)
            self.watched = [self.path] if self.path else []
            result = {"configuration": "disabled" if not self.path else "valid",
                      "configuration_error": None, "model_details": {
                          task: {"status": "not_configured", "loaded": False, "error": None}
                          for task in ("scope", "answer")},
                      "calibrations": {"scope": "not_configured", "similarity": "not_configured"}}
            result["calibration_sha256"] = {"scope": None, "similarity": None}
            if self.path:
                try:
                    from .formatting import INPUT_FORMAT_VERSION
                    from .neural import file_sha256, validate_artifact
                    config = json.loads(self.path.read_text(encoding="utf-8"))
                    if config.get("input_format_version") != INPUT_FORMAT_VERSION:
                        raise ValueError("Inference input format differs from runtime.")
                    for key in ("retrieval_k", "prompt_edit_k", "max_input_tokens", "max_new_tokens"):
                        if type(config.get(key)) is not int or config[key] <= 0:
                            raise ValueError("Inference retrieval and token budgets must be positive integers.")
                    if config.get("similarity_threshold") is not None:
                        raise ValueError("A similarity validation artifact is required instead of a bare threshold.")
                    result["configuration_sha256"] = file_sha256(self.path)
                    for task in ("scope", "answer"):
                        artifact = config.get(f"{task}_artifact")
                        if not artifact:
                            continue
                        root = Path(artifact)
                        self.watched.extend([root, root / "manifest.json", root / "model", root / "tokenizer"])
                        try:
                            raw_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
                            self.watched.extend(root / name for name in raw_manifest.get("files_sha256", raw_manifest.get("file_hashes", {})))
                            manifest = validate_artifact(root, task)
                            self.watched.extend(root / name for name in manifest.get("files_sha256", manifest.get("file_hashes", {})))
                            result["model_details"][task] = {"status": "trained", "loaded": False, "error": None,
                                "validation_basis": "configured_artifact_file_hashes",
                                **self.provenance(manifest, file_sha256(root / "manifest.json"))}
                        except Exception as exc:
                            result["model_details"][task].update(status="missing" if not (root / "manifest.json").is_file() else "invalid", error=str(exc))
                    for kind, key in (("scope", "calibration_artifact"), ("similarity", "similarity_calibration_artifact")):
                        if not config.get(key):
                            continue
                        calibration_path = Path(config[key])
                        self.watched.append(calibration_path)
                        try:
                            if kind == "similarity":
                                from .comparison import load_similarity_calibration
                                load_similarity_calibration(calibration_path, config)
                                self.watched.extend([calibration_path.parent / "validation_cosine_scores.jsonl",
                                    calibration_path.parent / "selection_grid.json", Path(__file__).parent / "comparison.py",
                                    Path(__file__).parent / "neural.py", Path(__file__).parent / "formatting.py",
                                    Path(__file__).resolve().parents[1] / "scripts/calibrate_similarity.py"])
                            else:
                                calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
                                if (result["model_details"]["scope"]["status"] != "trained"
                                    or calibration.get("purpose") != "validation_calibration" or calibration.get("split") != "validation"
                                    or calibration.get("status") != "frozen"
                                    or calibration.get("scope_manifest_sha256") != result["model_details"]["scope"].get("artifact_manifest_sha256")):
                                    raise ValueError("Scope calibration must be frozen validation selection bound to the trained artifact.")
                                for name in ("threshold", "ambiguity_margin", "temperature"):
                                    value = calibration.get(name)
                                    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                                        or (not 0 <= value <= 1 if name != "temperature" else value <= 0)):
                                        raise ValueError("Invalid scope calibration operating point.")
                                self.scope_runtime_evidence(calibration_path, calibration, self.watched)
                            result["calibrations"][kind] = "validated"
                            result["calibration_sha256"][kind] = file_sha256(calibration_path)
                        except Exception as exc:
                            result["calibrations"][kind] = "missing" if not calibration_path.is_file() else "invalid"
                            result.setdefault("calibration_errors", {})[kind] = str(exc)
                except Exception as exc:
                    result.update(configuration="invalid", configuration_error=str(exc))
            self.cached, self.signature = result, self._signature()
            return copy.deepcopy(result)


class DevelopmentReportCache:
    """Optional recorded validation results; no evaluation/model execution."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.watched = [self.path] if self.path else []
        self.signature = None
        self.cached = None
        self.lock = RLock()

    def _signature(self):
        result = []
        for path in self.watched:
            try:
                info = path.stat()
                result.append((str(path), info.st_mtime_ns, info.st_size))
            except OSError:
                result.append((str(path), None, None))
        return tuple(result)

    @staticmethod
    def rate(value):
        if not isinstance(value, dict) or not {"numerator", "denominator", "value"} <= value.keys():
            raise ValueError("Every reported rate requires numerator, denominator and value.")
        numerator, denominator, observed = value["numerator"], value["denominator"], value["value"]
        if (isinstance(numerator, bool) or not isinstance(numerator, (int, float)) or not math.isfinite(numerator)
                or type(denominator) is not int or not 0 <= numerator <= denominator):
            raise ValueError("Invalid rate counts.")
        if denominator == 0:
            if observed is not None:
                raise ValueError("A rate with zero denominator must be unavailable.")
        elif observed is None:
            if not isinstance(value.get("reason"), str) or not value["reason"].strip():
                raise ValueError("An unavailable metric with a nonzero population requires an explicit reason.")
        elif (isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed)
              or abs(observed - numerator / denominator) > 1e-8):
            raise ValueError("Rate value does not match its disclosed denominator.")

    @classmethod
    def scope_policy_metrics(cls, metrics, total, positives=None):
        if not isinstance(metrics, dict) or not isinstance(metrics.get("counts"), dict):
            raise ValueError("Scope policy requires explicit full-population confusion and clarification counts.")
        counts = metrics["counts"]
        required = ("total", "positive", "negative", "true_application", "false_application", "true_rejection",
                    "missed_positive", "clarification_positive", "clarification_negative")
        if any(type(counts.get(key)) is not int or not 0 <= counts[key] <= total for key in required):
            raise ValueError("Scope policy counts must be nonnegative integers within the pair population.")
        tp, fp, tn, fn, cp, cn = (counts[key] for key in required[3:])
        if (counts["total"] != total or counts["positive"] + counts["negative"] != total
            or tp + fn + cp != counts["positive"] or fp + tn + cn != counts["negative"]
            or (positives is not None and counts["positive"] != positives)):
            raise ValueError("Scope policy must retain all applicable, incompatible and clarified pairs.")
        expected = {"precision": (tp, tp + fp), "recall_including_clarification_as_missed": (tp, counts["positive"]),
            "false_application_rate": (fp, counts["negative"]), "coverage": (total - cp - cn, total),
            "clarification_rate": (cp + cn, total), "accuracy_including_clarification_as_unsuccessful": (tp + tn, total),
            "selective_accuracy": (tp + tn, total - cp - cn)}
        for key, (numerator, denominator) in expected.items():
            cls.rate(metrics.get(key))
            if (metrics[key]["numerator"] != numerator or metrics[key]["denominator"] != denominator
                or metrics[key]["value"] is None and denominator > 0):
                raise ValueError(f"Scope policy {key} differs from its full confusion and clarification counts.")

    def scope_policy(self, report):
        if "scope_policy" not in report:
            return
        from .neural import file_sha256
        policy = report["scope_policy"]
        if (not isinstance(policy, dict) or policy.get("routing_policy") != "approved_compatible_then_scope_uncertainty_v1"
            or policy.get("population") != "single_focal_question_correction_pairs"
            or policy.get("scope_manifest_sha256") != report["model_artifacts"]["scope"]["manifest_sha256"]
            or not re.fullmatch(r"[0-9a-f]{64}", str(policy.get("calibration_sha256")))):
            raise ValueError("Recorded scope policy must be bound to its exact scope model and pair-selection population.")
        path = Path(policy["calibration_path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.watched.append(path)
        if not path.is_file() or file_sha256(path) != policy["calibration_sha256"]:
            raise ValueError("Recorded scope policy calibration integrity failed.")
        calibration = json.loads(path.read_text(encoding="utf-8"))
        if (calibration.get("purpose") != "validation_calibration" or calibration.get("status") != "frozen"
            or calibration.get("split") != "validation" or calibration.get("dataset_id") != report["dataset_id"]):
            raise ValueError("Recorded scope policy must retain the same validation dataset.")
        for name in ("temperature", "threshold", "ambiguity_margin"):
            value = policy.get(name)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (not 0 <= value <= 1 if name != "temperature" else value <= 0)):
                raise ValueError("Recorded scope policy operating point is invalid.")
        for key in ("scope_manifest_sha256", "temperature", "threshold", "ambiguity_margin", "routing_policy",
                    "population", "raw_pair_metrics", "compatibility_gated_metrics"):
            if policy.get(key) != calibration.get(key):
                raise ValueError(f"Recorded scope policy differs from its hashed calibration: {key}.")
        for key in ("raw_pair_metrics", "compatibility_gated_metrics"):
            self.scope_policy_metrics(policy.get(key), report["questions"], report["oracle_questions"])
        ArtifactStatusCache.scope_runtime_evidence(path, calibration, self.watched)

    def snapshot(self, model_status):
        with self.lock:
            if self.cached is None or self._signature() != self.signature:
                self.watched = [self.path] if self.path else []
                report = None
                descriptor = {"status": "not_configured", "error": None}
                if self.path:
                    try:
                        from .neural import file_sha256
                        if self.path.stat().st_size > 2_000_000:
                            raise ValueError("Development summary exceeds the bounded 2 MB size.")
                        report = json.loads(self.path.read_text(encoding="utf-8"))
                        expected = {"artifact_type": "editguard_development_evaluation", "purpose": "validation_development",
                            "status": "completed", "review_policy": "automated_development", "human_reviewed_data": False,
                            "test_rows_read": False}
                        if any(report.get(key) != value for key, value in expected.items()):
                            raise ValueError("Only completed, explicitly automated development validation reports are accepted.")
                        if (not isinstance(report.get("dataset_id"), str) or not report["dataset_id"]
                            or type(report.get("questions")) is not int or not 1 <= report["questions"] <= 2000
                            or type(report.get("oracle_questions")) is not int or not 1 <= report["oracle_questions"] <= report["questions"]):
                            raise ValueError("Report populations must be finite and explicitly disclosed.")
                        for key in ("precision", "recall", "false_positive_rate", "brier", "ece"):
                            self.rate(report["scope"][key])
                        for key in ("accuracy", "direct_accuracy", "paraphrase_accuracy", "comparison_accuracy", "coverage"):
                            self.rate(report["oracle"][key])
                        seen = set()
                        from .comparison import CONFIGURATIONS
                        for item in report["configurations"]:
                            if item["configuration"] not in CONFIGURATIONS or item["configuration"] in seen:
                                raise ValueError("Comparison configurations must be known and unique.")
                            seen.add(item["configuration"])
                            for key in ("accuracy", "coverage", "out_of_scope_accuracy", "baseline_correct_retention", "false_application_rate"):
                                self.rate(item[key])
                            if item["accuracy"]["denominator"] != report["questions"] or item["coverage"]["denominator"] != report["questions"]:
                                raise ValueError("Accuracy and coverage must retain the full question denominator.")
                            for key in ("pending_review", "errors", "unavailable"):
                                if type(item[key]) is not int or not 0 <= item[key] <= report["questions"]:
                                    raise ValueError("Missing, failed and review-pending counts must remain explicit.")
                            for key in ("p50_ms", "p95_ms"):
                                value = item[key]
                                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                                    raise ValueError("Latency values must be finite nonnegative measurements or unavailable.")
                        if seen != set(CONFIGURATIONS):
                            raise ValueError("A matched development summary must include all five configurations.")
                        if (report["oracle"]["accuracy"]["denominator"] != report["oracle_questions"]
                            or report["oracle"]["coverage"]["denominator"] != report["oracle_questions"]
                            or sum(report["oracle"][key]["denominator"] for key in
                                   ("direct_accuracy", "paraphrase_accuracy", "comparison_accuracy")) != report["oracle_questions"]):
                            raise ValueError("Oracle routing requires its separate applicable-question denominator.")
                        for item in report["configurations"]:
                            out_of_scope = report["questions"] - report["oracle_questions"]
                            if (item["out_of_scope_accuracy"]["denominator"] != out_of_scope
                                or item["false_application_rate"]["denominator"] != out_of_scope
                                or item["baseline_correct_retention"]["denominator"] > out_of_scope):
                                raise ValueError("Out-of-scope correctness, application and retention need separate disclosed populations.")
                            if (item["p50_ms"] is None) != (item["p95_ms"] is None) or (
                                item["p50_ms"] is not None and item["p50_ms"] > item["p95_ms"]):
                                raise ValueError("Latency quantiles must be jointly available and ordered.")
                        if not isinstance(report["limitations"], list) or any(not isinstance(text, str) for text in report["limitations"]):
                            raise ValueError("Report limitations must be explicit text.")
                        for task in ("scope", "answer"):
                            artifact = report["model_artifacts"][task]
                            if (not re.fullmatch(r"[0-9a-f]{64}", str(artifact["manifest_sha256"]))
                                or type(artifact["completed_steps"]) is not int or artifact["completed_steps"] <= 0
                                or type(artifact["selected_step"]) is not int or not 1 <= artifact["selected_step"] <= artifact["completed_steps"]):
                                raise ValueError("Report trained-model identities must be explicit.")
                        self.scope_policy(report)
                        provenance = report["provenance"]
                        for key in ("config_sha256", "raw_manifest_sha256", "predictions_sha256"):
                            if not re.fullmatch(r"[0-9a-f]{64}", str(provenance[key])):
                                raise ValueError("Report lacks concrete provenance hashes.")
                        raw = Path(provenance["raw_manifest_path"])
                        if not raw.is_absolute():
                            raw = Path(__file__).resolve().parents[1] / raw
                        predictions = raw.parent / "predictions.jsonl"
                        self.watched.extend([raw, predictions])
                        if file_sha256(raw) != provenance["raw_manifest_sha256"] or file_sha256(predictions) != provenance["predictions_sha256"]:
                            raise ValueError("Raw validation evidence integrity failed.")
                        raw_manifest = json.loads(raw.read_text(encoding="utf-8"))
                        if (raw_manifest.get("purpose") != "validation_development" or raw_manifest.get("status") != "completed"
                            or raw_manifest.get("test_rows_read") is not False or raw_manifest.get("human_reviewed_data") is not False
                            or raw_manifest.get("review_policy") != "automated_development"
                            or raw_manifest.get("dataset_id") != report["dataset_id"] or raw_manifest.get("questions") != report["questions"]
                            or raw_manifest.get("configuration_sha256") != provenance["config_sha256"]
                            or raw_manifest.get("scorer_version") != provenance["scorer_version"]
                            or raw_manifest.get("predictions_sha256") != provenance["predictions_sha256"]):
                            raise ValueError("Raw manifest population or provenance differs from the summary.")
                        descriptor = {"status": "available", "error": None, "report_sha256": file_sha256(self.path)}
                    except Exception as exc:
                        descriptor = {"status": "missing" if not self.path.is_file() else "invalid", "error": str(exc)}
                        report = None
                self.cached, self.signature = (report, descriptor), self._signature()
            report, descriptor = copy.deepcopy(self.cached)
            if report:
                if model_status.get("restart_required"):
                    descriptor.update(status="incompatible", error="Configured inference dependencies changed after loading; restart before displaying this report against the current snapshot.")
                    return None, descriptor
                for task in ("scope", "answer"):
                    configured, measured = model_status["model_details"][task], report["model_artifacts"][task]
                    if (configured["status"] != "trained" or configured.get("artifact_manifest_sha256") != measured["manifest_sha256"]
                        or configured.get("completed_steps") != measured["completed_steps"] or configured.get("selected_step") != measured["selected_step"]):
                        descriptor.update(status="incompatible", error="Report model artifacts differ from the configured trained checkpoints.")
                        return None, descriptor
                if report.get("scope_policy") and (model_status.get("calibrations", {}).get("scope") != "validated"
                    or model_status.get("calibration_sha256", {}).get("scope") != report["scope_policy"]["calibration_sha256"]):
                    descriptor.update(status="incompatible", error="Report scope routing policy differs from the configured calibration.")
                    return None, descriptor
                descriptor["configuration_matches"] = model_status.get("configuration_sha256") == report["provenance"]["config_sha256"]
            return report, descriptor


def create_app(db_path=None, admin_token=None, comparison_runner=None, inference_config=None, evaluation_report=None):
    token = admin_token or os.environ.get("EDITGUARD_ADMIN_TOKEN", "local-demo-only")
    path = Path(db_path or os.environ.get("EDITGUARD_DB", "runtime/editguard.sqlite3"))
    path.parent.mkdir(parents=True, exist_ok=True)
    store = Store(path)
    store.seed()
    assistant = StructuredAssistant(store)
    neural_lock = RLock()
    runner = comparison_runner
    config_path = inference_config or os.environ.get("EDITGUARD_INFERENCE_CONFIG")
    artifact_status = ArtifactStatusCache(config_path)
    report_status = DevelopmentReportCache(evaluation_report or os.environ.get("EDITGUARD_EVALUATION_REPORT"))
    runner_validated = False
    load_error = None
    loaded_config_sha256 = None
    loaded_dependency_fingerprint = None

    def dependency_fingerprint(metadata):
        snapshot = {"configuration": metadata["configuration"],
            "configuration_sha256": metadata.get("configuration_sha256"),
            "models": {task: {"status": details["status"],
                "manifest_sha256": details.get("artifact_manifest_sha256")}
                for task, details in metadata["model_details"].items()},
            "calibrations": metadata["calibrations"], "calibration_sha256": metadata["calibration_sha256"]}
        return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _readiness():
        metadata = artifact_status.snapshot()
        changed_configuration = runner_validated and loaded_config_sha256 != metadata.get("configuration_sha256")
        changed_dependencies = runner_validated and loaded_dependency_fingerprint != dependency_fingerprint(metadata)
        restart_required = changed_configuration or changed_dependencies
        if restart_required:
            metadata["configuration_error"] = "Configured model, calibration or inference snapshot changed after loading. Restart the application to load that snapshot."
        details = metadata["model_details"]
        if runner:
            for task, model in (("scope", runner.scope), ("answer", runner.answer_model)):
                if model:
                    manifest = getattr(model, "manifest", None)
                    if runner_validated and manifest and not restart_required:
                        details[task] = {**details[task], **ArtifactStatusCache.provenance(manifest,
                            getattr(model, "artifact_manifest_sha256", details[task].get("artifact_manifest_sha256"))),
                            "status": "trained", "loaded": True, "validation_basis": "loaded_checkpoint"}
                    elif not runner_validated:
                        details[task].update(status="loaded_unvalidated", loaded=True,
                            error="Injected runner has no validated training-artifact evidence.")
        neural_state = "configuration_changed" if restart_required else "loaded" if runner else "unavailable" if load_error or metadata["configuration"] == "invalid" else "configured_not_loaded" if config_path else "disabled"
        modes = {"structured": {"status": "available", "available": True}}
        for mode in ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "learned"):
            if runner:
                eligible = (mode not in ("similarity_editor", "learned")
                    or (runner.answer_model and ((runner.scope and runner.calibration) if mode == "learned" else runner.similarity_threshold is not None)))
                status = "available" if eligible and not restart_required else "unavailable"
            else:
                eligible = metadata["configuration"] == "valid" and not load_error
                if mode in ("similarity_editor", "learned"):
                    eligible = eligible and details["answer"]["status"] == "trained" and (
                        details["scope"]["status"] == "trained" and metadata["calibrations"]["scope"] == "validated"
                        if mode == "learned" else metadata["calibrations"]["similarity"] == "validated")
                # from_config loads the configured artifacts and both calibrations,
                # so an invalid configured dependency prevents every neural mode.
                eligible = eligible and not any(item["status"] in ("missing", "invalid") for item in details.values()) and not any(
                    value in ("missing", "invalid") for value in metadata["calibrations"].values())
                status = "load_required" if eligible else "unavailable"
            modes[mode] = {"status": status, "available": True if status == "available" else None if status == "load_required" else False}
        return {**metadata, "models": {task: details[task]["status"] for task in details},
                "neural_inference": neural_state, "restart_required": restart_required,
                "load_error": load_error, "modes": modes}

    def readiness():
        with neural_lock:
            return _readiness()

    def neural_runner():
        nonlocal runner, runner_validated, load_error, loaded_config_sha256, loaded_dependency_fingerprint
        with neural_lock:
            if runner is None:
                if not config_path:
                    raise HTTPException(503, "Neural inference is not enabled. Set EDITGUARD_INFERENCE_CONFIG in the isolated ML environment.")
                from .comparison import ComparisonRunner
                try:
                    before_load = artifact_status.snapshot()
                    candidate = ComparisonRunner.from_config(config_path)
                    after_load = artifact_status.snapshot()
                    if dependency_fingerprint(before_load) != dependency_fingerprint(after_load):
                        raise ValueError("Configured dependencies changed while loading. Restart with a stable snapshot.")
                    runner = candidate
                    runner_validated = True
                    loaded_config_sha256 = after_load.get("configuration_sha256")
                    loaded_dependency_fingerprint = dependency_fingerprint(after_load)
                    load_error = None
                except Exception as exc:
                    load_error = str(exc)
                    raise HTTPException(503, f"Neural configuration could not load: {exc}") from exc
            return runner
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="EditGuard", version="0.3.0", lifespan=lifespan,
                  description="Controlled support knowledge editing; structured workflow and optional frozen neural comparisons.")
    app.state.store = store

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        return JSONResponse(status_code=exc.code, content={"detail": str(exc)})

    def administrator(x_admin_token: str = Header(default="")):
        if not secrets.compare_digest(x_admin_token, token):
            raise HTTPException(401, "Administrator token required.")

    @app.get("/api/health")
    def health():
        metadata = readiness()
        return {"status": "ok", "revision": store.revision(), **metadata,
                "answer_mode": "deterministic_structured", "final_evaluation": "pending"}

    @app.get("/api/facts")
    def facts():
        with store.snapshot():
            return {"facts": store.facts(), "revision": store.revision()}

    @app.get("/api/edits")
    def edits():
        with store.snapshot():
            return {"edits": store.edits(), "events": store.events(), "revision": store.revision()}

    @app.get("/api/conflicts")
    def conflicts():
        with store.snapshot():
            return {"conflicts": store.conflicts(), "revision": store.revision()}

    @app.post("/api/edits", status_code=201, dependencies=[Depends(administrator)])
    def create_edit(correction: CorrectionCreate):
        return store.create_correction(correction)

    @app.post("/api/edits/{edit_id}/approve", dependencies=[Depends(administrator)])
    def approve(edit_id: str):
        return store.approve(edit_id)

    @app.post("/api/edits/{edit_id}/withdraw", dependencies=[Depends(administrator)])
    def withdraw(edit_id: str):
        return store.withdraw(edit_id)

    @app.post("/api/ask")
    def ask(request: AskRequest):
        if request.mode == "structured":
            return assistant.ask(request)
        started = perf_counter()
        if readiness()["modes"][request.mode]["status"] == "unavailable":
            raise HTTPException(503, "Requested neural mode is unavailable. Inspect /api/health for missing, invalid or unconfigured dependencies.")
        comparison = neural_runner()
        with store.snapshot():
            facts, edits, conflicts, revision = store.facts(), store.edits(), store.conflicts(), store.revision()
        configuration = "trained_scope_editor" if request.mode == "learned" else request.mode
        context = request.model_dump(mode="json", exclude={"question", "mode"}, exclude_none=True)
        result = comparison.answer(request.question, context, facts, edits, configuration=configuration,
                                   revision=revision, conflicts=conflicts)
        if result["status"] == "unavailable":
            raise HTTPException(503, result["error"])
        before = result if configuration == "unchanged_base" else comparison.answer(
            request.question, context, facts, edits, configuration="unchanged_base", revision=revision)
        result.update(before_answer=before["answer"], before_answer_kind=before["answer_kind"],
                      before_trace=before, fact=None,
                      applicable_corrections=[edit for edit in edits if edit["edit_id"] == result["selected_edit_id"]],
                      explanation="Model-generated answer. Supplied sources and corrections do not verify generated prose.",
                      model_training=readiness()["model_details"],
                      api_latency_ms=round((perf_counter() - started) * 1000, 3))
        # Copy before_trace when it is the result itself, avoiding a JSON cycle.
        if before is result:
            result["before_trace"] = None
        return result

    @app.get("/api/evaluation")
    def evaluation():
        from .evaluation import run_slice_regression
        model_status = health()
        report, descriptor = report_status.snapshot(model_status)
        return {**run_slice_regression(), "revision": store.revision(), "model_readiness": model_status,
                "ml_results": report, "ml_report": descriptor, "final_evaluation": "pending"}

    @app.get("/api/export")
    def export():
        with store.snapshot():
            model_status = health()
            payload = {"schema_version": 1, "dataset": "controlled_fictional_demonstration",
                       "models": model_status["models"], "model_readiness": model_status,
                       "facts": store.facts(), "edits": store.edits(),
                       "events": store.events(), "revision": store.revision()}
        payload["evaluation"] = evaluation()
        return payload

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")
    return app

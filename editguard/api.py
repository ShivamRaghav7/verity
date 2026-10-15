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


class FinalReportCache(DevelopmentReportCache):
    """Read only completed, sealed final evidence; never run models or evaluation."""

    COUNTS = {"scope": 720, "answer": 288, "challenge": 96}

    def __init__(self, path, *, root=None):
        super().__init__(path)
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()

    @staticmethod
    def _json(path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate final evidence JSON key.")
                result[key] = value
            return result
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite final evidence.")))

    def _path(self, value):
        path = Path(value)
        path = (path if path.is_absolute() else self.root / path).resolve()
        if self.root not in path.parents:
            raise ValueError("Final evidence path escapes the workspace.")
        return path

    @staticmethod
    def _require(condition, message):
        if not condition:
            raise ValueError(message)

    def _population(self, population, count, *, oracle=False):
        from .comparison import CONFIGURATIONS
        self._require(population.get("questions") == count, "Final population count differs from its frozen denominator.")
        applicable = population.get("applicable_questions")
        self._require(type(applicable) is int and 0 <= applicable <= count, "Final applicability count is invalid.")
        rows = [population] if oracle else population["configurations"]
        expected = {"oracle_answer"} if oracle else set(CONFIGURATIONS)
        self._require(len(rows) == len(expected) and {row.get("configuration") for row in rows} == expected,
                      "Final comparisons require every configuration exactly once.")
        for row in rows:
            self._require(row.get("questions") == count and row.get("applicable_questions") == applicable,
                          "Final configuration omitted part of its cohort.")
            for key in ("accuracy", "coverage", "model_generated_answer_accuracy", "model_generated_answer_coverage"):
                self.rate(row[key])
                self._require(row[key]["denominator"] == count, "Final accuracy and coverage must retain all questions.")
            for key in ("pending_review", "abstentions", "errors", "unavailable"):
                self._require(type(row.get(key)) is int and 0 <= row[key] <= count,
                              "Final pending and nonanswer counts must remain explicit.")
            for key in ("out_of_scope_accuracy", "false_application_rate", "output_preservation"):
                self.rate(row[key])
                self._require(row[key]["denominator"] == count - applicable,
                              "Final out-of-scope denominator differs from its cohort.")
            self.rate(row["baseline_correct_retention"])
            self._require(row["baseline_correct_retention"]["denominator"] <= count - applicable,
                          "Final retention must use only baseline-correct out-of-scope questions.")
            latency = row["latency"]
            self._require(latency.get("total_count") == count and
                          type(latency.get("sample_count")) is int and type(latency.get("missing_count")) is int and
                          latency["sample_count"] >= 0 and latency["missing_count"] >= 0 and
                          latency["sample_count"] + latency["missing_count"] == count,
                          "Final latency population differs from its full cohort.")
            for key in ("p50_ms", "p95_ms"):
                value = row[key]
                self._require(value == latency[key] and (value is None or type(value) in (int, float)
                              and math.isfinite(value) and value >= 0), "Final latency quantiles are invalid.")
            if latency.get("complete") is True:
                self._require(latency["sample_count"] == count and row["p50_ms"] is not None and
                              row["p95_ms"] is not None and row["p50_ms"] <= row["p95_ms"],
                              "Complete latency requires the full measured population.")
            else:
                self._require(row["p50_ms"] is None and row["p95_ms"] is None,
                              "Incomplete latency cannot claim complete quantiles.")

    def _secondary(self, rows, count, *, oracle=False):
        from .comparison import CONFIGURATIONS
        expected = {"oracle_answer"} if oracle else set(CONFIGURATIONS)
        self._require(set(rows) == expected, "Secondary scoring configurations differ from the primary comparison.")
        for row in rows.values():
            item = row["all_requested_qa"]
            self._require(item.get("denominator") == count and all(type(item.get(key)) is int and item[key] >= 0
                          for key in ("correct", "incorrect", "pending_uncredited")) and
                          sum(item[key] for key in ("correct", "incorrect", "pending_uncredited")) == count,
                          "Secondary scoring must retain all outcomes in its separate denominator.")
            self.rate({"numerator": item["correct"], "denominator": count,
                       "value": item["accuracy_including_all_nonanswers_and_pending"]})

    def _compare_population(self, population, raw_metrics, *, oracle=False):
        rows = [population] if oracle else population["configurations"]
        for row in rows:
            recorded = raw_metrics["configurations"][row["configuration"]]
            for key in ("accuracy", "coverage", "model_generated_answer_accuracy", "model_generated_answer_coverage",
                        "baseline_correct_retention", "output_preservation", "retrieval_recall", "pending_review",
                        "abstentions", "errors", "unavailable", "status_counts", "latency"):
                self._require(row[key] == recorded[key], "Final summary metrics differ from the hashed raw metrics: " + key)
            self._require(row["out_of_scope_accuracy"] == recorded["subsets"]["out_of_scope"]["accuracy"]
                          and row["false_application_rate"] == recorded["false_correction_application_rate"],
                          "Final out-of-scope metrics differ from the hashed observations.")

    def _execution_recovery(self, report, bound, protocol):
        """Optional disclosures retain original observations and error counts."""
        recovery = report.get("execution_recovery", {})
        self._require(isinstance(recovery, dict) and set(recovery) <= {"io", "cuda_runtime_failure"},
                      "Unknown final execution-recovery disclosure.")
        if recovery.get("io") is not None:
            item = recovery["io"]
            declaration = self._json(bound(item["declaration_path"], item["declaration_sha256"]))
            bound(item["wrapper_path"], item["wrapper_sha256"])
            event_path = bound(item["events_path"], item["events_sha256"])
            self._require(declaration.get("purpose") == "frozen_final_io_recovery_declaration"
                          and declaration.get("protocol_sha256") == item["protocol_sha256"] == report["provenance"]["protocol_sha256"]
                          and declaration.get("wrapper_sha256") == item["wrapper_sha256"]
                          and declaration.get("frozen_evaluator_sha256") == item["frozen_evaluator_sha256"]
                          and item["frozen_evaluator_sha256"] == protocol["source_sha256"]["scripts/run_final_evaluation.py"]
                          and all(declaration.get(key) is False and item.get(key) is False for key in
                                  ("fresh_run_allowed", "inference_scoring_or_data_changes", "observations_rewritten"))
                          and item.get("events_frozen_at_summary_export") is True
                          and item.get("cause_status") == "not_established",
                          "IO recovery must be bound write-only evidence with its cause unresolved.")
            for key in ("max_write_attempts", "retry_delays_seconds", "mutable_write_targets"):
                self._require(item.get(key) == declaration.get(key), "IO retry bounds differ from their declaration.")
            self._require(item.get("max_write_attempts") == 8 and item.get("retry_delays_seconds") == [.05, .1, .2, .4, .4, .4, .4],
                          "IO retries must retain the finite declared schedule.")
            entries = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            counts = {Path(name).name: 0 for name in item["mutable_write_targets"]}
            delays = []
            for entry in entries:
                self._require(entry.get("declaration_sha256") == item["declaration_sha256"]
                              and entry.get("frozen_evaluator_sha256") == item["frozen_evaluator_sha256"]
                              and entry.get("inference_scoring_or_data_changes") is False,
                              "IO event differs from its recorded wrapper.")
                for retry in entry["write_retries"]:
                    target, attempt = retry.get("target"), retry.get("failed_attempt")
                    self._require(target in counts and type(attempt) is int and 1 <= attempt < 8
                                  and retry.get("delay_seconds") == item["retry_delays_seconds"][attempt - 1]
                                  and retry.get("error_type") == "PermissionError", "IO event contains an undeclared retry.")
                    counts[target] += 1
                    delays.append(retry["delay_seconds"])
            self._require(len(entries) == item["event_records"] and len(entries) > 0
                          and counts == item["retry_counts_by_target"] and sum(counts.values()) == item["recorded_write_retries"]
                          and math.isclose(math.fsum(delays), item["recorded_retry_delay_seconds"], abs_tol=1e-12),
                          "IO retry disclosure differs from the hashed event log.")
        if recovery.get("cuda_runtime_failure") is not None:
            item = recovery["cuda_runtime_failure"]
            disclosure = self._json(bound(item["disclosure_path"], item["disclosure_sha256"]))
            self._require({key: value for key, value in item.items()
                           if key not in ("disclosure_path", "disclosure_sha256", "original_results_replaced", "limitation")} == disclosure
                          and disclosure.get("purpose") == "final_runtime_failure_disclosure"
                          and disclosure.get("protocol_sha256") == report["provenance"]["protocol_sha256"]
                          and disclosure.get("observations_committed") == 4464
                          and disclosure.get("persistent_model_or_data_changes") is False
                          and item.get("original_results_replaced") is False
                          and isinstance(item.get("limitation"), str) and bool(item["limitation"].strip()),
                          "Runtime failure disclosure differs from its bound sidecar.")
            matched_status = {}
            for row in report["configurations"]:
                for status, count in row["status_counts"].items():
                    matched_status[status] = matched_status.get(status, 0) + count
            self._require(disclosure.get("status_by_kind") == {"matched": matched_status, "oracle": report["oracle"]["status_counts"]},
                          "Runtime failure disclosure must preserve original matched and oracle outcomes.")
            self._require(disclosure.get("original_gpu_instance_post_inference_fingerprint") == "unavailable_cuda_unknown_error"
                          and isinstance(disclosure.get("closure_recovery_policy"), str)
                          and isinstance(disclosure.get("oracle_measurement_policy"), str),
                          "Runtime failure must disclose the unavailable original GPU fingerprint and retained oracle errors.")

    def _validate(self, report):
        require = self._require
        expected = {"schema_version": 1, "artifact_type": "editguard_final_evaluation", "purpose": "frozen_final_evaluation",
                    "status": "completed", "review_policy": "automated_final", "human_reviewed_data": False,
                    "test_rows_read": True, "challenge_rows_read": True, "test_frozen": True,
                    "final_test_tuning_performed": False, "questions": 816, "oracle_questions": 384,
                    "population_counts": self.COUNTS}
        require(all(report.get(key) == value for key, value in expected.items()),
                "Only the complete frozen, automatically audited final population is accepted.")
        require(isinstance(report.get("dataset_id"), str) and bool(report["dataset_id"]), "Final dataset identity is required.")
        provenance = report["provenance"]
        evidence = provenance["evidence_files_sha256"]
        require(isinstance(evidence, dict) and bool(evidence), "Final evidence requires a concrete hashed file inventory.")
        inventory = {}
        for name, digest in evidence.items():
            path = self._path(name)
            require(path not in inventory and re.fullmatch(r"[0-9a-f]{64}", str(digest)) is not None,
                    "Final evidence inventory contains duplicate paths or invalid hashes.")
            inventory[path] = digest
        self.watched.extend(inventory)
        def bound(value, digest):
            path = self._path(value)
            require(inventory.get(path) == digest, "Final provenance is absent from its evidence inventory.")
            return path
        protocol_path = bound(provenance["protocol_path"], provenance["protocol_sha256"])
        lock_path = bound(provenance["test_lock_path"], provenance["test_lock_sha256"])
        packet_path = bound(provenance["packet_manifest_path"], provenance["packet_manifest_sha256"])
        raw_path = bound(provenance["raw_manifest_path"], provenance["raw_manifest_sha256"])
        ledger_path = bound(provenance["ledger_path"], provenance["ledger_sha256"])
        bound(provenance["config_path"], provenance["config_sha256"])
        for filename in ("predictions.jsonl", "oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json"):
            bound(raw_path.parent / filename, provenance[filename.removesuffix(".jsonl").removesuffix(".json") + "_sha256"])
        for path, digest in inventory.items():
            actual = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    actual.update(chunk)
            require(actual.hexdigest() == digest, "Final evidence integrity failed: " + path.name)
        protocol, lock, packet, raw, ledger = (self._json(path) for path in (protocol_path, lock_path, packet_path, raw_path, ledger_path))
        require(protocol.get("purpose") == "frozen_final_evaluation_protocol" and protocol.get("status") == "frozen"
                and protocol.get("protocol_id") == provenance["protocol_id"]
                and protocol.get("configuration", {}).get("sha256") == provenance["config_sha256"]
                and protocol.get("review_policy") == "automated_final" and protocol.get("human_reviewed_data") is False,
                "Final protocol differs from the summary.")
        from .comparison import CONFIGURATIONS
        evaluation = protocol["evaluation"]
        require(evaluation.get("question_counts") == self.COUNTS and evaluation.get("configurations") == list(CONFIGURATIONS)
                and evaluation.get("max_final_runs") == 1 and evaluation.get("matched_populations") == ["scope", "challenge"]
                and evaluation.get("oracle_populations") == ["answer", "challenge"],
                "Final protocol population or comparison rules differ.")
        require(lock.get("purpose") == "final_packet_lock_before_inference" and lock.get("status") == "frozen"
                and lock.get("protocol_sha256") == provenance["protocol_sha256"] and lock.get("counts") == self.COUNTS
                and lock.get("packet_manifest_sha256") == provenance["packet_manifest_sha256"]
                and lock.get("inference_started") is False and lock.get("review_counts") == {"accepted": 1104, "flagged": 0}
                and lock.get("review_policy") == "automated_final" and lock.get("human_reviewed_data") is False,
                "Final packet lacks its pre-inference automated audit lock.")
        require(packet.get("purpose") == "frozen_final_evaluation_packet" and packet.get("protocol_sha256") == provenance["protocol_sha256"]
                and packet.get("counts") == self.COUNTS and packet.get("files") == provenance["packet_files_sha256"]
                and packet.get("test_frozen") is True and packet.get("eligible_for_final_evaluation") is True,
                "Final packet differs from its frozen provenance.")
        for name, digest in packet["files"].items():
            require(Path(name).name == name, "Final packet contains an escaping filename.")
            bound(packet_path.parent / name, digest)
        require(raw.get("purpose") == "frozen_final_evaluation" and raw.get("status") == "completed"
                and raw.get("review_policy") == "automated_final" and raw.get("human_reviewed_data") is False
                and raw.get("protocol_sha256") == provenance["protocol_sha256"] and raw.get("protocol_id") == provenance["protocol_id"]
                and raw.get("configuration_sha256") == provenance["config_sha256"] and raw.get("population_counts") == self.COUNTS
                and raw.get("questions") == 816 and raw.get("matched_prediction_records") == 4080 and raw.get("oracle_records") == 384
                and raw.get("packet_manifest_sha256") == provenance["packet_manifest_sha256"]
                and raw.get("final_test_tuning_performed") is False and raw.get("fresh_final_runs") == 1
                and raw.get("base_weights_unchanged") is True
                and raw.get("base_parameter_sha256_before") == raw.get("base_parameter_sha256_after") == provenance["base_parameter_sha256"],
                "Final raw run is incomplete or differs from the declared comparison.")
        require(provenance.get("base_weights_unchanged") is True and ledger.get("status") == "completed"
                and ledger.get("fresh_final_runs") == 1 and ledger.get("result_manifest_sha256") == provenance["raw_manifest_sha256"],
                "Final run is not sealed by its single-run ledger.")
        for task in ("scope", "answer"):
            artifact = report["model_artifacts"][task]
            model_path = bound(Path(artifact["artifact_path"]) / "manifest.json", artifact["manifest_sha256"])
            model = self._json(model_path)
            require(model.get("purpose") == "trained" and model.get("status") == "completed" and model.get("task") == task
                    and model.get("completed_steps") == artifact["completed_steps"]
                    and model.get("selection", {}).get("global_step") == artifact["selected_step"]
                    and model.get("selected_model_state_sha256") == artifact["selected_model_state_sha256"]
                    and model.get("model_id") == artifact["model_id"] and model.get("model_revision") == artifact["model_revision"]
                    and protocol["models"][task]["manifest_sha256"] == artifact["manifest_sha256"]
                    and type(artifact["completed_steps"]) is int and 1 <= artifact["selected_step"] <= artifact["completed_steps"],
                    "Final trained artifact identity differs from its export.")
            for name, digest in model["files_sha256"].items():
                bound(model_path.parent / name, digest)
        self._population({"questions": report["questions"], "applicable_questions": report["matched_applicable_questions"],
                          "configurations": report["configurations"]}, 816)
        require(set(report["by_population"]) == {"scope", "challenge"}, "Final matched cohorts must remain separate.")
        for name in ("scope", "challenge"):
            self._population(report["by_population"][name], self.COUNTS[name])
        require(sum(item["applicable_questions"] for item in report["by_population"].values()) == report["matched_applicable_questions"],
                "Final cohort applicability counts differ from the combined population.")
        self._population(report["oracle"], 384, oracle=True)
        require(set(report["oracle"]["by_population"]) == {"answer", "challenge"}, "Final oracle cohorts must remain separate.")
        for name in ("answer", "challenge"):
            self._population(report["oracle"]["by_population"][name], self.COUNTS[name], oracle=True)
        for key in ("precision", "recall", "false_positive_rate", "brier", "ece"):
            self.rate(report["scope"][key])
        raw_metrics = self._json(raw_path.parent / "metrics.json")
        self._compare_population(report, raw_metrics)
        for name in ("scope", "challenge"):
            self._compare_population(report["by_population"][name], raw_metrics["by_population"][name])
        self._compare_population(report["oracle"], raw_metrics["oracle_answer_diagnostic"], oracle=True)
        for name in ("answer", "challenge"):
            self._compare_population(report["oracle"]["by_population"][name], raw_metrics["oracle_answer_by_population"][name], oracle=True)
        require(report["scope_full_metrics"] == raw_metrics["configurations"]["trained_scope_editor"]["scope"]
                and all(report["scope"][key] == report["scope_full_metrics"][key]
                        for key in ("precision", "recall", "false_positive_rate", "brier", "ece")),
                "Final scope diagnostics differ from the raw metrics.")
        secondary = report["secondary_short_qa"]
        require(secondary == self._json(raw_path.parent / "short_qa_metrics.json"),
                "Final secondary scores differ from their hashed separate artifact.")
        require(secondary.get("purpose") == "locked_final_secondary_short_qa" and secondary.get("primary_results_replaced") is False
                and secondary.get("post_hoc_policy") is False and secondary.get("protocol_sha256") == provenance["protocol_sha256"],
                "Final secondary scoring must retain its locked, separate definition.")
        self._secondary(secondary["configurations"], 816)
        for name in ("scope", "challenge"):
            self._secondary(secondary["by_population"][name], self.COUNTS[name])
        self._secondary(secondary["oracle"], 384, oracle=True)
        for name in ("answer", "challenge"):
            self._secondary(secondary["oracle_by_population"][name], self.COUNTS[name], oracle=True)
        require(isinstance(report.get("limitations"), list) and bool(report["limitations"])
                and all(isinstance(text, str) and bool(text.strip()) for text in report["limitations"]),
                "Final limitations must remain explicit.")
        self._execution_recovery(report, bound, protocol)
        if report.get("csv_export"):
            record = report["csv_export"]
            path = self._path(record["path"])
            self.watched.append(path)
            require(re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])) is not None
                    and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], "Final CSV export integrity failed.")

    def snapshot(self, model_status):
        with self.lock:
            if self.cached is None or self._signature() != self.signature:
                self.watched = [self.path] if self.path else []
                report, descriptor = None, {"status": "not_measured", "error": None}
                if self.path and self.path.is_file():
                    try:
                        self._require(self.path.stat().st_size <= 2_000_000, "Final summary exceeds the bounded 2 MB size.")
                        report = self._json(self.path)
                        self._validate(report)
                        descriptor = {"status": "available", "error": None,
                                      "report_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}
                    except Exception as exc:
                        report, descriptor = None, {"status": "invalid", "error": str(exc)}
                self.cached, self.signature = (report, descriptor), self._signature()
            report, descriptor = copy.deepcopy(self.cached)
            if report:
                if model_status.get("restart_required") or model_status.get("configuration_sha256") != report["provenance"]["config_sha256"]:
                    return None, {**descriptor, "status": "incompatible", "error": "Final evidence differs from the configured inference snapshot."}
                for task in ("scope", "answer"):
                    model = model_status.get("model_details", {}).get(task, {})
                    measured = report["model_artifacts"][task]
                    if model.get("status") != "trained" or model.get("artifact_manifest_sha256") != measured["manifest_sha256"]:
                        return None, {**descriptor, "status": "incompatible", "error": "Final model artifacts differ from the configured trained checkpoints."}
                descriptor["configuration_matches"] = True
            return report, descriptor


class OracleRecoveryReportCache(FinalReportCache):
    """Separate fixed-configuration operational supplement; originals stay visible."""

    def _interrupted(self, path, original):
        """Describe a bound attempted process without publishing recovery scores."""
        require = self._require
        require(path.stat().st_size <= 2_000_000, "Interrupted recovery metadata exceeds its bounded size.")
        manifest = self._json(path)
        provenance = original["provenance"]
        expected_originals = {"manifest.json": provenance["raw_manifest_sha256"],
            "oracle_predictions.jsonl": provenance["oracle_predictions_sha256"], "metrics.json": provenance["metrics_sha256"],
            "short_qa_metrics.json": provenance["short_qa_metrics_sha256"]}
        require(manifest.get("purpose") == "supplemental_oracle_runtime_recovery" and manifest.get("status") == "interrupted"
                and manifest.get("primary_results_replaced") is False and manifest.get("original_records_unchanged") is True
                and manifest.get("selection_used_correctness") is False and manifest.get("human_reviewed_data") is False
                and manifest.get("protocol_sha256") == provenance["protocol_sha256"]
                and manifest.get("configuration_sha256") == provenance["config_sha256"]
                and manifest.get("original_files_sha256") == expected_originals
                and manifest.get("oracle_questions") == 384 and manifest.get("selected_runtime_errors") == 319,
                "Interrupted recovery differs from the preserved original final evidence.")
        def bound(file, digest):
            file = self._path(file)
            self.watched.append(file)
            require(re.fullmatch(r"[0-9a-f]{64}", str(digest)) is not None
                    and hashlib.sha256(file.read_bytes()).hexdigest() == digest,
                    "Interrupted recovery integrity failed: " + file.name)
            return file
        for name, digest in expected_originals.items():
            bound(path.parent / "source" / name, digest)
        declaration_path = bound(path.parent / "source/declaration.json", manifest["declaration_sha256"])
        declaration = self._json(declaration_path)
        require(declaration.get("purpose") == "frozen_oracle_runtime_recovery_declaration" and declaration.get("status") == "frozen"
                and declaration.get("protocol_sha256") == provenance["protocol_sha256"]
                and declaration.get("configuration_sha256") == provenance["config_sha256"]
                and declaration.get("original_files_sha256") == expected_originals
                and declaration.get("maximum_new_generation_calls") == 319 and declaration.get("passes") == 1
                and declaration.get("primary_results_replaced") is False
                and declaration.get("training_or_parameter_selection") is False
                and declaration.get("supplement_source_sha256") == manifest["supplement_source_sha256"],
                "Interrupted recovery lacks its fixed-configuration declaration.")
        bound(declaration["supplement_source_path"], manifest["supplement_source_sha256"])
        require(set(manifest["outputs_sha256"]) == {"recovery_predictions.jsonl", "combined_oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json"},
                "Interrupted recovery output inventory is incomplete.")
        for name, digest in manifest["outputs_sha256"].items():
            bound(path.parent / name, digest)
        ledger_path = self._path(declaration["ledger_path"])
        self.watched.append(ledger_path)
        ledger = self._json(ledger_path)
        require(ledger.get("status") == "interrupted" and ledger.get("binding") == {
            "declaration_sha256": manifest["declaration_sha256"], "protocol_sha256": provenance["protocol_sha256"],
            "original_oracle_sha256": provenance["oracle_predictions_sha256"]},
            "Interrupted recovery ledger differs from its declared original evidence.")
        attempts = manifest.get("durable_attempt_reservations")
        require(type(attempts) is int and 0 <= attempts <= 319, "Interrupted recovery attempt count is invalid.")
        metrics = self._json(path.parent / "metrics.json")
        require(metrics.get("original_denominator") == 384 and metrics.get("supplemental_records") == attempts
                and metrics.get("not_attempted_selected_count") == manifest.get("not_attempted_selected_count") == 319 - attempts,
                "Interrupted recovery must account for attempted and unattempted generations separately.")
        failure = ledger.get("failure", {})
        return {"status": "interrupted", "error": f"A supplemental process was attempted and interrupted after {attempts} of 319 durable generation reservations. No completed recovery measurement is available; all original oracle errors remain unchanged.",
                "durable_attempt_reservations": attempts, "maximum_new_generation_calls": 319,
                "failure_type": failure.get("type"), "failure_reason": failure.get("reason"),
                "interrupted_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def _validate_recovery(self, manifest, original):
        require = self._require
        require(manifest.get("schema_version") == 1 and manifest.get("purpose") == "supplemental_oracle_runtime_recovery"
                and manifest.get("status") == "completed" and manifest.get("primary_results_replaced") is False
                and manifest.get("review_policy") == "automated_final" and manifest.get("human_reviewed_data") is False
                and manifest.get("original_records_unchanged") is True and manifest.get("selection_used_correctness") is False,
                "Only completed, separately named runtime-recovery evidence is accepted.")
        provenance = original["provenance"]
        require(manifest.get("protocol_sha256") == provenance["protocol_sha256"]
                and manifest.get("configuration_sha256") == provenance["config_sha256"]
                and manifest.get("oracle_questions") == 384 and manifest.get("selected_runtime_errors") == 319
                and manifest.get("maximum_new_generation_calls") == 319,
                "Oracle recovery differs from the original frozen model configuration or cohort.")
        def bound(value, digest):
            path = self._path(value)
            self.watched.append(path)
            require(re.fullmatch(r"[0-9a-f]{64}", str(digest)) is not None,
                    "Supplement evidence requires concrete SHA-256 identities.")
            observed = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    observed.update(chunk)
            require(observed.hexdigest() == digest, "Oracle recovery integrity failed: " + path.name)
            return path
        raw = self._path(provenance["raw_manifest_path"])
        expected_originals = {"manifest.json": provenance["raw_manifest_sha256"],
            "oracle_predictions.jsonl": provenance["oracle_predictions_sha256"],
            "metrics.json": provenance["metrics_sha256"], "short_qa_metrics.json": provenance["short_qa_metrics_sha256"]}
        require(manifest.get("original_files_sha256") == expected_originals,
                "Oracle recovery must preserve the exact original final evidence.")
        for name, digest in expected_originals.items():
            bound(raw.parent / name, digest)
            bound(self.path.parent / "source" / name, digest)
        declaration_path = bound(self.path.parent / "source/declaration.json", manifest["declaration_sha256"])
        declaration = self._json(declaration_path)
        require(declaration.get("purpose") == "frozen_oracle_runtime_recovery_declaration" and declaration.get("status") == "frozen"
                and declaration.get("protocol_sha256") == provenance["protocol_sha256"]
                and declaration.get("configuration_sha256") == provenance["config_sha256"]
                and declaration.get("original_files_sha256") == expected_originals
                and declaration.get("oracle_questions") == 384 and declaration.get("selected_questions") == 319
                and declaration.get("maximum_new_generation_calls") == 319 and declaration.get("passes") == 1
                and declaration.get("primary_results_replaced") is False and declaration.get("training_or_parameter_selection") is False
                and declaration.get("successful_original_records_reexecuted") is False
                and self._path(declaration["output_directory"]) == self.path.parent.resolve(),
                "Oracle recovery declaration must retain the original cohort and finite status-only selection.")
        selected = declaration["selected"]
        require(isinstance(selected, list) and len(selected) == 319 and len({row["id"] for row in selected}) == 319
                and all(row.get("population") in ("answer", "challenge") for row in selected),
                "Oracle recovery selection contains duplicate or incompatible question identities.")
        bound(declaration["supplement_source_path"], manifest["supplement_source_sha256"])
        require(declaration.get("supplement_source_sha256") == manifest["supplement_source_sha256"],
                "Oracle recovery source differs from its declaration.")
        ledger_path = self._path(declaration["ledger_path"])
        self.watched.append(ledger_path)
        ledger = self._json(ledger_path)
        require(ledger.get("status") == "completed" and ledger.get("result_manifest_sha256") == hashlib.sha256(self.path.read_bytes()).hexdigest()
                and ledger.get("binding") == {"declaration_sha256": manifest["declaration_sha256"],
                    "protocol_sha256": provenance["protocol_sha256"], "original_oracle_sha256": provenance["oracle_predictions_sha256"]},
                "Oracle recovery is not sealed by its separate execution ledger.")
        outputs = manifest["outputs_sha256"]
        require(set(outputs) == {"recovery_predictions.jsonl", "combined_oracle_predictions.jsonl", "metrics.json", "short_qa_metrics.json"},
                "Oracle recovery output inventory is incomplete.")
        for name, digest in outputs.items():
            bound(self.path.parent / name, digest)
        metrics = self._json(self.path.parent / "metrics.json")
        secondary = self._json(self.path.parent / "short_qa_metrics.json")
        require(metrics.get("purpose") == "supplemental_oracle_runtime_recovery" and metrics.get("primary_results_replaced") is False
                and metrics.get("original_denominator") == 384 and metrics.get("selected_runtime_error_count") == 319
                and metrics.get("original_successful_or_other_unselected_records_preserved") == 65
                and set(metrics["configurations"]) == {"oracle_answer"}
                and secondary.get("purpose") == "supplemental_oracle_runtime_recovery_secondary"
                and secondary.get("primary_results_replaced") is False,
                "Recovery metrics must keep all original successes and a separate full-cohort denominator.")
        for key in ("durable_attempt_reservations", "interrupted_reservation_count", "new_runtime_error_count", "not_attempted_selected_count"):
            require(type(manifest.get(key)) is int and 0 <= manifest[key] <= 319,
                    "Recovery attempt and failure counts must remain explicit and finite.")
        require(manifest["durable_attempt_reservations"] == 319 and manifest["not_attempted_selected_count"] == 0
                and metrics.get("supplemental_records") == 319
                and all(metrics.get(key) == manifest[key] for key in ("interrupted_reservation_count", "new_runtime_error_count", "not_attempted_selected_count")),
                "Completed recovery must account for every selected attempt, including interrupted reservations.")
        expected_base = provenance["base_parameter_sha256"]
        require(declaration.get("expected_base_parameter_fingerprint_sha256") == expected_base
                and isinstance(manifest.get("sessions"), list) and bool(manifest["sessions"]),
                "Recovery requires fixed base parameters and explicit process sessions.")
        for session in manifest["sessions"]:
            require(session.get("base_parameter_sha256_before") in (None, expected_base)
                    and session.get("base_parameter_sha256_after") in (None, expected_base),
                    "Recovery session parameters differ from the frozen base.")
        require(manifest["sessions"][-1].get("base_parameter_sha256_before") == expected_base
                and manifest["sessions"][-1].get("base_parameter_sha256_after") == expected_base,
                "Completed recovery requires its own before/after base checks; these do not establish the lost original GPU state.")
        def oracle_population(group, count):
            require(group.get("question_count") == count and group.get("record_count") == count
                    and set(group["configurations"]) == {"oracle_answer"}, "Recovery metrics lost oracle cohort observations.")
            current = group["configurations"]["oracle_answer"]
            require(current.get("count") == count and current.get("counts", {}).get("total") == count
                    and current["counts"].get("missing_records") == 0, "Recovery metrics omitted nonanswers.")
            result = {"configuration": "oracle_answer", "questions": count, "applicable_questions": count,
                **{key: current[key] for key in ("accuracy", "coverage", "model_generated_answer_accuracy", "model_generated_answer_coverage",
                    "baseline_correct_retention", "output_preservation", "retrieval_recall", "pending_review", "abstentions", "errors", "unavailable", "status_counts", "latency")},
                "out_of_scope_accuracy": current["subsets"]["out_of_scope"]["accuracy"],
                "false_application_rate": current["false_correction_application_rate"],
                "p50_ms": current["latency"]["p50_ms"], "p95_ms": current["latency"]["p95_ms"]}
            for style in ("direct", "paraphrase", "comparison", "challenge"):
                result[style + "_accuracy"] = current["subsets"].get(style, {"accuracy": {"numerator": 0, "denominator": 0, "value": None}})["accuracy"]
                self.rate(result[style + "_accuracy"])
            require(sum(result[style + "_accuracy"]["denominator"] for style in ("direct", "paraphrase", "comparison", "challenge")) == count,
                    "Recovery question-type populations must retain every oracle question.")
            self._population(result, count, oracle=True)
            return result
        oracle = oracle_population(metrics, 384)
        require(set(metrics["by_population"]) == {"answer", "challenge"}
                and set(secondary["by_population"]) == {"answer", "challenge"}, "Recovery cohorts must remain separate.")
        oracle["by_population"] = {name: oracle_population(metrics["by_population"][name], self.COUNTS[name]) for name in ("answer", "challenge")}
        self._secondary(secondary["configurations"], 384, oracle=True)
        for name in ("answer", "challenge"):
            self._secondary(secondary["by_population"][name], self.COUNTS[name], oracle=True)
        require(isinstance(manifest.get("limitations"), list) and bool(manifest["limitations"])
                and all(isinstance(value, str) and bool(value.strip()) for value in manifest["limitations"]),
                "Recovery limitations must remain explicit.")
        return {**manifest, "oracle": oracle, "secondary_short_qa": secondary,
                "provenance": {"manifest_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
                    "manifest_path": str(self.path), "declaration_sha256": manifest["declaration_sha256"],
                    "original_raw_manifest_sha256": provenance["raw_manifest_sha256"],
                    "protocol_sha256": provenance["protocol_sha256"], "configuration_sha256": provenance["config_sha256"]}}

    def snapshot(self, original):
        with self.lock:
            if original is None:
                return None, {"status": "original_unavailable", "error": "Compatible original final evidence is required before a recovery supplement can be displayed."}
            binding = (original["provenance"]["protocol_sha256"], original["provenance"]["config_sha256"], original["provenance"]["raw_manifest_sha256"])
            if self.cached is None or self._signature() != self.signature or getattr(self, "original_binding", None) != binding:
                interrupted_path = self.path.with_name("interrupted_manifest.json") if self.path else None
                self.watched = [self.path, interrupted_path] if self.path else []
                report, descriptor = None, {"status": "not_measured", "error": None}
                if self.path and self.path.is_file():
                    try:
                        self._require(self.path.stat().st_size <= 2_000_000, "Oracle recovery manifest exceeds its bounded size.")
                        report = self._validate_recovery(self._json(self.path), original)
                        descriptor = {"status": "available", "error": None, "manifest_sha256": report["provenance"]["manifest_sha256"]}
                    except Exception as exc:
                        report, descriptor = None, {"status": "invalid", "error": str(exc)}
                elif interrupted_path and interrupted_path.is_file():
                    try:
                        descriptor = self._interrupted(interrupted_path, original)
                    except Exception as exc:
                        descriptor = {"status": "invalid", "error": str(exc)}
                self.cached, self.signature, self.original_binding = (report, descriptor), self._signature(), binding
            return copy.deepcopy(self.cached)


def create_app(db_path=None, admin_token=None, comparison_runner=None, inference_config=None, evaluation_report=None,
               final_evaluation_report=None, oracle_recovery_report=None):
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
    final_report_status = FinalReportCache(final_evaluation_report or os.environ.get("EDITGUARD_FINAL_EVALUATION_REPORT")
                                         or Path(__file__).resolve().parents[1] / "reports/final/summary.json")
    oracle_recovery_status = OracleRecoveryReportCache(oracle_recovery_report or os.environ.get("EDITGUARD_ORACLE_RECOVERY_REPORT")
        or Path(__file__).resolve().parents[1] / "reports/final/oracle-runtime-recovery/manifest.json")
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
        _, final_descriptor = final_report_status.snapshot(metadata)
        return {"status": "ok", "revision": store.revision(), **metadata,
                "answer_mode": "deterministic_structured",
                "final_evaluation": "pending" if final_descriptor["status"] == "not_measured" else final_descriptor["status"],
                "final_report": final_descriptor}

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
                "ml_results": report, "ml_report": descriptor, "final_evaluation": model_status["final_evaluation"]}

    @app.get("/api/evaluation/final")
    def final_evaluation():
        metadata = readiness()
        report, descriptor = final_report_status.snapshot(metadata)
        return {"status": descriptor["status"], "report": report, "descriptor": descriptor,
                "disclaimer": "Recorded frozen test evidence only. This read does not execute inference or training. Labels and challenge wording received automated review; no independent human review is claimed."}

    @app.get("/api/evaluation/final/export/{kind}")
    def final_export(kind: str):
        if kind not in ("summary", "csv"):
            raise HTTPException(404, "Unknown final export.")
        report, descriptor = final_report_status.snapshot(readiness())
        if report is None:
            raise HTTPException(409, descriptor.get("error") or "Completed compatible final evidence is not available.")
        if kind == "summary":
            return FileResponse(final_report_status.path, media_type="application/json", filename="editguard-final-summary.json")
        if not report.get("csv_export"):
            raise HTTPException(404, "No final CSV export is recorded.")
        return FileResponse(final_report_status._path(report["csv_export"]["path"]), media_type="text/csv",
                            filename="editguard-final-metrics.csv")

    @app.get("/api/evaluation/final/oracle-recovery")
    def oracle_recovery():
        original, original_descriptor = final_report_status.snapshot(readiness())
        report, descriptor = oracle_recovery_status.snapshot(original)
        return {"status": descriptor["status"], "report": report, "descriptor": descriptor,
                "original_final_report": original_descriptor,
                "disclaimer": "Separately named fixed-configuration operational supplement. Original final scores and oracle runtime errors remain unchanged. Reading this endpoint does not execute inference or training."}

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

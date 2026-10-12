"""Synthetic planning fixtures only: no actual models, GPU or neural training."""

import copy
from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.training import canonical_hash, json_file_bytes
from scripts.prepare_data import canonical_bytes, content_hash, write_dataset
from scripts.prepare_pilot_runs import make_card, publish_cards
from scripts.profile_pilot_training import (current_source_hashes, reviewed_inputs,
                                          validate_output_paths)
from scripts.review_dataset import build_pilot, checked_rows, file_hash, read_json


ROOT = Path(__file__).resolve().parents[1]


class PilotPlanningTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "runtime/test-pilot-planning"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        full = self.root / "synthetic-full"
        write_dataset(full, family_count=20)
        self.data = self.root / "synthetic-pilot"
        build_pilot(full, self.data)
        manifest = read_json(self.data / "manifest.json")
        self.reviews = self.root / "synthetic-ai-decisions.json"
        rows = {row["id"]: {"decision": "accept", "reviewer": "Synthetic machine-review unit fixture only",
            "reviewer_type": "automated_development", "reviewed_at": "2026-10-12T08:00:00+05:30",
            "row_sha256": content_hash(row), "reason": "Synthetic fixture; not an actual project review decision."}
            for task in ("scope", "answer") for split in ("train", "validation")
            for row in checked_rows(self.data, task, split)}
        self.registry = {"dataset_id": manifest["dataset_id"],
            "origin_manifest_sha256": file_hash(self.data / "manifest.json"), "review_policy": "automated_development",
            "human_reviewed_data": False, "audit_sha256": "a" * 64, "rows": rows}
        self.reviews.write_bytes(json_file_bytes(self.registry))
        self.profiles = {}
        self.profile_paths = {}
        models = read_json(ROOT / "configs/models.json")
        for task, train_count, validation_count, samples, peak, checkpoint_size, export_size in (
            ("scope", 240, 120, [.33, .35, .34], 1.5, 800_000_000, 270_000_000),
            ("answer", 96, 48, [.40, .48, .44], 1.8, 1_055_000_000, 310_000_000)):
            environment = {"device": "cuda", "description": "SYNTHETIC UNIT FIXTURE; no device/model was run"}
            record = {"schema_version": 1, "purpose": "actual_trainer_development_profile",
                "status": "profile_complete", "runtime_eligible": False, "profile_artifact_purpose": "smoke_test",
                "review_policy": "automated_development", "human_reviewed_data": False, "test_rows_read": False,
                "dataset_id": manifest["dataset_id"], "manifest_sha256": file_hash(self.data / "manifest.json"),
                "reviews_sha256": file_hash(self.reviews), "input_format_version": INPUT_FORMAT_VERSION,
                "trainer_source_sha256": current_source_hashes(),
                "profile_script_sha256": file_hash(ROOT / "scripts/profile_pilot_training.py"),
                "model_config_sha256": file_hash(ROOT / "configs/models.json"),
                "environment": environment, "environment_sha256": canonical_hash(environment),
                "data_files_sha256": {split: file_hash(self.data / f"{task}_{split}.jsonl") for split in ("train", "validation")},
                "hardware": {"gpu_memory_gib": 6, "description": "SYNTHETIC measurement fixture only"},
                "profile": {"component": task, "backbone": models[f"{task}_candidate"],
                    "model_revision": models["model_revisions"][task], "device": "cuda", "batch_size": 2,
                    "input_length": 192, "target_length": 48 if task == "answer" else None,
                    "precision": "float32", "measured_optimizer_steps": 3, "step_seconds": samples,
                    "median_step_seconds": sorted(samples)[1], "train_rows": train_count,
                    "validation_rows": validation_count,
                    "full_validation_measurements": [{"global_step": step, "rows": validation_count, "seconds": seconds}
                        for step, seconds in ((2, .75 if task == "scope" else 1.1), (4, .78 if task == "scope" else 1.2))],
                    "checkpoint_measurements": [{"global_step": step, "seconds": seconds}
                        for step, seconds in ((2, 4.8 if task == "scope" else 7), (4, 5.2 if task == "scope" else 7.5))],
                    "load_tokenize_export_overhead_seconds": 18 if task == "scope" else 22,
                    "peak_allocated_gpu_gib": peak - .1, "peak_reserved_gpu_gib": peak,
                    "checkpoint_state_bytes": checkpoint_size, "export_bytes": export_size,
                    "limitations": "SYNTHETIC UNIT FIXTURE; invented numbers for validation branches, never actual profiling evidence."}}
            self.profiles[task] = record
            self.profile_paths[task] = self.root / f"synthetic-{task}-profile.json"
            self.profile_paths[task].write_bytes(json_file_bytes(record))

    def card(self, task):
        return make_card(task, self.data, self.reviews, self.profile_paths[task], self.root / f"model-{task}", "automated_development")

    def profile_change(self, task, change):
        record = copy.deepcopy(self.profiles[task])
        change(record)
        self.profile_paths[task].write_bytes(json_file_bytes(record))

    def test_imports_and_review_preflight_load_no_ml_package_and_create_no_outputs(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import scripts.profile_pilot_training; import scripts.prepare_pilot_runs; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        before = {path.name: file_hash(path) for path in self.data.iterdir()}
        _, populations = reviewed_inputs("scope", self.data, self.reviews, "automated_development")
        self.assertEqual([len(populations[split]) for split in ("train", "validation")], [240, 120])
        self.assertTrue(all(not row["human_reviewed"] for rows in populations.values() for row in rows))
        self.assertEqual(before, {path.name: file_hash(path) for path in self.data.iterdir()})
        self.assertFalse((self.root / "model-scope").exists())

    def test_old_format_checksum_pending_and_default_human_policy_fail_before_profiler(self):
        from scripts import profile_pilot_training
        output, report = self.root / "no-profile", self.root / "no-report.json"
        arguments = ["profile_pilot_training.py", "--task", "scope", "--dataset", str(self.data),
                     "--reviews", str(self.reviews), "--output-dir", str(output), "--report", str(report)]
        with (redirect_stderr(io.StringIO()), patch.object(sys, "argv", arguments), patch.object(profile_pilot_training, "hardware") as hardware,
              patch.object(profile_pilot_training, "run_training") as training):
            with self.assertRaises(SystemExit) as raised:
                profile_pilot_training.main()
            self.assertEqual(raised.exception.code, 2)
            hardware.assert_not_called()
            training.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(report.exists())
        self.registry["rows"].pop(next(iter(self.registry["rows"])))
        self.reviews.write_bytes(json_file_bytes(self.registry))
        with self.assertRaisesRegex(ValueError, "automated development review"):
            reviewed_inputs("scope", self.data, self.reviews, "automated_development")
        path = self.data / "scope_train.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "manifest"):
            reviewed_inputs("scope", self.data, self.reviews, "automated_development")
        manifest_path = self.data / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["input_format_version"] = "editguard-input-v1"
        manifest_path.write_bytes(json_file_bytes(manifest))
        with self.assertRaisesRegex(ValueError, "input format"):
            reviewed_inputs("scope", self.data, self.reviews, "automated_development")

    def test_registry_identity_and_profile_paths_protect_source_bytes(self):
        for change in ({"dataset_id": "other-dataset"}, {"origin_manifest_sha256": "0" * 64}):
            self.reviews.write_bytes(json_file_bytes({**self.registry, **change}))
            with self.assertRaisesRegex(ValueError, "bind the exact"):
                reviewed_inputs("scope", self.data, self.reviews, "automated_development")
        for output, report in ((self.data / "profile", self.root / "report.json"),
                               (self.root / "profile", self.data / "report.json"),
                               (self.root / "profile", self.root / "profile/report.json")):
            with self.assertRaises(ValueError):
                validate_output_paths(self.data, self.reviews, output, report)
            self.assertFalse(output.exists())
            self.assertFalse(report.exists())

    def test_current_pilot_caps_estimate_components_memory_and_checkpoint_disk_are_bound(self):
        scope, answer = self.card("scope"), self.card("answer")
        self.assertEqual((scope["max_steps"], answer["max_steps"]), (360, 240))
        self.assertEqual((scope["checkpoint_every_steps"], answer["checkpoint_every_steps"]), (120, 48))
        self.assertEqual((scope["estimated_total_seconds"], answer["estimated_total_seconds"]), (324, 362))
        self.assertEqual((scope["estimated_peak_gpu_gib"], answer["estimated_peak_gpu_gib"]), (2, 2.25))
        self.assertEqual((scope["checkpoint_plan"]["disk_budget_gib"], answer["checkpoint_plan"]["disk_budget_gib"]), (5, 6))
        self.assertEqual(scope["profile_environment_sha256"], self.profiles["scope"]["environment_sha256"])
        self.assertEqual(answer["profile_source_sha256"], current_source_hashes())
        self.assertFalse((self.root / "model-scope").exists())
        self.assertFalse((self.root / "model-answer").exists())

    def test_wrong_profile_status_task_data_source_environment_and_review_are_refused(self):
        changes = [lambda record: record.update(status="failed"),
            lambda record: record.update(profile=None),
            lambda record: record.update(profile_artifact_purpose="trained"),
            lambda record: record.update(human_reviewed_data=True),
            lambda record: record.update(reviews_sha256="0" * 64),
            lambda record: record.update(trainer_source_sha256={}),
            lambda record: record.update(environment_sha256="0" * 64),
            lambda record: record["profile"].update(component="answer"),
            lambda record: record["profile"].update(input_length=256),
            lambda record: record["data_files_sha256"].update(train="0" * 64),
            lambda record: record["profile"].update(train_rows=239)]
        for index, change in enumerate(changes):
            self.profile_change("scope", change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.card("scope")
        self.assertFalse((self.root / "model-scope").exists())

    def test_nonfinite_timing_partial_validation_and_impossible_memory_are_refused(self):
        changes = [lambda record: record["profile"].update(step_seconds=[.1, .2]),
            lambda record: record["profile"].update(step_seconds=[.1, -1, .2]),
            lambda record: record["profile"].update(median_step_seconds=.01),
            lambda record: record["profile"].update(full_validation_measurements=[{"global_step": 2, "rows": 120, "seconds": .5}]),
            lambda record: record["profile"]["full_validation_measurements"][0].update(rows=1),
            lambda record: record["profile"]["checkpoint_measurements"][0].update(seconds=0),
            lambda record: record["profile"].update(load_tokenize_export_overhead_seconds=-1),
            lambda record: record["profile"].update(peak_reserved_gpu_gib=5.5)]
        for index, change in enumerate(changes):
            self.profile_change("scope", change)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.card("scope")

    def test_both_cards_publish_only_after_both_preflights_and_invalid_second_leaves_no_partial_output(self):
        cards = {task: self.card(task) for task in ("scope", "answer")}
        invalid = copy.deepcopy(cards)
        invalid["answer"]["estimated_total_seconds"] = 1
        destination = self.root / "atomic-cards"
        with self.assertRaisesRegex(ValueError, "runtime estimate"):
            publish_cards(invalid, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".atomic-cards.pending-*")), [])
        results = publish_cards(cards, destination)
        self.assertEqual(set(path.name for path in destination.iterdir()), {"scope.json", "answer.json"})
        self.assertTrue(all(result["preflight_valid"] for result in results.values()))
        self.assertFalse(any(result["training_execution_requested"] for result in results.values()))
        with self.assertRaisesRegex(ValueError, "fresh"):
            publish_cards(cards, destination)

    def test_cli_rejects_invalid_answer_profile_before_publishing_either_card(self):
        from scripts import prepare_pilot_runs
        self.profile_change("answer", lambda record: record.update(status="failed"))
        destination = self.root / "no-cards"
        arguments = ["prepare_pilot_runs.py", "--dataset", str(self.data), "--reviews", str(self.reviews),
            "--review-policy", "automated_development", "--scope-profile", str(self.profile_paths["scope"]),
            "--answer-profile", str(self.profile_paths["answer"]), "--output", str(destination)]
        with redirect_stderr(io.StringIO()), patch.object(sys, "argv", arguments), self.assertRaises(SystemExit) as raised:
            prepare_pilot_runs.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(destination.exists())

    def test_profile_and_planned_artifacts_refuse_entity_leakage_and_input_descendants(self):
        rows = checked_rows(self.data, "scope", "validation")
        train_entity = checked_rows(self.data, "scope", "train")[0]["correction"]["entity_id"]
        rows[0]["expected_fact"]["entity_id"] = train_entity
        path = self.data / "scope_validation.jsonl"
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        manifest = read_json(self.data / "manifest.json")
        manifest["files"][path.name] = file_hash(path)
        (self.data / "manifest.json").write_bytes(json_file_bytes(manifest))
        self.registry["rows"][rows[0]["id"]]["row_sha256"] = content_hash(rows[0])
        self.registry["origin_manifest_sha256"] = file_hash(self.data / "manifest.json")
        self.reviews.write_bytes(json_file_bytes(self.registry))
        with self.assertRaisesRegex(ValueError, "entity, family or lineage"):
            reviewed_inputs("scope", self.data, self.reviews, "automated_development")
        with self.assertRaisesRegex(ValueError, "remain separate"):
            make_card("answer", self.data, self.reviews, self.profile_paths["answer"], self.data / "model", "automated_development")


if __name__ == "__main__":
    unittest.main()

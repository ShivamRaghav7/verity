import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import content_hash, write_dataset
from scripts.preflight_training import RESUME_FIELDS, file_hash, validate_run_card


class TrainingRunCardTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[1] / "runtime/test-training-contract"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        write_dataset(self.data, family_count=20)

    def make_card(self, task="scope"):
        reviews = self.root / f"{task}_reviews.json"
        records = {}
        for split in ("train", "validation"):
            for line in (self.data / f"{task}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                records[row["id"]] = {"decision": "accept", "reviewer": "synthetic test fixture only",
                                      "reviewed_at": "2026-10-11T12:00:00+05:30",
                                      "row_sha256": content_hash(row)}
        reviews.write_text(json.dumps({"rows": records}), encoding="utf-8")
        profile = self.root / f"{task}_profile.json"
        model_id = "distilbert/distilbert-base-uncased" if task == "scope" else "google/flan-t5-small"
        profile.write_text(json.dumps({"hardware": {"gpu_memory_gib": 6}, "profile": {
            "component": task, "backbone": model_id, "model_revision": "a" * 40,
            "batch_size": 2, "input_length": 192, "target_length": 48 if task == "answer" else None,
            "precision": "float32", "device": "cuda", "measured_optimizer_steps": 3,
            "median_step_seconds": 0.2, "peak_allocated_gpu_gib": 2.5,
            "limitations": "Synthetic test fixture, never an actual profile."}}), encoding="utf-8")
        card = {"schema_version": 1, "question": "Test fixture: are declared run inputs internally consistent?",
                "task": task, "dataset_id": json.loads((self.data / "manifest.json").read_text(encoding="utf-8"))["dataset_id"], "data_dir": str(self.data),
                "train_path": str(self.data / f"{task}_train.jsonl"),
                "validation_path": str(self.data / f"{task}_validation.jsonl"),
                "reviews_path": str(reviews), "profile_path": str(profile),
                "output_dir": str(self.root / "output"), "manifest_sha256": file_hash(self.data / "manifest.json"),
                "reviews_sha256": file_hash(reviews), "profile_sha256": file_hash(profile),
                "model_id": model_id, "model_revision": "a" * 40, "batch_size": 2,
                "input_length": 192, "target_length": 48 if task == "answer" else None,
                "precision": "float32", "max_steps": 20, "checkpoint_every_steps": 5,
                "keep_last_checkpoints": 2, "learning_rate": 2e-5, "seed": 42,
                "estimated_total_seconds": 10, "estimated_peak_gpu_gib": 3.5,
                "estimate_assumptions": "Synthetic unit-test fixture, not a released run estimate.",
                "checkpoint_plan": {"resume_policy": "exact_state_only", "state_fields": sorted(RESUME_FIELDS)},
                "expected_artifacts": ["model", "training_log.jsonl", "run_card.json", "checkpoints"],
                "resume_checkpoint": None}
        path = self.root / f"{task}_run_card.json"
        path.write_text(json.dumps(card), encoding="utf-8")
        return path, card

    def rewrite(self, path, card):
        path.write_text(json.dumps(card), encoding="utf-8")

    def test_both_tasks_validate_without_loading_models_or_creating_outputs(self):
        for task in ("scope", "answer"):
            path, card = self.make_card(task)
            result = validate_run_card(path)
            self.assertTrue(result["preflight_valid"])
            self.assertTrue(result["preflight_does_not_verify_training_loop"])
            self.assertFalse(result["training_execution_requested"])
            self.assertGreater(result["reviewed_train_rows"], 0)
            self.assertFalse(Path(card["output_dir"]).exists())

    def test_weighted_policy_requires_matching_profile_objective_before_release(self):
        from editguard.training import loss_objective
        path, card = self.make_card("answer")
        card["loss_policy"] = "comparison_first_token_8x"
        card["loss_objective"] = loss_objective("answer", card["loss_policy"])
        self.rewrite(path, card)
        with self.assertRaisesRegex(ValueError, "differs from the measured"):
            validate_run_card(path)
        profile = Path(card["profile_path"])
        record = json.loads(profile.read_text())
        record["profile"].update(loss_policy=card["loss_policy"], loss_objective=card["loss_objective"])
        profile.write_text(json.dumps(record), encoding="utf-8")
        card["profile_sha256"] = file_hash(profile)
        self.rewrite(path, card)
        self.assertTrue(validate_run_card(path)["preflight_valid"])
        card["loss_objective"]["comparison_first_token_weight"] = 1
        self.rewrite(path, card)
        with self.assertRaisesRegex(ValueError, "metadata must bind"):
            validate_run_card(path)

    def test_unreviewed_or_stale_decisions_cannot_release_run(self):
        path, card = self.make_card()
        reviews = Path(card["reviews_path"])
        records = json.loads(reviews.read_text(encoding="utf-8"))
        records["rows"].pop(next(iter(records["rows"])))
        reviews.write_text(json.dumps(records), encoding="utf-8")
        card["reviews_sha256"] = file_hash(reviews)
        self.rewrite(path, card)
        with self.assertRaisesRegex(ValueError, "human review"):
            validate_run_card(path)

    def test_input_hash_tampering_is_rejected(self):
        path, card = self.make_card()
        with Path(card["train_path"]).open("a", encoding="utf-8") as file:
            file.write("\n")
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_run_card(path)

    def test_old_input_format_cannot_be_relabeled_as_current_training(self):
        path, card = self.make_card()
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["input_format_version"] = "archived-input-format"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        card["manifest_sha256"] = file_hash(manifest_path)
        self.rewrite(path, card)
        with self.assertRaisesRegex(ValueError, "model input format"):
            validate_run_card(path)

    def test_reviewed_but_inconsistent_model_text_is_rejected(self):
        for task, field in (("scope", "correction_text"), ("answer", "input_text")):
            path, card = self.make_card(task)
            row_path = Path(card["train_path"])
            rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()]
            rows[0][field] = "Fixture wrong-format text; not a reviewed research label."
            row_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest_path = self.data / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][row_path.name] = file_hash(row_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            reviews_path = Path(card["reviews_path"])
            reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
            reviews["rows"][rows[0]["id"]]["row_sha256"] = content_hash(rows[0])
            reviews_path.write_text(json.dumps(reviews), encoding="utf-8")
            card.update(manifest_sha256=file_hash(manifest_path), reviews_sha256=file_hash(reviews_path))
            self.rewrite(path, card)
            with self.assertRaisesRegex(ValueError, "runtime formatter"):
                validate_run_card(path)

    def test_profile_backbone_revision_batch_lengths_precision_must_match(self):
        path, original = self.make_card("answer")
        changes = {"model_id": "other/model", "model_revision": "b" * 40,
                   "batch_size": 4, "input_length": 256, "target_length": 64,
                   "precision": "float16"}
        for key, value in changes.items():
            with self.subTest(key=key):
                card = {**original, key: value}
                self.rewrite(path, card)
                with self.assertRaisesRegex(ValueError, "differs"):
                    validate_run_card(path)

    def test_test_set_paths_and_input_overwrite_are_rejected(self):
        path, original = self.make_card()
        for change in ({"train_path": str(self.data / "scope_test.jsonl")},
                       {"validation_path": str(self.data / "scope_test.jsonl")},
                       {"output_dir": str(self.data)}):
            self.rewrite(path, {**original, **change})
            with self.assertRaises(ValueError):
                validate_run_card(path)

    def test_stopping_cap_checkpoint_and_estimate_bounds(self):
        path, original = self.make_card()
        for change in ({"max_steps": 0}, {"max_steps": 100001},
                       {"checkpoint_every_steps": 21}, {"keep_last_checkpoints": 0},
                       {"estimated_total_seconds": 1}, {"estimated_peak_gpu_gib": 1},
                       {"estimated_peak_gpu_gib": 7}, {"learning_rate": float("nan")},
                       {"model_revision": "main"}):
            with self.subTest(change=change):
                self.rewrite(path, {**original, **change})
                with self.assertRaises(ValueError):
                    validate_run_card(path)

    def test_exact_state_plan_and_resume_metadata_are_required(self):
        path, original = self.make_card()
        card = {**original, "checkpoint_plan": {"resume_policy": "weights_only", "state_fields": []}}
        self.rewrite(path, card)
        with self.assertRaisesRegex(ValueError, "exact_state_only"):
            validate_run_card(path)
        checkpoint = Path(original["output_dir"]) / "checkpoints/step-5"
        checkpoint.mkdir(parents=True)
        self.rewrite(path, {**original, "resume_checkpoint": str(checkpoint)})
        with self.assertRaisesRegex(ValueError, "Resume metadata"):
            validate_run_card(path)


if __name__ == "__main__":
    unittest.main()

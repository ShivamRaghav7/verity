"""Synthetic typed-review propagation checks; no model packages execute."""

import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from editguard.comparison import ComparisonRunner
from editguard.formatting import INPUT_FORMAT_VERSION
from editguard.neural import file_sha256
from scripts.calibrate_scope import reviewed_inputs as scope_inputs
from scripts.calibrate_similarity import reviewed_inputs as similarity_inputs
from scripts.prepare_data import content_hash, write_dataset
from scripts.run_comparison import main as comparison_main, reviewed_validation


ROOT = Path(__file__).resolve().parents[1]


class ReviewPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "data"
        write_dataset(self.dataset, family_count=20)
        self.rows = [json.loads(line) for line in (self.dataset / "scope_validation.jsonl").read_text().splitlines()]
        records = {row["id"]: {"decision": "accept", "reviewer": "Synthetic policy unit fixture only",
                   "reviewed_at": "2026-10-11T15:00:00+05:30", "row_sha256": content_hash(row)} for row in self.rows}
        self.reviews = self.root / "reviews.json"
        self.reviews.write_text(json.dumps({"rows": records}), encoding="utf-8")
        self.config = self.root / "inference.json"
        config = json.loads((ROOT / "configs/inference.json").read_text())
        config["input_format_version"] = INPUT_FORMAT_VERSION
        self.config.write_text(json.dumps(config), encoding="utf-8")
        self.scope = self.root / "synthetic-scope"
        for name, text in (("model/config.json", "{}"), ("model/model.safetensors", "synthetic fixture, not weights"),
                           ("tokenizer/tokenizer_config.json", "{}")):
            path = self.scope / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed", "task": "scope",
                    "dataset_id": json.loads((self.dataset / "manifest.json").read_text())["dataset_id"],
                    "manifest_sha256": file_sha256(self.dataset / "manifest.json"),
                    "input_format_version": INPUT_FORMAT_VERSION, "input_length": 192,
                    "review_policy": "automated_development", "human_reviewed_data": False,
                    "files_sha256": {path.relative_to(self.scope).as_posix(): file_sha256(path)
                                     for path in self.scope.rglob("*") if path.is_file()}}
        (self.scope / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def automated_registry(self):
        registry = json.loads(self.reviews.read_text())
        registry.update(review_policy="automated_development", human_reviewed_data=False, audit_sha256="b" * 64)
        for record in registry["rows"].values():
            record["reviewer_type"] = "automated_development"
        self.reviews.write_text(json.dumps(registry), encoding="utf-8")
        return registry

    def gated(self, policy="human"):
        scope = scope_inputs(self.scope, self.dataset, self.reviews, max_rows=2000, review_policy=policy)
        similarity = similarity_inputs(self.config, self.dataset, self.reviews, max_rows=2000, review_policy=policy)
        comparison = reviewed_validation(self.dataset, self.reviews, review_policy=policy)
        return scope, similarity, comparison

    def test_human_default_retains_original_row_hashes_and_training_policy_separation(self):
        scope, similarity, comparison = self.gated()
        for bindings in (scope[1], similarity[2]):
            self.assertEqual(bindings["review_policy"], "human")
            self.assertTrue(bindings["human_reviewed_data"])
            self.assertEqual(bindings["accepted_row_sha256"][self.rows[0]["id"]], content_hash(self.rows[0]))
        self.assertFalse(scope[1]["scope_training_human_reviewed_data"])
        self.assertEqual(scope[1]["scope_training_review_policy"], "automated_development")
        self.assertEqual(comparison[1], self.rows)

    def test_automated_registry_is_refused_by_default_and_explicitly_bound_when_selected(self):
        self.automated_registry()
        for function, arguments in ((scope_inputs, (self.scope, self.dataset, self.reviews)),
                                    (similarity_inputs, (self.config, self.dataset, self.reviews))):
            with self.assertRaisesRegex(ValueError, "human review"):
                function(*arguments, max_rows=2000)
        with self.assertRaisesRegex(ValueError, "human review"):
            reviewed_validation(self.dataset, self.reviews)
        scope, similarity, comparison = self.gated("automated_development")
        for bindings in (scope[1], similarity[2]):
            self.assertEqual(bindings["review_policy"], "automated_development")
            self.assertFalse(bindings["human_reviewed_data"])
            self.assertEqual(bindings["review_audit_sha256"], "b" * 64)
        self.assertEqual(comparison[1], self.rows)

    def test_machine_policy_requires_typed_registry_and_audit_before_model_loading(self):
        for policy in ("automated_development", "unspecified_machine_policy"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                self.gated(policy)
        registry = self.automated_registry()
        registry["human_reviewed_data"] = True
        self.reviews.write_text(json.dumps(registry), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "machine-review registry"):
            self.gated("automated_development")

    def test_preflight_clis_label_automated_validation_without_loading_or_outputs(self):
        self.automated_registry()
        for script, option, source in (("calibrate_scope.py", "--scope-artifact", self.scope),
                                       ("calibrate_similarity.py", "--config", self.config)):
            output = self.root / (script + "-output")
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), option, str(source),
                "--dataset", str(self.dataset), "--reviews", str(self.reviews), "--review-policy", "automated_development",
                "--output", str(output), "--preflight-only"], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            self.assertEqual(record["review_policy"], "automated_development")
            self.assertFalse(record["human_reviewed_data"])
            self.assertFalse(record["model_execution_requested"])
            self.assertFalse(output.exists())

    def fixture_comparison(self, *, smoke=False):
        class SyntheticBase:
            device = "cpu"
            last_trace = {"purpose": "synthetic unit fixture; no model execution"}

            def parameter_fingerprint(self):
                return "c" * 64

            def generate(self, text):
                return "Synthetic unit fixture; no factual accuracy claim."

        class SyntheticRetriever:
            def rank(self, question, records, formatter, k=3):
                return [{"record": item, "score": .5} for item in records[:k]]

        output = self.root / ("smoke-output" if smoke else "validation-output")
        argv = ["run_comparison.py", "--config", str(self.config), "--output", str(output),
                "--configurations", "unchanged_base", "--max-questions", "2"]
        argv += ["--smoke"] if smoke else ["--dataset", str(self.dataset), "--reviews", str(self.reviews),
                                            "--review-policy", "automated_development"]
        runner = ComparisonRunner(SyntheticBase(), SyntheticRetriever())
        fake_packages = {"torch": SimpleNamespace(__version__="synthetic-fixture"),
                         "psutil": SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0)))}
        with patch.object(sys, "argv", argv), patch("scripts.run_comparison.ComparisonRunner.from_config", return_value=runner), \
                patch.dict(sys.modules, fake_packages), patch("sys.stdout", io.StringIO()):
            comparison_main()
        manifest = json.loads((output / "manifest.json").read_text())
        metrics = json.loads((output / "metrics.json").read_text())
        self.assertEqual(metrics["review_policy"], manifest["review_policy"])
        self.assertEqual(metrics["human_reviewed_data"], manifest["human_reviewed_data"])
        self.assertEqual(metrics["reviewed_data"], manifest["reviewed_data"])
        predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
        return manifest, predictions

    def test_comparison_manifest_and_predictions_cannot_present_automated_as_human(self):
        self.automated_registry()
        manifest, predictions = self.fixture_comparison()
        self.assertEqual(manifest["review_policy"], "automated_development")
        self.assertFalse(manifest["human_reviewed_data"])
        self.assertTrue(manifest["reviewed_data"])
        self.assertEqual(manifest["review_audit_sha256"], "b" * 64)
        self.assertEqual(len(predictions), 2)
        original = {row["id"]: content_hash(row) for row in self.rows}
        for record in predictions:
            self.assertEqual(record["review_policy"], "automated_development")
            self.assertFalse(record["human_reviewed_data"])
            self.assertEqual(record["row_sha256"], original[record["id"]])

    def test_engineering_smoke_is_never_labeled_as_human_review(self):
        manifest, predictions = self.fixture_comparison(smoke=True)
        self.assertIsNone(manifest["review_policy"])
        self.assertFalse(manifest["human_reviewed_data"])
        self.assertFalse(manifest["reviewed_data"])
        self.assertTrue(all(record["review_policy"] is None and not record["human_reviewed_data"] for record in predictions))


if __name__ == "__main__":
    unittest.main()

"""Refused resumes preserve saved evidence before any ML initialization."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from editguard.training import run_training


class TrainingResumeGuardTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "run"
        self.output.mkdir()
        self.checkpoint = self.output / "checkpoints/step-00000120"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "state.pt").write_bytes(b"synthetic checkpoint; never loaded")
        self.write_metadata(120)
        self.manifest_path = self.output / "manifest.json"
        self.manifest_path.write_text(json.dumps({"status": "interrupted", "purpose": "synthetic refusal fixture"}), encoding="utf-8")
        self.weights = self.output / "model/model.safetensors"
        self.weights.parent.mkdir()
        self.weights.write_bytes(b"saved synthetic export bytes, not trained weights")
        self.card = {"task": "answer", "max_steps": 240, "seed": 42, "output_dir": str(self.output),
                     "manifest_sha256": "a" * 64, "reviews_sha256": "b" * 64}
        # No real dataset, review registry or model is used by these refusal checks.
        self.rows = [{"reviewed": True, "review_status": "accepted", "fixture": "synthetic only"}]

    def write_metadata(self, step):
        (self.checkpoint / "metadata.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")

    def snapshot(self):
        return {path.relative_to(self.output).as_posix(): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def invoke(self, checkpoint=None):
        return run_training(self.card, train_rows=self.rows, validation_rows=self.rows, purpose="trained",
                            run_card_sha256="c" * 64, resume_checkpoint=checkpoint or self.checkpoint)

    def refuse(self, message, checkpoint=None):
        before = self.snapshot()
        with patch("editguard.training.configure_determinism", side_effect=AssertionError("ML initialization was reached")) as configure:
            with self.assertRaisesRegex(ValueError, message):
                self.invoke(checkpoint)
        configure.assert_not_called()
        self.assertEqual(self.snapshot(), before)

    def test_completed_manifest_and_weights_remain_byte_identical(self):
        self.manifest_path.write_text(json.dumps({"status": "completed", "files_sha256": {"model": "synthetic"}}), encoding="utf-8")
        self.write_metadata(240)
        self.refuse("Completed training artifacts cannot be resumed")
        self.assertEqual(json.loads(self.manifest_path.read_text())["status"], "completed")

    def test_checkpoint_at_or_beyond_cap_is_refused_before_ml_or_mutation(self):
        for step in (240, 241, 1000):
            with self.subTest(step=step):
                self.write_metadata(step)
                self.refuse("below the unchanged stopping cap")

    def test_missing_negative_boolean_or_noninteger_step_is_refused(self):
        for step in (-1, True, 120.0, "120", None):
            with self.subTest(step=step):
                self.write_metadata(step)
                self.refuse("nonnegative integer")

    def test_outside_missing_or_nondirectory_checkpoint_is_refused(self):
        outside = self.root / "outside-checkpoint"
        outside.mkdir()
        (outside / "metadata.json").write_text('{"global_step": 120}', encoding="utf-8")
        for path in (outside, self.output / "missing", self.weights, self.output):
            with self.subTest(path=path):
                self.refuse("existing directory inside output_dir", path)

    def test_malformed_metadata_and_manifest_fail_without_mutation_or_ml(self):
        original_manifest = self.manifest_path.read_bytes()
        for path in (self.manifest_path, self.checkpoint / "metadata.json"):
            with self.subTest(path=path):
                self.manifest_path.write_bytes(original_manifest)
                self.write_metadata(120)
                path.write_bytes(b"malformed synthetic JSON")
                before = self.snapshot()
                with patch("editguard.training.configure_determinism", side_effect=AssertionError("No ML")) as configure:
                    with self.assertRaises(json.JSONDecodeError):
                        self.invoke()
                configure.assert_not_called()
                self.assertEqual(self.snapshot(), before)

    def test_nonobject_manifest_and_metadata_are_refused_before_ml(self):
        self.manifest_path.write_text("[]", encoding="utf-8")
        self.refuse("manifest must be a JSON object")
        self.manifest_path.write_text('{"status": "interrupted"}', encoding="utf-8")
        (self.checkpoint / "metadata.json").write_text("null", encoding="utf-8")
        self.refuse("metadata must be a JSON object")

    def test_valid_intermediate_metadata_can_reach_setup_and_preserves_prior_evidence(self):
        for step in (0, 120, 239):
            with self.subTest(step=step):
                self.write_metadata(step)
                before = self.snapshot()
                with patch("editguard.training.configure_determinism", side_effect=RuntimeError("synthetic setup boundary")) as configure:
                    with self.assertRaisesRegex(RuntimeError, "synthetic setup boundary"):
                        self.invoke()
                configure.assert_called_once()
                self.assertEqual(self.snapshot(), before)

    def test_completed_resume_is_refused_with_no_torch_or_transformers_import(self):
        self.manifest_path.write_text('{"status": "completed"}', encoding="utf-8")
        card = self.root / "card.json"
        card.write_text(json.dumps(self.card), encoding="utf-8")
        command = "import sys,json; from pathlib import Path; from editguard.training import run_training; " \
            "card=json.loads(Path(sys.argv[1]).read_text()); rows=[{'reviewed':True,'review_status':'accepted'}]; " \
            "\ntry: run_training(card,train_rows=rows,validation_rows=rows,purpose='trained',run_card_sha256='c'*64,resume_checkpoint=sys.argv[2])" \
            "\nexcept ValueError as exc: assert 'Completed training' in str(exc)" \
            "\nelse: raise AssertionError('Expected completed-resume refusal')" \
            "\nassert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        before = self.snapshot()
        result = subprocess.run([sys.executable, "-c", command, str(card), str(self.checkpoint)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()

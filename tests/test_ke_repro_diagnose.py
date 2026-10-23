"""Finite diagnostic-controller tests; no checkpoint, training or GPU needed."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ke_repro.diagnose import controller, diagnostic_operation, formatted_cases, plan


def cases():
    return [{"id": name, "image": "fixture.png", "question": f"Question {name}?", "expected": "target"}
            for name in ("direct", "paraphrase", "alternate_image", "neighbor")]


class FakeBackend:
    def __init__(self, *, fail_after_load=False):
        self.loaded = False
        self.fail_after_load = fail_after_load
        self.generated, self.scores, self.tokenizations = [], [], []
        self.loads = self.rollbacks = 0

    def weight_hashes(self):
        return {"matrix": "patched" if self.loaded else "base"}

    def generate(self, image, question, max_new_tokens):
        self.generated.append((self.loaded, question, max_new_tokens))
        if self.loaded and self.fail_after_load:
            raise RuntimeError("Deliberate generation failure")
        return "patched answer" if self.loaded else "base answer"

    def score_target(self, image, question, target, *, target_alignment):
        self.scores.append((self.loaded, question, target_alignment))
        return {"conditioning": "teacher_forced_full_target_in_input", "is_generated_answer": False,
                "nll": 2., "token_accuracy": 0., "target_alignment": target_alignment}

    def tokenization_report(self, question, target):
        self.tokenizations.append((question, target))
        return {"standalone_labels_match_input_suffix": False}

    def load_patch(self, path):
        self.loads += 1
        self.loaded = True
        return {"source": str(path)}

    def rollback(self):
        self.rollbacks += 1
        self.loaded = False
        return {"exact_weight_restore": True}

    def edit(self, *args, **kwargs):
        raise AssertionError("Diagnostic must never fit a new edit")


class DiagnosticOperationTests(unittest.TestCase):
    def test_fixed_formats_counts_separate_teacher_forcing_and_exact_restore(self):
        backend = FakeBackend()
        result = diagnostic_operation(backend, cases(), Path("old-patch"))
        self.assertEqual(len(backend.generated), 16)
        self.assertEqual(len(backend.scores), 8)
        self.assertEqual(len(backend.tokenizations), 2)
        self.assertTrue(all(row[2] == 24 for row in backend.generated))
        self.assertEqual({row[2] for row in backend.scores}, {"release", "context"})
        self.assertEqual(backend.loads, 1)
        self.assertEqual(backend.rollbacks, 1)
        self.assertFalse(backend.loaded)
        self.assertEqual(result["edit_count"], 0)
        self.assertFalse(result["training_executed"])
        self.assertIsNone(result["accuracy_estimate"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["baseline"]["raw"][0]["question"], "Question direct?")
        self.assertEqual(result["loaded_patch"]["vqa"][0]["question"], "Question: Question direct? Answer:")
        self.assertFalse(result["teacher_forced"]["baseline"]["raw"]["release"]["is_generated_answer"])
        json.dumps(result, allow_nan=False)

    def test_failure_after_loading_still_rolls_back(self):
        backend = FakeBackend(fail_after_load=True)
        with self.assertRaisesRegex(RuntimeError, "Deliberate"):
            diagnostic_operation(backend, cases(), Path("old-patch"))
        self.assertEqual(backend.loads, 1)
        self.assertEqual(backend.rollbacks, 1)
        self.assertFalse(backend.loaded)

    def test_no_new_prompt_formats_or_case_selection(self):
        with self.assertRaises(ValueError):
            formatted_cases(cases(), "try_another")
        with self.assertRaises(ValueError):
            diagnostic_operation(FakeBackend(), cases()[:1], Path("old-patch"))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        old_patch = root / "old-patch"
        old_patch.write_bytes(b"tiny saved patch")
        self.args = argparse.Namespace(upstream=root / "upstream", assets=root / "assets",
                                       manifest=root / "manifest.json", output=root / "diagnostic",
                                       patch=old_patch, device="cuda", fixture=None, timeout_seconds=1800)

    def test_process_failure_overrides_optimistic_worker_report(self):
        def process(command, **kwargs):
            self.assertIsInstance(command, list)
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["timeout"], 1800)
            self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertIn("--expected-patch-sha256", command)
            (self.args.output / "diagnostic.json").write_text(json.dumps({"execution_status": "completed"}))
            return SimpleNamespace(returncode=-9)

        with patch("ke_repro.diagnose.preflight", return_value={"assets_verified": True}), \
             patch("ke_repro.diagnose.subprocess.run", side_effect=process), redirect_stdout(StringIO()):
            self.assertEqual(controller(self.args), 1)
        result = json.loads((self.args.output / "diagnostic.json").read_text())
        self.assertEqual(result["worker_reported_status"], "completed")
        self.assertEqual(result["execution_status"], "process_failed")
        self.assertEqual(result["worker_return_code"], -9)

    def test_timeout_without_report_is_preserved(self):
        with patch("ke_repro.diagnose.preflight", return_value={"assets_verified": True}), \
             patch("ke_repro.diagnose.subprocess.run", side_effect=subprocess.TimeoutExpired("worker", 1800)), \
             redirect_stdout(StringIO()):
            self.assertEqual(controller(self.args), 1)
        result = json.loads((self.args.output / "diagnostic.json").read_text())
        self.assertEqual(result["execution_status"], "timed_out")
        self.assertIsNone(result["worker_return_code"])

    def test_fresh_output_required_and_preflight_failure_never_starts_worker(self):
        with patch("ke_repro.diagnose.preflight", side_effect=ValueError("bad assets")), \
             patch("ke_repro.diagnose.subprocess.run") as process:
            with self.assertRaisesRegex(ValueError, "bad assets"):
                controller(self.args)
            process.assert_not_called()
        result = json.loads((self.args.output / "diagnostic.json").read_text())
        self.assertEqual(result["execution_status"], "preflight_failed")
        self.assertFalse(result["models_loaded"])
        with self.assertRaisesRegex(ValueError, "Output already exists"):
            controller(self.args)

    def test_declared_plan_is_finite_and_contains_no_training(self):
        value = plan(self.args)
        self.assertEqual(value["generated_answer_count"], 16)
        self.assertEqual(value["teacher_forced_score_count"], 8)
        self.assertEqual(value["edit_count"], 0)
        self.assertFalse(value["training_executed"])
        self.assertEqual(value["maximum_worker_seconds"], 1800)


if __name__ == "__main__":
    unittest.main()

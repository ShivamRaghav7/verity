import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ke_repro.smoke import (answer_map, combine, edit_phase, exact_short_answer,
                            format_cases, load_cases, main, reload_phase, write_json)


class FakeBackend:
    """A lifecycle simulator, never evidence about the released model."""

    compatibility = {"simulation": True}

    def __init__(self):
        self.active = False

    def weight_hashes(self):
        return {"matrix": "edited" if self.active else "base"}

    def generate(self, image, question, max_new_tokens):
        return "target" if self.active and question != "neighbor?" else "original"

    def edit(self, *args, **kwargs):
        self.active = True
        return {"target_alignment": kwargs["target_alignment"]}

    def save_patch(self, path):
        path.mkdir()
        return {}

    def load_patch(self, path):
        self.active = True

    def rollback(self):
        self.active = False
        return {}


class SmokeContractTests(unittest.TestCase):
    def test_rejected_or_incidental_target_is_not_answer_match(self):
        self.assertTrue(exact_short_answer("Albany, New York.", "Albany_New_York"))
        for text in ("Not Albany, New York", "Albany, New York is an example", "", "Denton or Albany New York"):
            self.assertFalse(exact_short_answer(text, "Albany New York"))

    def test_fixture_requires_training_split_and_prevents_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            (root / "images/a.jpg").write_bytes(b"fixture")
            record = {"image": "a.jpg", "image_rephrase": "a.jpg", "m_loc": "a.jpg",
                      "src": "question", "rephrase": "different wording", "m_loc_q": "neighbor?",
                      "alt": "target", "m_loc_a": "original"}
            path = root / "train-fixture.json"
            fixture = {"upstream_split": "train", "purpose": "development_smoke_only", "record": record}
            write_json(path, fixture)
            _, cases = load_cases(root, path)
            self.assertEqual(len(cases), 4)
            fixture["upstream_split"] = "eval"
            write_json(path, fixture)
            with self.assertRaisesRegex(ValueError, "TRAIN"):
                load_cases(root, path)
            fixture["upstream_split"] = "train"
            fixture["record"]["image"] = "../../outside.jpg"
            write_json(path, fixture)
            with self.assertRaisesRegex(ValueError, "escapes"):
                load_cases(root, path)

    def test_controller_keeps_technical_and_answer_outcomes_separate(self):
        cases = [{"id": name, "image": "a", "question": "neighbor?" if name == "neighbor" else "question?",
                  "expected": "original" if name == "neighbor" else "target"}
                 for name in ("direct", "paraphrase", "alternate_image", "neighbor")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = edit_phase(FakeBackend(), cases, root, 24)
            first["execution_status"] = "completed"
            write_json(root / "edit.json", first)
            fresh = reload_phase(FakeBackend(), cases, root, 24)
            fresh["execution_status"] = "completed"
            summary = combine(first, fresh)
            self.assertTrue(summary["technical_gate_passed"])
            self.assertTrue(summary["direct_edit_target_matched"])
            self.assertTrue(summary["neighbor_answer_preserved"])
            self.assertIsNone(summary["accuracy_estimate"])
            first["edited"][0]["strict_target_match"] = False
            summary = combine(first, fresh)
            self.assertTrue(summary["technical_gate_passed"])
            self.assertFalse(summary["direct_edit_target_matched"])
            fresh["checks"]["loaded_editable_matrices_match"] = False
            self.assertFalse(combine(first, fresh)["technical_gate_passed"])

    def test_failed_reload_cannot_pass(self):
        self.assertFalse(combine({"execution_status": "completed", "checks": {"a": True}}, None)["technical_gate_passed"])

    def test_plan_requires_no_torch_or_assets(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["plan"]), 0)

    def test_prompt_and_alignment_are_explicit_and_do_not_change_target(self):
        cases = [{"id": "direct", "question": "Which city?", "expected": "Albany"}]
        self.assertEqual(format_cases(cases, "raw")[0]["question"], "Which city?")
        prompted = format_cases(cases, "vqa")[0]
        self.assertEqual(prompted["question"], "Question: Which city? Answer:")
        self.assertEqual(prompted["raw_question"], cases[0]["question"])
        self.assertEqual(prompted["expected"], "Albany")
        self.assertEqual(cases[0]["question"], "Which city?")
        with redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(main(["plan", "--target-alignment", "context", "--question-format", "vqa"]), 0)
        plan = json.loads(captured.getvalue())
        self.assertEqual(plan["target_alignment"], "context")
        self.assertEqual(plan["question_format"], "vqa")

    def test_completed_report_cannot_override_timeout_or_nonzero_exit(self):
        for failure in ("timeout", "nonzero"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "run"

                def simulate(command, **kwargs):
                    phase = command[command.index("--phase") + 1]
                    indices = range(3) if phase == "edit" else range(3, 8)
                    report = {"execution_status": "completed", "checks": {str(index): True for index in indices}}
                    # Eight successful checks cannot excuse a failed process exit.
                    write_json(output / (phase + ".json"), report)
                    if phase == "reload":
                        if failure == "timeout":
                            raise subprocess.TimeoutExpired(command, 1)
                        return SimpleNamespace(returncode=2)
                    return SimpleNamespace(returncode=0)

                with patch("ke_repro.smoke.preflight", return_value={"assets_verified": True}), \
                        patch("ke_repro.smoke.subprocess.run", side_effect=simulate), \
                        redirect_stdout(io.StringIO()):
                    code = main(["smoke", "--output", str(output)])
                result = json.loads((output / "smoke.json").read_text())
                self.assertEqual(code, 1)
                self.assertFalse(result["technical_gate_passed"])
                self.assertEqual(result["execution_status"], "failed")


if __name__ == "__main__":
    unittest.main()

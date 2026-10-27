"""Execute generated launch guards with process fixtures; never load a model."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_depth_notebook import build as depth_notebook
from scripts.build_reasonke_notebook import build as chat_notebook
from tests.test_operations_notebooks import (COMMIT, ProcessFixture, SubprocessFixture,
                                             cell_with_flag, code_for, source)


class DepthNotebookTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.process = SubprocessFixture()
        self.notebook = depth_notebook(COMMIT)
        self.namespace = {"DIRECTORY": self.directory, "REPO": self.directory,
                          "PYTHON": "fixture-python", "subprocess": self.process}

    def execute(self, item, *, enable=None):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_for(item, enable=enable), self.namespace)

    def profile(self):
        return next(item for item in self.notebook["cells"]
                    if item["cell_type"] == "code" and
                    "DEPTH_JOB = subprocess.Popen" in source(item))

    def test_generated_cells_parse_and_source_is_pinned(self):
        for notebook in (self.notebook, chat_notebook(COMMIT, context_policy="iterative")):
            restored = json.loads(json.dumps(notebook))
            for index, item in enumerate(restored["cells"]):
                if item["cell_type"] == "code":
                    with self.subTest(name=restored["metadata"]["colab"]["name"], index=index):
                        code_for(item)
                        self.assertEqual(item["outputs"], [])
            setup = ast.parse(source(restored["cells"][1]))
            pin = next(node for node in setup.body if isinstance(node, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id == "SOURCE_COMMIT" for t in node.targets))
            self.assertEqual(ast.literal_eval(pin.value), COMMIT)
        for invalid in ("main", "shortsha", "x" * 40):
            with self.subTest(pin=invalid), self.assertRaises(ValueError):
                depth_notebook(invalid)

    def test_run_all_launches_profile_only_and_reuses_running_process(self):
        self.execute(self.profile())
        self.execute(self.profile())
        for flag in ("RUN_DEVELOPMENT", "RUN_FREEZE", "RUN_FINAL", "RUN_SCORE"):
            self.execute(cell_with_flag(self.notebook, flag))
            self.assertFalse(self.namespace[flag])
        self.assertEqual(len(self.process.launches), 1)
        self.assertIn("profile", self.process.launches[0])
        self.assertEqual(self.process.commands, [])
        self.assertFalse((self.directory / "protocol.json").exists())
        self.assertFalse((self.directory / "results.json").exists())

    def test_enabled_development_cannot_pass_a_running_or_failed_profile(self):
        stage = cell_with_flag(self.notebook, "RUN_DEVELOPMENT")
        for result in (None, 1):
            self.namespace["DEPTH_JOB"] = ProcessFixture(result)
            with self.subTest(exit=result), self.assertRaisesRegex(RuntimeError, "Complete the profile"):
                self.execute(stage, enable="RUN_DEVELOPMENT")
        self.assertEqual(self.process.launches, [])
        self.namespace["DEPTH_JOB"] = ProcessFixture(0)
        self.execute(stage, enable="RUN_DEVELOPMENT")
        self.assertEqual(len(self.process.launches), 1)
        self.assertIn("development", self.process.launches[0])

    def test_final_requires_protocol_and_idle_worker(self):
        stage = cell_with_flag(self.notebook, "RUN_FINAL")
        self.namespace.update(PROTOCOL=self.directory / "protocol.json", DEPTH_JOB=ProcessFixture(0))
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            self.execute(stage, enable="RUN_FINAL")
        # Notebook only gates the file here. The real runner verifies all bound
        # inputs and source hashes; this fixture tests dispatch, not protocol validity.
        self.namespace["PROTOCOL"].write_text("fixture-not-a-real-protocol")
        self.namespace["DEPTH_JOB"] = ProcessFixture()
        with self.assertRaisesRegex(RuntimeError, "phase is already running"):
            self.execute(stage, enable="RUN_FINAL")
        self.assertEqual(self.process.launches, [])
        self.namespace["DEPTH_JOB"] = ProcessFixture(0)
        self.execute(stage, enable="RUN_FINAL")
        command = self.process.launches[0]
        self.assertIn("final", command)
        self.assertEqual(command[command.index("--protocol") + 1], str(self.namespace["PROTOCOL"]))

    def test_india_launcher_is_separate_and_keeps_optional_generations_disabled(self):
        original = chat_notebook(COMMIT)
        india = chat_notebook(COMMIT, context_policy="iterative")
        self.assertEqual(original["metadata"]["colab"]["name"], "04_reasonke_chat.ipynb")
        self.assertEqual(india["metadata"]["colab"]["name"], "07_india_chat.ipynb")
        joined = "\n".join(source(item) for item in india["cells"])
        self.assertIn("'--context-policy', 'iterative'", joined)
        self.assertNotIn("'--context-policy', 'retrieved'", joined)
        self.assertIn("'--context-policy', 'retrieved'", "\n".join(source(i) for i in original["cells"]))
        self.assertIn("'artifacts/reasonke/india-chat-1'", joined)
        self.assertIn("fictional", source(india["cells"][0]))
        for flag in ("RUN_STARTUP_CHECKS", "RUN_LIVE_CHECKS"):
            tree = ast.parse(source(cell_with_flag(india, flag)))
            assignment = next(node for node in tree.body if isinstance(node, ast.Assign) and
                              any(isinstance(t, ast.Name) and t.id == flag for t in node.targets))
            self.assertFalse(ast.literal_eval(assignment.value))
        with self.assertRaises(ValueError):
            chat_notebook(COMMIT, context_policy="unbounded")

    def test_scoring_requires_successful_final_and_preserves_existing_scorecard(self):
        stage = cell_with_flag(self.notebook, "RUN_SCORE")
        self.namespace["PROTOCOL"] = self.directory / "protocol.json"
        for result in (None, 1):
            self.namespace["DEPTH_JOB"] = ProcessFixture(result)
            with self.subTest(exit=result), self.assertRaisesRegex(RuntimeError, "Complete the final phase"):
                self.execute(stage, enable="RUN_SCORE")
        self.assertEqual(self.process.commands, [])
        self.namespace["DEPTH_JOB"] = ProcessFixture(0)
        with patch.object(self.process, "run") as run:
            self.execute(stage, enable="RUN_SCORE")
            command = run.call_args.args[0]
            self.assertIn("editguard_depth.metrics", command)
            self.assertEqual(command[command.index("--protocol") + 1], str(self.namespace["PROTOCOL"]))
            self.assertEqual(command[command.index("--output") + 1], str(self.directory / "results.json"))
            self.assertIn(str(self.directory / "final-reasonke.json"), command)
            self.assertIn(str(self.directory / "final-instruction.json"), command)
            self.assertTrue(run.call_args.kwargs["check"])
            run.assert_called_once()
        scorecard = self.directory / "results.json"
        scorecard.write_text("preserved-existing-evidence")
        self.execute(stage, enable="RUN_SCORE")
        self.assertEqual(scorecard.read_text(), "preserved-existing-evidence")
        self.assertEqual(self.process.commands, [])

    def test_status_prints_measured_budget_only_for_complete_profile(self):
        status = next(item for item in self.notebook["cells"] if item["cell_type"] == "code"
                      and "Current phase exit:" in source(item))
        self.namespace.update(DEPTH_JOB=ProcessFixture(0), json=json)
        (self.directory / "pipeline.log").write_text("fixture-status")
        receipt = {"status": "completed", "records": [
            {"outcome": "wrong", "pipeline_seconds": n,
             "response": {"generation": {"peak_allocated_bytes": 100 + n}}} for n in range(1, 9)],
            "sessions": [{"load_wall_seconds": 15, "runtime": {"load_seconds": 12,
                          "hardware": {"name": "CPU test fixture", "load_peak_allocated_bytes": 99}}}]}
        profile = self.directory / "profile-reasonke.json"
        profile.write_text(json.dumps(receipt))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code_for(status), self.namespace)
        text = output.getvalue()
        budget = json.loads(text[text.index('{'):])
        self.assertEqual(budget["pipeline_sum_seconds"], 36)
        self.assertEqual(budget["pipeline_mean_seconds"], 4.5)
        self.assertEqual(budget["forecast_272_mean_seconds"], 1224)
        self.assertEqual(budget["forecast_272_max_seconds"], 2176)
        self.assertEqual(budget["generation_peak_allocated_bytes"], 108)
        self.assertEqual(budget["load_sessions"][0]["load_wall_seconds"], 15)
        self.assertIn("Model loading, downloads", budget["forecast_excludes"])
        receipt["records"].pop()
        profile.write_text(json.dumps(receipt))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(code_for(status), self.namespace)
        self.assertNotIn('"profile_budget"', output.getvalue())
        self.assertIn("No complete eight-response profile budget yet", output.getvalue())


if __name__ == "__main__":
    unittest.main()

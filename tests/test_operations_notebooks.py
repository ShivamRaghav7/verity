"""Execute notebook stage guards with explicit process fixtures, never a GPU model."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.build_reasonke_notebook import build as chat_notebook
from scripts.build_operations_notebook import build as operations_notebook

COMMIT = "edb7131cc55ed94675407ad9e4c1dfa8a479b61f"


def source(cell):
    return "".join(cell["source"])


def cell_with_flag(notebook, flag):
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and any(
                isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == flag
                                                     for target in node.targets)
                for node in ast.parse(source(cell)).body):
            return cell
    raise AssertionError("Missing explicit notebook stage: " + flag)


def code_for(cell, *, enable=None):
    tree = ast.parse(source(cell))
    if enable:
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == enable for target in node.targets)]
        if len(assignments) != 1:
            raise AssertionError("Expected one stage switch: " + enable)
        # Simulate the operator changing the exposed switch in the actual generated
        # cell. The remaining executable code and ordering stay unchanged.
        assignments[0].value = ast.Constant(True)
        ast.fix_missing_locations(tree)
    return compile(tree, "<generated-notebook-cell>", "exec")


class ProcessFixture:
    def __init__(self, exit_code=None):
        self.returncode = exit_code

    def poll(self):
        return self.returncode


class SubprocessFixture:
    STDOUT = -2

    def __init__(self):
        self.launches = []
        self.commands = []
        self.head = COMMIT

    def Popen(self, command, **kwargs):
        self.launches.append(list(command))
        return ProcessFixture()

    def check_output(self, command, **kwargs):
        self.commands.append(list(command))
        return self.head + "\n"

    def run(self, command, **kwargs):
        self.commands.append(list(command))
        if "--freeze-profiles" not in command:
            raise AssertionError("Unexpected executable command in notebook fixture: " + str(command))
        # The real freeze subprocess owns the initial file; the notebook then
        # annotates it with development selection and writes it again.
        target = Path(command[command.index("--output") + 1])
        target.write_text(json.dumps({"schema": "fixture-protocol", "status": "frozen", "sources": {}}) + "\n",
                          encoding="utf-8")
        return SimpleNamespace(stdout="Fixture freeze completed", stderr="", check_returncode=lambda: None)


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.process = SubprocessFixture()
        self.namespace = {"Path": Path, "json": json, "os": os, "secrets": secrets, "time": time,
                          "REPO": self.repo, "PYTHON": "explicit-notebook-test-python",
                          "subprocess": self.process, "inventory": {"fixture": True}}
        self.operations = operations_notebook(COMMIT)
        self.chat = chat_notebook(COMMIT)

    def execute(self, cell, *, enable=None):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code_for(cell, enable=enable), self.namespace)

    def stage(self, flag):
        return cell_with_flag(self.operations, flag)

    def test_all_generated_cells_parse_and_reject_unpinned_source(self):
        for notebook in (self.chat, self.operations):
            # JSON serialization/reopening must preserve valid executable cells,
            # including escapes nested inside generated notebook strings.
            reopened = json.loads(json.dumps(notebook))
            for index, cell in enumerate(reopened["cells"]):
                if cell["cell_type"] == "code":
                    with self.subTest(notebook=notebook["metadata"]["colab"]["name"], cell=index):
                        code_for(cell)
            setup = source(notebook["cells"][1])
            assignment = next(node for node in ast.parse(setup).body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "SOURCE_COMMIT" for t in node.targets))
            self.assertEqual(ast.literal_eval(assignment.value), COMMIT)
        for invalid in ("main", "edb7131", "x" * 40):
            with self.subTest(commit=invalid), self.assertRaises(ValueError):
                chat_notebook(invalid)
            with self.subTest(commit=invalid), self.assertRaises(ValueError):
                operations_notebook(invalid)

    def test_source_mismatch_refuses_setup_before_install_or_checkout(self):
        self.namespace["SOURCE_COMMIT"] = COMMIT
        self.process.head = "0" * 40
        with self.assertRaisesRegex(RuntimeError, "Existing checkout differs"):
            self.execute(self.chat["cells"][2])
        self.assertEqual(len(self.process.commands), 1)
        self.assertEqual(self.process.commands[0][-2:], ["rev-parse", "HEAD"])
        self.assertEqual(self.process.launches, [])

    def test_chat_default_launches_one_server_without_optional_generations(self):
        worker = cell_with_flag(self.chat, "RUN_STARTUP_CHECKS")
        self.execute(worker)
        self.assertFalse(self.namespace["RUN_STARTUP_CHECKS"])
        self.assertEqual(len(self.process.launches), 1)
        command = self.process.launches[0]
        self.assertIn("--serve", command)
        self.assertIn("editguard_reasonke", command)
        self.assertNotIn("--profile", command)
        self.assertNotIn("--output", command)
        self.execute(worker)
        self.assertEqual(len(self.process.launches), 1, "Re-running must reuse the still-running worker")
        self.execute(cell_with_flag(self.chat, "RUN_LIVE_CHECKS"))
        self.assertFalse(self.namespace["RUN_LIVE_CHECKS"])
        self.assertEqual(self.process.commands, [])
        self.assertEqual(len(self.process.launches), 1)

    def test_run_all_starts_profiles_only_and_keeps_later_stages_disabled(self):
        # Execute actual launch cells in notebook order, while the profile process
        # deliberately remains running. No import, install, sleep or model occurs.
        profile = next(cell for cell in self.operations["cells"] if cell["cell_type"] == "code"
                       and any(isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "DIRECTORY"
                                                                 for t in node.targets)
                               for node in ast.parse(source(cell)).body))
        ordered = [profile, self.stage("RUN_DEVELOPMENT"), self.stage("RUN_FREEZE"), self.stage("RUN_FINAL")]
        indices = [self.operations["cells"].index(cell) for cell in ordered]
        self.assertEqual(indices, sorted(indices))
        for cell in ordered:
            self.execute(cell)
        self.assertEqual(len(self.process.launches), 1)
        command = self.process.launches[0]
        self.assertEqual(command[command.index("--phase") + 1], "profile")
        self.assertEqual(self.process.commands, [])
        self.assertFalse(self.namespace["RUN_DEVELOPMENT"])
        self.assertFalse(self.namespace["RUN_FREEZE"])
        self.assertFalse(self.namespace["RUN_FINAL"])
        self.assertFalse(self.namespace["PROTOCOL"].exists())

    def test_enabled_development_and_freeze_cannot_jump_past_running_predecessors(self):
        directory = self.repo / "run"
        directory.mkdir()
        self.namespace.update(DIRECTORY=directory, PROFILE_JOB=ProcessFixture())
        with self.assertRaisesRegex(RuntimeError, "Both profile jobs"):
            self.execute(self.stage("RUN_DEVELOPMENT"), enable="RUN_DEVELOPMENT")
        self.assertEqual(self.process.launches, [])
        self.namespace["PROFILE_JOB"].returncode = 0
        self.execute(self.stage("RUN_DEVELOPMENT"), enable="RUN_DEVELOPMENT")
        self.assertEqual(len(self.process.launches), 1)
        self.assertIn("development", self.process.launches[0])
        with patch("editguard_benchmark.metrics.aggregate") as aggregate:
            with self.assertRaisesRegex(RuntimeError, "Complete both development jobs"):
                self.execute(self.stage("RUN_FREEZE"), enable="RUN_FREEZE")
            aggregate.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            self.execute(self.stage("RUN_FINAL"), enable="RUN_FINAL")
        self.assertEqual(len(self.process.launches), 1)
        self.assertEqual(self.process.commands, [])

    def test_enabled_freeze_roundtrips_json_with_actual_newline_and_binds_development(self):
        directory = self.repo / "run"
        directory.mkdir()
        self.namespace.update(DIRECTORY=directory, DEVELOPMENT_JOB=ProcessFixture(0))
        for name in ("reasonke", "instruction"):
            (directory / ("development-" + name + ".json")).write_text(json.dumps({"fixture_model": name}) + "\n",
                                                                        encoding="utf-8")
        development = {"execution_status": "completed", "conditions": {
            key: {"overall": {"counts": {"correct": value}}} for key, value in
            (("R-RAG", 24), ("R-ALL", 21), ("G-RAG", 24), ("G-ALL", 22))}}
        with patch("editguard_benchmark.metrics.aggregate", return_value=development) as aggregate:
            self.execute(self.stage("RUN_FREEZE"), enable="RUN_FREEZE")
            aggregate.assert_called_once_with([directory / "development-reasonke.json", directory / "development-instruction.json"])
        protocol = self.namespace["PROTOCOL"]
        raw = protocol.read_bytes()
        self.assertTrue(raw.endswith(b"\n"), "A literal backslash-n after the JSON would make the protocol invalid")
        reopened = json.loads(raw)
        selection = reopened["development_selection"]
        self.assertEqual(selection["reasonke_context_policy"], "retrieved")
        self.assertEqual(selection["reasonke_retrieved_correct"], 24)
        self.assertEqual(selection["reasonke_full_memory_correct"], 21)
        for name, expected in selection["receipt_sha256"].items():
            self.assertEqual(expected, hashlib.sha256((directory / ("development-" + name + ".json")).read_bytes()).hexdigest())
        self.assertEqual(reopened["explanation_audit"]["assigned_raw_traces"], 32)
        self.assertEqual(self.process.launches, [], "Freeze performs no model process launch")
        with patch("editguard_benchmark.metrics.aggregate", side_effect=AssertionError("Do not overwrite a frozen protocol")):
            self.execute(self.stage("RUN_FREEZE"), enable="RUN_FREEZE")
        self.assertEqual(protocol.read_bytes(), raw)
        self.assertEqual(len(self.process.commands), 1)
        self.execute(self.stage("RUN_FINAL"), enable="RUN_FINAL")
        self.assertEqual(len(self.process.launches), 1)
        command = self.process.launches[0]
        self.assertEqual(command[command.index("--phase") + 1], "final")
        self.assertEqual(command[command.index("--protocol") + 1], str(protocol))

    def test_incomplete_development_cannot_produce_protocol(self):
        directory = self.repo / "run"
        directory.mkdir()
        self.namespace.update(DIRECTORY=directory, DEVELOPMENT_JOB=ProcessFixture(0))
        with patch("editguard_benchmark.metrics.aggregate", return_value={"execution_status": "incomplete"}):
            with self.assertRaisesRegex(RuntimeError, "unexecuted outcomes"):
                self.execute(self.stage("RUN_FREEZE"), enable="RUN_FREEZE")
        self.assertFalse((directory / "protocol.json").exists())
        self.assertEqual(self.process.commands, [])
        self.assertEqual(self.process.launches, [])


if __name__ == "__main__":
    unittest.main()

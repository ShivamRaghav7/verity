"""Depth-study execution contracts using CPU process fixtures, never model calls."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from editguard_benchmark.data import encoded
from editguard_benchmark.run import source_identity
from editguard_benchmark.scoring import score
from editguard_depth import data, metrics, run as runner
from editguard_reasonke.runtime import extract_answer
from scripts import run_depth_pipeline


ROOT = Path(__file__).resolve().parents[1]


class _Process:
    def __init__(self):
        self.started = False
        self.alive = False
        self.terminated = False
        self.joins = []

    def start(self):
        self.started = self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        # An intentionally uncooperative fixture exercises forced termination.
        self.joins.append(timeout)

    def terminate(self):
        self.terminated = True
        self.alive = False


class _Pipe:
    def __init__(self, model_manifest, targets, outcomes=(), polls=()):
        self.manifest, self.targets = model_manifest, targets
        self.outcomes, self.polls = iter(outcomes), iter(polls)
        self.sent, self.poll_budgets = [], []
        self.loaded = self.closed = False

    def poll(self, timeout):
        self.poll_budgets.append(timeout)
        return next(self.polls, True)

    def recv(self):
        if not self.loaded:
            self.loaded = True
            return {"ready": True, "runtime": {
                "ready": True, "model_id": self.manifest["model_id"],
                "revision": self.manifest["revision"], "load_seconds": 0.01,
                "hardware": {"name": "Explicit CPU process fixture; no GPU"},
                "software_fixture": True}}
        outcome = next(self.outcomes, "abstain")
        if outcome == "error":
            return {"error": {"type": "RuntimeError", "message": "Explicit CPU error fixture"}}
        payload = self.sent[-1]
        text = (self.targets[payload["question"]] if outcome == "correct" else
                "Explicit incorrect CPU fixture" if outcome == "wrong" else "UNKNOWN")
        raw = "<answer>" + text + "</answer>"
        content, fmt = extract_answer(raw)
        return {"response": {"content": content, "raw_output": raw, "format": fmt,
            "generation": {"model": self.manifest["model_id"], "revision": self.manifest["revision"],
                "fact_ids": [fact["id"] for fact in payload["facts"]],
                "max_new_tokens": payload["max_new_tokens"], "max_input_tokens": 4096,
                "do_sample": False, "software_fixture": True}}}

    def send(self, value):
        self.sent.append(copy.deepcopy(value))

    def close(self):
        self.closed = True


class _Context:
    def __init__(self, model_manifest, targets, outcomes=(), polls=()):
        self.parent = _Pipe(model_manifest, targets, outcomes, polls)
        self.child = _Pipe(model_manifest, targets)
        self.process = _Process()
        self.process_arguments = None

    def Pipe(self):
        return self.parent, self.child

    def Process(self, **kwargs):
        self.process_arguments = kwargs
        return self.process


class DepthRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Author a disposable packet rather than relying on a GPU-run directory.
        cls.packet = tempfile.TemporaryDirectory()
        cls.fixture_root = Path(cls.packet.name)
        paths = list((ROOT / "editguard_depth").glob("*.py"))
        paths += list((ROOT / "editguard_benchmark").glob("*.py"))
        paths += [ROOT / name for name in ("editguard_reasonke/runtime.py", "configs/depth-evaluation.json",
                  "configs/reasonke-model.json", "configs/instruction-model.json")]
        config = json.loads((ROOT / "configs/depth-evaluation.json").read_text())
        if config.get("retrieval_selection"):
            paths.append(ROOT / config["retrieval_selection"])
        for path in paths:
            destination = cls.fixture_root / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        data.write_dataset(cls.fixture_root / "data/reasonke/depth-v1")

    @classmethod
    def tearDownClass(cls):
        cls.packet.cleanup()

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)
        self.config_path = self.fixture_root / "configs/depth-evaluation.json"
        self.config = json.loads(self.config_path.read_text())
        for module in (runner, metrics):
            binding = patch.object(module, "ROOT", self.fixture_root)
            binding.start()
            self.addCleanup(binding.stop)

    def context(self, model="reasonke", phase="profile", outcomes=(), polls=()):
        _, _, tasks = runner.tasks_for(self.config, model, phase)
        manifest = json.loads((self.fixture_root / self.config[model + "_manifest"]).read_text())
        return _Context(manifest, {task["question"]["question"]: task["question"]["expected_answer"]
                                   for task in tasks}, outcomes, polls)

    def execute(self, path, *, model="reasonke", phase="profile", outcomes=(), polls=(), resume=False, protocol=None):
        context = self.context(model, phase, outcomes, polls)
        with patch.object(runner.multiprocessing, "get_context", return_value=context), contextlib.redirect_stdout(io.StringIO()):
            result = runner.run(self.config_path, model, phase, path, protocol=protocol, resume=resume)
        self.assertTrue(context.process.started)
        self.assertTrue(context.parent.closed)
        self.assertTrue(context.child.closed)
        self.assertIs(context.process_arguments["target"], runner._worker)
        self.assertTrue(context.process_arguments["daemon"])
        return result, context

    def inputs(self, directory=None):
        directory = Path(directory or self.output)
        for model in ("reasonke", "instruction"):
            for phase in ("profile", "development"):
                self.execute(directory / f"{phase}-{model}.json", model=model, phase=phase)
        return directory

    def sealed(self):
        directory = self.inputs()
        path = directory / "protocol.json"
        return path, runner.freeze(directory, path, self.config_path)

    @staticmethod
    def save(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_finite_populations_and_preselected_oracle_are_matched(self):
        for phase, expected in (("profile", 8), ("development", 24), ("final", 272)):
            _, _, reasonke = runner.tasks_for(self.config, "reasonke", phase)
            _, _, instruction = runner.tasks_for(self.config, "instruction", phase)
            self.assertEqual(len(reasonke), expected)
            self.assertEqual(len(instruction), expected)
            self.assertEqual([row["question"] for row in reasonke], [row["question"] for row in instruction])
            if phase != "final":
                self.assertEqual({row["question"]["split"] for row in reasonke}, {"development"})
                self.assertEqual(len({row["question"]["category"] for row in reasonke}), 8)
                continue
            main = [row for row in reasonke if row["condition"] == "R-RAG"]
            oracle = [row for row in reasonke if row["condition"] == "R-ORACLE"]
            self.assertEqual((len(main), len(oracle)), (256, 16))
            expected_questions = sorted((row["question"] for row in main if row["question"]["depth"] == 3),
                key=lambda question: hashlib.sha256(("depth-oracle-43:" + question["id"]).encode()).hexdigest())[:16]
            self.assertEqual([row["question"] for row in oracle], expected_questions)
            self.assertEqual({row["question"]["split"] for row in reasonke}, {"final"})

    def test_fact_selection_and_budgets_match_without_gold_target_leakage(self):
        self.assertEqual(self.config["retrieval_k"], 4)
        self.assertEqual(self.config["max_new_tokens"], 512)
        self.assertEqual(self.config["request_watchdog_seconds"], 240)
        self.assertEqual(self.config["model_load_watchdog_seconds"], 1800)
        for phase in ("profile", "development", "final"):
            _, worlds, left = runner.tasks_for(self.config, "reasonke", phase)
            _, _, right = runner.tasks_for(self.config, "instruction", phase)
            for one, two in zip(left, right):
                with self.subTest(phase=phase, case=one["id"]):
                    payload, routing = runner.request_for(one, worlds, self.config)
                    other, other_routing = runner.request_for(two, worlds, self.config)
                    self.assertEqual(payload, other)
                    self.assertEqual(routing, other_routing)
                    self.assertEqual(payload["history"], [])
                    self.assertEqual(payload["max_new_tokens"], 512)
                    if one["condition"].endswith("ORACLE"):
                        self.assertTrue(routing["complete"])
                        self.assertTrue(routing["gold_support_used"])
                        self.assertEqual(len(payload["facts"]), 3)
                    else:
                        self.assertFalse(routing["gold_support_used"])
                        self.assertLessEqual(len(payload["facts"]), 4)
                        poisoned = copy.deepcopy(one)
                        poisoned["question"].update(expected_answer="HIDDEN_TARGET_FIXTURE",
                            expected_aliases=["HIDDEN_TARGET_FIXTURE"], required_fact_ids=["hidden-support-fixture"])
                        hidden_payload, _ = runner.request_for(poisoned, worlds, self.config)
                        self.assertEqual(payload, hidden_payload)
                    self.assertNotIn("expected_answer", payload)

    def test_new_study_does_not_change_completed_operations_identities(self):
        old_path = ROOT / "configs/operations-evaluation.json"
        frozen = json.loads((ROOT / "reports/reasonke/operations-final/protocol.json").read_text())
        self.assertEqual(source_identity(old_path), frozen["sources"])
        depth_sources = runner.identities(self.config_path)
        self.assertIn("editguard_depth/run.py", depth_sources)
        self.assertIn("editguard_depth/data.py", depth_sources)
        self.assertIn("configs/depth-evaluation.json", depth_sources)
        self.assertIn("data/reasonke/depth-v1/manifest.json", depth_sources)
        self.assertFalse(any(name.startswith("editguard_depth/") for name in source_identity(old_path)))
        self.assertEqual(source_identity(old_path), frozen["sources"])

    def test_final_requires_valid_protocol_before_final_data_or_worker(self):
        with patch.object(runner, "tasks_for") as tasks, patch.object(runner.multiprocessing, "get_context") as launch:
            with self.assertRaises(ValueError):
                runner.run(self.config_path, "reasonke", "final", self.output / "no-protocol.json")
            tasks.assert_not_called()
            launch.assert_not_called()
        invalid = self.output / "invalid-protocol.json"
        self.save(invalid, {"status": "frozen", "sources": runner.identities(self.config_path)})
        with patch.object(runner, "tasks_for") as tasks, patch.object(runner.multiprocessing, "get_context") as launch:
            with self.assertRaises(ValueError):
                runner.run(self.config_path, "reasonke", "final", self.output / "bad-plan.json", protocol=invalid)
            tasks.assert_not_called()
            launch.assert_not_called()
        self.assertFalse((self.output / "no-protocol.json").exists())
        self.assertFalse((self.output / "bad-plan.json").exists())

    def test_every_completed_outcome_is_retained_and_resume_loads_no_model(self):
        path = self.output / "outcomes.json"
        receipt, context = self.execute(path, outcomes=("correct", "wrong", "abstain", "error") * 2)
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual((receipt["recorded"], receipt["unexecuted"]), (8, 0))
        self.assertEqual([row["outcome"] for row in receipt["records"]],
                         ["correct", "wrong", "abstain", "execution_error"] * 2)
        self.assertEqual(sum(isinstance(payload, dict) for payload in context.parent.sent), 8)
        aggregate = metrics.audit_and_aggregate([path], config_path=self.config_path)
        summary = aggregate["conditions"]["R-RAG"]["overall"]
        self.assertEqual((summary["assigned"], summary["accuracy"], summary["coverage"]), (8, .25, .5))
        before = path.read_bytes()
        with patch.object(runner.multiprocessing, "get_context") as launch:
            resumed = runner.run(self.config_path, "reasonke", "profile", path, resume=True)
        launch.assert_not_called()
        self.assertEqual(resumed["records"], receipt["records"])
        self.assertEqual(path.read_bytes(), before)
        with self.assertRaises(ValueError):
            runner.run(self.config_path, "reasonke", "profile", path)

    def test_hard_request_error_is_preserved_and_only_unfinished_records_resume(self):
        path = self.output / "timeout.json"
        interrupted, context = self.execute(path, polls=(True, False))
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertEqual((interrupted["recorded"], interrupted["unexecuted"]), (1, 7))
        self.assertEqual(interrupted["records"][0]["outcome"], "execution_error")
        self.assertEqual(context.parent.poll_budgets, [1800, 240])
        self.assertTrue(context.process.terminated)
        saved_error = copy.deepcopy(interrupted["records"][0])
        completed, resumed = self.execute(path, resume=True)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["records"][0], saved_error)
        sent = [payload for payload in resumed.parent.sent if isinstance(payload, dict)]
        self.assertEqual(len(sent), 7)
        self.assertNotIn(saved_error["question"]["question"], [payload["question"] for payload in sent])
        self.assertEqual((completed["recorded"], completed["unexecuted"]), (8, 0))

    def test_load_watchdog_stops_owned_worker_with_all_cases_unexecuted(self):
        receipt, context = self.execute(self.output / "load-timeout.json", polls=(False,))
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual((receipt["recorded"], receipt["unexecuted"]), (0, 8))
        self.assertEqual(context.parent.poll_budgets, [1800])
        self.assertTrue(context.process.terminated)
        self.assertFalse(any(isinstance(payload, dict) for payload in context.parent.sent))

    def test_inflight_transport_failure_is_retained_without_retrying_that_question(self):
        path = self.output / "transport-failure.json"
        context = self.context()
        receive = context.parent.recv
        calls = 0
        def lost_connection():
            nonlocal calls
            calls += 1
            if calls == 1:
                return receive()
            raise EOFError("Explicit worker disconnect fixture")
        with patch.object(runner.multiprocessing, "get_context", return_value=context), \
                patch.object(context.parent, "recv", side_effect=lost_connection), \
                contextlib.redirect_stdout(io.StringIO()):
            receipt = runner.run(self.config_path, "reasonke", "profile", path)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual((receipt["recorded"], receipt["unexecuted"]), (1, 7))
        failed = copy.deepcopy(receipt["records"][0])
        self.assertEqual(failed["outcome"], "execution_error")
        self.assertEqual(failed["error"]["type"], "EOFError")
        completed, resumed = self.execute(path, resume=True)
        self.assertEqual(completed["records"][0], failed)
        self.assertEqual(sum(isinstance(payload, dict) for payload in resumed.parent.sent), 7)

    def test_audit_checks_actual_budgets_greedy_setting_status_and_latency(self):
        source = self.output / "provenance-source.json"
        receipt, _ = self.execute(source)
        def output_budget(value): value["records"][0]["response"]["generation"]["max_new_tokens"] = 1024
        def input_budget(value): value["records"][0]["response"]["generation"]["max_input_tokens"] = 8192
        def sampling(value): value["records"][0]["response"]["generation"]["do_sample"] = True
        def phase(value): value["plan"]["phase"] = "unknown"
        def status(value): value["status"] = "unknown"
        def latency(value): value["records"][0]["pipeline_seconds"] = float("nan")
        for mutate in (output_budget, input_budget, sampling, phase, status, latency):
            with self.subTest(change=mutate.__name__):
                value = copy.deepcopy(receipt)
                mutate(value)
                path = self.output / ("audit-" + mutate.__name__ + ".json")
                self.save(path, value)
                with self.assertRaises(ValueError):
                    metrics.audit_and_aggregate([path], config_path=self.config_path)
        with self.assertRaises(ValueError):
            metrics.audit_and_aggregate([], config_path=self.config_path)

    def test_completed_status_cannot_hide_unfinished_assigned_records(self):
        path = self.output / "mislabelled-completion.json"
        receipt, _ = self.execute(path, polls=(False,))
        receipt["status"] = "completed"
        self.save(path, receipt)
        try:
            report = metrics.audit_and_aggregate([path], config_path=self.config_path)
        except ValueError:
            return  # Refusing the inconsistent receipt also satisfies the gate.
        self.assertNotEqual(report["execution_status"], "completed")

    def test_finite_model_wall_budget_stops_before_another_request(self):
        clock = iter((0.0, 1.0, 30000.0))
        with patch.object(runner.time, "perf_counter", side_effect=lambda: next(clock, 30001.0)):
            receipt, context = self.execute(self.output / "wall-timeout.json")
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual((receipt["recorded"], receipt["unexecuted"]), (0, 8))
        self.assertFalse(any(isinstance(payload, dict) for payload in context.parent.sent))
        self.assertTrue(context.process.terminated)

    def test_resume_rejects_resealed_labels_context_provenance_or_score_changes(self):
        source = self.output / "original.json"
        receipt, _ = self.execute(source)
        def question(value): value["records"][0]["question"]["expected_answer"] = "Changed target"
        def condition(value): value["records"][0]["condition"] = "G-RAG"
        def facts(value): value["records"][0]["supplied_facts"][0]["answer"] = "Changed fact"
        def routing(value): value["records"][0]["retrieval"]["selected_fact_ids"] = ["Changed ID"]
        def extraction(value): value["records"][0]["response"]["content"] = "Changed extraction"
        def provenance(value): value["records"][0]["response"]["generation"]["revision"] = "0" * 40
        def scoring(value): value["records"][0].update(outcome="correct", correct=True, answered=True)
        def dataset(value): value["dataset"]["dataset_id"] = "Changed dataset"
        def plan(value): value["plan"]["task_ids_sha256"] = "0" * 64
        def unexecuted_with_response(value): value["records"][0]["outcome"] = "unexecuted"
        for mutate in (question, condition, facts, routing, extraction, provenance, scoring, dataset, plan, unexecuted_with_response):
            with self.subTest(change=mutate.__name__):
                value = copy.deepcopy(receipt)
                mutate(value)
                path = self.output / (mutate.__name__ + ".json")
                self.save(path, value)
                before = path.read_bytes()
                with patch.object(runner.multiprocessing, "get_context") as launch:
                    with self.assertRaises(ValueError):
                        runner.run(self.config_path, "reasonke", "profile", path, resume=True)
                launch.assert_not_called()
                self.assertEqual(path.read_bytes(), before)

    def test_freeze_binds_all_four_exact_receipts_and_measured_budget(self):
        path, protocol = self.sealed()
        self.assertEqual(protocol["final_assigned"], 544)
        self.assertEqual((protocol["per_main_condition"], protocol["oracle_per_model"]), (256, 16))
        expected = {f"{phase}-{model}.json" for phase in ("profile", "development")
                    for model in ("reasonke", "instruction")}
        self.assertEqual(set(protocol["input_receipts"]), expected)
        for name in expected:
            self.assertEqual(hashlib.sha256((path.parent / name).read_bytes()).hexdigest(), protocol["input_receipts"][name])
        for profile in protocol["profiles"].values():
            self.assertEqual((profile["n"], profile["final_assigned"]), (8, 272))
            self.assertAlmostEqual(profile["forecast_seconds_mean"], 272 * profile["mean_seconds"])
            self.assertAlmostEqual(profile["forecast_seconds_max"], 272 * profile["max_seconds"])
        self.assertEqual(runner.verify_protocol(path, self.config_path), protocol)
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            runner.freeze(path.parent, path, self.config_path)
        self.assertEqual(path.read_bytes(), before)

    def test_freeze_keeps_protocol_beside_its_bound_input_receipts(self):
        destination = self.output / "elsewhere" / "protocol.json"
        with self.assertRaises(ValueError):
            runner.freeze(self.output, destination, self.config_path)
        self.assertFalse(destination.exists())

    def test_freeze_rejects_forged_tasks_scores_or_invalid_profile_measurement(self):
        directory = self.inputs()
        profile = directory / "profile-reasonke.json"
        original = profile.read_bytes()
        for change in ("question", "score", "task_digest", "negative_timing", "missing_generation"):
            with self.subTest(change=change):
                value = json.loads(original)
                row = value["records"][0]
                if change == "question": row["question"]["expected_answer"] = "Changed target"
                elif change == "score": row.update(outcome="correct", correct=True, answered=True)
                elif change == "task_digest": value["plan"]["task_ids_sha256"] = "0" * 64
                elif change == "negative_timing": row["pipeline_seconds"] = -1.0
                else:
                    row.pop("response")
                    row["error"] = {"type": "RuntimeError", "message": "Explicit failed profile fixture"}
                    row.update(score(row["question"], error=row["error"]))
                self.save(profile, value)
                protocol = directory / ("rejected-" + change + ".json")
                with self.assertRaises(ValueError):
                    runner.freeze(directory, protocol, self.config_path)
                self.assertFalse(protocol.exists())
                profile.write_bytes(original)

    def test_protocol_rejects_missing_inputs_tampering_and_inconsistent_forecasts(self):
        path, protocol = self.sealed()
        def schema(value): value["schema"] = "not-the-depth-protocol"
        def sources(value): value["sources"]["editguard_depth/run.py"] = "0" * 64
        def population(value): value["final_assigned"] = 545
        def main(value): value["per_main_condition"] = 255
        def oracle(value): value["oracle_per_model"] = 17
        def profiles(value): value["profiles"].pop("instruction")
        def samples(value): value["profiles"]["reasonke"]["n"] = 7
        def forecast(value): value["profiles"]["reasonke"]["forecast_seconds_mean"] += 1
        def inputs(value): value["input_receipts"].pop("development-instruction.json")
        def checksum(value): value["input_receipts"]["profile-reasonke.json"] = "0" * 64
        for mutate in (schema, sources, population, main, oracle, profiles, samples, forecast, inputs, checksum):
            with self.subTest(change=mutate.__name__):
                value = copy.deepcopy(protocol)
                mutate(value)
                destination = path.parent / ("invalid-" + mutate.__name__ + ".json")
                self.save(destination, value)
                with self.assertRaises(ValueError):
                    runner.verify_protocol(destination, self.config_path)
        selected = path.parent / "development-reasonke.json"
        original = selected.read_bytes()
        selected.unlink()
        with self.assertRaises(ValueError): runner.verify_protocol(path, self.config_path)
        selected.write_bytes(original)

    def test_resealed_wrong_phase_cannot_recurse_into_protocol_validation(self):
        path, protocol = self.sealed()
        selected = path.parent / "profile-reasonke.json"
        receipt = json.loads(selected.read_text())
        receipt["plan"]["phase"] = "final"
        self.save(selected, receipt)
        protocol["input_receipts"][selected.name] = hashlib.sha256(selected.read_bytes()).hexdigest()
        self.save(path, protocol)
        with self.assertRaises(ValueError):
            runner.verify_protocol(path, self.config_path)

    def test_final_run_preserves_all_272_records_and_protocol_identity(self):
        protocol, _ = self.sealed()
        path = self.output / "final-reasonke.json"
        result, context = self.execute(path, phase="final", protocol=protocol)
        self.assertEqual((result["recorded"], result["unexecuted"]), (272, 0))
        self.assertEqual(result["plan"]["protocol_sha256"], hashlib.sha256(protocol.read_bytes()).hexdigest())
        self.assertEqual(sum(isinstance(payload, dict) for payload in context.parent.sent), 272)
        audit = metrics.audit_and_aggregate([path], config_path=self.config_path, protocol=protocol)
        self.assertEqual(audit["assigned_records"], 272)
        self.assertEqual(audit["conditions"]["R-RAG"]["overall"]["assigned"], 256)
        self.assertEqual(audit["conditions"]["R-ORACLE"]["overall"]["assigned"], 16)
        self.assertEqual(audit["audit_model_calls"], 0)

    def test_sequential_pipeline_stops_before_second_model_on_failure(self):
        from types import SimpleNamespace
        arguments = ["run_depth_pipeline", "--phase", "profile", "--directory", str(self.output)]
        with patch("sys.argv", arguments), patch.object(run_depth_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit): run_depth_pipeline.main()
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("--resume", command)
        self.assertEqual(command[command.index("--model") + 1], "reasonke")
        self.assertEqual(run.call_args.kwargs["timeout"], 31000)

    def test_pipeline_final_requires_protocol_and_success_stays_sequential(self):
        from types import SimpleNamespace
        with patch("sys.argv", ["run_depth_pipeline", "--phase", "final", "--directory", str(self.output)]), patch.object(run_depth_pipeline.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): run_depth_pipeline.main()
            run.assert_not_called()
        arguments = ["run_depth_pipeline", "--phase", "development", "--directory", str(self.output)]
        with patch("sys.argv", arguments), patch.object(run_depth_pipeline.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, contextlib.redirect_stdout(io.StringIO()):
            run_depth_pipeline.main()
        self.assertEqual(run.call_count, 2)
        self.assertEqual([call.args[0][call.args[0].index("--model") + 1] for call in run.call_args_list],
                         ["reasonke", "instruction"])


if __name__ == "__main__":
    unittest.main()

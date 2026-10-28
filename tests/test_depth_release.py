"""Post-study launch safety and command contracts; no GPU/model requests."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from scripts.finish_depth_release import launch, require_completed


def completed_audit():
    return {"phase": "final", "execution_status": "completed", "assigned_records": 544,
        "independent_lineages": 32, "conditions": {name: {"overall": {"assigned": n}}
            for name, n in (("R-RAG", 256), ("G-RAG", 256), ("R-ORACLE", 16), ("G-ORACLE", 16))}}


def acceptance_result(*, error=False):
    return {"status": "completed", "outcome_counts": {"correct": 12, "wrong": 1,
        "abstain": 1, "execution_error": int(error), "unexecuted": 0},
        "verified_neural_response_count": 14, "consistency_checks": {"software_fixture": True},
        "all_consistency_checks_passed": True, "initial_handbook_restored": True}


class DepthReleaseTests(unittest.TestCase):
    def test_partial_receipt_cannot_load_another_model_or_create_output(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.finish_depth_release.subprocess.Popen") as starter:
            root = Path(folder)
            (root / "final-reasonke.json").write_text(json.dumps({"status": "running", "records": []}))
            with self.assertRaisesRegex(ValueError, "must finish"):
                launch(root / "new-demo", root)
            starter.assert_not_called()
            self.assertFalse((root / "new-demo").exists())

    def test_completed_inputs_require_frozen_audit_and_exact_population(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for model in ("reasonke", "instruction"):
                (root / f"final-{model}.json").write_text(json.dumps({"status": "completed",
                    "records": [{"outcome": "wrong"}] * 272}))
            with patch("scripts.finish_depth_release.audit_and_aggregate", side_effect=ValueError("Changed source")) as audit:
                with self.assertRaisesRegex(ValueError, "Changed source"):
                    require_completed(root)
                self.assertEqual(audit.call_args.kwargs["protocol"], root / "protocol.json")
            wrong = completed_audit()
            wrong["assigned_records"] = 543
            with patch("scripts.finish_depth_release.audit_and_aggregate", return_value=wrong):
                with self.assertRaisesRegex(ValueError, "population"):
                    require_completed(root)
            with patch("scripts.finish_depth_release.audit_and_aggregate", return_value=completed_audit()):
                self.assertEqual(require_completed(root)["assigned_records"], 544)

    def test_completed_status_with_unexecuted_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.finish_depth_release.audit_and_aggregate") as audit:
            root = Path(folder)
            (root / "final-reasonke.json").write_text(json.dumps({"status": "completed",
                "records": [{"outcome": "correct"}] * 271 + [{"outcome": "unexecuted"}]}))
            with self.assertRaisesRegex(ValueError, "must finish"):
                require_completed(root)
            audit.assert_not_called()

    def test_existing_output_or_busy_runtime_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.finish_depth_release.require_completed", return_value=completed_audit()), patch("scripts.finish_depth_release.subprocess.Popen") as starter:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                launch(root, root / "benchmark")
            with patch("scripts.finish_depth_release.ensure_idle", side_effect=ValueError("Port occupied")):
                with self.assertRaisesRegex(ValueError, "occupied"):
                    launch(root / "new-demo", root / "benchmark")
            starter.assert_not_called()
            self.assertFalse((root / "new-demo").exists())

    def test_untrusted_proxy_origin_is_rejected_before_audit(self):
        with patch("scripts.finish_depth_release.require_completed") as audit:
            with self.assertRaises(ValueError):
                launch("unused-demo", "unused-benchmark", trusted_origin="https://example.com")
            audit.assert_not_called()

    def _mock_launch(self, root, *, acceptance_error=False):
        directory = root / "new-demo"
        worker, measuring, demo = (Mock(pid=n) for n in (410, 411, 412))

        def verification(command, **kwargs):
            output = Path(command[command.index("--output") + 1])
            output.write_text(json.dumps(acceptance_result(error=acceptance_error)))
            return Mock(returncode=int(acceptance_error))

        with patch("scripts.finish_depth_release.require_completed", return_value=completed_audit()), \
             patch("scripts.finish_depth_release.ensure_idle"), \
             patch("scripts.finish_depth_release.subprocess.check_output", return_value="fixture-commit\n"), \
             patch("scripts.finish_depth_release.subprocess.Popen", side_effect=[worker, measuring, demo]) as popen, \
             patch("scripts.finish_depth_release.wait_ready") as ready, \
             patch("scripts.finish_depth_release.stop_owned") as stop, \
             patch("scripts.finish_depth_release.subprocess.run", side_effect=verification) as runner:
            if acceptance_error:
                with self.assertRaisesRegex(RuntimeError, "all outcomes retained"):
                    launch(directory, root / "benchmark", python="fixture-python")
                result = json.loads((directory / "launch.json").read_text())
            else:
                result = launch(directory, root / "benchmark", python="fixture-python",
                    trusted_origin="https://fixture-8005.prod.colab.dev")
        return result, popen, ready, stop, runner

    def test_single_finite_workload_then_fresh_iterative_demo_no_secret_or_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, popen, ready, stop, runner = self._mock_launch(root)
            self.assertEqual(result["status"], "demo_ready")
            self.assertFalse(result["benchmark_retried"])
            self.assertFalse(result["primary_scoring_changed"])
            self.assertEqual(result["assigned_http_generations"], 14)
            # One wrong answer and one abstention stay in the recorded outcomes.
            self.assertEqual(result["http_outcome_counts"]["wrong"], 1)
            self.assertEqual(result["http_outcome_counts"]["abstain"], 1)
            runner.assert_called_once()
            self.assertEqual(runner.call_args.kwargs["timeout"], 3600)
            self.assertIn("scripts.verify_india_http", runner.call_args.args[0])
            self.assertEqual(popen.call_count, 3)
            commands = [call.args[0] for call in popen.call_args_list]
            for command in commands[1:]:
                self.assertEqual(command[command.index("--context-policy") + 1], "iterative")
            storages = [command[command.index("--storage") + 1] for command in commands[1:]]
            self.assertNotEqual(*storages)
            self.assertTrue(storages[0].endswith("http-workspace"))
            self.assertTrue(storages[1].endswith("demo-workspace"))
            self.assertIn("--trusted-origin", commands[2])
            secret = popen.call_args_list[0].kwargs["env"]["EDITGUARD_REASONKE_TOKEN"]
            self.assertNotIn(secret, json.dumps(result))
            self.assertFalse(any(secret in str(arg) for command in commands for arg in command))
            self.assertEqual(ready.call_args_list[0].kwargs["seconds"], 1800)
            self.assertEqual((root / "new-demo/prelaunch-audit.json").exists(), True)

    def test_execution_failure_preserves_receipt_and_does_not_launch_demo(self):
        with tempfile.TemporaryDirectory() as folder:
            result, popen, ready, stop, runner = self._mock_launch(Path(folder), acceptance_error=True)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(result["http_outcome_counts"]["execution_error"], 1)
            self.assertEqual(popen.call_count, 2)
            runner.assert_called_once()
            self.assertTrue(stop.called)


if __name__ == "__main__":
    unittest.main()

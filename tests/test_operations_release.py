"""GPU workflow gates using explicit process fixtures; no models or weights."""
import json
from pathlib import Path
import tempfile
import zipfile
import unittest
from unittest.mock import patch

from scripts.finish_operations_release import launch, require_completed
from scripts.finalize_operations_release import finalize


class OperationsReleaseTests(unittest.TestCase):
    def test_incomplete_final_receipt_cannot_start_another_gpu_process(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.finish_operations_release.subprocess.Popen") as starter:
            path = Path(temporary)
            (path / "final-reasonke.json").write_text(json.dumps({"status": "running", "records": [{"outcome": "unexecuted"}]}))
            with self.assertRaisesRegex(ValueError, "must finish"):
                launch(path / "new-demo", path)
            starter.assert_not_called()
            self.assertFalse((path / "new-demo").exists())

    def test_completed_receipts_require_successful_integrity_audit_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.finish_operations_release.subprocess.Popen") as starter:
            path = Path(temporary)
            for model in ("reasonke", "instruction"):
                (path / ("final-" + model + ".json")).write_text(json.dumps({"status": "completed", "records": [{"outcome": "wrong"}]}))
            with patch("scripts.finish_operations_release.audit_receipts", side_effect=ValueError("Changed source identity")) as audit:
                with self.assertRaisesRegex(ValueError, "Changed source"):
                    launch(path / "new-demo", path)
                audit.assert_called_once()
            starter.assert_not_called()

    def test_existing_output_is_never_overwritten_even_after_completed_audit(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.finish_operations_release.require_completed", return_value={}), patch("scripts.finish_operations_release.subprocess.Popen") as starter:
            with self.assertRaisesRegex(ValueError, "never overwritten"):
                launch(Path(temporary), Path(temporary) / "benchmark")
            starter.assert_not_called()

    def test_finalization_cannot_write_or_launch_before_integrity_gate(self):
        with tempfile.TemporaryDirectory() as temporary, patch("scripts.finalize_operations_release.require_completed", side_effect=ValueError("Incomplete final run")), patch("scripts.finalize_operations_release.launch") as starter:
            path = Path(temporary) / "new-run"
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                finalize(Path(temporary) / "benchmark", path)
            self.assertFalse(path.exists())
            starter.assert_not_called()

    def test_completed_finalization_packages_evidence_without_chats_or_reexecuting_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, run_dir = root / "raw", root / "private-run"
            source.mkdir()
            for name in ("final-reasonke.json", "final-instruction.json", "protocol.json"):
                (source / name).write_text('{"explicit_software_fixture":true}')
            results = {"execution_status": "completed", "assigned_records": 1312, "independent_lineages": 32,
                "conditions": {"R-RAG": {"overall": {"assigned": 256, "counts": {"correct": 200, "wrong": 50, "abstain": 6, "execution_error": 0, "unexecuted": 0}}}}}
            def figures(paths, output):
                output.mkdir()
                (output / "fixture-figure.svg").write_text("Software fixture, no plotted measurements")
            def site(run, benchmark, **kwargs):
                self.assertTrue((root / "reports/reasonke/operations-final/host-audit.json").exists())
                run.mkdir()
                for name in ("http-acceptance.json", "history-acceptance.json", "launch.json", "prelaunch-audit.json"):
                    (run / name).write_text('{"explicit_software_fixture":true}')
                (run / "private-chat.sqlite3").write_bytes(b"must not be exported")
                return {"status": "demo_ready"}
            with patch("scripts.finalize_operations_release.ROOT", root), patch("scripts.finalize_operations_release.require_completed", return_value={"status": "passed"}), patch("scripts.finalize_operations_release.aggregate", return_value=results), patch("scripts.finalize_operations_release.audit_receipt", return_value={"assigned_traces": 32}), patch("scripts.finalize_operations_release.generate", side_effect=figures), patch("scripts.finalize_operations_release.launch", side_effect=site) as launch_fixture:
                outcome = finalize(source, run_dir)
            launch_fixture.assert_called_once()
            self.assertEqual(outcome["status"], "demo_ready")
            self.assertFalse(outcome["benchmark_retried"])
            self.assertFalse(outcome["primary_scoring_changed"])
            with zipfile.ZipFile(outcome["bundle"]) as bundle:
                self.assertIn("final/final-reasonke.json", bundle.namelist())
                self.assertIn("final/explanation-audit.json", bundle.namelist())
                self.assertIn("final/http-acceptance.json", bundle.namelist())
                self.assertIn("final/history-acceptance.json", bundle.namelist())
                self.assertFalse(any("sqlite" in name for name in bundle.namelist()))
                self.assertIn("200/256", bundle.read("final/README.md").decode())


if __name__ == "__main__":
    unittest.main()

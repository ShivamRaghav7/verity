"""One-episode controller contracts; no model or evaluation execution."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.profile_vlkeb_pipeline import execution_success, main, run
from tests.test_ke_repro_evaluate import FakeBackend, edit, evaluate_edit


class PipelineProfileTests(unittest.TestCase):
    def test_default_plan_is_inert_and_finite(self):
        output = io.StringIO()
        with patch("scripts.profile_vlkeb_pipeline.preflight") as preflight, redirect_stdout(output):
            self.assertEqual(main([]), 0)
        preflight.assert_not_called()
        plan = json.loads(output.getvalue())
        self.assertEqual((plan["generated_queries"], plan["ft_optimizer_steps"], plan["maximum_worker_seconds"]), (25, 15, 600))

    def test_wrong_answers_and_abstentions_are_outcomes_but_execution_errors_fail(self):
        row = evaluate_edit(FakeBackend(), edit(), Path("assets"), "development_not_final_test")
        self.assertFalse(row["methods"]["base"][0]["strict_target_match"])
        self.assertTrue(execution_success(row))
        row["methods"]["base"][0].update(status="abstention", answer="")
        self.assertTrue(execution_success(row))
        row["methods"]["base"][0]["status"] = "generation_error"
        self.assertFalse(execution_success(row))
        row["methods"]["base"][0]["status"] = "completed"
        row["exact_restore"] = False
        self.assertFalse(execution_success(row))

    def test_nonzero_worker_overrides_completed_claim_and_preserves_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(output=Path(temporary) / "profile", upstream=Path("upstream"),
                assets=Path("assets"), manifest=Path("manifest"), rag_root=Path("rag"),
                rag_calibration=Path("calibration"), device="cuda")
            row = evaluate_edit(FakeBackend(), edit(), Path("assets"), "development_not_final_test")
            def process(*unused, **kwargs):
                (args.output / "profile.json").write_text(json.dumps({"execution_status": "completed", "unit_row": row}))
                return SimpleNamespace(returncode=-9)
            with patch("scripts.profile_vlkeb_pipeline.preflight", return_value={}), \
                 patch("scripts.profile_vlkeb_pipeline.validate_rag", return_value={"selection": {"threshold": .8}}), \
                 patch("scripts.profile_vlkeb_pipeline.fixed_fixture"), \
                 patch("scripts.profile_vlkeb_pipeline.source_receipts", return_value={}), \
                 patch("scripts.profile_vlkeb_pipeline.subprocess.run", side_effect=process):
                self.assertEqual(run(args), 1)
            report = json.loads((args.output / "profile.json").read_text())
            self.assertEqual(report["execution_status"], "failed")
            self.assertEqual(report["worker_return_code"], -9)
            self.assertIn("unit_row", report)


if __name__ == "__main__":
    unittest.main()

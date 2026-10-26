"""GPU workflow gates using explicit process fixtures; no models or weights."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.finish_operations_release import launch, require_completed


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


if __name__ == "__main__":
    unittest.main()

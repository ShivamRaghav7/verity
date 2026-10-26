"""Reject altered answer evidence before reporting metrics; no neural inference."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_operations_evidence import audit_receipts

ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = [ROOT / "reports/reasonke/operations-development-v1" / ("development-" + model + ".json")
            for model in ("reasonke", "instruction")]


class OperationsEvidenceTests(unittest.TestCase):
    def test_real_receipts_recompute_without_new_model_calls(self):
        result = audit_receipts(RECEIPTS)
        self.assertEqual(result["assigned"], 120)
        self.assertEqual(result["new_model_generations"], 0)
        self.assertFalse(result["human_reviewed"])

    def test_changed_prompt_generated_provenance_and_scoring_cannot_pass_audit(self):
        original = json.loads(RECEIPTS[0].read_text())
        for field in ("request_sha256", "model", "outcome"):
            changed = json.loads(json.dumps(original))
            row = changed["records"][0]
            if field == "request_sha256":
                row[field] = "0" * 64
            elif field == "model":
                row["response"]["generation"]["model"] = "fabricated/model"
            else:
                row[field] = "wrong" if row[field] == "correct" else "correct"
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "altered-receipt.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(ValueError):
                    audit_receipts([path, RECEIPTS[1]])


if __name__ == "__main__":
    unittest.main()

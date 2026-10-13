"""Machine-review integrity and source-boundary regressions, without ML.

The immutable development archive supplies controlled test fixtures. No human
decisions or test/challenge corpus are read or invented.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.audit_pilot_v3 import audit, file_hash, preflight_directory
from scripts.balance_scope_questions import balance_scope_questions
from scripts.register_automated_review import register

ROOT = Path(__file__).resolve().parents[1]


class PilotV3ReviewTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "runtime/test-pilot-v3-review"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.origin = self.root / "origin"
        self.origin.mkdir()
        with zipfile.ZipFile(ROOT / "data/pilots/pilot-v2-source.zip") as archive:
            for name in archive.namelist():
                target = (self.origin / name).resolve()
                self.assertEqual(target.parent, self.origin.resolve())
                target.write_bytes(archive.read(name))
        self.dataset = self.root / "revised"
        balance_scope_questions(self.origin, self.dataset)
        self.snapshot = self.hashes()
        self.review = audit(self.dataset, self.origin, "2026-10-13T12:00:00+05:30")
        self.audit_path = self.root / "audit.json"
        self.output = self.root / "registry.json"

    def hashes(self):
        return {directory.name + "/" + path.name: file_hash(path)
                for directory in (self.origin, self.dataset) for path in directory.iterdir() if path.is_file()}

    def write_audit(self, value):
        self.audit_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def test_complete_fresh_machine_review_recomputes_labels_and_preserves_both_sources(self):
        self.assertEqual(self.review["decision_counts"], {"accept": 504, "flagged": 0})
        self.assertTrue(all(self.review["global_checks"].values()))
        self.assertFalse(self.review["human_review_performed"])
        self.assertFalse(self.review["test_or_challenge_rows_read"])
        for split, expected in (("train", {"positive": 24, "negative": 36}),
                                ("validation", {"positive": 12, "negative": 18})):
            self.assertTrue(all(value == expected for value in self.review["scope_style_balance"][split].values()))
        self.assertTrue(all(item["computed_polarity"] in ("yes", "no") for item in self.review["decisions"]
                            if item["computed_question_style"].startswith("comparison")))
        self.write_audit(self.review)
        result = register(self.dataset, self.audit_path, self.output)
        self.assertEqual((result["accepted"], result["rejected"], result["target_checks"]), (504, 0, 144))
        registry = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertFalse(registry["human_reviewed_data"])
        self.assertEqual(registry["audit_sha256"], file_hash(self.audit_path))
        self.assertTrue(all(key.startswith("pilot-v3:") for key in registry["rows"]))
        self.assertEqual(self.hashes(), self.snapshot)

    def test_deterministic_audit_with_an_explicit_timestamp(self):
        repeated = audit(self.dataset, self.origin, self.review["reviewed_at"])
        self.assertEqual(self.review, repeated)
        self.assertEqual(self.hashes(), self.snapshot)

    def test_registration_refuses_unbound_origin_missing_balance_and_false_style(self):
        for mutation in ("origin", "balance", "style", "human"):
            with self.subTest(mutation=mutation):
                altered = copy.deepcopy(self.review)
                if mutation == "origin":
                    altered["origin_source_hashes"]["data/generated/pilot-v2/scope_train.jsonl"] = "0" * 64
                elif mutation == "balance":
                    altered["global_checks"].pop("matched_style_prior:train")
                elif mutation == "style":
                    decision = next(item for item in altered["decisions"] if item["computed_question_style"] == "comparison_yes")
                    decision["computed_question_style"] = "comparison_no"
                else:
                    altered["human_review_performed"] = True
                self.write_audit(altered)
                with self.assertRaises(ValueError):
                    register(self.dataset, self.audit_path, self.output)
                self.assertFalse(self.output.exists())
        self.assertEqual(self.hashes(), self.snapshot)

    def test_scope_label_is_recomputed_when_a_tampered_row_claims_applicability(self):
        source = self.dataset / "scope_train.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        row = next(item for item in rows if item["label"] == 0)
        identity = row["id"]
        row["label"] = 1
        source.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][source.name] = file_hash(source)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reviewed = audit(self.dataset, self.origin, self.review["reviewed_at"])
        decision = next(item for item in reviewed["decisions"] if item["id"] == identity)
        self.assertEqual(decision["computed_scope_label"], 0)
        self.assertFalse(decision["checks"]["applicability_label_or_answer"])
        self.assertEqual(decision["decision"], "flagged")
        self.assertFalse(reviewed["global_checks"]["complete36_edit_worlds"])
        self.assertEqual(reviewed["decision_counts"]["accept"], 0)

    def test_quantity_comparison_is_recomputed_instead_of_trusting_polarity(self):
        source = self.dataset / "scope_validation.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        row = next(item for item in rows if item["label"] == 0 and item.get("expected_polarity") == "no")
        identity = row["id"]
        row["expected_polarity"] = "yes"
        source.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        manifest_path = self.dataset / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"][source.name] = file_hash(source)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        reviewed = audit(self.dataset, self.origin, self.review["reviewed_at"])
        decision = next(item for item in reviewed["decisions"] if item["id"] == identity)
        self.assertEqual(decision["computed_polarity"], "no")
        self.assertFalse(decision["checks"]["independent_comparison_polarity"])
        self.assertEqual(decision["decision"], "flagged")

    def test_unsupported_test_named_file_is_refused_before_its_content_is_read(self):
        # Marker bytes are synthetic; an attempted JSON read would raise a different error.
        marker = self.dataset / "scope_test.jsonl"
        marker.write_bytes(b"do not parse this synthetic unsupported-file marker")
        with self.assertRaisesRegex(ValueError, "no test/challenge/review"):
            preflight_directory(self.dataset, 3)
        self.assertEqual(marker.read_bytes(), b"do not parse this synthetic unsupported-file marker")

    def test_v2_decisions_cannot_be_migrated_to_the_revision(self):
        old = json.loads((ROOT / "reports/pilot-v2-independent-review.json").read_text(encoding="utf-8"))
        self.write_audit(old)
        with self.assertRaisesRegex(ValueError, "explicitly automated development review"):
            register(self.dataset, self.audit_path, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.hashes(), self.snapshot)


if __name__ == "__main__":
    unittest.main()

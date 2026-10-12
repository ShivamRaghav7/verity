"""Exercise registration of the archived finite development pilot, without ML.

These tests use the actual 504 reviewed development rows and their recorded audit.
Tampered audits are isolated negative fixtures; no human labels are manufactured.
"""

import copy
import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.prepare_data import content_hash, load_reviewed_rows
from scripts.register_automated_review import register


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/pilots/pilot-v2-source.zip"
AUDIT = ROOT / "reports/pilot-v2-independent-review.json"
REGISTRY = ROOT / "reports/pilot-v2-reviews.automated.json"
FILES = {f"{task}_{split}.jsonl" for task in ("scope", "answer") for split in ("train", "validation")}
FILES |= {"catalogue.json", "manifest.json", "derivation.json", "selection.json", "split_assignments.json"}


class AutomatedRegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "runtime/test-automated-registration"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "development-data"
        self.dataset.mkdir()
        with zipfile.ZipFile(ARCHIVE) as archive:
            self.assertEqual(set(archive.namelist()), FILES)
            for name in archive.namelist():
                target = (self.dataset / name).resolve()
                self.assertEqual(target.parent, self.dataset.resolve())
                target.write_bytes(archive.read(name))
        self.original_source = self.snapshot()
        self.audit_data = json.loads(AUDIT.read_text(encoding="utf-8"))
        self.audit = self.root / "audit.json"
        self.write_audit(self.audit_data)
        self.output = self.root / "decisions.automated.json"

    def snapshot(self):
        return {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                for file in self.dataset.iterdir() if file.is_file()}

    def write_audit(self, data):
        self.audit.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def assert_rejected(self, data, reason=None):
        self.write_audit(data)
        manager = self.assertRaisesRegex(ValueError, reason) if reason else self.assertRaises(ValueError)
        with manager:
            register(self.dataset, self.audit, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot(), self.original_source)

    def test_full_archive_registration_preserves_sources_and_explicit_machine_provenance(self):
        audit_before = self.audit.read_bytes()
        result = register(self.dataset, self.audit, self.output)
        self.assertEqual({key: result[key] for key in ("rows_checked", "accepted", "rejected", "target_checks")},
                         {"rows_checked": 504, "accepted": 504, "rejected": 0, "target_checks": 144})
        self.assertFalse(result["human_reviewed_data"])
        registry = json.loads(self.output.read_text(encoding="utf-8"))
        self.assertEqual(registry["review_policy"], "automated_development")
        self.assertFalse(registry["human_reviewed_data"])
        self.assertEqual(registry["audit_sha256"], hashlib.sha256(audit_before).hexdigest())
        self.assertEqual(registry["origin_manifest_sha256"], self.original_source["manifest.json"])
        self.assertEqual(len(registry["rows"]), 504)
        self.assertTrue(all(row["reviewer_type"] == "automated_development" for row in registry["rows"].values()))
        self.assertEqual(self.snapshot(), self.original_source)
        self.assertEqual(self.audit.read_bytes(), audit_before)

    def test_default_human_gate_rejects_machine_registry_and_explicit_development_gate_is_typed(self):
        register(self.dataset, self.audit, self.output)
        path = self.dataset / "scope_train.jsonl"
        with self.assertRaisesRegex(ValueError, "human review"):
            load_reviewed_rows(path, self.output)
        rows = load_reviewed_rows(path, self.output, review_policy="automated_development")
        self.assertEqual(len(rows), 240)
        self.assertTrue(all(row["reviewed"] and row["human_reviewed"] is False
                            and row["review_policy"] == "automated_development" for row in rows))
        self.assertEqual(self.snapshot(), self.original_source)

    def test_stale_row_hash_and_missing_or_duplicate_decisions_fail(self):
        for modification in ("stale", "missing", "duplicate"):
            with self.subTest(modification=modification):
                audit = copy.deepcopy(self.audit_data)
                if modification == "stale":
                    audit["decisions"][0]["row_sha256"] = "0" * 64
                elif modification == "missing":
                    audit["decisions"].pop()
                else:
                    audit["decisions"][-1] = copy.deepcopy(audit["decisions"][0])
                self.assert_rejected(audit)

    def test_false_missing_or_malformed_independent_checks_fail(self):
        for modification in ("false", "missing", "malformed", "issues"):
            with self.subTest(modification=modification):
                audit = copy.deepcopy(self.audit_data)
                decision = audit["decisions"][0]
                if modification == "false":
                    decision["checks"]["independently_selected_numeric_value"] = False
                elif modification == "missing":
                    decision["checks"].pop("independently_selected_numeric_value")
                elif modification == "malformed":
                    decision["checks"] = [True]
                else:
                    decision["issues"] = ["unresolved contradictory label"]
                self.assert_rejected(audit, "complete passing independent checks")

    def test_recorded_scope_polarity_and_fact_cannot_disagree_despite_true_check_flags(self):
        for field in ("computed_scope_label", "computed_applicable", "computed_polarity", "computed_expected_fact"):
            with self.subTest(field=field):
                audit = copy.deepcopy(self.audit_data)
                decision = audit["decisions"][0]
                if field == "computed_scope_label":
                    decision[field] = 1 - decision[field]
                elif field == "computed_applicable":
                    decision[field] = not decision[field]
                elif field == "computed_polarity":
                    decision[field] = "no" if decision[field] != "no" else "yes"
                else:
                    decision[field]["value"] = "999"
                self.assert_rejected(audit, "independent applicability, polarity or fact")

    def test_recorded_question_and_target_endorsements_are_bound_to_the_actual_fact(self):
        for field in ("parsed_question_fact", "parsed_target"):
            with self.subTest(field=field):
                audit = copy.deepcopy(self.audit_data)
                decision = next(row for row in audit["decisions"] if row["file"].startswith("answer"))
                decision[field]["entity_id"] = "different-product"
                self.assert_rejected(audit, "parsed question fact" if field == "parsed_question_fact" else "target assertion")

    def test_stale_source_hash_and_method_source_hash_fail(self):
        for modification in ("source_hash", "method_hash", "method_source", "empty_method", "missing_method"):
            with self.subTest(modification=modification):
                audit = copy.deepcopy(self.audit_data)
                if modification == "source_hash":
                    audit["source_hashes"]["data/generated/pilot-v2/scope_train.jsonl"] = "0" * 64
                elif modification == "method_hash":
                    audit["implementation"]["sha256"] = "0" * 64
                elif modification == "method_source":
                    audit["implementation"]["source"] += "\n# modified procedure"
                elif modification == "empty_method":
                    audit["implementation"]["source"] = ""
                    audit["implementation"]["sha256"] = hashlib.sha256(b"").hexdigest()
                else:
                    audit["implementation"] = None
                self.assert_rejected(audit)

    def test_global_checks_must_be_complete_and_passing(self):
        for modification in ("false", "missing", "malformed"):
            with self.subTest(modification=modification):
                audit = copy.deepcopy(self.audit_data)
                if modification == "false":
                    audit["global_checks"]["disjoint:family"] = False
                elif modification == "missing":
                    audit["global_checks"].pop("disjoint:family")
                else:
                    audit["global_checks"] = "claimed passing"
                self.assert_rejected(audit, "dataset/hash/split checks")

    def test_machine_audit_cannot_claim_human_review_at_either_level(self):
        for field in ("human_review_performed", "human_approval_claimed", "reviewer_type", "decision_reviewer_type"):
            with self.subTest(field=field):
                audit = copy.deepcopy(self.audit_data)
                if field == "decision_reviewer_type":
                    audit["decisions"][0]["reviewer_type"] = "human"
                elif field == "reviewer_type":
                    audit[field] = "human"
                else:
                    audit[field] = True
                self.assert_rejected(audit)

    def test_immutable_fresh_filename_and_source_or_audit_paths_are_protected(self):
        self.output.write_bytes(b"preserved registry evidence")
        with self.assertRaisesRegex(ValueError, "fresh"):
            register(self.dataset, self.audit, self.output)
        self.assertEqual(self.output.read_bytes(), b"preserved registry evidence")
        for output in (self.dataset / "new-registry.json", self.dataset / "nested/new-registry.json", self.dataset, self.audit):
            with self.subTest(output=str(output)), self.assertRaises(ValueError):
                register(self.dataset, self.audit, output)
        self.assertEqual(self.snapshot(), self.original_source)

    def test_frozen_or_non_development_manifest_is_refused_without_reading_any_test_rows(self):
        path = self.dataset / "manifest.json"
        original = path.read_bytes()
        for field, value in (("test_frozen", True), ("test_files_included", True),
                             ("purpose", "final_evaluation"), ("pilot_version", 1)):
            with self.subTest(field=field):
                manifest = json.loads(original)
                manifest[field] = value
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "unfrozen pilot-v2 development"):
                    register(self.dataset, self.audit, self.output)
                self.assertFalse(self.output.exists())
        path.write_bytes(original)
        self.assertEqual(self.snapshot(), self.original_source)

    def test_tracked_registry_binds_the_actual_audit_and_all504_original_rows_read_only(self):
        before = REGISTRY.read_bytes()
        registry = json.loads(before)
        self.assertEqual(registry["audit_sha256"], hashlib.sha256(AUDIT.read_bytes()).hexdigest())
        self.assertEqual(registry["origin_manifest_sha256"], self.original_source["manifest.json"])
        self.assertEqual(len(registry["rows"]), 504)
        self.assertFalse(registry["human_reviewed_data"])
        for task in ("scope", "answer"):
            for split in ("train", "validation"):
                for line in (self.dataset / f"{task}_{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    decision = registry["rows"][row["id"]]
                    self.assertEqual(decision["row_sha256"], content_hash(row))
                    self.assertEqual(decision["decision"], "accept")
                    self.assertEqual(decision["reviewer_type"], "automated_development")
        self.assertEqual(REGISTRY.read_bytes(), before)
        self.assertEqual(self.snapshot(), self.original_source)


if __name__ == "__main__":
    unittest.main()

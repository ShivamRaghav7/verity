import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_data import content_hash, load_reviewed_rows, write_dataset
from scripts.review_dataset import (build_pilot, checked_rows, export_html, file_hash,
                                    import_decisions, integrity_flags, read_json,
                                    validate_decision)


class ReviewToolingTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1] / "runtime/test-review-tooling"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        write_dataset(self.source, family_count=20)

    def decision(self, row, decision="accept"):
        record = {"decision": decision, "reviewer": "Synthetic unit-test fixture only",
                  "reviewed_at": "2026-10-11T16:00:00+05:30", "row_sha256": content_hash(row)}
        if decision == "reject":
            record["reason"] = "Fixture rejection, not a human dataset label."
        return record

    def incoming(self, records):
        path = self.root / "incoming.json"
        path.write_text(json.dumps({"rows": records}), encoding="utf-8")
        return path

    def test_default_pilot_is_finite_and_contains_no_test_rows_or_files(self):
        for name in ("scope_test.jsonl", "answer_test.jsonl", "challenge_drafts.jsonl"):
            (self.source / name).unlink()
        output = self.root / "pilot"
        manifest = build_pilot(self.source, output)
        self.assertEqual(manifest["counts"], {"scope": {"train": 240, "validation": 120},
                                               "answer": {"train": 96, "validation": 48}})
        self.assertFalse(manifest["test_files_included"])
        self.assertTrue(all("test" not in path.name for path in output.iterdir()))
        catalogue = {row["fact_id"]: row for row in read_json(output / "catalogue.json")}
        entity_families = {row["entity_id"]: row["family_id"] for row in catalogue.values()}
        selected = read_json(output / "selection.json")["selected_families"]
        for task in ("scope", "answer"):
            for split in ("train", "validation"):
                for row in checked_rows(output, task, split):
                    self.assertIn(row["family_id"], selected[split])
                    self.assertIn(entity_families[row["expected_fact"]["entity_id"]], selected[split])
                    self.assertEqual(integrity_flags(row, catalogue), [])
                    self.assertFalse(row["reviewed"])

    def test_pilot_records_source_and_new_hashes_and_preserves_all_edit_variants(self):
        output = self.root / "pilot"
        build_pilot(self.source, output)
        source_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                       for row in checked_rows(self.source, task, split)}
        pilot_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                      for row in checked_rows(output, task, split)}
        records = read_json(output / "derivation.json")
        self.assertEqual(len(records), 504)
        self.assertGreater(sum(record["foreign_negative_rerouted"] for record in records), 0)
        for record in records:
            self.assertEqual(record["source_row_sha256"], content_hash(source_rows[record["source_row_id"]]))
            self.assertEqual(record["new_row_sha256"], content_hash(pilot_rows[record["id"]]))
        counts = {}
        for row in checked_rows(output, "scope", "train"):
            counts.setdefault(row["edit_id"], []).append(row)
        self.assertEqual(len(counts), 24)
        for rows in counts.values():
            self.assertEqual(sum(row["label"] for row in rows), 4)
            self.assertEqual(len(rows), 10)
            self.assertEqual({row["expected_polarity"] for row in rows if row.get("expected_polarity")}, {"yes", "no"})

    def test_pilot_is_reproducible_and_never_silently_overwrites(self):
        first, second = self.root / "first", self.root / "second"
        left, right = build_pilot(self.source, first), build_pilot(self.source, second)
        self.assertEqual(left, right)
        self.assertEqual({path.name: file_hash(path) for path in first.iterdir()},
                         {path.name: file_hash(path) for path in second.iterdir()})
        with self.assertRaisesRegex(ValueError, "new or empty"):
            build_pilot(self.source, first)
        (second / "test_lock.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "frozen"):
            build_pilot(self.source, second)

    def test_optional_closure_exposes_large_cyclic_component(self):
        with self.assertRaisesRegex(ValueError, "closure exceeds"):
            build_pilot(self.source, self.root / "pilot", task="scope", policy="closure", maximum_families_per_split=8)
        self.assertFalse((self.root / "pilot").exists())

    def test_single_task_pilot_import_does_not_require_other_task_files(self):
        output = self.root / "pilot"
        manifest = build_pilot(self.source, output, task="scope")
        self.assertEqual(set(manifest["counts"]), {"scope"})
        self.assertFalse((output / "answer_train.jsonl").exists())
        row = checked_rows(output, "scope", "train")[0]
        result = import_decisions(output, self.incoming({row["id"]: self.decision(row)}), output / "reviews.json")
        self.assertEqual(result["added_decisions"], 1)

    def test_explicit_import_preserves_unknown_registry_fields_and_is_idempotent(self):
        row = checked_rows(self.source, "scope", "train")[0]
        registry = self.root / "reviews.json"
        unknown = {"dataset_id": "editguard-controlled-v1", "future_metadata": {"team": "fixture"},
                   "rows": {"foreign-preserved-id": {"future_field": True}}}
        registry.write_text(json.dumps(unknown), encoding="utf-8")
        incoming = self.incoming({row["id"]: self.decision(row)})
        result = import_decisions(self.source, incoming, registry)
        self.assertEqual(result["added_decisions"], 1)
        merged = read_json(registry)
        self.assertEqual(merged["future_metadata"], unknown["future_metadata"])
        self.assertEqual(merged["rows"]["foreign-preserved-id"], unknown["rows"]["foreign-preserved-id"])
        digest = file_hash(registry)
        self.assertEqual(import_decisions(self.source, incoming, registry)["added_decisions"], 0)
        self.assertEqual(file_hash(registry), digest)

    def test_stale_hash_naive_time_missing_reason_and_conflicts_are_rejected(self):
        row = checked_rows(self.source, "scope", "train")[0]
        original = self.decision(row)
        for change in ({"row_sha256": "0" * 64}, {"reviewed_at": "2026-10-11T12:00:00"},
                       {"reviewer": ""}, {"decision": "reject"}):
            with self.subTest(change=change):
                with self.assertRaises(ValueError):
                    validate_decision(row, {**original, **change})
        registry = self.root / "reviews.json"
        import_decisions(self.source, self.incoming({row["id"]: original}), registry)
        before = registry.read_bytes()
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            import_decisions(self.source, self.incoming({row["id"]: self.decision(row, "reject")}), registry)
        self.assertEqual(before, registry.read_bytes())

    def test_duplicate_json_ids_and_test_decisions_are_rejected(self):
        path = self.root / "duplicates.json"
        path.write_text('{"rows":{"duplicate":{},"duplicate":{}}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_json(path)
        test_row = json.loads((self.source / "scope_test.jsonl").read_text(encoding="utf-8").splitlines()[0])
        with self.assertRaisesRegex(ValueError, "Test changes"):
            import_decisions(self.source, self.incoming({test_row["id"]: self.decision(test_row)}), self.root / "reviews.json")

    def test_frozen_registry_snapshot_cannot_be_modified(self):
        rows = checked_rows(self.source, "scope", "train")[:2]
        registry = self.root / "reviews.json"
        import_decisions(self.source, self.incoming({rows[0]["id"]: self.decision(rows[0])}), registry)
        (self.source / "test_lock.json").write_text(json.dumps({"review_registry_sha256": file_hash(registry)}), encoding="utf-8")
        before = registry.read_bytes()
        with self.assertRaisesRegex(ValueError, "immutable"):
            import_decisions(self.source, self.incoming({rows[1]["id"]: self.decision(rows[1])}), registry)
        self.assertEqual(before, registry.read_bytes())

    def test_portable_html_is_escaped_and_has_no_autoaccept(self):
        path = self.source / "scope_train.jsonl"
        rows = checked_rows(self.source, "scope", "train")
        rows[0]["question"] = '</script><img src=x onerror="alert(1)">'
        path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
        manifest_path = self.source / "manifest.json"
        manifest = read_json(manifest_path)
        manifest["files"][path.name] = file_hash(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / "review.html"
        result = export_html(self.source, output, "scope", "train", limit=1)
        page = output.read_text(encoding="utf-8")
        self.assertEqual(result["autoaccepted"], 0)
        self.assertNotIn(rows[0]["question"], page)
        self.assertIn("\\u003c/script\\u003e", page)
        self.assertIn('["","Pending"]', page)
        self.assertIn("pre.textContent", page)
        self.assertNotIn("innerHTML", page)

    def test_integrity_flags_find_bad_scope_polarity_truth_and_support_without_review(self):
        row = next(row for row in checked_rows(self.source, "answer", "train")
                   if row["question_type"] == "comparison")
        malformed = {**row, "expected_polarity": "no" if row["expected_polarity"] == "yes" else "yes",
                     "expected_fact": {**row["expected_fact"], "value": "999"},
                     "supporting_context": [{**row["correction"], "value": row["correction"]["original_value"]}]}
        flags = integrity_flags(malformed)
        self.assertIn("applicable_ground_truth_value_mismatch", flags)
        self.assertIn("support_contains_edited_relation", flags)
        self.assertIn("target_subject_relation_or_value_mismatch", flags)
        self.assertFalse(malformed["reviewed"])

    def test_existing_loader_now_rejects_naive_review_timestamps(self):
        row = checked_rows(self.source, "scope", "train")[0]
        path = self.root / "one-row.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        registry = self.incoming({row["id"]: {**self.decision(row), "reviewed_at": "2026-10-11T12:00:00"}})
        with self.assertRaises(ValueError):
            load_reviewed_rows(path, registry)


if __name__ == "__main__":
    unittest.main()

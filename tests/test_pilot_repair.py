"""Immutable grammar derivation checks on generated, unreviewed test fixtures."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard.formatting import (INPUT_FORMAT_VERSION, conditional_input,
                                  correction_text, display_unit, fact_text)
from scripts.prepare_data import canonical_bytes, content_hash, question_for, write_dataset
from scripts.repair_pilot import (_legacy_correction, _legacy_fact, _semantic_record,
                                  repair_pilot)
from scripts.review_dataset import build_pilot, checked_rows, file_hash, read_json


ROOT = Path(__file__).resolve().parents[1]


class PilotRepairTests(unittest.TestCase):
    def setUp(self):
        temporary_root = ROOT / "runtime/test-pilot-repair"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        full_source = self.root / "synthetic-full"
        write_dataset(full_source, family_count=20)
        self.source = self.root / "synthetic-legacy-pilot"
        build_pilot(full_source, self.source)
        catalogue = {fact["fact_id"]: fact for fact in read_json(self.source / "catalogue.json")}
        # Explicit synthetic legacy fixtures: never regenerate or mutate the
        # project's historical source/test dataset during regression tests.
        old_rows = {}
        for task in ("scope", "answer"):
            for split in ("train", "validation"):
                path = self.source / f"{task}_{split}.jsonl"
                rows = checked_rows(self.source, task, split)
                for row in rows:
                    row.pop("input_format_version", None)
                    row["question"] = row["question"].replace("connect 1 device on", "connect 1 devices on").replace(
                        "at least 1 support hour per day", "at least 1 support hours per day")
                    row["correction_text"] = _legacy_correction(row["correction"])
                    if task == "answer":
                        fact = catalogue[row["expected_fact"]["fact_id"]]
                        row["target_text"] = _legacy_fact(fact, row["expected_fact"]["value"], row.get("expected_polarity"))
                        row["input_text"] = (f"answer correction: {row['correction_text']} Question: {row['question']} "
                            f"As of: {row['request_context']['as_of']}. Unchanged context: "
                            + " ".join(_legacy_fact(fact) for fact in row["supporting_context"]))
                    old_rows[row["id"]] = row
                path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        derivation = read_json(self.source / "derivation.json")
        for record in derivation:
            record["new_row_sha256"] = content_hash(old_rows[record["id"]])
        (self.source / "derivation.json").write_text(json.dumps(derivation), encoding="utf-8")
        manifest = read_json(self.source / "manifest.json")
        manifest.pop("input_format_version", None)
        manifest["files"] = {name: file_hash(self.source / name) for name in manifest["files"]}
        (self.source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.output = self.root / "pilot-v2"

    def rewrite_rows(self, task, split, change):
        path = self.source / f"{task}_{split}.jsonl"
        rows = checked_rows(self.source, task, split)
        change(rows)
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        manifest = read_json(self.source / "manifest.json")
        manifest["files"][path.name] = file_hash(path)
        (self.source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_original_pilot_and_separate_synthetic_test_artifacts_stay_immutable(self):
        before = {str(path.relative_to(self.root)): file_hash(path) for path in self.root.rglob("*") if path.is_file()}
        manifest = repair_pilot(self.source, self.output)
        self.assertEqual(manifest["human_review_decisions"], {"pending": 504, "accept": 0, "reject": 0})
        self.assertEqual(manifest["input_format_version"], INPUT_FORMAT_VERSION)
        self.assertFalse(manifest["eligible_for_training"])
        for name, digest in before.items():
            self.assertEqual(file_hash(self.root / name), digest, name)
        for name in ("catalogue.json", "selection.json", "split_assignments.json"):
            self.assertEqual((self.output / name).read_bytes(), (self.source / name).read_bytes())
        self.assertEqual(set(path.name for path in self.output.iterdir()), set(path.name for path in self.source.iterdir()))

    def test_derivation_binds_new_ids_content_lineages_and_every_unchanged_semantic_field(self):
        manifest = repair_pilot(self.source, self.output)
        self.assertNotEqual(manifest["dataset_id"], read_json(self.source / "manifest.json")["dataset_id"])
        source_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                       for row in checked_rows(self.source, task, split)}
        new_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                    for row in checked_rows(self.output, task, split)}
        self.assertFalse(set(source_rows) & set(new_rows))
        derivation = read_json(self.output / "derivation.json")
        self.assertEqual(len(derivation), 504)
        for record in derivation:
            original, row = source_rows[record["source_row_id"]], new_rows[record["id"]]
            self.assertEqual(record["source_row_sha256"], content_hash(original))
            self.assertEqual(record["new_row_sha256"], content_hash(row))
            self.assertEqual(record["semantic_fields_sha256"], content_hash(_semantic_record(original)))
            self.assertEqual(_semantic_record(original), _semantic_record(row))
            self.assertEqual(original["lineage_id"], row["lineage_id"])
            self.assertEqual(original["edit_id"], row["edit_id"])
            self.assertFalse(row["reviewed"])
            self.assertEqual(row["review_status"], "pending")
        for name, digest in manifest["origin_hashes"].items():
            self.assertEqual(digest, file_hash(self.source / name))

    def test_repaired_targets_and_runtime_conditioned_inputs_use_the_identical_formatter(self):
        manifest = repair_pilot(self.source, self.output)
        self.assertGreater(manifest["text_change_counts"].get("question", 0), 0)
        self.assertGreater(manifest["text_change_counts"].get("target_text", 0), 0)
        catalogue = {fact["fact_id"]: fact for fact in read_json(self.output / "catalogue.json")}
        for split in ("train", "validation"):
            for row in checked_rows(self.output, "answer", split):
                self.assertEqual(row["correction_text"], correction_text(row["correction"]))
                self.assertEqual(row["input_text"], conditional_input(row["question"], row["correction"],
                    row["supporting_context"], row["request_context"]["as_of"]))
                fact = catalogue[row["expected_fact"]["fact_id"]]
                expected = (f"{row['expected_polarity'].capitalize()}. " if row.get("expected_polarity") else "")
                expected += fact_text({**fact, "value": row["expected_fact"]["value"]})
                self.assertEqual(row["target_text"], expected)
                self.assertNotIn("1 devices", row["target_text"])
                self.assertNotIn("connect 1 devices", row["question"])

    def test_quantity_one_inflects_device_and_hour_without_changing_unit_schema(self):
        fact = {"entity_name": "Synthetic", "plan": "Basic", "version": "v2", "value": "1", "relation": "device_limit", "unit": "devices", "aliases": ["S"]}
        original = copy.deepcopy(fact)
        self.assertIn("supports 1 device.", fact_text(fact))
        self.assertIn("connect 1 device", question_for(fact, 2, 1))
        self.assertEqual(display_unit("devices", "1.0"), "device")
        self.assertEqual(display_unit("devices", "2"), "devices")
        self.assertEqual(fact, original)
        hour_fact = {**fact, "relation": "support_hours", "unit": "hours/day"}
        self.assertIn("1 support hour per day", fact_text(hour_fact))
        self.assertIn("at least 1 support hour per day", question_for(hour_fact, 2, 1))
        self.assertEqual(display_unit("hours/day", "1"), "hour/day")
        self.assertEqual(hour_fact["unit"], "hours/day")

    def test_derivation_is_reproducible_and_existing_or_partial_output_is_refused(self):
        first = repair_pilot(self.source, self.output)
        second_output = self.root / "second"
        second = repair_pilot(self.source, second_output)
        self.assertEqual(first, second)
        self.assertEqual({path.name: file_hash(path) for path in self.output.iterdir()},
                         {path.name: file_hash(path) for path in second_output.iterdir()})
        with self.assertRaisesRegex(ValueError, "existing output"):
            repair_pilot(self.source, self.output)
        partial = self.root / "partial-output"
        partial.mkdir()
        (partial / "scope_train.jsonl").write_text("incomplete")
        with self.assertRaisesRegex(ValueError, "partial/existing"):
            repair_pilot(self.source, partial)
        self.assertEqual((partial / "scope_train.jsonl").read_text(), "incomplete")

    def test_checksum_mismatch_frozen_test_or_registry_source_is_refused_without_output(self):
        path = self.source / "scope_train.jsonl"
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            repair_pilot(self.source, self.output)
        path.write_bytes(original)
        for filename in ("test_lock.json", "scope_test.jsonl", "challenge_drafts.jsonl", "reviews.json"):
            sentinel = self.source / filename
            sentinel.write_text("do not read or copy this sentinel")
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                repair_pilot(self.source, self.output)
            self.assertFalse(self.output.exists())
            self.assertEqual(sentinel.read_text(), "do not read or copy this sentinel")
            sentinel.unlink()

    def test_source_partial_wrong_split_reviewed_or_semantically_invalid_rows_are_refused(self):
        partial = self.source / "answer_validation.jsonl"
        partial_bytes = partial.read_bytes()
        partial.unlink()
        with self.assertRaises(ValueError):
            repair_pilot(self.source, self.output)
        partial.write_bytes(partial_bytes)
        path = self.source / "scope_train.jsonl"
        original_bytes = path.read_bytes()
        manifest_path = self.source / "manifest.json"
        original_manifest = manifest_path.read_bytes()
        for change in ({"split": "test"}, {"reviewed": True}, {"review_status": "accepted"}, {"label": 7}):
            self.rewrite_rows("scope", "train", lambda rows: rows[0].update(change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                repair_pilot(self.source, self.output)
            self.assertFalse(self.output.exists())
            path.write_bytes(original_bytes)
            manifest_path.write_bytes(original_manifest)

    def test_publication_failure_preserves_source_and_never_leaves_partial_output(self):
        before = {path.name: file_hash(path) for path in self.source.iterdir()}
        with patch("scripts.repair_pilot.os.rename", side_effect=OSError("synthetic publication failure")), self.assertRaises(OSError):
            repair_pilot(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".pilot-v2.pending-*")), [])
        self.assertEqual(before, {path.name: file_hash(path) for path in self.source.iterdir()})


if __name__ == "__main__":
    unittest.main()

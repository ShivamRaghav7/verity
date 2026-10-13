"""Bounded scope revisions use synthetic development fixtures, never final data."""

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard.formatting import INPUT_FORMAT_VERSION
from scripts.balance_scope_questions import (STYLE_NAMES, balance_scope_questions,
                                            immutable_fields, question_variant)
from scripts.prepare_data import NEGATIVE_TYPES, canonical_bytes, content_hash, write_dataset
from scripts.review_dataset import build_pilot, checked_rows, file_hash, integrity_flags, read_json

ROOT = Path(__file__).resolve().parents[1]


class ScopeBalancingTests(unittest.TestCase):
    def setUp(self):
        # Keep large synthetic publications outside cloud-synced workspaces.
        # OneDrive can temporarily lock freshly written staging directories,
        # making an otherwise deterministic atomic-rename fixture fail.
        self.temporary = tempfile.TemporaryDirectory(prefix="editguard-scope-balancing-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        full = self.root / "synthetic-full"
        write_dataset(full, family_count=20)
        self.source = self.root / "synthetic-pilot-v2"
        build_pilot(full, self.source)
        # The current generator uses runtime-v2 grammar. Assign fresh synthetic
        # v2 identities without consulting the project's historical data.
        manifest = read_json(self.source / "manifest.json")
        new_rows = {}
        for task in ("scope", "answer"):
            for split in ("train", "validation"):
                path = self.source / f"{task}_{split}.jsonl"
                rows = checked_rows(self.source, task, split)
                for row in rows:
                    row["id"] = row["id"].replace("pilot-v1:", "pilot-v2:", 1)
                    row["input_format_version"] = INPUT_FORMAT_VERSION
                    new_rows[row["id"]] = row
                path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        derivations = read_json(self.source / "derivation.json")
        for record in derivations:
            record["id"] = record["id"].replace("pilot-v1:", "pilot-v2:", 1)
            record["new_row_sha256"] = content_hash(new_rows[record["id"]])
        (self.source / "derivation.json").write_text(json.dumps(derivations), encoding="utf-8")
        manifest.update(pilot_version=2, input_format_version=INPUT_FORMAT_VERSION, eligible_for_final_evaluation=False,
                        dataset_id=manifest["dataset_id"].replace("-pilot-v1-", "-pilot-v2-"))
        manifest["files"] = {name: file_hash(self.source / name) for name in manifest["files"]}
        (self.source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.output = self.root / "pilot-v3"

    def rewrite_rows(self, name, mutate):
        task, split = name.removesuffix(".jsonl").split("_")
        rows = checked_rows(self.source, task, split)
        mutate(rows)
        path = self.source / name
        path.write_bytes(b"".join(canonical_bytes(row) + b"\n" for row in rows))
        manifest = read_json(self.source / "manifest.json")
        manifest["files"][name] = file_hash(path)
        (self.source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_four_forms_have_same_label_prior_without_adding_rows(self):
        manifest = balance_scope_questions(self.source, self.output)
        self.assertEqual(manifest["pilot_version"], 3)
        self.assertEqual(manifest["input_format_version"], "editguard-input-v2")
        self.assertEqual(manifest["counts"], {"scope": {"train": 240, "validation": 120},
                                            "answer": {"train": 96, "validation": 48}})
        catalogue = {fact["fact_id"]: fact for fact in read_json(self.output / "catalogue.json")}
        derivations = {record["id"]: record for record in read_json(self.output / "derivation.json")}
        for split, positive, negative in (("train", 24, 36), ("validation", 12, 18)):
            rows = checked_rows(self.output, "scope", split)
            counts = Counter((derivations[row["id"]]["question_style"], row["label"]) for row in rows)
            for style in STYLE_NAMES:
                self.assertEqual(counts[(style, 1)], positive)
                self.assertEqual(counts[(style, 0)], negative)
                self.assertEqual(manifest["scope_style_balance"][split][style]["positive_prior"], .4)
            self.assertEqual(Counter(row["negative_type"] for row in rows if not row["label"]),
                             {kind: positive for kind in NEGATIVE_TYPES})
            for row in rows:
                self.assertEqual(integrity_flags(row, catalogue), [])

    def test_source_bytes_facts_scope_truth_and_answer_targets_remain_fixed(self):
        before = {str(path.relative_to(self.root)): file_hash(path) for path in self.root.rglob("*") if path.is_file()}
        manifest = balance_scope_questions(self.source, self.output)
        for path, digest in before.items():
            self.assertEqual(file_hash(self.root / path), digest, path)
        for name in ("catalogue.json", "selection.json", "split_assignments.json"):
            self.assertEqual((self.output / name).read_bytes(), (self.source / name).read_bytes())
        self.assertFalse(manifest["eligible_for_training"])
        self.assertFalse(manifest["eligible_for_final_evaluation"])
        self.assertEqual(manifest["human_review_decisions"], {"pending": 504, "accept": 0, "reject": 0})
        original_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                         for row in checked_rows(self.source, task, split)}
        derived_rows = {row["id"]: row for task in ("scope", "answer") for split in ("train", "validation")
                        for row in checked_rows(self.output, task, split)}
        self.assertFalse(set(original_rows) & set(derived_rows))
        for record in read_json(self.output / "derivation.json"):
            old, row = original_rows[record["source_row_id"]], derived_rows[record["id"]]
            self.assertEqual(content_hash(old), record["source_row_sha256"])
            self.assertEqual(content_hash(row), record["new_row_sha256"])
            self.assertEqual(immutable_fields(old), immutable_fields(row))
            self.assertEqual(content_hash(immutable_fields(row)), record["immutable_fields_sha256"])
            self.assertEqual(old["expected_fact"], row["expected_fact"])
            self.assertEqual(old["correction"], row["correction"])
            self.assertFalse(row["reviewed"])
            self.assertEqual(row["review_status"], "pending")
            if "target_text" in row or row.get("label") == 1:
                self.assertEqual({key: value for key, value in old.items() if key != "id"},
                                 {key: value for key, value in row.items() if key != "id"})

    def test_negative_aliases_comparisons_and_price_direction_use_original_queried_value(self):
        balance_scope_questions(self.source, self.output)
        catalogue = {fact["fact_id"]: fact for fact in read_json(self.output / "catalogue.json")}
        records = {record["id"]: record for record in read_json(self.output / "derivation.json")}
        observed = Counter()
        for split in ("train", "validation"):
            for row in checked_rows(self.output, "scope", split):
                if row["label"]:
                    continue
                fact = catalogue[row["expected_fact"]["fact_id"]]
                style = STYLE_NAMES.index(records[row["id"]]["question_style"])
                question, question_type, polarity = question_variant(fact, style)
                self.assertEqual(row["question"], question)
                self.assertEqual(row["question_type"], question_type)
                self.assertEqual(row["expected_polarity"], polarity)
                if style == 1:
                    self.assertIn(fact["aliases"][0], question)
                    observed["alias"] += 1
                if style >= 2:
                    self.assertEqual(polarity, "yes" if style == 2 else "no")
                    observed[(fact["relation"], polarity)] += 1
        self.assertGreater(observed["alias"], 0)
        for relation in ("device_limit", "price_monthly", "storage_gb", "support_hours"):
            self.assertGreater(observed[(relation, "yes")], 0)
            self.assertGreater(observed[(relation, "no")], 0)
        price = {"entity_name": "Synthetic", "aliases": ["S"], "plan": "Basic", "version": "v2", "relation": "price_monthly", "value": "12"}
        self.assertIn("at most 12 USD", question_variant(price, 2)[0])
        self.assertIn("at most 11 USD", question_variant(price, 3)[0])
        self.assertEqual(question_variant(price, 3)[2], "no")
        with self.assertRaises(ValueError):
            question_variant(price, True)

    def test_reproducibility_and_refusal_of_existing_output_and_test_unknown_sources(self):
        first = balance_scope_questions(self.source, self.output)
        second = self.root / "replica"
        self.assertEqual(first, balance_scope_questions(self.source, second))
        self.assertEqual({path.name: file_hash(path) for path in self.output.iterdir()},
                         {path.name: file_hash(path) for path in second.iterdir()})
        with self.assertRaisesRegex(ValueError, "Existing or partial"):
            balance_scope_questions(self.source, self.output)
        for name in ("test_lock.json", "scope_test.jsonl", "challenge_drafts.jsonl", "reviews.json", "unknown.json"):
            sentinel = self.source / name
            sentinel.write_text("never read or migrate this sentinel")
            with self.subTest(name=name), self.assertRaises(ValueError):
                balance_scope_questions(self.source, self.root / "refused")
            self.assertFalse((self.root / "refused").exists())
            sentinel.unlink()

    def test_invalid_integrity_or_semantic_source_is_refused_without_partial_publication(self):
        path = self.source / "scope_train.jsonl"
        original, manifest = path.read_bytes(), (self.source / "manifest.json").read_bytes()
        path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            balance_scope_questions(self.source, self.output)
        path.write_bytes(original)
        for mutate in (lambda rows: rows[0].update(split="test"), lambda rows: rows[0].update(label=0),
                       lambda rows: rows[4].update(question="Unrecognized question form"),
                       lambda rows: rows[4]["expected_fact"].update(value="999"),
                       lambda rows: rows[0].update(reviewed=True)):
            self.rewrite_rows("scope_train.jsonl", mutate)
            with self.assertRaises(ValueError):
                balance_scope_questions(self.source, self.output)
            self.assertFalse(self.output.exists())
            path.write_bytes(original)
            (self.source / "manifest.json").write_bytes(manifest)

    def test_atomic_publication_failure_preserves_source_and_cleans_only_staging(self):
        before = {path.name: file_hash(path) for path in self.source.iterdir()}
        with patch("scripts.balance_scope_questions.os.rename", side_effect=OSError("synthetic failure")), self.assertRaises(OSError):
            balance_scope_questions(self.source, self.output)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".pilot-v3.pending-*")), [])
        self.assertEqual(before, {path.name: file_hash(path) for path in self.source.iterdir()})


if __name__ == "__main__":
    unittest.main()

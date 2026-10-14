import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.rescore_validation import rescore


class ValidationRescoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.old_scorer = self.root / "old.py"
        self.old_scorer.write_bytes(b"# exact previous grammar\n")
        fact = {"entity_id": "fixture", "entity_name": "Fixture", "plan": "Basic", "version": "v2",
                "relation": "storage_gb", "value": "20", "unit": "GB"}
        self.records = []
        for config in ("unchanged_base", "updated_rag", "prompt_only", "similarity_editor", "trained_scope_editor"):
            for identity, applicable in (("one", True), ("two", False)):
                self.records.append({"id": identity, "configuration": config, "question": "Can I store 21 GB?",
                                     "answer": "No. Fixture Basic version v2 supports 20 GB of storage.",
                                     "answer_kind": "model_generated", "status": "answered", "family_id": "f",
                                     "question_type": "comparison", "request_context": {"entity_id": "fixture"},
                                     "expected_fact": fact, "expected_polarity": "no", "applicable": applicable,
                                     "review_policy": "automated_development", "human_reviewed_data": False,
                                     "row_sha256": identity, "score": {"correct": None, "scorer_version": "old-v2"}})
        self.oracle = [{**self.records[0], "configuration": "oracle_answer"}]
        self.manifest = {"status": "completed", "purpose": "validation_development",
                         "review_policy": "automated_development", "human_reviewed_data": False,
                         "test_rows_read": False, "base_weights_unchanged": True,
                         "scorer_version": "old-v2", "scorer_sha256": hashlib.sha256(self.old_scorer.read_bytes()).hexdigest(),
                         "questions": 2, "configurations": list(dict.fromkeys(r["configuration"] for r in self.records)),
                         "accepted_row_sha256": {"one": "one", "two": "two"}}
        self.save()

    def save(self):
        for name, records in (("predictions.jsonl", self.records), ("oracle_predictions.jsonl", self.oracle)):
            (self.source / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        self.manifest["predictions_sha256"] = hashlib.sha256((self.source / "predictions.jsonl").read_bytes()).hexdigest()
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.source / "metrics.json").write_text("{}", encoding="utf-8")

    def test_rescore_preserves_all_inference_and_original_files(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        output = self.root / "derived"
        report = rescore(self.source, output, self.old_scorer)
        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["record_count"], 10)
        actual = [json.loads(r) for r in (output / "predictions.jsonl").read_text().splitlines()]
        for old, new in zip(self.records, actual):
            self.assertEqual(new["source_score"], old["score"])
            self.assertEqual({k: v for k, v in new.items() if k not in ("score", "source_score")},
                             {k: v for k, v in old.items() if k != "score"})
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.source.iterdir()})
        self.assertFalse(json.loads((output / "manifest.json").read_text())["derivation"]["inference_executed"])

    def test_nonanswers_remain_uncredited_in_full_denominator(self):
        self.records[0].update(status="clarification", answer=None)
        self.oracle[0].update(status="clarification", answer=None)
        self.save()
        report = rescore(self.source, self.root / "derived", self.old_scorer)
        metric = report["configurations"]["unchanged_base"]
        self.assertEqual(metric["accuracy"]["denominator"], 2)
        self.assertLessEqual(metric["accuracy"]["numerator"], 1)

    def test_hash_tampering_is_rejected_before_output(self):
        with (self.source / "predictions.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaises(ValueError):
            rescore(self.source, self.root / "derived", self.old_scorer)
        self.assertFalse((self.root / "derived").exists())

    def test_wrong_archived_scorer_is_rejected(self):
        self.old_scorer.write_text("different")
        with self.assertRaises(ValueError):
            rescore(self.source, self.root / "derived", self.old_scorer)

    def test_missing_configuration_case_is_rejected(self):
        self.records.pop()
        self.save()
        with self.assertRaises(ValueError):
            rescore(self.source, self.root / "derived", self.old_scorer)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "derived"
        output.mkdir()
        with self.assertRaises(ValueError):
            rescore(self.source, output, self.old_scorer)


if __name__ == "__main__":
    unittest.main()

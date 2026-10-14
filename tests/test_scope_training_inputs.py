"""Scope preprocessing training contracts without changing raw reviewed records."""

import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from editguard.scope_inputs import scope_pair, scope_policy_identity
from editguard.training import (canonical_hash, scope_input_overlap, scope_training_identity,
                                scope_training_pairs, validate_identity)


def row(name="Nimbus", alias="NMB", *, label=1):
    return {"label": label, "question": f"How many devices does {name} Basic version v2 support?",
        "correction_text": f"Correction for {name} (aliases: {alias}), Basic, version v2: device_limit changed from 5 to 3 devices, effective from 2026-10-01. Source: controlled://corrections/{alias}/devices.",
        "id": "reviewed-" + alias, "reviewed": True, "review_status": "accepted"}


class ScopeTrainingInputContractTests(unittest.TestCase):
    def test_raw_default_preserves_exact_strings_and_reviewed_rows(self):
        rows = [{"question": "  Unparsed free-form question?  ", "correction_text": "  free-form original descriptor  ", "label": 0}]
        original = copy.deepcopy(rows)
        self.assertEqual(scope_training_pairs(rows), [(rows[0]["question"], rows[0]["correction_text"])])
        self.assertEqual(rows, original)
        self.assertEqual(scope_training_identity("scope", {})["scope_input_policy"], "raw_v1")

    def test_neutral_pairs_follow_shared_runtime_function_without_mutating_rows(self):
        rows = [row(), row("Aster 044", "A044")]
        before = canonical_hash(rows)
        pairs = scope_training_pairs(rows, "entity_neutral_v1")
        self.assertEqual(pairs, [scope_pair(r["question"], r["correction_text"], "entity_neutral_v1") for r in rows])
        self.assertEqual(pairs[0], pairs[1])
        self.assertEqual(canonical_hash(rows), before)

    def test_answer_rejects_all_explicit_scope_metadata(self):
        self.assertEqual(scope_training_identity("answer", {}), {})
        for key, value in scope_policy_identity("raw_v1").items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "only for the scope"):
                scope_training_identity("answer", {key: value})

    def test_answer_checkpoint_identity_rejects_scope_metadata_before_ml_import(self):
        from editguard.training import train_loop
        with self.assertRaisesRegex(ValueError, "only for the scope"):
            train_loop(None, {}, {}, config={"task": "answer"}, identity=scope_policy_identity("raw_v1"), output_dir="unused")

    def test_scope_identity_binds_policy_version_and_source(self):
        expected = scope_policy_identity("entity_neutral_v1")
        self.assertEqual(scope_training_identity("scope", expected), expected)
        for key in ("scope_input_policy_version", "scope_input_source_sha256"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "version/source"):
                scope_training_identity("scope", {**expected, key: "changed"})
        with self.assertRaisesRegex(ValueError, "complete"):
            scope_training_identity("scope", {"scope_input_policy": "entity_neutral_v1"})
        with self.assertRaises(ValueError):
            scope_training_identity("scope", {"scope_input_policy": "unknown"})
        with self.assertRaises(ValueError):
            scope_training_identity("scope", {"scope_input_policy": None})

    def test_contradictory_pair_rejected_within_and_across_partitions(self):
        positive, negative = row(), row("Aster 044", "A044", label=0)
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            scope_training_pairs([positive, negative], "entity_neutral_v1")
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            scope_input_overlap([positive], [negative], "entity_neutral_v1")
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            scope_training_pairs([positive, {**positive, "label": 0}], "raw_v1")

    def test_same_label_abstraction_overlap_reports_actual_pairs_and_rows(self):
        training = [row(), row("Aster 044", "A044")]
        validation = [row("Aster 099", "A099")]
        report = scope_input_overlap(training, validation, "entity_neutral_v1")
        self.assertEqual(report["train"]["unique_pairs"], 1)
        self.assertEqual(report["train"]["same_label_duplicate_rows"], 1)
        self.assertEqual(report["train"]["same_label_duplicate_groups"], 1)
        self.assertEqual(report["validation"]["same_label_duplicate_rows"], 0)
        self.assertEqual(report["cross_split"], {"identical_input_pairs": 1, "same_label_pairs": 1,
            "contradictory_label_pairs": 0, "train_rows_with_shared_pair": 2, "validation_rows_with_shared_pair": 1})
        self.assertEqual(scope_input_overlap(training, validation, "raw_v1")["cross_split"]["identical_input_pairs"], 0)

    def test_malformed_scope_labels_rejected_before_tokenization(self):
        for label in (True, False, 2, "1", None):
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, "integer"):
                scope_training_pairs([row(label=label)], "raw_v1")

    def test_scope_policy_change_rejects_checkpoint_identity(self):
        with self.assertRaisesRegex(ValueError, "scope_input"):
            validate_identity(scope_policy_identity("raw_v1"), scope_policy_identity("entity_neutral_v1"))


@unittest.skipUnless(os.environ.get("EDITGUARD_RUN_ML_TESTS") == "1", "Optional CPU tokenization/recovery checks; set EDITGUARD_RUN_ML_TESTS=1")
class ScopeTrainingTokenizationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        self.torch = torch
        torch.set_num_threads(1)

    def tokenizer(self):
        torch = self.torch

        class CapturingTokenizer:
            def __init__(self):
                self.calls = []

            def __call__(self, first, second=None, **kwargs):
                self.calls.append((first, second, kwargs))
                if isinstance(first, str):
                    return {"input_ids": [101, len(first.split()), len(second.split()), 102]}
                return {"input_ids": torch.tensor([[len(q.split()), len(c.split())] for q, c in zip(first, second)]),
                        "attention_mask": torch.ones((len(first), 2), dtype=torch.long)}

        return CapturingTokenizer()

    def test_both_length_and_batch_tokenization_use_transformed_pairs(self):
        from editguard.training import tokenize_rows
        rows, tokenizer = [row(), row("Aster 044", "A044")], self.tokenizer()
        encoded, statistics = tokenize_rows(rows, task="scope", tokenizer=tokenizer, input_length=192,
                                            scope_input_policy="entity_neutral_v1")
        expected = scope_training_pairs(rows, "entity_neutral_v1")
        self.assertEqual([(call[0], call[1]) for call in tokenizer.calls[:2]], expected)
        self.assertEqual(tokenizer.calls[-1][:2], ([pair[0] for pair in expected], [pair[1] for pair in expected]))
        self.assertEqual(encoded["labels"].tolist(), [1, 1])
        self.assertEqual(statistics["scope_pair_text_sha256"], canonical_hash(expected))
        self.assertEqual(statistics["scope_same_label_duplicate_rows"], 1)
        self.assertEqual(statistics["scope_input_source_sha256"], scope_policy_identity()["scope_input_source_sha256"])

    def test_raw_tokenization_retains_original_batch_strings(self):
        from editguard.training import tokenize_rows
        rows, tokenizer = [row(), row("Aster 044", "A044")], self.tokenizer()
        _, stats = tokenize_rows(rows, task="scope", tokenizer=tokenizer, input_length=192)
        self.assertEqual(tokenizer.calls[-1][:2], ([r["question"] for r in rows], [r["correction_text"] for r in rows]))
        self.assertEqual(stats["scope_input_policy"], "raw_v1")

    def test_explicit_answer_policy_rejected_before_tokenizer_is_called(self):
        from editguard.training import tokenize_rows
        tokenizer = self.tokenizer()
        for policy in ("raw_v1", "entity_neutral_v1"):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "only for the scope"):
                tokenize_rows([], task="answer", tokenizer=tokenizer, input_length=192, target_length=48,
                              scope_input_policy=policy)
        self.assertEqual(tokenizer.calls, [])

    def test_neutral_scope_checkpoint_recovery_is_exact_and_policy_change_rejected(self):
        from editguard.training import configure_determinism, train_loop
        torch = self.torch

        class TinyScope(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2)
                self.dropout = torch.nn.Dropout(.25)

            def forward(self, input_ids, labels):
                return SimpleNamespace(loss=torch.nn.functional.cross_entropy(self.layer(self.dropout(input_ids)), labels))

        encoded = {"input_ids": torch.tensor([[1., 0.], [0., 1.], [.5, .5]]), "labels": torch.tensor([0, 1, 1])}
        config = {"task": "scope", **scope_policy_identity("entity_neutral_v1"), "device": "cpu", "seed": 42,
            "batch_size": 2, "max_steps": 4, "learning_rate": .01, "checkpoint_every_steps": 2,
            "validation_every_steps": 2, "keep_last_checkpoints": 2}
        identity = {"purpose": "smoke_test", "configuration_hash": canonical_hash(config), **scope_policy_identity("entity_neutral_v1")}
        temporary_root = Path(__file__).resolve().parents[1] / "runtime/test-scope-inputs"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
            root = Path(temporary)
            configure_determinism(42, 1)
            complete = train_loop(TinyScope(), encoded, encoded, config=config, identity=identity, output_dir=root / "complete")
            configure_determinism(42, 1)
            partial = train_loop(TinyScope(), encoded, encoded, config=config, identity=identity,
                                 output_dir=root / "resumed", stop_after_steps=2)
            resumed = train_loop(TinyScope(), encoded, encoded, config=config, identity=identity,
                output_dir=root / "resumed", resume_checkpoint=partial["last_checkpoint"])
            self.assertEqual(complete["final_state_sha256"], resumed["final_state_sha256"])
            self.assertEqual(complete["best_validation"], resumed["best_validation"])
            changed = {**config, **scope_policy_identity("raw_v1")}
            changed_identity = {**identity, **scope_policy_identity("raw_v1"), "configuration_hash": canonical_hash(changed)}
            with self.assertRaisesRegex(ValueError, "identity differs"):
                train_loop(TinyScope(), encoded, encoded, config=changed, identity=changed_identity,
                    output_dir=root / "resumed", resume_checkpoint=partial["last_checkpoint"])


if __name__ == "__main__":
    unittest.main()

"""Answer-objective contracts and optional tiny CPU numerical/resume regression."""

import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from editguard.training import canonical_hash, loss_objective, validate_identity


class AnswerObjectiveContractTests(unittest.TestCase):
    def test_uniform_is_default_and_scope_only_accepts_uniform(self):
        self.assertEqual(loss_objective("answer")["policy"], "uniform")
        self.assertEqual(loss_objective("scope")["normalization"], "original_model_loss")
        with self.assertRaisesRegex(ValueError, "only for the answer"):
            loss_objective("scope", "comparison_first_token_8x")

    def test_unknown_or_untyped_policy_is_rejected(self):
        for policy in (None, "weighted", "comparison_first_token_9x", 8, {}, False):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "Unknown loss_policy"):
                loss_objective("answer", policy)

    def test_objective_policy_and_normalization_are_identity_bound(self):
        uniform, weighted = loss_objective("answer"), loss_objective("answer", "comparison_first_token_8x")
        self.assertNotEqual(canonical_hash(uniform), canonical_hash(weighted))
        for alteration in ({**weighted, "comparison_first_token_weight": 1},
                           {**weighted, "normalization": "divide_by_number_of_rows"}):
            with self.assertRaisesRegex(ValueError, "loss_objective"):
                validate_identity({"loss_objective": weighted}, {"loss_objective": alteration})


@unittest.skipUnless(os.environ.get("EDITGUARD_RUN_ML_TESTS") == "1", "Optional CPU objective checks; set EDITGUARD_RUN_ML_TESTS=1 in the ML environment")
class AnswerObjectiveNumericalTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(str(exc))
        self.torch = torch
        torch.set_num_threads(1)

    def tokenizer(self, *, padding_side="right"):
        class TinyTokenizer:
            pad_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": {"Yes": [1], "No": [2]}[text]}

        tokenizer = TinyTokenizer()
        tokenizer.padding_side = padding_side
        return tokenizer

    def rows(self):
        return [{"reviewed": True, "review_status": "accepted", "question_type": kind, "expected_polarity": polarity}
                for kind, polarity in (("comparison", "yes"), ("comparison", "no"), ("direct", None))]

    def labels(self):
        return self.torch.tensor([[1, 3, 4, -100], [2, 3, 4, -100], [3, 4, -100, -100]], dtype=self.torch.long)

    def test_equal_yes_no_weighting_and_padding_denominator(self):
        from editguard.training import comparison_loss_weights
        weights = comparison_loss_weights(self.rows(), self.labels(), self.tokenizer())
        self.assertEqual(weights.tolist(), [[8., 1., 1., 0.], [8., 1., 1., 0.], [1., 1., 0., 0.]])
        self.assertEqual(float(weights.sum()), 22.)

    def test_wrong_polarity_malformed_metadata_and_unreviewed_rows_rejected(self):
        from editguard.training import comparison_loss_weights
        mutations = ({"expected_polarity": "no"}, {"expected_polarity": "unknown"},
                     {"expected_polarity": []}, {"question_type": "direct"},
                     {"question_type": "dialogue"}, {"reviewed": False}, {"review_status": "pending"})
        for mutation in mutations:
            rows = self.rows()
            rows[0] = {**rows[0], **mutation}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                comparison_loss_weights(rows, self.labels(), self.tokenizer())

    def test_noncomparison_polarity_and_left_padding_rejected(self):
        from editguard.training import comparison_loss_weights
        rows = self.rows()
        rows[2]["expected_polarity"] = "yes"
        with self.assertRaisesRegex(ValueError, "Non-comparison"):
            comparison_loss_weights(rows, self.labels(), self.tokenizer())
        with self.assertRaisesRegex(ValueError, "right-padded"):
            comparison_loss_weights(self.rows(), self.labels(), self.tokenizer(padding_side="left"))

    def test_multitoken_polarity_and_missing_position_zero_rejected(self):
        from editguard.training import comparison_loss_weights
        tokenizer = self.tokenizer()
        with patch.object(type(tokenizer), "__call__", return_value={"input_ids": [1, 2]}), self.assertRaisesRegex(ValueError, "single-token"):
            comparison_loss_weights(self.rows(), self.labels(), tokenizer)
        labels = self.labels()
        labels[0, 0] = -100
        with self.assertRaisesRegex(ValueError, "position zero"):
            comparison_loss_weights(self.rows(), labels, self.tokenizer())

    def test_invalid_shape_weights_and_interior_padding_rejected(self):
        from editguard.training import comparison_loss_weights, validate_loss_weights
        labels = self.labels()
        weights = comparison_loss_weights(self.rows(), labels, self.tokenizer())
        invalid = []
        modified = weights.clone(); modified[0, 3] = 1.; invalid.append((labels, modified))
        modified = weights.clone(); modified[0, 1] = 8.; invalid.append((labels, modified))
        modified = weights.clone(); modified[0, 0] = float("nan"); invalid.append((labels, modified))
        modified = weights.clone(); modified[0, 0] = 2.; invalid.append((labels, modified))
        modified_labels = labels.clone(); modified_labels[0, 1] = -100; invalid.append((modified_labels, weights))
        modified_labels = labels.clone(); modified_labels[0, 0] = -2; invalid.append((modified_labels, weights))
        invalid.append((labels, weights[:, :2]))
        invalid.append((labels, weights.long()))
        for invalid_labels, invalid_weights in invalid:
            with self.subTest(labels=invalid_labels.tolist(), weights=invalid_weights.tolist()), self.assertRaises(ValueError):
                validate_loss_weights(invalid_labels, invalid_weights)

    def logits_model(self, logits):
        torch = self.torch

        class LogitsModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(logits.clone())

            def forward(self, input_ids, labels):
                nll = torch.nn.functional.cross_entropy(self.logits.transpose(1, 2), labels, ignore_index=-100)
                return SimpleNamespace(logits=self.logits, loss=nll)

        return LogitsModel()

    def test_weighted_numerical_loss_and_gradient_match_independent_formula(self):
        from editguard.training import comparison_loss_weights, objective_batch_loss
        torch, labels = self.torch, self.labels()
        values = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5) / 20
        weights = comparison_loss_weights(self.rows(), labels, self.tokenizer())
        model = self.logits_model(values)
        loss, denominator = objective_batch_loss(model, {"input_ids": torch.zeros((3, 2)), "labels": labels,
            "loss_weights": weights}, task="answer", policy="comparison_first_token_8x")
        loss.backward()
        independent_logits = values.clone().requires_grad_(True)
        log_probabilities = independent_logits.log_softmax(dim=-1)
        expected = sum(-log_probabilities[row, token, int(labels[row, token])] * float(weights[row, token])
                       for row in range(3) for token in range(4) if labels[row, token] != -100) / 22
        expected.backward()
        self.assertEqual(denominator, 22.)
        self.assertTrue(torch.allclose(loss.detach(), expected.detach(), atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(model.logits.grad, independent_logits.grad, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.equal(model.logits.grad[0, 3], torch.zeros(5)))
        # Equal logits: an eight-weight first token has eight times the same-class gradient magnitude.
        self.assertAlmostEqual(float(model.logits.grad[0, 0, 0] / model.logits.grad[0, 1, 0]), 8., places=5)

    def test_uniform_returns_original_model_loss_object_exactly(self):
        from editguard.training import objective_batch_loss
        torch = self.torch
        sentinel = torch.tensor(7.25, requires_grad=True)
        model = lambda **kwargs: SimpleNamespace(loss=sentinel)
        loss, denominator = objective_batch_loss(model, {"input_ids": torch.zeros((3, 2)), "labels": self.labels()}, task="answer")
        self.assertIs(loss, sentinel)
        self.assertEqual(denominator, 8)
        with self.assertRaisesRegex(ValueError, "Uniform"):
            objective_batch_loss(model, {"labels": self.labels(), "loss_weights": torch.ones((3, 4))}, task="answer")

    def test_uniform_optimizer_matches_original_direct_model_loss(self):
        from editguard.training import configure_determinism, train_loop
        torch = self.torch

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = torch.nn.Linear(2, 4)
                self.dropout = torch.nn.Dropout(.25)
                self.head = torch.nn.Linear(4, 2)

            def forward(self, input_ids, labels):
                logits = self.head(self.dropout(self.hidden(input_ids)))
                return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels))

        encoded = {"input_ids": torch.tensor([[1., 0.], [0., 1.], [.4, .6]]), "labels": torch.tensor([0, 1, 1])}
        config = {"task": "scope", "device": "cpu", "seed": 42, "batch_size": 2, "max_steps": 4,
            "learning_rate": .01, "checkpoint_every_steps": 2, "validation_every_steps": 2, "keep_last_checkpoints": 2}
        identity = {"configuration_hash": canonical_hash(config), "purpose": "smoke_test"}

        def old_direct_loss(model, batch, *, task, policy):
            # Literal original uniform forward and validation denominator.
            return model(**batch).loss, len(batch["labels"])

        # Cloud sync can lock checkpoint directories during their atomic rename.
        with tempfile.TemporaryDirectory(prefix="editguard-answer-objective-") as temporary:
            root = Path(temporary)
            configure_determinism(42, 1)
            revised = TinyClassifier()
            current = train_loop(revised, encoded, encoded, config=config, identity=identity, output_dir=root / "current")
            configure_determinism(42, 1)
            original = TinyClassifier()
            with patch("editguard.training.objective_batch_loss", side_effect=old_direct_loss):
                previous = train_loop(original, encoded, encoded, config=config, identity=identity, output_dir=root / "previous")
            self.assertEqual(current["final_state_sha256"], previous["final_state_sha256"])
            self.assertEqual(current["best_validation"], previous["best_validation"])
            for key, tensor in original.state_dict().items():
                self.assertTrue(torch.equal(tensor, revised.state_dict()[key]), key)

    def test_validation_aggregates_weight_denominators_across_different_length_batches(self):
        from editguard.training import validation_loss
        torch = self.torch

        class VariableBatch(torch.nn.Module):
            def forward(self, input_ids, labels):
                logits = torch.stack((input_ids, -input_ids, input_ids * 0), dim=-1)
                return SimpleNamespace(logits=logits, loss=torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels))

        encoded = {"input_ids": torch.tensor([[.1, .2, .3], [.2, .3, .4], [.8, .5, .2]]),
                   "labels": torch.tensor([[1, 2, -100], [2, 1, 2], [1, -100, -100]]),
                   "loss_weights": torch.tensor([[8., 1., 0.], [8., 1., 1.], [1., 0., 0.]])}
        value, denominator = validation_loss(VariableBatch(), encoded, 2, "cpu", "answer",
            loss_policy="comparison_first_token_8x", return_denominator=True)
        logits = torch.stack((encoded["input_ids"], -encoded["input_ids"], encoded["input_ids"] * 0), dim=-1)
        nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), encoded["labels"], reduction="none")
        self.assertEqual(denominator, 20.)
        self.assertAlmostEqual(value, float((nll * encoded["loss_weights"]).sum() / 20), places=6)

    def test_invalid_objective_identity_and_shape_fail_before_optimizer(self):
        from editguard.training import train_loop
        config = {"task": "answer", "loss_policy": "comparison_first_token_8x"}
        weights = self.torch.ones_like(self.labels(), dtype=self.torch.float32)
        encoded = {"labels": self.labels(), "loss_weights": weights}
        with patch("torch.optim.AdamW", side_effect=AssertionError("optimizer must not be constructed")):
            with self.assertRaisesRegex(ValueError, "identity"):
                train_loop(None, encoded, encoded, config=config, identity={}, output_dir="unused")
            with self.assertRaisesRegex(ValueError, "Padding weight"):
                train_loop(None, encoded, encoded, config=config,
                           identity={"loss_objective": loss_objective("answer", config["loss_policy"])}, output_dir="unused")

    def test_weighted_exact_resume_matches_uninterrupted_and_rejects_policy_change(self):
        from editguard.training import configure_determinism, train_loop
        torch = self.torch

        class TinyAnswer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(5, 4)
                self.dropout = torch.nn.Dropout(.25)
                self.head = torch.nn.Linear(4, 5)

            def forward(self, input_ids, labels):
                logits = self.head(self.dropout(self.embedding(input_ids)))
                return SimpleNamespace(logits=logits,
                    loss=torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100))

        encoded = {"input_ids": torch.tensor([[1, 2, 3], [2, 3, 4], [3, 4, 1]]),
            "labels": torch.tensor([[1, 3, -100], [2, 3, 4], [3, 4, -100]]),
            "loss_weights": torch.tensor([[8., 1., 0.], [8., 1., 1.], [1., 1., 0.]])}
        config = {"task": "answer", "loss_policy": "comparison_first_token_8x", "device": "cpu", "seed": 42,
            "batch_size": 2, "max_steps": 4, "learning_rate": .01, "checkpoint_every_steps": 2,
            "validation_every_steps": 2, "keep_last_checkpoints": 2}
        identity = {"purpose": "smoke_test", "configuration_hash": canonical_hash(config),
            "loss_objective": loss_objective("answer", config["loss_policy"])}
        with tempfile.TemporaryDirectory(prefix="editguard-answer-resume-") as temporary:
            root = Path(temporary)
            configure_determinism(42, 1)
            uninterrupted = TinyAnswer()
            complete = train_loop(uninterrupted, encoded, encoded, config=config, identity=identity, output_dir=root / "complete")
            configure_determinism(42, 1)
            partial = train_loop(TinyAnswer(), encoded, encoded, config=config, identity=identity,
                                 output_dir=root / "resumed", stop_after_steps=2)
            resumed_model = TinyAnswer()
            resumed = train_loop(resumed_model, encoded, encoded, config=config, identity=identity,
                output_dir=root / "resumed", resume_checkpoint=partial["last_checkpoint"])
            self.assertEqual(complete["final_state_sha256"], resumed["final_state_sha256"])
            self.assertEqual(complete["best_validation"], resumed["best_validation"])
            self.assertEqual(complete["best_validation"]["metric"], "validation_weighted_token_loss")
            for key, tensor in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(tensor, resumed_model.state_dict()[key]), key)
            changed = {**config, "loss_policy": "uniform"}
            changed_identity = {**identity, "configuration_hash": canonical_hash(changed), "loss_objective": loss_objective("answer")}
            unweighted = {key: value for key, value in encoded.items() if key != "loss_weights"}
            with self.assertRaisesRegex(ValueError, "identity differs"):
                train_loop(TinyAnswer(), unweighted, unweighted, config=changed, identity=changed_identity,
                    output_dir=root / "resumed", resume_checkpoint=partial["last_checkpoint"])


if __name__ == "__main__":
    unittest.main()

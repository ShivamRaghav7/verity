"""Synthetic run-release refusal checks; no actual labels, models or GPU."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from editguard.training import file_hash, json_file_bytes, loss_objective
from scripts.prepare_answer_revision import POLICY, publish, revision_card
from tests import test_pilot_planning as planning_fixture


class AnswerRevisionPlanningTests(unittest.TestCase):
    def setUp(self):
        fixture = planning_fixture.PilotPlanningTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        manifest_path = fixture.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["pilot_version"] = 2
        manifest_path.write_bytes(json_file_bytes(manifest))
        fixture.registry["origin_manifest_sha256"] = file_hash(manifest_path)
        fixture.reviews.write_bytes(json_file_bytes(fixture.registry))
        record = fixture.profiles["answer"]
        record.update(manifest_sha256=file_hash(manifest_path), reviews_sha256=file_hash(fixture.reviews))
        record["profile"].update(loss_policy=POLICY, loss_objective=loss_objective("answer", POLICY))
        fixture.profile_paths["answer"].write_bytes(json_file_bytes(record))
        self.initial = self.root / "synthetic-initial-answer"
        self.initial.mkdir()
        (self.initial / "manifest.json").write_bytes(b"synthetic metadata, not model weights")
        self.prior = {"dataset_id": manifest["dataset_id"], "manifest_sha256": file_hash(manifest_path),
                      "reviews_sha256": file_hash(fixture.reviews)}
        self.output = self.root / "answer-revision"

    def card(self):
        with patch("scripts.prepare_answer_revision.validate_artifact", return_value=self.prior):
            return revision_card(self.fixture.data, self.fixture.reviews,
                self.fixture.profile_paths["answer"], self.initial, self.output)

    def test_single_objective_revision_binds_profile_and_keeps_cap_and_initial(self):
        before = (self.initial / "manifest.json").read_bytes()
        card = self.card()
        self.assertEqual(card["max_steps"], 240)
        self.assertEqual(card["checkpoint_every_steps"], 48)
        self.assertEqual(card["loss_objective"], loss_objective("answer", POLICY))
        self.assertEqual(card["review_policy"], "automated_development")
        result = publish(card, self.root / "cards/answer.json")
        self.assertTrue(result["preflight_valid"])
        self.assertFalse(result["training_execution_requested"])
        self.assertFalse(self.output.exists())
        self.assertEqual((self.initial / "manifest.json").read_bytes(), before)

    def test_different_initial_dataset_or_registry_is_refused(self):
        for key in ("manifest_sha256", "reviews_sha256", "dataset_id"):
            original = self.prior[key]
            self.prior[key] = "unrelated-source"
            with self.assertRaisesRegex(ValueError, "exact reviewed data"):
                self.card()
            self.prior[key] = original

    def test_unweighted_or_different_normalization_profile_is_refused(self):
        path = self.fixture.profile_paths["answer"]
        original = path.read_bytes()
        for change in ({"loss_policy": "uniform"}, {"loss_objective": {"policy": POLICY, "normalization": "wrong"}}):
            record = json.loads(original)
            record["profile"].update(change)
            path.write_bytes(json_file_bytes(record))
            with self.assertRaisesRegex(ValueError, "measured objective"):
                self.card()
        path.write_bytes(original)

    def test_prior_descendant_and_existing_model_outputs_are_protected(self):
        self.output = self.initial / "revision"
        with self.assertRaisesRegex(ValueError, "separate from the initial"):
            self.card()
        self.output = self.root / "already-exists"
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh answer revision"):
            self.card()

    def test_failed_preflight_and_existing_card_preserve_evidence(self):
        card = self.card()
        target = self.root / "cards/refusal.json"
        with patch("scripts.prepare_answer_revision.validate_run_card", side_effect=ValueError("fixture refusal")):
            with self.assertRaisesRegex(ValueError, "fixture refusal"):
                publish(card, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.glob("*.pending-*")), [])
        target.write_bytes(b"prior evidence")
        with self.assertRaisesRegex(ValueError, "fresh run-card"):
            publish(card, target)
        self.assertEqual(target.read_bytes(), b"prior evidence")

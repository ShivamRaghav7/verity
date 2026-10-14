"""Synthetic finite revision contract; no actual dataset review or GPU run."""

import json
import unittest
from unittest.mock import patch

from editguard.scope_inputs import scope_policy_identity
from editguard.training import file_hash, json_file_bytes
from scripts.prepare_scope_neutral_revision import POLICY, publish, revision_card
from tests import test_pilot_planning as planning_fixture


class ScopeNeutralRevisionPlanningTests(unittest.TestCase):
    def setUp(self):
        fixture = planning_fixture.PilotPlanningTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.root = fixture, fixture.root
        path = fixture.data / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest.update(pilot_version=3, revision_version=1)
        path.write_bytes(json_file_bytes(manifest))
        fixture.registry["origin_manifest_sha256"] = file_hash(path)
        fixture.reviews.write_bytes(json_file_bytes(fixture.registry))
        record = fixture.profiles["scope"]
        record.update(manifest_sha256=file_hash(path), reviews_sha256=file_hash(fixture.reviews))
        record["profile"].update(scope_policy_identity(POLICY))
        fixture.profile_paths["scope"].write_bytes(json_file_bytes(record))
        self.prior = self.root / "synthetic-prior-scope"
        self.prior.mkdir()
        (self.prior / "manifest.json").write_bytes(b"synthetic metadata; not weights")
        self.metadata = {"dataset_id": manifest["dataset_id"], "manifest_sha256": file_hash(path),
                         "reviews_sha256": file_hash(fixture.reviews)}
        self.output = self.root / "scope-final-revision"

    def card(self):
        with patch("scripts.prepare_scope_neutral_revision.validate_artifact", return_value=self.metadata):
            return revision_card(self.fixture.data, self.fixture.reviews,
                self.fixture.profile_paths["scope"], self.prior, self.output)

    def test_last_revision_changes_only_input_policy_with_same_budget(self):
        before = (self.prior / "manifest.json").read_bytes()
        card = self.card()
        self.assertEqual((card["revision_number"], card["maximum_targeted_revisions"]), (2, 2))
        self.assertEqual((card["max_steps"], card["checkpoint_every_steps"]), (360, 120))
        self.assertEqual(card["scope_input_policy"], POLICY)
        self.assertEqual(card["scope_input_source_sha256"], scope_policy_identity(POLICY)["scope_input_source_sha256"])
        result = publish(card, self.root / "cards/scope.json")
        self.assertTrue(result["preflight_valid"])
        self.assertFalse(result["training_execution_requested"])
        self.assertFalse(self.output.exists())
        self.assertEqual((self.prior / "manifest.json").read_bytes(), before)

    def test_different_reviewed_population_and_prior_identity_are_refused(self):
        for key in ("dataset_id", "manifest_sha256", "reviews_sha256"):
            original = self.metadata[key]
            self.metadata[key] = "different-data"
            with self.assertRaisesRegex(ValueError, "exact reviewed data"):
                self.card()
            self.metadata[key] = original

    def test_profile_policy_or_preprocessing_source_mismatch_is_refused(self):
        path = self.fixture.profile_paths["scope"]
        original = path.read_bytes()
        for change in ({"scope_input_policy": "raw_v1"}, {"scope_input_source_sha256": "0" * 64}):
            record = json.loads(original)
            record["profile"].update(change)
            path.write_bytes(json_file_bytes(record))
            with self.assertRaisesRegex(ValueError, "measured input adaptation"):
                self.card()
        path.write_bytes(original)

    def test_existing_and_prior_nested_outputs_are_protected(self):
        self.output = self.prior / "revision"
        with self.assertRaisesRegex(ValueError, "separate from earlier"):
            self.card()
        self.output = self.root / "already-exists"
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "fresh scope revision"):
            self.card()

    def test_failed_publication_and_old_card_preserve_evidence(self):
        card = self.card()
        target = self.root / "cards/failure.json"
        with patch("scripts.prepare_scope_neutral_revision.validate_run_card", side_effect=ValueError("fixture refusal")):
            with self.assertRaisesRegex(ValueError, "fixture refusal"):
                publish(card, target)
        self.assertFalse(target.exists())
        self.assertEqual(list(target.parent.glob("*.pending-*")), [])
        target.write_bytes(b"older evidence")
        with self.assertRaisesRegex(ValueError, "fresh run-card"):
            publish(card, target)
        self.assertEqual(target.read_bytes(), b"older evidence")

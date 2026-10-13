"""Synthetic provenance/refusal fixtures; no model, real labels or GPU run."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_scope_revision import publish, revision_card


class ScopeRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "synthetic-data"
        self.dataset.mkdir()
        self.initial = self.root / "initial"
        self.initial.mkdir()
        (self.initial / "manifest.json").write_text("synthetic metadata", encoding="utf-8")
        self.output = self.root / "model"
        self.manifest = {
            "pilot_version": 3, "revision_version": 1, "scope_labels_changed": False,
            "structured_fact_truth_changed": False, "positive_questions_changed": False,
            "answer_inputs_or_targets_changed": False,
            "counts": {"scope": {"train": 240, "validation": 120}, "answer": {"train": 96, "validation": 48}},
            "origin_dataset_id": "synthetic-initial", "origin_hashes": {"manifest.json": "a" * 64},
            "scope_style_balance": {split: {style: {"positive": p, "negative": n, "positive_prior": .4}
                for style in ("direct", "paraphrase", "comparison_yes", "comparison_no")}
                for split, p, n in (("train", 24, 36), ("validation", 12, 18))}}
        self.prior = {"dataset_id": "synthetic-initial", "manifest_sha256": "a" * 64}
        self.write_manifest()

    def write_manifest(self):
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def prepare(self):
        with (patch("scripts.prepare_scope_revision.validate_artifact", return_value=self.prior),
              patch("scripts.prepare_scope_revision.make_card", return_value={"max_steps": 360, "checkpoint_every_steps": 120}) as planner):
            card = revision_card(self.dataset, self.root / "reviews.json", self.root / "profile.json", self.initial, self.output)
        return card, planner

    def test_changed_facts_or_wrong_origin_fail_before_planning(self):
        original = copy.deepcopy(self.manifest)
        for change in ({"scope_labels_changed": True}, {"answer_inputs_or_targets_changed": True},
                       {"origin_dataset_id": "other"}, {"revision_version": 3},
                       {"origin_hashes": {"manifest.json": "b" * 64}}):
            self.manifest = {**original, **change}
            self.write_manifest()
            with (patch("scripts.prepare_scope_revision.validate_artifact", return_value=self.prior),
                  patch("scripts.prepare_scope_revision.make_card") as planner,
                  self.assertRaisesRegex(ValueError, "bind the completed")):
                revision_card(self.dataset, self.root / "r", self.root / "p", self.initial, self.output)
            planner.assert_not_called()

    def test_balance_is_required_and_prior_model_is_never_overwritten(self):
        self.manifest["scope_style_balance"]["train"]["direct"]["negative"] = 35
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "40%"):
            self.prepare()
        self.manifest["scope_style_balance"]["train"]["direct"]["negative"] = 36
        self.write_manifest()
        self.output = self.initial / "revision"
        with self.assertRaisesRegex(ValueError, "separate from initial"):
            self.prepare()

    def test_revision_is_single_scope_run_with_fixed_cap_and_machine_policy(self):
        card, planner = self.prepare()
        self.assertEqual(card["revision_number"], 1)
        self.assertEqual(card["maximum_targeted_revisions"], 2)
        self.assertEqual(card["max_steps"], 360)
        self.assertEqual(planner.call_args.args[-1], "automated_development")
        self.assertIn("Revised negatives differ", card["comparison_plan"])
        self.assertFalse(self.output.exists())

    def test_failed_preflight_leaves_no_published_card_and_existing_card_is_preserved(self):
        path = self.root / "cards/scope.json"
        card = {key: str(self.root / key) for key in (
            "data_dir", "reviews_path", "profile_path", "initial_scope_artifact", "output_dir")}
        with patch("scripts.prepare_scope_revision.validate_run_card", side_effect=ValueError("fixture refusal")):
            with self.assertRaisesRegex(ValueError, "fixture refusal"):
                publish(card, path)
        self.assertFalse(path.exists())
        self.assertEqual(list(path.parent.glob("*.pending-*")), [])
        path.write_bytes(b"immutable prior card")
        with self.assertRaisesRegex(ValueError, "fresh run-card"):
            publish(card, path)
        self.assertEqual(path.read_bytes(), b"immutable prior card")


if __name__ == "__main__":
    unittest.main()

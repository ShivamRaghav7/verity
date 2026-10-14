"""Paired scope diagnosis keeps abstractions, labels and inference distinct."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard.neural import file_sha256
from editguard.scope_inputs import scope_policy_identity
from editguard.training import scope_input_overlap
from scripts.diagnose_scope_neutral import checked_models, overlap_groups, summarize
from scripts.prepare_data import content_hash


def row(name, alias, value="3", label=1, identifier=None):
    return {"id": identifier or alias, "label": label,
        "question": f"How many devices does {name} Basic version v2 support?",
        "correction_text": f"Correction for {name} (aliases: {alias}), Basic, version v2: device_limit changed from 5 to {value} devices, effective from 2026-10-01. Source: controlled://corrections/{alias}/devices."}


def record(label, probability, group):
    return {"label": label, "raw_probability": probability, "neutral_overlap_group": group,
        "question_style": "direct", "negative_type": "applicable_positive" if label else "wrong_version",
        "relation": "device_limit", "family_id": "demo"}


class ScopeNeutralDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.partitions = {"train": [row("Nimbus", "NMB")],
            "validation": [row("Aster 044", "A044"), row("Aster 099", "A099", "4")]}

    def test_overlap_groups_bind_identical_inputs_without_mutating_rows(self):
        original = copy.deepcopy(self.partitions)
        groups, report = overlap_groups(self.partitions)
        self.assertEqual(groups, {"A044": "shared_with_train", "A099": "not_shared_with_train"})
        self.assertEqual(report["cross_split"]["validation_rows_with_shared_pair"], 1)
        self.assertEqual(self.partitions, original)

    def test_raw_and_neutral_overlap_reported_separately(self):
        _, raw = overlap_groups(self.partitions, "raw_v1")
        _, neutral = overlap_groups(self.partitions)
        self.assertEqual(raw["cross_split"]["identical_input_pairs"], 0)
        self.assertEqual(neutral["cross_split"]["identical_input_pairs"], 1)

    def test_contradictory_abstraction_is_refused(self):
        self.partitions["validation"][0]["label"] = 0
        with self.assertRaisesRegex(ValueError, "contradictory"):
            overlap_groups(self.partitions)

    def test_duplicate_validation_ids_are_refused(self):
        self.partitions["validation"][1]["id"] = "A044"
        with self.assertRaisesRegex(ValueError, "unique"):
            overlap_groups(self.partitions)

    def test_group_denominators_sum_to_full_population(self):
        records = [record(1, .9, "shared_with_train"), record(1, .3, "not_shared_with_train"),
                   record(0, .1, "not_shared_with_train")]
        result = summarize(records)
        self.assertEqual(result["raw"]["rows"], 3)
        groups = result["by"]["neutral_overlap_group"]
        self.assertEqual(groups["shared_with_train"]["rows"], 1)
        self.assertEqual(groups["not_shared_with_train"]["rows"], 2)
        self.assertEqual(groups["not_shared_with_train"]["threshold_0_5"]["recall"]["value"], 0)
        self.assertIsNone(groups["shared_with_train"]["auroc"]["value"])

    def test_empty_subset_refused(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            summarize([])

    def manifests(self, dataset, reviews):
        common = {"dataset_id": "pilot", "manifest_sha256": file_sha256(dataset / "manifest.json"),
            "reviews_sha256": file_sha256(reviews), "review_policy": "automated_development",
            "human_reviewed_data": False, "input_format": "question_then_correction_pair",
            "completed_steps": 360, "selection": {"metric": "validation_loss"},
            "model_id": "fixed-backbone", "model_revision": "pinned", "pretrained_state_sha256": "initial",
            "input_length": 192, "positive_class_index": 1,
            "data_hashes": {split + "_rows": content_hash(rows) for split, rows in self.partitions.items()}}
        return {"raw": copy.deepcopy(common), "neutral": {**copy.deepcopy(common),
            **scope_policy_identity("entity_neutral_v1"), "scope_input_overlap": scope_input_overlap(
                self.partitions["train"], self.partitions["validation"], "entity_neutral_v1")}}

    def checked_fixture(self, edit=None):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            (dataset / "manifest.json").write_text(json.dumps({"dataset_id": "pilot"}), encoding="utf-8")
            reviews = dataset / "reviews.json"
            reviews.write_text("{}", encoding="utf-8")
            manifests = self.manifests(dataset, reviews)
            if edit:
                edit(manifests)
            artifacts = {"revision1_raw": "raw", "revision2_neutral": "neutral"}
            with patch("scripts.diagnose_scope_neutral.validate_artifact", side_effect=lambda path, task: manifests[path]):
                return checked_models(artifacts, dataset, reviews, self.partitions)

    def test_exact_dataset_and_policy_binding_is_accepted(self):
        result = self.checked_fixture()
        self.assertEqual(set(result), {"revision1_raw", "revision2_neutral"})

    def test_data_hash_backbone_policy_and_overlap_changes_refused(self):
        def changed_hash(manifests):
            manifests["neutral"]["data_hashes"]["validation_rows"] = "changed"

        def changed_overlap(manifests):
            manifests["neutral"]["scope_input_overlap"]["cross_split"]["identical_input_pairs"] += 1

        def changed_policy(manifests):
            manifests["neutral"].update(scope_policy_identity("raw_v1"))

        for mutation in (changed_hash, changed_overlap, changed_policy,
                         lambda m: m["neutral"].update(model_id="other"),
                         lambda m: m["raw"].update(human_reviewed_data=True),
                         lambda m: m["neutral"].update(reviews_sha256="changed")):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.checked_fixture(mutation)


if __name__ == "__main__":
    unittest.main()

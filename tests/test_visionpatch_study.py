"""Model-free scoring/seal regressions; no downloads or encoder inference."""

import itertools
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from visionpatch.data import file_hash
from visionpatch.model import MODEL_ID, MODEL_REVISION
from visionpatch.study import CODE_NAMES, METHODS, _features, develop, load_and_validate_seal, weighted_retention
from visionpatch.study_metrics import grouped_accuracy_interval, select_global_candidate, summarize_study
from visionpatch.study_protocol import EDIT_WORDS

try:
    import torch
except ImportError:
    torch = None


def object_rows():
    rows = []
    for source, label in (("photo-a", "cup"), ("photo-b", "camera")):
        rows.append({"task_id": source + "|clean", "image_id": source, "class_id": label,
                     "variant": "clean", "word": None, "base": label, "edit": label})
        for word in EDIT_WORDS:
            for position in ("top", "center"):
                for variant in ("blank", "distractor"):
                    base = word if variant == "distractor" else label
                    rows.append({"task_id": f"{source}|{word}|{position}|{variant}",
                                 "image_id": source, "class_id": label, "variant": variant,
                                 "word": word, "base": base, "edit": label})
    return rows


def card_rows():
    return [{"task_id": f"card-{word}-{i}", "class_id": word,
             "base": word, "edit": word if word != "camera" else "cup"}
            for word in ("camera", "cup") for i in range(3)]


class StudyMetricTests(unittest.TestCase):
    def test_explicit_counts_distinguish_photos_variants_and_baseline_preservation(self):
        summary = summarize_study(object_rows(), card_rows(), "edit", "camera")
        self.assertEqual(summary["original_photos"], 2)
        self.assertEqual(summary["object_questions"], 26)
        self.assertEqual(summary["target_distractor_accuracy"], {"numerator": 4, "denominator": 4, "rate": 1})
        self.assertEqual(summary["other_word_distractor_accuracy"]["denominator"], 8)
        self.assertEqual(summary["genuine_target_clean_accuracy"]["denominator"], 1)
        self.assertEqual(summary["baseline_attack_induced_errors"]["denominator"], 2)
        self.assertEqual(summary["repair_rate"]["numerator"], 2)
        self.assertEqual(summary["all_baseline_wrong_target_repair_rate"]["denominator"], 2)
        self.assertEqual(summary["target_distractor_breakdowns"]["class_id"]["cup"]["denominator"], 2)
        self.assertEqual(summary["legitimate_text_accuracy"]["rate"], .5)
        self.assertEqual(summary["preservation"]["legitimate_text"]["new_errors"]["numerator"], 3)

    def test_abstentions_remain_in_denominator_and_missing_predictions_raise(self):
        rows = object_rows()
        rows[0]["edit"] = None
        result = summarize_study(rows, [], "edit", "camera")
        self.assertEqual(result["clean_object_accuracy"]["denominator"], 2)
        self.assertEqual(result["clean_object_accuracy"]["rate"], .5)
        self.assertEqual(result["abstentions_scored_as_incorrect"], 1)
        del rows[0]["edit"]
        with self.assertRaises(KeyError):
            summarize_study(rows, [], "edit", "camera")

    def test_empty_text_has_null_accuracy_and_cannot_enter_selection(self):
        result = summarize_study(object_rows(), [], "edit", "camera")
        self.assertIsNone(result["legitimate_text_accuracy"]["rate"])
        with self.assertRaisesRegex(ValueError, "positive denominators"):
            select_global_candidate(["one"], {"one": [result]})

    def test_repeated_tasks_or_missing_clean_sources_are_rejected(self):
        rows = object_rows()
        with self.assertRaisesRegex(ValueError, "Repeated"):
            summarize_study(rows + [rows[0]], [], "edit", "camera")
        with self.assertRaisesRegex(ValueError, "clean task"):
            summarize_study(rows[1:], [], "edit", "camera")

    def test_paired_bootstrap_uses_photo_groups_not_overlay_count(self):
        rows = [row for row in object_rows() if row["variant"] == "distractor"]
        result = grouped_accuracy_interval(rows, "edit", samples=100)
        self.assertEqual(result["groups"], 2)
        self.assertEqual(result["questions"], 12)
        self.assertAlmostEqual(result["difference"], 5 / 6)
        self.assertEqual(result, grouped_accuracy_interval(rows, "edit", samples=100))
        cards = grouped_accuracy_interval(card_rows(), "edit", group_key="class_id", samples=100)
        self.assertEqual(cards["groups"], 2)
        self.assertEqual(cards["questions"], 6)
        self.assertAlmostEqual(cards["difference"], -.5)

    def test_global_validation_selection_ties_use_new_errors_then_declared_index(self):
        def summary(score, errors):
            return {key: {"rate": score} for key in (
                "target_distractor_accuracy", "clean_object_accuracy", "legitimate_text_accuracy")} | {
                    "preservation": {"clean_object": {"new_errors": {"numerator": errors}}}}
        values = {"first": [summary(.7, 2)] * 3, "second": [summary(.7, 1)] * 3,
                  "third": [summary(.7, 1)] * 3}
        result = select_global_candidate(list(values), values)
        self.assertEqual(result["selected"], "second")
        values["first"] = [summary(.8, 9)] * 3
        self.assertEqual(select_global_candidate(list(values), values)["selected"], "first")


class StudySealTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.versions = {"mock": "1"}
        self.names = {"config", "manifest", "profile", "development", "train_features", "validation_features"}
        self.names.update("code/" + name for name in CODE_NAMES)
        self.names.update("font/" + name for name in ("train_sans", "validation_serif", "final_sans_bold"))
        self.names.update(("model/config.json", "model/model.safetensors", "model/vocab.json",
                           "model/merges.txt", "model/tokenizer_config.json"))
        for word in EDIT_WORDS:
            for candidate in ("object-only", "protected-text-1", "protected-text-4", "finetune-0", "finetune-1"):
                self.names.add(f"candidate/{word}/{candidate}")
        sequential = []
        for order in itertools.permutations(EDIT_WORDS):
            stages = [{"stage": index, "word": word} for index, word in enumerate(order, 1)]
            sequential.append({"order": list(order), "stages": stages})
            self.names.update(f"sequential/{'-'.join(order)}/{index}" for index in range(1, 4))
        bindings = {}
        for index, name in enumerate(sorted(self.names)):
            target = self.directory / (str(index) + ".json")
            target.write_text(json.dumps({"selected": {}, "sequential": sequential}) if name == "development" else "fixed")
            bindings[name] = {"path": str(target), "sha256": file_hash(target)}
        self.seal = {"format_version": 2, "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
                     "methods": list(METHODS), "edit_words": list(EDIT_WORDS),
                     "runtime_versions": self.versions, "bindings": bindings,
                     "selected": {}, "sequential": sequential}
        self.path = self.directory / "seal.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.seal))

    def tearDown(self):
        self.temporary.cleanup()

    def test_complete_seal_validates_without_loading_model(self):
        with patch("visionpatch.study._versions", return_value=self.versions), patch("visionpatch.study.load_clip") as loader:
            self.assertEqual(load_and_validate_seal(self.path), self.seal)
            loader.assert_not_called()

    def test_changed_bound_file_is_rejected(self):
        Path(self.seal["bindings"]["config"]["path"]).write_text("changed")
        with patch("visionpatch.study._versions", return_value=self.versions):
            with self.assertRaisesRegex(ValueError, "Sealed input changed"):
                load_and_validate_seal(self.path)

    def test_incomplete_models_stages_runtime_and_edited_selection_are_rejected(self):
        original = json.loads(json.dumps(self.seal))
        mutations = [lambda s: s["bindings"].pop("model/model.safetensors"),
                     lambda s: s["bindings"].pop("candidate/camera/protected-text-1"),
                     lambda s: s["sequential"].pop(),
                     lambda s: s.update(runtime_versions={"mock": "2"}),
                     lambda s: s.update(selected={"camera": "unfrozen"})]
        for mutate in mutations:
            self.seal = json.loads(json.dumps(original))
            mutate(self.seal)
            self.save()
            with self.subTest(seal=self.seal), patch("visionpatch.study._versions", return_value=self.versions):
                with self.assertRaises(ValueError):
                    load_and_validate_seal(self.path)


@unittest.skipIf(torch is None, "Optional tensor pipeline tests require PyTorch")
class StudyPipelineTests(unittest.TestCase):
    def test_weighted_retention_covariance_matches_declared_object_plus_text(self):
        objects = torch.tensor([[2., 0.], [0., 1.]], dtype=torch.float64)
        cards = torch.tensor([[1., 2.], [2., 1.], [0., 1.]], dtype=torch.float64)
        for weight in (0, 1, 4):
            combined = weighted_retention(objects, cards, weight)
            expected = objects.T @ objects / len(objects) + weight * cards.T @ cards / len(cards)
            torch.testing.assert_close(combined.T @ combined / len(combined), expected, atol=1e-12, rtol=0)

    def test_feature_cache_rejects_stale_fingerprint_before_encoder_load(self):
        from visionpatch.study import _save_tensor_new

        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(artifacts=Path(directory), dataset=Path(directory))
            _save_tensor_new(args.artifacts / "train-features.pt", {"fingerprint": {"stale": True}, "rows": []})
            with patch("visionpatch.study._fingerprint", return_value={"expected": True}), \
                 patch("visionpatch.study.iter_photo_tasks", return_value=[]), \
                 patch("visionpatch.study.text_task_rows", return_value=[]), \
                 patch("visionpatch.study.load_clip") as loader:
                with self.assertRaisesRegex(ValueError, "inputs changed"):
                    _features(args, {}, {}, "train")
                loader.assert_not_called()

    def test_develop_refuses_stale_profile_before_any_model_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(artifacts=root / "artifacts", reports=root / "reports")
            args.reports.mkdir()
            (args.reports / "profile.json").write_text(json.dumps({"fingerprint": {"old": 1}}))
            with patch("visionpatch.study._inputs", return_value=({}, {})), \
                 patch("visionpatch.study._fingerprint", return_value={"new": 1}), \
                 patch("visionpatch.study.load_clip") as loader:
                with self.assertRaisesRegex(ValueError, "Profile is stale"):
                    develop(args)
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

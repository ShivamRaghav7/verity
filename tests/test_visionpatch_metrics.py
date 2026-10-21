"""Hand-calculated metric denominators and source-grouped uncertainty checks."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from visionpatch.metrics import paired_accuracy_interval, ratio, summarize


def source(identifier, label, base, edited):
    """Create clean/blank/distractor observations in that fixed order."""
    return [{"image_id": identifier, "class_id": label, "variant": variant,
             "base": baseline, "projection_edit": prediction}
            for variant, baseline, prediction in zip(("clean", "blank", "distractor"), base, edited)]


def counted_examples():
    # One repair, one unrepaired induced error, one collateral error, one
    # correction of an already-wrong baseline, and genuine-distractor damage.
    return (
        source("a", "cup", ["cup", "cup", "camera"], ["cup", "cup", "cup"])
        + source("b", "laptop", ["laptop", "laptop", "camera"], ["laptop", "laptop", "camera"])
        + source("c", "watch", ["watch", "watch", "watch"], ["elephant", "watch", "camera"])
        + source("d", "airplanes", ["camera", "camera", "camera"], ["airplanes", "airplanes", "airplanes"])
        + source("e", "camera", ["camera", "camera", "camera"], ["cup", "cup", "cup"])
    )


class VisualMetricDenominatorTests(unittest.TestCase):
    def test_repairs_exclude_wrong_baselines_and_genuine_distractor(self):
        result = summarize(counted_examples(), "projection_edit")
        self.assertEqual(result["sources"], 5)
        self.assertEqual(result["baseline_attack_induced_errors"], ratio(2, 3))
        self.assertEqual(result["repair_rate"], ratio(1, 2))
        self.assertEqual(result["repairs_per_all_sources"], ratio(1, 5))
        self.assertEqual(result["genuine_distractor_clean_accuracy"], ratio(0, 1))

    def test_accuracy_and_collateral_error_use_different_declared_denominators(self):
        result = summarize(counted_examples(), "projection_edit")
        self.assertEqual(result["clean_accuracy"], ratio(3, 5))
        self.assertEqual(result["blank_accuracy"], ratio(4, 5))
        self.assertEqual(result["distractor_accuracy"], ratio(2, 5))
        self.assertEqual(result["clean_new_errors"], ratio(2, 4))
        self.assertEqual(result["blank_new_errors"], ratio(1, 4))
        self.assertEqual(result["distractor_new_errors"], ratio(2, 2))

    def test_unchanged_baseline_has_zero_repairs_and_zero_new_errors(self):
        result = summarize(counted_examples(), "base")
        self.assertEqual(result["repair_rate"], ratio(0, 2))
        for variant, correct in (("clean", 4), ("blank", 4), ("distractor", 2)):
            self.assertEqual(result[f"{variant}_new_errors"], ratio(0, correct))

    def test_zero_denominators_remain_explicit_and_do_not_become_success(self):
        rows = source("one", "cup", ["camera"] * 3, ["cup"] * 3)
        result = summarize(rows, "projection_edit")
        for key in ("repair_rate", "baseline_attack_induced_errors", "clean_new_errors",
                    "blank_new_errors", "distractor_new_errors", "genuine_distractor_clean_accuracy"):
            self.assertEqual(result[key], ratio(0, 0))
            self.assertIsNone(result[key]["rate"])
        empty = summarize([], "base")
        self.assertEqual(empty["sources"], 0)
        self.assertEqual(empty["clean_accuracy"], ratio(0, 0))

    def test_duplicate_variant_missing_variant_and_inconsistent_truth_rejected(self):
        rows = source("one", "cup", ["cup"] * 3, ["cup"] * 3)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(rows + [copy.deepcopy(rows[0])], "projection_edit")
        with self.assertRaisesRegex(ValueError, "three declared"):
            summarize(rows[:2], "projection_edit")
        inconsistent = copy.deepcopy(rows)
        inconsistent[1]["class_id"] = "laptop"
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            summarize(inconsistent, "projection_edit")

    def test_custom_distractor_category_is_used_for_eligibility(self):
        rows = source("genuine", "watch", ["watch"] * 3, ["camera"] * 3)
        result = summarize(rows, "projection_edit", distractor="watch")
        self.assertEqual(result["baseline_attack_induced_errors"], ratio(0, 0))
        self.assertEqual(result["genuine_distractor_clean_accuracy"], ratio(0, 1))


class PairedSourceBootstrapTests(unittest.TestCase):
    def test_three_variants_count_as_one_independent_source_each(self):
        rows = counted_examples()
        before = copy.deepcopy(rows)
        result = paired_accuracy_interval(rows, "projection_edit", samples=100, seed=71)
        self.assertEqual(result["sources"], 5)
        # Attacked differences: +1, 0, -1, +1, -1.
        self.assertEqual(result["difference"], 0)
        self.assertLessEqual(result["interval_95"][0], 0)
        self.assertGreaterEqual(result["interval_95"][1], 0)
        self.assertEqual(result, paired_accuracy_interval(rows, "projection_edit", samples=100, seed=71))
        self.assertEqual(rows, before)

    def test_uniform_repair_and_unchanged_predictions_have_exact_point_intervals(self):
        rows = source("one", "cup", ["cup", "cup", "camera"], ["cup"] * 3)
        repaired = paired_accuracy_interval(rows, "projection_edit", samples=40)
        self.assertEqual(repaired["difference"], 1)
        self.assertEqual(repaired["interval_95"], [1, 1])
        unchanged = paired_accuracy_interval(rows, "base", samples=40)
        self.assertEqual(unchanged["difference"], 0)
        self.assertEqual(unchanged["interval_95"], [0, 0])

    def test_duplicate_attacked_rows_cannot_inflate_sample_size(self):
        rows = counted_examples()
        duplicate = next(row for row in rows if row["variant"] == "distractor")
        with self.assertRaisesRegex(ValueError, "Duplicate attacked"):
            paired_accuracy_interval(rows + [copy.deepcopy(duplicate)], "projection_edit")

    def test_empty_input_and_invalid_bootstrap_budget(self):
        self.assertEqual(paired_accuracy_interval([], "base"),
                         {"sources": 0, "difference": None, "interval_95": None})
        for samples in (0, -1, True, 1.5):
            with self.subTest(samples=samples), self.assertRaisesRegex(ValueError, "positive integer"):
                paired_accuracy_interval(counted_examples(), "projection_edit", samples=samples)


class ProfileOutputGuardTests(unittest.TestCase):
    def test_existing_profile_refused_before_loading_gpu_or_importing_optional_torch(self):
        from visionpatch import profile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text("preserved evidence", encoding="utf-8")
            with (patch.object(sys, "argv", ["profile", "--output", str(path)]),
                  patch.object(profile, "load_clip") as loader):
                with self.assertRaisesRegex(FileExistsError, "already exists"):
                    profile.main()
                loader.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "preserved evidence")


if __name__ == "__main__":
    unittest.main()

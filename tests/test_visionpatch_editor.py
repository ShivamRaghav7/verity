"""Meaningful synthetic checks for the optional projection-editing baseline."""

from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from visionpatch.editor import (
        apply_projection_patch_, fit_projection_patch, load_projection_patch,
        projection_sha256, rollback_projection_patch_, save_projection_patch,
    )


@unittest.skipIf(torch is None, "Optional visual editor tests require PyTorch")
class ProjectionEditorTests(unittest.TestCase):
    def setUp(self):
        self.weight = torch.eye(3, dtype=torch.float64)
        self.keys = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        self.targets = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float64)
        self.retention = torch.eye(3, dtype=torch.float64)

    def fit(self, **kwargs):
        return fit_projection_patch(self.weight, self.keys, self.targets, self.retention,
                                    fit_ridge=1e-12, **kwargs)

    def test_single_edit_fits_target_and_preserves_orthogonal_directions(self):
        original = self.weight.clone()
        patch = self.fit()
        edited = patch.edited_weight()
        torch.testing.assert_close(self.keys @ edited.T, self.targets, atol=1e-10, rtol=0)
        torch.testing.assert_close(edited[:, 1:], original[:, 1:], atol=0, rtol=0)
        self.assertEqual(patch.rank_bound, 1)
        self.assertTrue(torch.equal(self.weight, original), "fitting must not mutate the model")

    def test_multiple_keys_and_rectangular_weight_fit_without_transpose_error(self):
        weight = torch.zeros((2, 4), dtype=torch.float32)
        keys = torch.eye(4, dtype=torch.float32)[:2]
        targets = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
        patch = fit_projection_patch(weight, keys, targets, torch.eye(4), fit_ridge=1e-10)
        torch.testing.assert_close(keys @ patch.edited_weight().T, targets)
        self.assertEqual(patch.left_factor.shape, (2, 2))
        self.assertEqual(patch.right_factor.shape, (2, 4))

    def test_retention_covariance_reduces_change_on_frequent_direction(self):
        weight = torch.zeros((1, 2), dtype=torch.float64)
        keys = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        targets = torch.ones((1, 1), dtype=torch.float64)
        retention = torch.tensor([[100.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
        patch = fit_projection_patch(weight, keys, targets, retention, fit_ridge=1e-12)
        edited = patch.edited_weight()
        self.assertLess(abs(float(edited[0, 0])), abs(float(edited[0, 1])) / 100)
        torch.testing.assert_close(keys @ edited.T, targets, atol=1e-8, rtol=0)

    def test_apply_is_once_and_rollback_restores_exact_parameter_bytes(self):
        self.weight = self.weight.float()
        original = self.weight.clone()
        patch = self.fit()
        apply_projection_patch_(self.weight, patch)
        self.assertEqual(projection_sha256(self.weight), patch.after_sha256)
        with self.assertRaisesRegex(ValueError, "source state"):
            apply_projection_patch_(self.weight, patch)
        rollback_projection_patch_(self.weight, patch)
        self.assertTrue(torch.equal(self.weight, original))
        self.assertEqual(projection_sha256(self.weight), patch.before_sha256)
        with self.assertRaisesRegex(ValueError, "edited state"):
            rollback_projection_patch_(self.weight, patch)
        apply_projection_patch_(self.weight, patch)
        self.assertEqual(projection_sha256(self.weight), patch.after_sha256)

    def test_stale_state_rejected_without_mutating_it(self):
        patch = self.fit()
        stale = self.weight.clone()
        stale[0, 1] = 0.1
        before = stale.clone()
        for operation in (apply_projection_patch_, rollback_projection_patch_):
            with self.assertRaisesRegex(ValueError, "state"):
                operation(stale, patch)
            self.assertTrue(torch.equal(stale, before))

    def test_serialization_round_trip_preserves_factors_hashes_and_rollback(self):
        patch = self.fit(metadata={"support_ids": ["development-image-1"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.pt"
            save_projection_patch(patch, path)
            # The serialized object can be loaded by the restricted loader.
            self.assertIsInstance(torch.load(path, weights_only=True), dict)
            restored = load_projection_patch(path)
        self.assertEqual(restored.before_sha256, patch.before_sha256)
        self.assertEqual(restored.after_sha256, patch.after_sha256)
        self.assertEqual(restored.metadata, patch.metadata)
        torch.testing.assert_close(restored.edited_weight(), patch.edited_weight(), atol=0, rtol=0)
        apply_projection_patch_(self.weight, restored)
        rollback_projection_patch_(self.weight, restored)
        self.assertTrue(torch.equal(self.weight, torch.eye(3, dtype=torch.float64)))

    def test_tampered_factor_and_snapshot_are_rejected_before_apply(self):
        patch = self.fit()
        patch.left_factor[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "integrity"):
            apply_projection_patch_(self.weight, patch)
        patch = self.fit()
        patch.before_weight[0, 0] += 1
        with self.assertRaisesRegex(ValueError, "integrity"):
            apply_projection_patch_(self.weight, patch)
        self.assertTrue(torch.equal(self.weight, torch.eye(3, dtype=torch.float64)))

    def test_invalid_dimensions_nonfinite_values_and_regularization_rejected(self):
        for kwargs in ({"covariance_ridge": 0}, {"covariance_ridge": float("nan")},
                       {"fit_ridge": -1}, {"fit_ridge": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_projection_patch(self.weight, self.keys, self.targets, self.retention, **kwargs)
        for keys, targets, retention in (
            (torch.ones(1, 2), self.targets, self.retention),
            (self.keys, torch.ones(2, 3), self.retention),
            (self.keys, self.targets, torch.ones(2, 2)),
            (self.keys * float("nan"), self.targets, self.retention),
            (torch.empty(0, 3), torch.empty(0, 3), self.retention),
        ):
            with self.subTest(shape=keys.shape), self.assertRaises(ValueError):
                fit_projection_patch(self.weight, keys, targets, retention)

    def test_dtype_is_bound_and_zero_edit_is_rejected(self):
        self.assertNotEqual(projection_sha256(self.weight), projection_sha256(self.weight.float()))
        patch = self.fit()
        with self.assertRaisesRegex(ValueError, "source state"):
            apply_projection_patch_(self.weight.float(), patch)
        with self.assertRaisesRegex(ValueError, "no representable"):
            fit_projection_patch(self.weight, self.keys, self.keys @ self.weight.T, self.retention)

    def test_duplicate_support_and_empty_signal_retention_remain_finite(self):
        patch = fit_projection_patch(self.weight, self.keys.repeat(2, 1),
                                     self.targets.repeat(2, 1), torch.zeros_like(self.retention))
        self.assertTrue(bool(torch.isfinite(patch.edited_weight()).all()))
        torch.testing.assert_close(self.keys @ patch.edited_weight().T,
                                   self.targets, atol=1e-5, rtol=0)

    def test_loaded_artifact_rejects_corruption_and_unknown_format(self):
        patch = self.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch.pt"
            save_projection_patch(patch, path)
            original_payload = torch.load(path, weights_only=True)
            for field, value in (("format_version", True), ("method", "unknown"),
                                 ("before_sha256", "0" * 64),
                                 ("metadata_json", "null")):
                payload = dict(original_payload)
                payload[field] = value
                torch.save(payload, path)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    load_projection_patch(path)


if __name__ == "__main__":
    unittest.main()

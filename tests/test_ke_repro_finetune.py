"""Synthetic per-edit optimization and transactional restoration, not model scores."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("FT tests require the isolated ML environment") from error

from ke_repro.bridge import ParameterPatch, VLKEBBackend, tensor_hash
from ke_repro.finetune import PARAMETER_NAMES, finetune_last_layer


class TinyFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.untouched = torch.nn.Parameter(torch.tensor([1., 2.]))
        self.opt_model = torch.nn.Module()
        self.opt_model.model = torch.nn.Module()
        self.opt_model.model.decoder = torch.nn.Module()
        self.opt_model.model.decoder.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(32)])
        last = self.opt_model.model.decoder.layers[31]
        last.fc1 = torch.nn.Linear(4, 4, bias=False)
        last.fc2 = torch.nn.Linear(4, 3, bias=False)

    def forward(self, batch):
        last = self.opt_model.model.decoder.layers[31]
        return SimpleNamespace(logits=last.fc2(torch.tanh(last.fc1(batch["features"]))).unsqueeze(1))


def nll(logits, labels):
    return torch.nn.functional.cross_entropy(logits.reshape(-1, 3).float(), labels.reshape(-1))


class FineTuneTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = TinyFTModel().half()
        dict(self.model.named_parameters())[PARAMETER_NAMES[1]].requires_grad_(False)
        self.batch = {"features": torch.ones(1, 4, dtype=torch.float16),
                      "labels": torch.tensor([[2]]),
                      "alignment_metadata": {"target_alignment": "context"}}
        self.before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        self.flags = {name: value.requires_grad for name, value in self.model.named_parameters()}
        self.patch = ParameterPatch(self.model, PARAMETER_NAMES, identity={"fixture": "synthetic"})

    def run_update(self, loss=nll):
        return self.patch.capture_update(lambda: finetune_last_layer(self.model, self.batch, loss))

    def assert_restored(self):
        for name, value in self.model.named_parameters():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))
            self.assertEqual(value.requires_grad, self.flags[name])
            self.assertIsNone(value.grad)

    def test_fixed_fifteen_updates_reduce_actual_cross_entropy_and_freeze_other_weights(self):
        initial = float(nll(self.model(self.batch).logits, self.batch["labels"]).item())
        result = self.run_update()
        self.assertEqual(result["completed_steps"], 15)
        self.assertEqual(len(result["loss_per_step"]), 15)
        self.assertLess(result["final_teacher_forced_nll"], initial)
        self.assertEqual(result["optimizer_recipe"]["lr"], 2e-4)
        self.assertEqual(result["optimizer_recipe"]["weight_decay"], 0.01)
        self.assertEqual(result["master_dtype"], "torch.float32")
        self.assertFalse(result["full_fp32_upstream_reproduction"])
        self.assertFalse(result["kl_regularization"])
        for name, value in self.model.named_parameters():
            self.assertEqual(value.dtype, torch.float16)
            self.assertEqual(value.requires_grad, self.flags[name])
            self.assertIsNone(value.grad)
            if name not in PARAMETER_NAMES:
                self.assertTrue(torch.equal(value, self.before[name]))
        self.patch.rollback()
        self.assert_restored()

    def test_export_reload_and_rollback_restore_exact_original_half_bytes(self):
        self.run_update()
        edited = self.patch.weight_hashes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patch"
            self.patch.save(path)
            self.patch.rollback()
            self.assert_restored()
            self.patch.load(path)
            self.assertEqual(self.patch.weight_hashes(), edited)
            self.patch.rollback()
            self.assert_restored()

    def test_mid_run_failure_restores_weights_flags_and_hook_tensors(self):
        calls = 0

        def failing_loss(logits, labels):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("deliberate third-step failure")
            return nll(logits, labels)

        for parameter in self.model.parameters():
            parameter.__x__, parameter.__delta__ = torch.ones(1), torch.ones(1)
        with self.assertRaisesRegex(RuntimeError, "third-step"):
            self.run_update(failing_loss)
        self.assert_restored()
        self.assertIsNone(self.patch.active)
        self.assertTrue(all(not hasattr(p, "__x__") and not hasattr(p, "__delta__")
                            for p in self.model.parameters()))

    def test_capture_rejects_nonfinite_update_and_restores_originals(self):
        def corrupt():
            with torch.no_grad():
                dict(self.model.named_parameters())[PARAMETER_NAMES[0]].fill_(float("inf"))
            return {}

        with self.assertRaisesRegex(RuntimeError, "finite"):
            self.patch.capture_update(corrupt)
        self.assert_restored()
        self.assertIsNone(self.patch.active)

    def test_context_labels_and_exact_parameter_names_are_required(self):
        self.batch["alignment_metadata"]["target_alignment"] = "release"
        with self.assertRaisesRegex(ValueError, "context-aligned"):
            self.run_update()
        self.assert_restored()
        self.batch["alignment_metadata"]["target_alignment"] = "context"
        with self.assertRaisesRegex(ValueError, "Missing exact FT parameters"):
            finetune_last_layer(torch.nn.Linear(2, 2), self.batch, nll)

    def test_backend_supervises_newline_suffix_and_preserves_desired_answer_separately(self):
        class CharacterTokenizer:
            def encode(self, text, add_special_tokens=True):
                return ([1] if add_special_tokens else []) + [ord(char) for char in text]

        self.model.opt_tokenizer = CharacterTokenizer()
        self.model.device = torch.device("cpu")
        backend = object.__new__(VLKEBBackend)
        backend.model, backend.patches = self.model, self.patch
        backend.editor = SimpleNamespace(config=None, edit_loss_fn=None)
        backend._image = lambda image: image

        def inspect_batch(model, batch, loss_function):
            self.assertEqual(batch["text_input"], ["Question? Albany\n"])
            self.assertEqual(batch["labels"][0, -1].item(), ord("\n"))
            self.assertEqual(batch["alignment_metadata"]["answer_termination_supervision"], "newline")
            return {"answer_termination_supervision": "newline"}

        with patch("ke_repro.finetune.finetune_last_layer", side_effect=inspect_batch):
            result = backend.edit_finetune(None, "Question?", "Albany")
        self.assertEqual(result["answer_termination_supervision"], "newline")
        self.assertEqual(self.patch.active["metadata"]["desired_target"], "Albany")
        self.assertEqual(self.patch.active["metadata"]["target"], "Albany")
        self.assertEqual(self.patch.active["metadata"]["training_target"], "Albany\n")
        self.assertTrue(self.patch.active["metadata"]["supervised_target_includes_terminator"])
        self.patch.rollback()


if __name__ == "__main__":
    unittest.main()

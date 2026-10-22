"""Tiny tensor checks for patch lifecycle, not released-model performance."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Bridge adapter tests require the isolated ML environment") from exc

from ke_repro.bridge import ParameterPatch, _low_memory_opt_loader, extract_mend_factors, tensor_hash


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 3, bias=False)
        self.second = torch.nn.Linear(3, 2, bias=False)
        self.untouched = torch.nn.Parameter(torch.tensor([1., 2.]))

    def forward(self, batch):
        return self.second(torch.tanh(self.first(batch["features"])))


class IdentityFactors(torch.nn.Module):
    def forward(self, u, v, index=None):
        return u.reshape(-1, u.shape[-1]).float(), v.reshape(-1, v.shape[-1]).float()


class TinyEditor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = SimpleNamespace(inner_params=["first.weight", "second.weight"], shared=False)
        self.mend = torch.nn.ModuleDict({name.replace(".", "#"): IdentityFactors()
                                        for name in self.config.inner_params})
        self.edit_loss_fn = lambda config, logits, labels: {
            "nll": torch.nn.functional.mse_loss(logits, labels)
        }
        for name in self.config.inner_params:
            module = getattr(model, name.split(".")[0])
            module.register_forward_hook(self._forward_hook)
            module.register_full_backward_hook(self._backward_hook)

    @staticmethod
    def _forward_hook(module, inputs, output):
        module.weight.__x__ = inputs[0].detach()

    @staticmethod
    def _backward_hook(module, inputs, outputs):
        module.weight.__delta__ = outputs[0].detach()


class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = TinyModel().half()
        self.names = ["first.weight", "second.weight"]
        self.patch = ParameterPatch(self.model, self.names, identity={"release": "synthetic"})
        self.before = {name: value.detach().clone() for name, value in self.model.named_parameters()}
        self.factors = {"first.weight": (torch.randn(5, 4), torch.randn(5, 3)),
                        "second.weight": (torch.randn(5, 3), torch.randn(5, 2))}
        self.rates = torch.tensor([.0123, -.0246])

    def test_factor_orientation_rate_sign_and_exact_rollback(self):
        result = self.patch.apply(self.factors, self.rates, lr_scale=1.25)
        self.assertEqual(result["changed_parameters"], self.names)
        parameters = dict(self.model.named_parameters())
        for index, name in enumerate(self.names):
            u, v = self.factors[name]
            expected = self.before[name].clone()
            expected.add_((v.T @ u) * self.rates[index] * 1.25)
            self.assertEqual(tensor_hash(parameters[name]), tensor_hash(expected))
        self.assertEqual(tensor_hash(parameters["untouched"]), tensor_hash(self.before["untouched"]))
        restored = self.patch.rollback()
        self.assertTrue(restored["exact_weight_restore"])
        for name, value in parameters.items():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))

    def test_failed_second_update_restores_first_parameter(self):
        broken = dict(self.factors)
        broken["second.weight"] = (torch.randn(5, 8), torch.randn(5, 2))
        with self.assertRaisesRegex(ValueError, "Factor shape"):
            self.patch.apply(broken, self.rates)
        self.assertIsNone(self.patch.active)
        for name, value in self.model.named_parameters():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))

    def test_nonfinite_update_is_rejected_transactionally(self):
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            self.patch.apply(self.factors, torch.tensor([.01, float("nan")]))
        for name, value in self.model.named_parameters():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))

    def test_save_reload_and_original_bytes_survive_disk(self):
        self.patch.apply(self.factors, self.rates, metadata={"source": "tiny fixture"})
        edited_hashes = self.patch.weight_hashes()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.pt"
            saved = self.patch.save(path)
            self.assertGreater(saved["bytes"], 0)
            self.assertEqual(len(saved["sha256"]), 64)
            with self.assertRaises(FileExistsError):
                self.patch.save(path)
            with self.assertRaisesRegex(RuntimeError, "active"):
                self.patch.load(path)
            self.patch.rollback()
            self.assertEqual(self.patch.load(path)["weight_hashes"], edited_hashes)
            self.patch.rollback()
        for name, value in self.model.named_parameters():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))

    def test_reload_refuses_different_base_or_identity(self):
        self.patch.apply(self.factors, self.rates)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "patch.pt"
            self.patch.save(path)
            self.patch.rollback()
            with torch.no_grad():
                self.model.first.weight[0, 0].add_(1)
            with self.assertRaisesRegex(ValueError, "base weights"):
                self.patch.load(path)
            other = ParameterPatch(self.model, self.names, identity={"release": "different"})
            with self.assertRaisesRegex(ValueError, "different model"):
                other.load(path)

    def test_reload_rejects_dtype_and_nonfinite_tensor(self):
        self.patch.apply(self.factors, self.rates)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pt"
            payload = dict(self.patch.active)
            payload["after"] = dict(payload["after"])
            payload["after"][self.names[0]] = payload["after"][self.names[0]].float()
            torch.save(payload, path)
            self.patch.rollback()
            with self.assertRaisesRegex(ValueError, "dtype/values"):
                self.patch.load(path)
        for name, value in self.model.named_parameters():
            self.assertEqual(tensor_hash(value), tensor_hash(self.before[name]))

    def test_one_active_patch_and_explicit_lifecycle(self):
        with self.assertRaisesRegex(RuntimeError, "no active"):
            self.patch.rollback()
        self.patch.apply(self.factors, self.rates)
        with self.assertRaisesRegex(RuntimeError, "active"):
            self.patch.apply(self.factors, self.rates)


class FactorTests(unittest.TestCase):
    def test_identity_transform_matches_autograd_gradient_without_mutation(self):
        torch.manual_seed(19)
        model = TinyModel()
        editor = TinyEditor(model)
        features = torch.randn(2, 5, 4)
        labels = torch.randn(2, 5, 2)
        batch = {"features": features, "labels": labels}
        before = {name: tensor_hash(value) for name, value in model.named_parameters()}
        logits = model(batch)
        torch.nn.functional.mse_loss(logits, labels).backward()
        expected = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
                    if name in editor.config.inner_params}
        # Deliberately leave gradients to prove the extraction clears accumulation.
        factors, metadata = extract_mend_factors(editor, batch)
        self.assertGreater(metadata["inner_nll"], 0)
        self.assertFalse(editor.training)
        for name, (u, v) in factors.items():
            self.assertTrue(torch.allclose(v.T @ u, expected[name], atol=1e-6, rtol=1e-5))
        for name, value in model.named_parameters():
            self.assertEqual(tensor_hash(value), before[name])
            self.assertIsNone(value.grad)
            self.assertFalse(hasattr(value, "__x__"))
            self.assertFalse(hasattr(value, "__delta__"))

    def test_shared_transform_receives_parameter_mode_and_stays_in_eval(self):
        torch.manual_seed(23)
        model = TinyModel()
        model.first = torch.nn.Linear(4, 4, bias=False)
        model.second = torch.nn.Linear(4, 4, bias=False)
        editor = TinyEditor(model)
        editor.config.shared = True
        editor.get_shape = lambda parameter: (parameter.shape[1], parameter.shape[0])
        editor.shape_dict = {(4, 4): editor.config.inner_params}

        class ModeFactors(IdentityFactors):
            def __init__(self):
                super().__init__()
                self.modes = []

            def forward(self, u, v, index=None):
                if self.training:
                    raise AssertionError("Inference must not update training normalization statistics")
                self.modes.append(index)
                return super().forward(u, v, index)

        transform = ModeFactors()
        editor.mend = torch.nn.ModuleDict({"(4, 4)": transform})
        factors, _ = extract_mend_factors(editor, {"features": torch.randn(2, 3, 4),
                                                 "labels": torch.randn(2, 3, 4)})
        self.assertEqual(list(factors), editor.config.inner_params)
        self.assertEqual(transform.modes, [0, 1])


class LoaderTests(unittest.TestCase):
    def test_low_memory_loader_preserves_class_arguments_and_restores_descriptor(self):
        class Original:
            calls = []

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return cls(), {"missing_keys": [], "mismatched_keys": [], "error_msgs": []}

        descriptor = Original.__dict__["from_pretrained"]
        report = {}
        with _low_memory_opt_loader(Original, report):
            result = Original.from_pretrained("weights", torch_dtype=torch.float16)
            self.assertIsInstance(result, Original)
        self.assertIs(Original.__dict__["from_pretrained"], descriptor)
        self.assertEqual(Original.calls[0][0], ("weights",))
        self.assertEqual(Original.calls[0][1]["torch_dtype"], torch.float16)
        self.assertTrue(Original.calls[0][1]["low_cpu_mem_usage"])
        self.assertTrue(Original.calls[0][1]["output_loading_info"])

    def test_loader_refuses_missing_weights_and_restores_after_error(self):
        class Original:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls(), {"missing_keys": ["model.decoder.layers.29.fc1.weight"]}

        descriptor = Original.__dict__["from_pretrained"]
        with self.assertRaisesRegex(RuntimeError, "completely populate"):
            with _low_memory_opt_loader(Original, {}):
                Original.from_pretrained("weights")
        self.assertIs(Original.__dict__["from_pretrained"], descriptor)


if __name__ == "__main__":
    unittest.main()

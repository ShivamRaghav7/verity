"""Core resume checks without ML imports; optional real CPU optimizer regression."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from editguard.training import DeterministicSampler, atomic_json, canonical_hash, validate_identity


ROOT = Path(__file__).resolve().parents[1]


class TrainerContractTests(unittest.TestCase):
    def test_training_import_does_not_load_ml_packages(self):
        process = subprocess.run([sys.executable, "-c",
            "import sys; import editguard.training; import scripts.train_auxiliary; "
            "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr)

    def test_sampler_resume_matches_multiple_future_epoch_orders(self):
        original = DeterministicSampler(5, 42)
        original.next_indices(2)
        original.next_indices(2)
        restored = DeterministicSampler(5, 999)
        restored.load_state_dict(original.state_dict())
        self.assertEqual([original.next_indices(2) for _ in range(12)],
                         [restored.next_indices(2) for _ in range(12)])

    def test_sampler_rejects_wrong_population_or_invalid_cursor(self):
        sampler = DeterministicSampler(3, 42)
        for change in ({"size": 4}, {"order": [0, 1, 1]}, {"cursor": 4}, {"epoch": -1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                sampler.load_state_dict({**sampler.state_dict(), **change})

    def test_exact_identity_rejects_run_data_and_environment_changes(self):
        identity = {"configuration_hash": "configuration", "environment_hash": "environment",
                    "data_hashes": {"train": "train", "validation": "validation"}, "reviews_sha256": "reviews"}
        for key in identity:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_identity(identity, {**identity, key: "changed"})
        self.assertEqual(canonical_hash({"a": 1, "b": 2}), canonical_hash({"b": 2, "a": 1}))

    def test_failed_atomic_json_replacement_preserves_old_file(self):
        # Keep atomic replacement fixtures outside cloud-synchronized checkouts.
        with tempfile.TemporaryDirectory(prefix="editguard-trainer-contract-") as temporary:
            path = Path(temporary) / "manifest.json"
            atomic_json(path, {"status": "original"})
            with patch("editguard.training.os.replace", side_effect=OSError("simulated replacement failure")), self.assertRaises(OSError):
                atomic_json(path, {"status": "changed"})
            self.assertEqual(json.loads(path.read_text()), {"status": "original"})
            self.assertEqual(list(Path(temporary).glob("*.tmp-*")), [])


@unittest.skipUnless(os.environ.get("EDITGUARD_RUN_ML_TESTS") == "1", "Optional real CPU optimizer checks; set EDITGUARD_RUN_ML_TESTS=1 in the ML environment")
class ExactOptimizerResumeTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import numpy
        except ImportError as exc:
            self.skipTest(str(exc))
        self.torch = torch
        self.numpy = numpy
        # OneDrive can briefly lock freshly written checkpoint directories and
        # reject their atomic rename. OS scratch tests the trainer without that
        # unrelated synchronization service competing for its files.
        self.temporary = tempfile.TemporaryDirectory(prefix="editguard-trainer-ml-")
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)

    def make_model(self):
        from types import SimpleNamespace
        torch = self.torch

        class TinyClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden = torch.nn.Linear(4, 6)
                self.dropout = torch.nn.Dropout(.35)
                self.output = torch.nn.Linear(6, 2)

            def forward(self, input_ids, labels):
                logits = self.output(self.dropout(torch.tanh(self.hidden(input_ids))))
                return SimpleNamespace(loss=torch.nn.functional.cross_entropy(logits, labels))

        return TinyClassifier()

    def inputs(self):
        torch = self.torch
        return {"input_ids": torch.tensor([[1., 0., .1, .2], [0., 1., .2, .1], [.2, .3, 1., 0.],
                                           [.1, .2, 0., 1.], [.5, .5, .4, .4]]),
                "labels": torch.tensor([0, 1, 0, 1, 0])}

    def config(self):
        return {"task": "scope", "device": "cpu", "seed": 42, "batch_size": 2,
                "max_steps": 4, "learning_rate": .01, "checkpoint_every_steps": 2,
                "validation_every_steps": 2, "keep_last_checkpoints": 2, "max_grad_norm": 1.0}

    def identity(self):
        return {"purpose": "smoke_test", "configuration_hash": canonical_hash(self.config()),
                "environment_hash": "controlled-real-cpu-test", "data_hashes": {"fixture": "tiny-nonproduction"}}

    def test_resumed_parameters_losses_rng_and_selection_match_uninterrupted(self):
        from editguard.training import configure_determinism, train_loop
        torch, np = self.torch, self.numpy
        configure_determinism(42, 1)
        uninterrupted = self.make_model()
        complete = train_loop(uninterrupted, self.inputs(), self.inputs(), config=self.config(),
                              identity=self.identity(), output_dir=self.output / "complete")
        expected_parameters = {key: value.detach().clone() for key, value in uninterrupted.state_dict().items()}
        expected_rng_draw = (random.random(), float(np.random.random()), torch.rand(3))

        configure_determinism(42, 1)
        partial_model = self.make_model()
        partial = train_loop(partial_model, self.inputs(), self.inputs(), config=self.config(),
                             identity=self.identity(), output_dir=self.output / "resumed", stop_after_steps=2)
        self.assertEqual(partial["status"], "interrupted")
        # Initialization and unrelated RNG consumption must not affect resume.
        random.random(); np.random.random(); torch.rand(10)
        resumed_model = self.make_model()
        resumed = train_loop(resumed_model, self.inputs(), self.inputs(), config=self.config(),
                             identity=self.identity(), output_dir=self.output / "resumed",
                             resume_checkpoint=partial["last_checkpoint"])
        self.assertEqual(resumed["status"], "completed")
        for key, actual in resumed_model.state_dict().items():
            self.assertTrue(torch.equal(actual, expected_parameters[key]), key)
        self.assertEqual(resumed["final_state_sha256"], complete["final_state_sha256"])
        self.assertEqual(resumed["best_validation"], complete["best_validation"])
        actual_rng_draw = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(actual_rng_draw[:2], expected_rng_draw[:2])
        self.assertTrue(torch.equal(actual_rng_draw[2], expected_rng_draw[2]))
        logs = [[json.loads(line) for line in (self.output / run / "training_log.jsonl").read_text().splitlines()]
                for run in ("complete", "resumed")]
        for first, second in zip(*logs):
            for key in ("global_step", "loss", "validation_loss", "learning_rate_after_step", "batch_indices"):
                self.assertEqual(first[key], second[key], key)

    def test_corrupt_checkpoint_is_refused_before_restore(self):
        from editguard.training import configure_determinism, train_loop
        configure_determinism(42, 1)
        partial = train_loop(self.make_model(), self.inputs(), self.inputs(), config=self.config(),
                             identity=self.identity(), output_dir=self.output / "corrupt", stop_after_steps=2)
        path = Path(partial["last_checkpoint"]) / "state.pt"
        with path.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            train_loop(self.make_model(), self.inputs(), self.inputs(), config=self.config(), identity=self.identity(),
                       output_dir=self.output / "corrupt", resume_checkpoint=partial["last_checkpoint"])


if __name__ == "__main__":
    unittest.main()

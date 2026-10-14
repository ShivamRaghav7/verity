"""Small saved-byte fixtures for independent audit refusal and shared weights."""

import io
from pathlib import Path
import pickle
import struct
import tempfile
import unittest
import zipfile

from scripts.audit_scope_revision import DescriptorUnpickler, StorageKind, StorageRef, TensorRef
from scripts.audit_targeted_runs import (ARCHIVE_NAMES, archive_exact, expected_objective,
    safe_relative, selected_export_comparison)


class TargetedTrainingAuditTests(unittest.TestCase):
    @staticmethod
    def check(name, condition):
        if not condition:
            raise ValueError(name)

    def test_checkpoint_unpickler_rejects_executable_globals(self):
        import os
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint global"):
            DescriptorUnpickler(io.BytesIO(pickle.dumps(os.system))).load()

    def test_declared_objectives_are_distinct_and_task_bound(self):
        self.assertEqual(expected_objective("scope")["comparison_first_token_weight"], 1)
        weighted = expected_objective("answer")
        self.assertEqual(weighted["comparison_first_token_weight"], 8)
        self.assertEqual(weighted["normalization"], "sum_weighted_token_nll_over_sum_weights")
        with self.assertRaisesRegex(ValueError, "Only the declared"):
            expected_objective("arbitrary")

    def test_source_paths_cannot_escape_the_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(safe_relative(root, "sub/data.json"), root / "sub/data.json")
            for name in ("../data.json", ".", str(root.parent / "data.json")):
                with self.assertRaisesRegex(ValueError, "escaped"):
                    safe_relative(root, name)

    def test_differing_archive_preserved_without_partial_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, destination = root / "model", root / "archive"
            artifact.mkdir()
            destination.mkdir()
            for name in ARCHIVE_NAMES:
                (artifact / name).write_bytes(b"new bytes")
            (destination / ARCHIVE_NAMES[-1]).write_bytes(b"immutable prior bytes")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                archive_exact(artifact, destination, root)
            self.assertEqual(list(destination.iterdir()), [destination / ARCHIVE_NAMES[-1]])
            self.assertEqual((destination / ARCHIVE_NAMES[-1]).read_bytes(), b"immutable prior bytes")

    def test_identical_archive_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, destination = root / "model", root / "archive"
            artifact.mkdir()
            for name in ARCHIVE_NAMES:
                (artifact / name).write_bytes(name.encode())
            first = archive_exact(artifact, destination, root)
            second = archive_exact(artifact, destination, root)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 4)

    def t5_fixture(self, corrupt_alias=False, extra_export=False):
        value = struct.pack("<f", 3.0)
        buffer = io.BytesIO()
        state = {}
        with zipfile.ZipFile(buffer, "w") as archive:
            for index, name in enumerate(("shared.weight", "encoder.embed_tokens.weight",
                                          "decoder.embed_tokens.weight", "lm_head.weight")):
                key = str(index)
                archive.writestr("checkpoint/data/" + key,
                    struct.pack("<f", 4.0) if corrupt_alias and name.startswith("encoder") else value)
                state[name] = TensorRef(StorageRef(StorageKind("FloatStorage", "torch.float32", 4), key, "cpu", 1),
                                        0, (1,), (1,))
        safe = {"shared.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
                "lm_head.weight": {"dtype": "F32", "shape": [1], "data_offsets": [4, 8]}}
        if extra_export:
            safe["invented.weight"] = safe["shared.weight"]
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            return selected_export_comparison(io.BytesIO(value * 2), safe, 0, state, archive,
                                              "checkpoint/", "answer", self.check)

    def test_t5_omitted_shared_embeddings_are_each_byte_checked(self):
        result = self.t5_fixture()
        self.assertEqual(result["checkpoint_tensor_count"], 4)
        self.assertEqual(result["export_tensor_count"], 2)
        self.assertEqual(len(result["shared_embedding_aliases_verified"]), 2)
        with self.assertRaisesRegex(ValueError, "selected_tensor:encoder.embed_tokens.weight"):
            self.t5_fixture(corrupt_alias=True)

    def test_export_cannot_add_an_unrecorded_tensor(self):
        with self.assertRaisesRegex(ValueError, "selected_export_no_extra_keys"):
            self.t5_fixture(extra_export=True)


if __name__ == "__main__":
    unittest.main()

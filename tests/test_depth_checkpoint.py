"""Portable recovery of explicit CPU ledgers; no model loading or inference."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from editguard_benchmark.data import encoded
from editguard_benchmark.scoring import score
from editguard_depth import metrics, run as runner
from editguard_reasonke.runtime import extract_answer
from scripts import restore_depth_checkpoint as checkpoint


ROOT = Path(__file__).resolve().parents[1]


def _bytes(value):
    return (json.dumps(value, indent=2) + "\n").encode()


def receipt(model="reasonke", phase="profile", *, unfinished=False, protocol=None):
    """Actual assigned tasks with obvious fabricated CPU-process outputs."""
    config_path = ROOT / "configs/depth-evaluation.json"
    config = json.loads(config_path.read_text())
    dataset, worlds, tasks = runner.tasks_for(config, model, phase)
    model_manifest = json.loads((ROOT / config[model + "_manifest"]).read_text())
    for index, row in enumerate(tasks):
        if unfinished and index > 0:
            continue
        request, routing = runner.request_for(row, worlds, config)
        raw = "<answer>Explicit incorrect CPU fixture</answer>" if index == 0 else "<answer>UNKNOWN</answer>"
        content, fmt = extract_answer(raw)
        response = {"content": content, "raw_output": raw, "format": fmt,
                    "generation": {"model": model_manifest["model_id"], "revision": model_manifest["revision"],
                                   "fact_ids": routing["selected_fact_ids"], "max_new_tokens": config["max_new_tokens"],
                                   "max_input_tokens": 4096, "do_sample": False, "software_fixture": True}}
        row.update(request_sha256=hashlib.sha256(encoded(request)).hexdigest(),
                   supplied_facts=request["facts"], retrieval=routing,
                   pipeline_seconds=1.0, response=response)
        row.update(score(row["question"], response))
    return {"schema": "editguard-depth-inference-v1", "status": "interrupted" if unfinished else "completed",
            "plan": {"model": model, "phase": phase, "sources": runner.identities(config_path),
                     "task_ids_sha256": hashlib.sha256(encoded([row["id"] for row in tasks])).hexdigest(),
                     "protocol_sha256": hashlib.sha256(protocol).hexdigest() if protocol else None},
            "dataset": dataset, "records": tasks, "sessions": [], "software_fixture": True}


class DepthCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = {f"profile-{model}.json": _bytes(receipt(model)) for model in ("reasonke", "instruction")}

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="editguard-depth-checkpoint-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bundle = self.root / "evidence.zip"
        self.target = self.root / "restored"

    def archive(self, files=None):
        with zipfile.ZipFile(self.bundle, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, raw in (files or self.profile).items():
                archive.writestr(name, raw)
        return self.bundle

    def frozen(self):
        source = self.root / "bound-inputs"
        source.mkdir()
        files = dict(self.profile)
        files.update({f"development-{model}.json": _bytes(receipt(model, "development"))
                      for model in ("reasonke", "instruction")})
        for name, raw in files.items():
            (source / name).write_bytes(raw)
        runner.freeze(source, source / "protocol.json", ROOT / "configs/depth-evaluation.json")
        files["protocol.json"] = (source / "protocol.json").read_bytes()
        return files

    def test_profiles_restore_exact_bytes_and_disclose_missing_development(self):
        result = checkpoint.restore(self.archive(), self.target, recovery_note="Old development runtime expired; outputs were not backed up.")
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["publication"], "atomic_new_directory")
        self.assertFalse(result["missing_receipts_mean_no_generations"])
        self.assertEqual(result["missing_development_receipts"], ["development-instruction.json", "development-reasonke.json"])
        note = Path(result["recovery_receipt"])
        self.assertEqual(note.parent, self.target.parent)
        self.assertNotEqual(note.parent, self.target)
        self.assertEqual(json.loads(note.read_text())["recovery_note"], result["recovery_note"])
        self.assertEqual(set(path.name for path in self.target.iterdir()), set(self.profile))
        for name, raw in self.profile.items():
            self.assertEqual((self.target / name).read_bytes(), raw)
            audit = metrics.audit_and_aggregate([self.target / name])
            self.assertEqual(next(iter(audit["conditions"].values()))["overall"]["counts"]["wrong"], 1)

    def test_preserve_existing_is_explicit_and_idempotent(self):
        self.archive()
        self.target.mkdir()
        sentinel = self.target / "owned-model.log"
        sentinel.write_text("preserve unrelated runtime work")
        with self.assertRaisesRegex(ValueError, "preserve-existing"):
            checkpoint.restore(self.bundle, self.target)
        first = checkpoint.restore(self.bundle, self.target, preserve_existing=True)
        second = checkpoint.restore(self.bundle, self.target, preserve_existing=True)
        self.assertEqual(first["created_files"], sorted(self.profile))
        self.assertEqual(second["created_files"], [])
        self.assertEqual(second["preserved_existing_files"], sorted(self.profile))
        self.assertEqual(sentinel.read_text(), "preserve unrelated runtime work")

    def test_conflicting_existing_outcomes_are_never_replaced(self):
        self.archive()
        self.target.mkdir()
        name = "profile-reasonke.json"
        old = self.profile[name] + b" "
        (self.target / name).write_bytes(old)
        with self.assertRaisesRegex(ValueError, "outcomes differ"):
            checkpoint.restore(self.bundle, self.target, preserve_existing=True)
        self.assertEqual((self.target / name).read_bytes(), old)
        self.assertFalse((self.target / "profile-instruction.json").exists())

    def test_unknown_traversal_duplicate_symlink_and_oversized_members_refused(self):
        for name in ("../profile-reasonke.json", "/profile-reasonke.json", "recovery.json", "nested/profile-reasonke.json"):
            with self.subTest(name=name):
                self.archive({name: b"{}"})
                with self.assertRaises(ValueError): checkpoint.restore(self.bundle, self.target)
                self.assertFalse(self.target.exists())
        with warnings.catch_warnings(), zipfile.ZipFile(self.bundle, "w") as archive:
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(2): archive.writestr("profile-reasonke.json", self.profile["profile-reasonke.json"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            checkpoint.restore(self.bundle, self.target)
        with zipfile.ZipFile(self.bundle, "w") as archive:
            linked = zipfile.ZipInfo("profile-reasonke.json")
            linked.create_system = 3
            linked.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(linked, "elsewhere")
        with self.assertRaisesRegex(ValueError, "linked"):
            checkpoint.restore(self.bundle, self.target)
        self.archive()
        with patch.object(checkpoint, "MAX_FILE_BYTES", 20), self.assertRaisesRegex(ValueError, "oversized"):
            checkpoint.restore(self.bundle, self.target)
        self.assertFalse(self.target.exists())

    def test_source_label_and_filename_corruption_fail_before_publication(self):
        original = json.loads(self.profile["profile-reasonke.json"])
        cases = []
        bad = copy.deepcopy(original)
        bad["plan"]["sources"]["editguard_depth/run.py"] = "0" * 64
        cases.append({"profile-reasonke.json": _bytes(bad)})
        bad = copy.deepcopy(original)
        bad["records"][0]["question"]["expected_answer"] = "Changed target"
        cases.append({"profile-reasonke.json": _bytes(bad)})
        cases.append({"profile-instruction.json": self.profile["profile-reasonke.json"]})
        for files in cases:
            with self.subTest(files=list(files)):
                self.archive(files)
                with self.assertRaises(ValueError): checkpoint.restore(self.bundle, self.target)
                self.assertFalse(self.target.exists())

    def test_interrupted_single_ledger_retains_all_assigned_outcomes(self):
        raw = _bytes(receipt(unfinished=True))
        self.archive({"profile-reasonke.json": raw})
        result = checkpoint.restore(self.bundle, self.target)
        self.assertEqual(result["progress"]["profile-reasonke.json"],
                         {"status": "interrupted", "assigned": 8, "recorded": 1, "unexecuted": 7})
        self.assertEqual((self.target / "profile-reasonke.json").read_bytes(), raw)

    def test_protocol_requires_all_four_exact_bound_inputs(self):
        files = self.frozen()
        self.archive({name: raw for name, raw in files.items() if name != "development-instruction.json"})
        with self.assertRaisesRegex(ValueError, "four bound"):
            checkpoint.restore(self.bundle, self.target)
        self.assertFalse(self.target.exists())
        self.archive(files)
        result = checkpoint.restore(self.bundle, self.target)
        self.assertEqual(result["missing_development_receipts"], [])
        for name, raw in files.items(): self.assertEqual((self.target / name).read_bytes(), raw)
        runner.verify_protocol(self.target / "protocol.json", ROOT / "configs/depth-evaluation.json")

    def test_existing_inputs_participate_in_validation_before_new_protocol_is_added(self):
        files = self.frozen()
        self.target.mkdir()
        for name, raw in files.items():
            if name != "protocol.json": (self.target / name).write_bytes(raw)
        self.archive({"protocol.json": files["protocol.json"]})
        result = checkpoint.restore(self.bundle, self.target, preserve_existing=True)
        self.assertEqual(result["created_files"], ["protocol.json"])
        self.assertEqual(set(result["preserved_existing_files"]), checkpoint.FROZEN_INPUTS)

    def test_final_and_results_require_protocol_and_completed_final_pair(self):
        self.archive({"final-reasonke.json": _bytes(receipt(phase="final"))})
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            checkpoint.restore(self.bundle, self.target)
        self.archive({"results.json": b'{"accuracy":1.0}'})
        with self.assertRaisesRegex(ValueError, "both completed"):
            checkpoint.restore(self.bundle, self.target)
        self.assertFalse(self.target.exists())

    def test_complete_scorecard_is_recomputed_and_location_paths_are_portable(self):
        files = self.frozen()
        source = self.root / "bound-inputs"
        for model in ("reasonke", "instruction"):
            name = f"final-{model}.json"
            files[name] = _bytes(receipt(model, "final", protocol=files["protocol.json"]))
            (source / name).write_bytes(files[name])
        report = metrics.audit_and_aggregate([source / "final-reasonke.json", source / "final-instruction.json"], protocol=source / "protocol.json")
        report["receipt_audit"][0]["receipt"] = "/different/runtime/final-reasonke.json"
        report["receipt_audit"][1]["receipt"] = "C:\\different\\runtime\\final-instruction.json"
        files["results.json"] = _bytes(report)
        self.archive(files)
        checkpoint.restore(self.bundle, self.target)
        self.assertEqual((self.target / "results.json").read_bytes(), files["results.json"])
        report["conditions"]["R-RAG"]["overall"]["counts"]["correct"] += 1
        files["results.json"] = _bytes(report)
        self.archive(files)
        with self.assertRaisesRegex(ValueError, "results differ"):
            checkpoint.restore(self.bundle, self.root / "corrupted")
        self.assertFalse((self.root / "corrupted").exists())

    def test_duplicate_json_keys_are_refused(self):
        self.archive({"profile-reasonke.json": b'{"plan":{},"plan":{}}'})
        with self.assertRaisesRegex(ValueError, "Repeated JSON key"):
            checkpoint.restore(self.bundle, self.target)
        self.assertFalse(self.target.exists())

    def test_merge_publication_failure_rolls_back_only_new_files(self):
        self.archive()
        self.target.mkdir()
        sentinel = self.target / "private-log.txt"
        sentinel.write_text("keep")
        link = os.link
        def interrupted(source, target):
            if Path(target).name == "profile-reasonke.json":
                raise OSError("Explicit publication fixture")
            return link(source, target)
        with patch.object(checkpoint.os, "link", side_effect=interrupted), self.assertRaisesRegex(OSError, "publication fixture"):
            checkpoint.restore(self.bundle, self.target, preserve_existing=True)
        self.assertEqual([path.name for path in self.target.iterdir()], ["private-log.txt"])
        self.assertEqual(sentinel.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()

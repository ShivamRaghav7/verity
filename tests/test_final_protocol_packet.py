"""Synthetic final-stage freeze/derivation/refusal tests; no real held-out input."""

from collections import Counter
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import final_protocol as protocol_tool
from scripts import prepare_final_packet as packet_tool
from scripts.prepare_data import build_dataset, dataset_files


class FinalProtocolPacketTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = {"python_version": "synthetic Python never runs inference",
                            "runtime_packages": {name: "synthetic-1" for name in protocol_tool.RUNTIME_PACKAGES}}
        self.enterContext(patch("scripts.final_protocol.runtime_environment", return_value=self.environment))
        self.original, self.dev = self.root / "original", self.root / "development"
        self.original.mkdir()
        self.dev.mkdir()
        self.dataset = build_dataset(seed=917, family_count=20)
        files = dataset_files(self.dataset)
        for name, contents in files.items():
            (self.original / name).write_bytes(contents)
        original_manifest = {"dataset_id": "editguard-controlled-v1", "files": {name: protocol_tool.file_hash(self.original / name) for name in files},
            "counts": {task: dict(Counter(row["split"] for row in self.dataset[task])) for task in ("scope", "answer")},
            "challenge_drafts": len(self.dataset["challenges"])}
        self.write(self.original / "manifest.json", original_manifest)
        selected = {split: sorted(family for family, value in self.dataset["family_splits"].items() if value == split)[:1]
                    for split in ("train", "validation")}
        dev_facts = [fact for fact in self.dataset["facts"] if fact["split"] != "test"]
        self.write(self.dev / "catalogue.json", dev_facts)
        self.write(self.dev / "selection.json", {"selected_families": selected})
        self.write(self.dev / "split_assignments.json", {family: split for family, split in self.dataset["family_splits"].items() if split != "test"})
        for task in ("scope", "answer"):
            for split in ("train", "validation"):
                rows = [row for row in self.dataset[task] if row["family_id"] in selected[split]]
                (self.dev / f"{task}_{split}.jsonl").write_bytes(packet_tool.rows_bytes(rows))
        dev_manifest = {"dataset_id": "synthetic-development", "test_files_included": False,
                        "files": {path.name: protocol_tool.file_hash(path) for path in self.dev.iterdir()}}
        self.write(self.dev / "manifest.json", dev_manifest)
        self.write(self.root / "machine-reviews.json", {"purpose": "synthetic fixture registry"})
        for name in protocol_tool.SOURCE_NAMES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((protocol_tool.ROOT / name).read_bytes() if (protocol_tool.ROOT / name).exists() else b"synthetic runner pending\n")
        for task in ("scope", "answer"):
            artifact = self.root / "models" / task
            (artifact / "model").mkdir(parents=True)
            weights = artifact / "model/model.safetensors"
            weights.write_bytes(b"synthetic weights never loaded")
            self.write(artifact / "profile.json", {"purpose": "synthetic, no neural library imported"})
            training_names = ("editguard/training.py", "scripts/train_auxiliary.py", "scripts/prepare_data.py",
                              "scripts/preflight_training.py", "editguard/formatting.py", "editguard/scope_inputs.py")
            card = {"data_dir": str(self.dev), "reviews_path": str(self.root / "machine-reviews.json"),
                    "profile_path": str(artifact / "profile.json"), "profile_sha256": protocol_tool.file_hash(artifact / "profile.json"),
                    "profile_source_sha256": {name: protocol_tool.file_hash(self.root / name) for name in training_names}}
            self.write(artifact / "run_card.json", card)
            manifest = {"task": task, "status": "completed", "purpose": "trained", "input_format_version": "editguard-input-v2",
                "files_sha256": {"model/model.safetensors": protocol_tool.file_hash(weights)},
                "selected_model_state_sha256": "a" * 64, "model_id": "synthetic-model", "model_revision": "b" * 40,
                "review_policy": "automated_development", "human_reviewed_data": False,
                "dataset_id": "synthetic-development", "manifest_sha256": protocol_tool.file_hash(self.dev / "manifest.json"),
                "reviews_sha256": protocol_tool.file_hash(self.root / "machine-reviews.json"),
                "run_card_sha256": protocol_tool.file_hash(artifact / "run_card.json")}
            if task == "scope":
                manifest.update(scope_input_policy="entity_neutral_v1", scope_input_overlap={"cross_split": {"same_label_pairs": 0}})
            self.write(artifact / "manifest.json", manifest)
        self.write(self.root / "calibrations/scope/calibration.json", {"status": "frozen", "split": "validation",
            "scope_manifest_sha256": protocol_tool.file_hash(self.root / "models/scope/manifest.json")})
        self.write(self.root / "calibrations/similarity/calibration.json", {"status": "frozen", "split": "validation"})
        self.config = self.root / "config.json"
        self.write(self.config, {"scope_artifact": "models/scope", "answer_artifact": "models/answer",
            "calibration_artifact": "calibrations/scope/calibration.json", "similarity_calibration_artifact": "calibrations/similarity/calibration.json",
            "base_revision": "c" * 40, "embedding_revision": "d" * 40,
            "input_format_version": "editguard-input-v2", "local_files_only": True, "max_new_tokens": 48})
        self.development_evaluation = self.root / "development-evaluation.json"
        self.write(self.development_evaluation, {"purpose": "validation_development", "status": "completed",
            "base_weights_unchanged": True, "base_parameter_sha256_before": "f" * 64, "base_parameter_sha256_after": "f" * 64,
            "inference_config": protocol_tool.read_json(self.config), "scorer_version": protocol_tool.SCORER_VERSION})
        self.protocol = self.root / "protocol.json"
        self.packet = self.root / "packet"
        self.original_hashes = {path.name: protocol_tool.file_hash(path) for path in self.original.iterdir()}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(packet_tool.json_bytes(value))

    def freeze(self):
        return protocol_tool.freeze_protocol(self.config, self.original, self.dev, self.protocol,
            self.root / "evaluation-output", self.root / "ledger.json", root=self.root,
            original_manifest_sha256=self.original_hashes["manifest.json"], development_evaluation=self.development_evaluation)

    def prepare(self):
        self.freeze()
        return packet_tool.prepare_packet(self.protocol, protocol_tool.file_hash(self.protocol), self.packet, root=self.root)

    def validate(self):
        return packet_tool.validate_packet(self.packet, self.protocol, protocol_tool.file_hash(self.protocol), root=self.root)

    def rehash_packet(self, changed_name):
        manifest = protocol_tool.read_json(self.packet / "manifest.json")
        manifest["files"][changed_name] = protocol_tool.file_hash(self.packet / changed_name)
        self.write(self.packet / "manifest.json", manifest)

    def test_actual_frozen_packet_handoff_matches_runner_dependency_inventory(self):
        from scripts.run_final_evaluation import locked_inputs, REQUIRED_SOURCES
        self.prepare()
        protocol, manifest, facts, populations = locked_inputs(
            self.protocol, protocol_tool.file_hash(self.protocol), self.packet, root=self.root)
        self.assertEqual(REQUIRED_SOURCES, set(protocol_tool.SOURCE_NAMES))
        self.assertEqual(set(protocol["source_sha256"]), REQUIRED_SOURCES)
        self.assertEqual({name: len(rows) for name, rows in populations.items()}, manifest["counts"])
        self.assertTrue(facts)
        self.assertEqual(protocol["evaluation"]["matched_questions"], len(populations["scope"]) + len(populations["challenge"]))

    def test_freeze_does_not_deserialize_questions_and_binds_single_attempt(self):
        original_reader = protocol_tool.read_json
        forbidden = {"scope_test.jsonl", "answer_test.jsonl", "challenge_drafts.jsonl"}
        def guard(path):
            self.assertNotIn(Path(path).name, forbidden)
            return original_reader(path)
        with patch("scripts.final_protocol.read_json", side_effect=guard):
            frozen = self.freeze()
        self.assertFalse(frozen["original_question_content_deserialized_during_freeze"])
        self.assertEqual(frozen["evaluation"]["max_final_runs"], 1)
        self.assertEqual(frozen["evaluation"]["matched_populations"], ["scope", "challenge"])
        self.assertEqual(frozen["evaluation"]["oracle_populations"], ["answer", "challenge"])
        self.assertFalse(frozen["human_reviewed_data"])
        self.assertIn("scripts/review_dataset.py", frozen["source_sha256"])
        self.assertIn("editguard/__init__.py", frozen["source_sha256"])
        self.assertIn("editguard/training.py", frozen["source_sha256"])
        self.assertEqual(frozen["runtime_packages"], self.environment["runtime_packages"])
        self.assertEqual(frozen["evaluation"]["expected_base_parameter_fingerprint_sha256"], "f" * 64)
        self.assertIn("training_implementation_sha256", frozen["models"]["answer"])

    def test_packet_refuses_before_any_row_read_without_valid_protocol(self):
        self.freeze()
        with patch("scripts.prepare_final_packet.read_rows") as reader:
            with self.assertRaisesRegex(ValueError, "protocol checksum"):
                packet_tool.prepare_packet(self.protocol, "0" * 64, self.packet, root=self.root)
            reader.assert_not_called()
        self.assertFalse(self.packet.exists())

    def test_complete_derivation_preserves_originals_labels_counts_and_questions(self):
        manifest = self.prepare()
        validated, facts, scope, answer, challenge = self.validate()
        self.assertEqual(validated, manifest)
        self.assertTrue(manifest["eligible_for_final_evaluation"])
        self.assertEqual(manifest["review_counts"]["flagged"], 0)
        for population, rows, source_name in (("scope", scope, "scope_test.jsonl"), ("answer", answer, "answer_test.jsonl"), ("challenge", challenge, "challenge_drafts.jsonl")):
            originals = {row["id"]: row for row in packet_tool.read_rows(self.original / source_name)}
            self.assertEqual(len(rows), len(originals))
            for row in rows:
                for key in ("question", "expected_fact", "request_context", "family_id", "lineage_id", "split"):
                    self.assertEqual(row[key], originals[row["id"]][key])
                if population == "scope":
                    self.assertEqual(row["label"], originals[row["id"]]["label"])
                else:
                    self.assertEqual(row["label"], 1)
        self.assertEqual({path.name: protocol_tool.file_hash(path) for path in self.original.iterdir()}, self.original_hashes)
        self.assertFalse(manifest["eligible_for_training"])
        self.assertFalse(manifest["human_reviewed_data"])
        self.assertEqual(len(facts), sum(fact["split"] == "test" for fact in self.dataset["facts"]))

    def test_frozen_config_model_source_or_original_tamper_refused(self):
        self.freeze()
        digest = protocol_tool.file_hash(self.protocol)
        for path in (self.config, self.root / "models/scope/model/model.safetensors",
                     self.root / "scripts/final_label_audit.py", self.original / "scope_test.jsonl",
                     self.development_evaluation, self.root / "models/answer/profile.json"):
            before = path.read_bytes()
            try:
                path.write_bytes(before + b"tamper")
                with self.assertRaises(ValueError):
                    protocol_tool.validate_protocol(self.protocol, digest, root=self.root)
            finally:
                path.write_bytes(before)
        protocol_tool.validate_protocol(self.protocol, digest, root=self.root)

    def test_runtime_version_drift_is_refused_before_any_heldout_rows(self):
        self.freeze()
        drifted = copy.deepcopy(self.environment)
        drifted["runtime_packages"]["transformers"] = "synthetic-2"
        with patch("scripts.final_protocol.runtime_environment", return_value=drifted), \
                patch("scripts.prepare_final_packet.read_rows") as reader:
            with self.assertRaisesRegex(ValueError, "distribution versions differ"):
                packet_tool.prepare_packet(self.protocol, protocol_tool.file_hash(self.protocol), self.packet, root=self.root)
            reader.assert_not_called()

    def test_validation_evidence_requires_same_config_and_unchanged_base(self):
        evidence = protocol_tool.read_json(self.development_evaluation)
        for field, value in (("base_weights_unchanged", False), ("base_parameter_sha256_after", "e" * 64),
                             ("inference_config", {}), ("scorer_version", "unsupported")):
            modified = {**evidence, field: value}
            self.write(self.development_evaluation, modified)
            with self.assertRaisesRegex(ValueError, "matching validation evidence"):
                self.freeze()
            self.assertFalse(self.protocol.exists())
        self.write(self.development_evaluation, evidence)

    def test_frozen_protocol_tamper_even_with_new_external_hash_refused(self):
        self.freeze()
        protocol = protocol_tool.read_json(self.protocol)
        protocol["evaluation"]["max_final_runs"] = 2
        self.write(self.protocol, protocol)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            protocol_tool.validate_protocol(self.protocol, protocol_tool.file_hash(self.protocol), root=self.root)

    def test_packet_and_protocol_overwrite_refused(self):
        self.prepare()
        before = protocol_tool.file_hash(self.protocol)
        with self.assertRaisesRegex(ValueError, "fresh immutable"):
            self.freeze()
        with self.assertRaisesRegex(ValueError, "fresh final packet"):
            packet_tool.prepare_packet(self.protocol, before, self.packet, root=self.root)
        self.assertEqual(protocol_tool.file_hash(self.protocol), before)

    def test_row_tamper_cannot_pass_even_after_all_self_declared_hashes_rewritten(self):
        self.prepare()
        path = self.packet / "scope_test.jsonl"
        original_bytes = path.read_bytes()
        base_manifest = protocol_tool.read_json(self.packet / "manifest.json")
        for field, replacement in (("label", 0), ("question", "A new question?"),
                                   ("expected_fact", {"value": "999"})):
            rows = packet_tool.read_rows(path)
            positive = next(row for row in rows if row["label"] == 1)
            positive[field] = replacement
            path.write_bytes(packet_tool.rows_bytes(rows))
            self.rehash_packet(path.name)
            # An attacker can rewrite registry and recorded audit hashes too;
            # replaying original derivation still rejects their changed row.
            registry = protocol_tool.read_json(self.packet / "reviews.json")
            audit = protocol_tool.read_json(self.packet / "audit.json")
            derivations = protocol_tool.read_json(self.packet / "derivation.json")
            registry["rows"][positive["id"]]["row_sha256"] = protocol_tool.canonical_hash(positive)
            audit["rows"][positive["id"]]["row_sha256"] = protocol_tool.canonical_hash(positive)
            next(item for item in derivations if item["id"] == positive["id"])["derived_row_sha256"] = protocol_tool.canonical_hash(positive)
            self.write(self.packet / "audit.json", audit)
            registry["audit_sha256"] = protocol_tool.file_hash(self.packet / "audit.json")
            self.write(self.packet / "reviews.json", registry)
            self.write(self.packet / "derivation.json", derivations)
            for name in ("audit.json", "reviews.json", "derivation.json"):
                self.rehash_packet(name)
            with self.assertRaisesRegex(ValueError, "changed original questions, labels, facts"):
                self.validate()
            path.write_bytes(original_bytes)
            self.write(self.packet / "manifest.json", base_manifest)
            # Each next fixture uses the same id; reset every byte-bound record.
            original_rows = packet_tool.read_rows(path)
            id_row = next(row for row in original_rows if row["id"] == positive["id"])
            restored_hash = protocol_tool.canonical_hash(id_row)
            registry["rows"][positive["id"]]["row_sha256"] = restored_hash
            audit["rows"][positive["id"]]["row_sha256"] = restored_hash
            next(item for item in derivations if item["id"] == positive["id"])["derived_row_sha256"] = restored_hash
            self.write(self.packet / "audit.json", audit)
            registry["audit_sha256"] = protocol_tool.file_hash(self.packet / "audit.json")
            self.write(self.packet / "reviews.json", registry)
            self.write(self.packet / "derivation.json", derivations)

    def test_raw_family_entity_lineage_leakage_is_refused(self):
        self.freeze()
        protocol = protocol_tool.read_json(self.protocol)
        training = packet_tool.read_rows(self.dev / "scope_train.jsonl")
        leaked = copy.deepcopy(training[0])
        with self.assertRaisesRegex(ValueError, "leaks raw"):
            packet_tool.leakage_and_overlap({"scope": [leaked], "answer": [], "challenge": []},
                protocol_tool.read_json(self.dev / "catalogue.json"), protocol, self.root)

    def test_rehashed_false_human_claim_or_forged_reviewer_is_refused(self):
        self.prepare()
        registry_path = self.packet / "reviews.json"
        original_registry = protocol_tool.read_json(registry_path)
        original_manifest = protocol_tool.read_json(self.packet / "manifest.json")
        modified = copy.deepcopy(original_registry)
        modified["human_approval_claimed"] = True
        self.write(registry_path, modified)
        self.rehash_packet("reviews.json")
        with self.assertRaisesRegex(ValueError, "stale automated review"):
            self.validate()
        modified = copy.deepcopy(original_registry)
        next(iter(modified["rows"].values()))["reviewer"] = "Invented independent reviewer"
        self.write(registry_path, modified)
        self.rehash_packet("reviews.json")
        with self.assertRaisesRegex(ValueError, "review record differs"):
            self.validate()
        self.write(registry_path, original_registry)
        self.write(self.packet / "manifest.json", original_manifest)
        self.validate()

    def test_rehashed_audit_cannot_approve_a_changed_truth_or_counts(self):
        self.prepare()
        manifest_path = self.packet / "manifest.json"
        manifest = protocol_tool.read_json(manifest_path)
        changed_manifest = copy.deepcopy(manifest)
        changed_manifest["review_counts"]["accepted"] -= 1
        self.write(manifest_path, changed_manifest)
        with self.assertRaisesRegex(ValueError, "stale automated review"):
            self.validate()
        self.write(manifest_path, manifest)
        audit_path = self.packet / "audit.json"
        audit = protocol_tool.read_json(audit_path)
        next(iter(audit["rows"].values()))["computed_expected_fact"]["value"] = "999"
        self.write(audit_path, audit)
        registry_path = self.packet / "reviews.json"
        registry = protocol_tool.read_json(registry_path)
        registry["audit_sha256"] = protocol_tool.file_hash(audit_path)
        self.write(registry_path, registry)
        for name in ("audit.json", "reviews.json"):
            self.rehash_packet(name)
        with self.assertRaisesRegex(ValueError, "independent audit cannot be reproduced"):
            self.validate()

    def test_transformed_same_label_overlap_disclosed_and_contradiction_refused(self):
        self.prepare()
        protocol = protocol_tool.read_json(self.protocol)
        facts = protocol_tool.read_json(self.packet / "catalogue.json")
        populations = {name: packet_tool.read_rows(self.packet / path) for name, path in packet_tool.ROW_NAMES.items()}
        duplicate = copy.deepcopy(next(row for row in populations["scope"] if row["label"] == 1))
        duplicate["id"] += ":synthetic-duplicate"
        duplicate["label"] = 0
        populations["scope"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "contradictory labels"):
            packet_tool.leakage_and_overlap(populations, facts, protocol, self.root)


if __name__ == "__main__":
    unittest.main()

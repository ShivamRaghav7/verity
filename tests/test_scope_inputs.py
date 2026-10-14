from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import RLock
import unittest
from unittest.mock import patch

from editguard.formatting import INPUT_FORMAT_VERSION, correction_text
from editguard.neural import ModelUnavailable, ScopeCheckpoint, file_sha256, validate_artifact
from editguard.scope_inputs import (DEFAULT_SCOPE_INPUT_POLICY, SCOPE_INPUT_POLICIES,
    SCOPE_INPUT_POLICY_VERSION, scope_pair, scope_policy_identity, validate_scope_policy)


def descriptor(**changes):
    edit = {"entity_name": "Nimbus", "entity_aliases": ["Nimbus Cloud", "NC"], "plan": "Basic", "version": "v2",
        "relation": "device_limit", "original_value": "5", "corrected_value": "3", "unit": "devices",
        "effective_from": "2026-09-01", "source": "controlled://corrections/family-015/device_limit/0"}
    edit.update(changes)
    return correction_text(edit)


class ScopeInputTests(unittest.TestCase):
    def neutral(self, question, correction=None):
        return scope_pair(question, correction or descriptor(), "entity_neutral_v1")

    def test_raw_default_preserves_every_input_byte_and_legacy_shape(self):
        question, correction = "  Nimbus\tBasic v2?\n", "old legacy descriptor without aliases"
        self.assertEqual(scope_pair(question, correction), (question, correction))
        self.assertEqual(scope_pair(question, correction, "raw_v1"), (question, correction))
        self.assertEqual(DEFAULT_SCOPE_INPUT_POLICY, "raw_v1")

    def test_nimbus_direct_and_alias_paraphrase_use_the_same_reference_selectors(self):
        question, correction = self.neutral("How many devices does Nimbus Basic version v2 support?")
        self.assertEqual(question, "How many devices does this product this plan this version support?")
        self.assertEqual(correction, "Correction for this product, this plan, this version: device limit changed from 5 to 3 devices.")
        question, _ = self.neutral("What is the device allowance on NC Basic version v2?")
        self.assertEqual(question, "What is the device allowance on this product this plan this version?")

    def test_longest_overlapping_alias_is_replaced_as_one_reference(self):
        question, _ = self.neutral("How many devices does Nimbus Cloud Basic v2 support?")
        self.assertEqual(question, "How many devices does this product this plan this version support?")
        self.assertNotIn("Cloud", question)

    def test_unfamiliar_product_names_produce_the_same_positive_pair(self):
        first = self.neutral("How many devices does Nimbus Basic version v2 support?")
        second = self.neutral("How many devices does Helios Connect Basic version v2 support?",
            descriptor(entity_name="Helios Connect", entity_aliases=["HC"]))
        self.assertEqual(first, second)

    def test_numeric_comparison_and_relation_words_remain_visible(self):
        question, correction = self.neutral("Can I connect 4 devices on Nimbus Basic version v2?")
        self.assertEqual(question, "Can I connect 4 devices on this product this plan this version?")
        self.assertIn("changed from 5 to 3 devices", correction)
        question, _ = self.neutral("What is the monthly price of Nimbus Basic version v2?")
        self.assertIn("monthly price", question)
        self.assertNotIn("devices", question)

    def test_wrong_product_tier_and_version_are_not_authorized_or_erased(self):
        question, _ = self.neutral("How many devices does Helios Basic version v2 support?")
        self.assertIn("Helios", question)
        self.assertNotIn("this product", question)
        question, _ = self.neutral("How many devices does Nimbus Pro version v2 support?")
        self.assertIn(" Pro ", question)
        self.assertNotIn("this plan", question)
        question, _ = self.neutral("How many devices does Nimbus Basic version v1 support?")
        self.assertIn("version v1", question)
        self.assertNotIn("this version", question)

    def test_wrong_entity_token_continuations_preserve_identity(self):
        for wrong in ("NimbusMini", "Nimbus-Plus", "Nimbus/Plus", "Nimbus+", "Nimbus.Enterprise", "PréNimbus"):
            question, _ = self.neutral(f"How many devices does {wrong} Basic v2 support?")
            with self.subTest(wrong=wrong):
                self.assertIn(wrong, question)
                self.assertNotIn("this product", question)

    def test_numbered_alias_plus_is_not_confused_with_core_alias(self):
        correction = descriptor(entity_name="Aster 015", entity_aliases=["A015"])
        question, _ = self.neutral("Can I connect 4 devices on A015+ Basic version v2?", correction)
        self.assertIn("A015+", question)
        self.assertNotIn("this product", question)
        question, _ = self.neutral("Can I connect 4 devices on A015 Basic version v2?", correction)
        self.assertIn("this product", question)

    def test_wrong_version_continuations_remain_visible(self):
        for version in ("v20", "v2.1", "v2-beta", "v2/legacy", "v2+", "v2_legacy"):
            question, _ = self.neutral(f"How many devices does Nimbus Basic version {version} support?")
            with self.subTest(version=version):
                self.assertIn(version, question)
                self.assertNotIn("this version", question)

    def test_wrong_plan_prefixes_are_not_masked(self):
        for plan in ("BasicPlus", "Basic-Pro", "Basic/Legacy"):
            question, _ = self.neutral(f"How many devices does Nimbus {plan} v2 support?")
            with self.subTest(plan=plan):
                self.assertIn(plan, question)
                self.assertNotIn("this plan", question)

    def test_case_punctuation_and_possessive_literal_boundaries(self):
        question, _ = self.neutral("What's NIMBUS Basic's device allowance on V2?")
        self.assertEqual(question, "What's this product this plan's device allowance on this version?")
        question, _ = self.neutral("Ask Nimbus.")
        self.assertEqual(question, "Ask this product.")

    def test_source_date_and_opaque_family_ids_do_not_enter_neutral_descriptor(self):
        question = "Can I connect 4 devices on Nimbus Basic version v2?"
        first = self.neutral(question)
        second = self.neutral(question, descriptor(source="https://fictional.example/handbook/opaque-77a9?case=39",
                                                  effective_from="2027-03-01"))
        self.assertEqual(first, second)
        for token in ("Source", "controlled", "family-015", "2026", "https", "2027"):
            self.assertNotIn(token, first[1])

    def test_all_declared_relations_and_units_keep_property_and_values(self):
        examples = (("price_monthly", "USD/month", "12", "9", "price monthly"),
                    ("storage_gb", "GB", "50", "75", "storage gb"),
                    ("support_hours", "hours/day", "8", "12", "support hours"))
        for relation, unit, old, new, rendered in examples:
            with self.subTest(relation=relation):
                _, correction = self.neutral("Nimbus Basic v2?", descriptor(relation=relation, unit=unit,
                    original_value=old, corrected_value=new))
                self.assertIn(rendered, correction)
                self.assertIn(f"{old} to {new} {unit}", correction)

    def test_unknown_policy_bad_descriptor_and_ambiguous_aliases_are_rejected(self):
        for policy in ("unknown", None, True, "RAW_V1"):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                scope_pair("Question", descriptor(), policy)
        cases = ("Correction for Nimbus: device_limit now 3.", descriptor(effective_from="2026-02-30"),
                 descriptor(entity_aliases=["Basic"]), descriptor(entity_aliases=["NC,Extra"]),
                 descriptor(relation="unknown_property"), descriptor(unit="GB"))
        for correction in cases:
            with self.subTest(correction=correction), self.assertRaises(ValueError):
                self.neutral("Nimbus Basic v2?", correction)
        with self.assertRaises(ValueError):
            self.neutral(" ")

    def test_policy_identity_is_a_hash_of_exact_preprocessing_source(self):
        identity = scope_policy_identity("entity_neutral_v1")
        import editguard.scope_inputs as module
        self.assertEqual(identity["scope_input_source_sha256"], file_sha256(Path(module.__file__)))
        self.assertEqual(identity["scope_input_policy_version"], SCOPE_INPUT_POLICY_VERSION)
        self.assertEqual(identity["scope_input_policy"], "entity_neutral_v1")
        self.assertEqual(set(SCOPE_INPUT_POLICIES), {"raw_v1", "entity_neutral_v1"})

    def test_shared_metadata_validator_accepts_legacy_and_refuses_partial_answer_or_stale_identity(self):
        self.assertEqual(validate_scope_policy({"task": "scope"}), "raw_v1")
        self.assertEqual(validate_scope_policy({"component": "answer"}), "raw_v1")
        identity = scope_policy_identity("entity_neutral_v1")
        self.assertEqual(validate_scope_policy({"task": "scope", **identity}), "entity_neutral_v1")
        for record in ({"scope_input_policy": "entity_neutral_v1"}, {"scope_input_policy_version": "unknown"},
                       {"task": "scope", **identity, "scope_input_source_sha256": "f" * 64},
                       {"component": "answer", **scope_policy_identity("raw_v1")}, None):
            with self.subTest(record=record), self.assertRaises(ValueError):
                validate_scope_policy(record)


class ScopeRuntimePolicyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name, contents in (("model/config.json", "{}"), ("model/model.safetensors", "synthetic fixture, never loaded"),
                              ("tokenizer/tokenizer_config.json", "{}")):
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text(contents, encoding="utf-8")
        self.manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed", "task": "scope",
            "input_format_version": INPUT_FORMAT_VERSION, "input_length": 192,
            "files_sha256": {path.relative_to(self.root).as_posix(): file_sha256(path)
                             for path in self.root.rglob("*") if path.is_file()}}
        self.save()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_legacy_artifact_has_no_new_mandatory_metadata(self):
        self.assertEqual(validate_artifact(self.root, "scope"), self.manifest)

    def test_opt_in_artifact_requires_exact_policy_version_and_source(self):
        self.manifest.update(scope_policy_identity("entity_neutral_v1"))
        self.save()
        self.assertEqual(validate_artifact(self.root, "scope"), self.manifest)
        original = dict(self.manifest)
        for changes in ({"scope_input_source_sha256": "f" * 64}, {"scope_input_policy_version": "unknown"},
                        {"scope_input_policy": "unknown"}):
            self.manifest = {**original, **changes}
            self.save()
            with self.assertRaises(ModelUnavailable):
                validate_artifact(self.root, "scope")
        self.manifest = dict(original)
        self.manifest.pop("scope_input_source_sha256")
        self.save()
        with self.assertRaises(ModelUnavailable):
            validate_artifact(self.root, "scope")

    def test_invalid_scope_policy_is_rejected_before_ml_imports(self):
        self.manifest["scope_input_policy"] = "unknown"
        self.save()
        command = "import sys; from editguard.neural import ScopeCheckpoint, ModelUnavailable; " \
            "\ntry: ScopeCheckpoint(sys.argv[1])\nexcept ModelUnavailable: pass\nelse: raise AssertionError('Expected policy refusal')\n" \
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", command, str(self.root)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_normalizes_exactly_once_and_legacy_raw_pair_stays_identical(self):
        class Probability:
            def cpu(self):
                return .75
        class Logits:
            def softmax(self, dim):
                return self
            def __getitem__(self, index):
                self.index = index
                return Probability()
        class Output:
            logits = Logits()
        class Torch:
            @staticmethod
            def inference_mode():
                return nullcontext()
        calls = []
        # Match the real tokenizer return object's .to without any Torch dependency.
        class Batch(dict):
            def to(self, device):
                return self
        def tokenize(question, correction, **kwargs):
            calls.append((question, correction))
            return Batch(encoded=True)
        model = ScopeCheckpoint.__new__(ScopeCheckpoint)
        model.scope_input_policy = "entity_neutral_v1"
        model.torch, model.device, model.input_length = Torch(), "cpu", 192
        model.tokenizer, model.model, model._lock = tokenize, lambda **kwargs: Output(), RLock()
        question, correction = "How many devices does Nimbus Basic version v2 support?", descriptor()
        with patch("editguard.neural.scope_pair", wraps=scope_pair) as normalization:
            self.assertEqual(model.score(question, correction), .75)
            self.assertEqual(normalization.call_count, 1)
        self.assertEqual(calls[-1], scope_pair(question, correction, "entity_neutral_v1"))
        model.scope_input_policy = "raw_v1"
        model.score(question, correction)
        self.assertEqual(calls[-1], (question, correction))

    def test_scope_policy_does_not_become_an_answer_artifact_requirement(self):
        self.manifest["task"] = "answer"
        self.save()
        self.assertEqual(validate_artifact(self.root, "answer"), self.manifest)


if __name__ == "__main__":
    unittest.main()

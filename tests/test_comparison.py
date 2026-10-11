import copy
import json
import tempfile
import unittest
from pathlib import Path

from editguard.comparison import ComparisonRunner, active_memory, resolve_context, updated_snapshot
from editguard.formatting import INPUT_FORMAT_VERSION, conditional_input, correction_text
from editguard.neural import ModelUnavailable, calibrated_probability, file_sha256, validate_artifact
from scripts.run_comparison import oracle_answer_result, scenario_memory, score_record, smoke_scenarios


class FakeGenerator:
    def __init__(self):
        self.inputs = []
        self.last_trace = {"all_parameters_frozen": True}

    def generate(self, text):
        self.inputs.append(text)
        return "Synthetic generator fixture; not an ML result."


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def rank(self, question, records, text_fn, k=3):
        self.calls.append([text_fn(record) for record in records])
        # Deliberately simple fixture; not a real embedding baseline.
        return [{"record": record, "score": .9 - index * .01} for index, record in enumerate(records[:k])]


class FakeScope:
    def __init__(self, probability):
        self.probability = probability

    def score(self, question, correction):
        return self.probability


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.facts, self.rows = smoke_scenarios()
        self.row = self.rows[0]
        self.edit = self.row["correction"]
        self.base, self.retriever = FakeGenerator(), FakeRetriever()
        self.runner = ComparisonRunner(self.base, self.retriever)

    def ask(self, configuration, **kwargs):
        return self.runner.answer(self.row["question"], self.row["request_context"], self.facts,
                                  [self.edit], configuration=configuration, **kwargs)

    def test_updated_rag_retrieves_corrected_snapshot_base_uses_initial(self):
        unchanged = self.ask("unchanged_base")
        updated = self.ask("updated_rag")
        self.assertIn("supports 5 devices", unchanged["input_text"])
        self.assertNotIn("to 3 devices", unchanged["input_text"])
        self.assertIn("supports 3 devices", updated["input_text"])
        self.assertNotIn("supports 5 devices", updated["input_text"])
        self.assertEqual(self.facts[0]["value"], "5")
        self.assertFalse(unchanged["retrieval_candidates_raw"])
        self.assertEqual(updated["retrieved_facts"][0]["applied_edit_id"], self.edit["edit_id"])

    def test_prompt_baseline_receives_both_initial_facts_and_same_approved_edit(self):
        result = self.ask("prompt_only")
        self.assertIn("supports 5 devices", result["input_text"])
        self.assertIn("changed from 5 to 3", result["input_text"])
        self.assertEqual(result["answer_kind"], "model_generated")
        self.assertEqual(result["answer"], self.base.generate("fixture"))

    def test_missing_auxiliaries_are_explicit_unavailable(self):
        for configuration in ("similarity_editor", "trained_scope_editor"):
            result = self.ask(configuration)
            self.assertEqual(result["status"], "unavailable")
        self.assertFalse(self.base.inputs)

    def test_unavailable_metadata_keeps_matched_aggregation_conditions(self):
        from editguard.metrics import aggregate_predictions
        records = [score_record(self.ask(configuration), self.row, self.facts, self.runner)
                   for configuration in ("unchanged_base", "updated_rag", "trained_scope_editor")]
        self.assertTrue(all(record["request_context"] == self.row["request_context"] for record in records))
        metrics = aggregate_predictions(records)
        self.assertEqual(metrics["configurations"]["trained_scope_editor"]["counts"]["unavailable"], 1)

    def test_retrieval_and_scope_failures_are_countable_error_records(self):
        def broken(*args, **kwargs):
            raise RuntimeError("fixture inference failure")
        self.retriever.rank = broken
        self.assertEqual(self.ask("updated_rag")["status"], "error")
        self.assertFalse(self.base.inputs)

    def test_trained_scope_routes_to_answer_model_and_retains_support(self):
        answer = FakeGenerator()
        self.runner = ComparisonRunner(self.base, self.retriever, scope=FakeScope(.95), answer_model=answer,
                                       calibration={"threshold": .8, "ambiguity_margin": .03, "temperature": 1})
        result = self.ask("trained_scope_editor")
        self.assertEqual(result["selected_edit_id"], self.edit["edit_id"])
        self.assertEqual(len(answer.inputs), 1)
        self.assertFalse(self.base.inputs)
        self.assertIn("changed from 5 to 3", result["input_text"])
        self.assertNotIn("supports 5 devices", result["input_text"])
        self.assertEqual(result["input_text"], conditional_input(self.row["question"], self.edit,
                                                               result["supporting_context"], "2026-10-11"))

    def test_uncertainty_abstains_and_low_score_falls_back_to_base(self):
        self.runner = ComparisonRunner(self.base, self.retriever, scope=FakeScope(.81), answer_model=FakeGenerator(),
                                       calibration={"threshold": .8, "ambiguity_margin": .03, "temperature": 1})
        self.assertEqual(self.ask("trained_scope_editor")["status"], "clarification")
        self.runner.scope.probability = .2
        self.assertEqual(self.ask("trained_scope_editor")["route"], "frozen_base")

    def test_version_tier_entity_and_date_gates_preserve_scopes(self):
        for row in self.rows[4:]:
            result = self.runner.answer(row["question"], row["request_context"], self.facts, [self.edit], configuration="prompt_only")
            self.assertNotIn("Approved corrections", result["input_text"])
        old_context = {**self.row["request_context"], "as_of": "2026-09-01"}
        self.assertEqual(updated_snapshot(self.facts, [self.edit], "2026-09-01")[0]["value"], "5")
        mini = resolve_context("Nimbus Mini Basic v2 devices", {}, self.facts)
        self.assertEqual(mini["entity_id"], "nimbus-mini")
        with self.assertRaises(ValueError):
            resolve_context("Nimbus Basic v1 devices", self.row["request_context"], self.facts)
        with self.assertRaises(ValueError):
            resolve_context("What did Nimbus Basic previously support?", {}, self.facts)
        for version in ("v2.1", "v2-beta", "version 2.1"):
            with self.assertRaises(ValueError):
                resolve_context(f"Nimbus Basic {version} devices", {}, self.facts)
        for question in ("How many devices does Nimbus Enterprise version v2 support?",
                         "How many devices does UnknownCo Basic version v2 support?",
                         "How many devices with Enterprise plan on Nimbus v2?"):
            with self.assertRaises(ValueError):
                resolve_context(question, self.row["request_context"], self.facts)
        self.assertEqual(resolve_context("How many devices does Basic support?",
                                         {"entity_id": "nimbus"}, self.facts)["plan"], "Basic")

    def test_unresolved_conflicts_and_withdrawal(self):
        conflicting = {**self.edit, "edit_id": "other", "corrected_value": "4"}
        result = self.runner.answer(self.row["question"], self.row["request_context"], self.facts,
                                    [self.edit, conflicting], configuration="updated_rag")
        self.assertEqual(result["status"], "conflict")
        self.assertFalse(self.base.inputs)
        withdrawn = {**self.edit, "status": "withdrawn"}
        self.assertEqual(updated_snapshot(self.facts, [withdrawn], "2026-10-11")[0]["value"], "5")

    def test_date_scoped_supersession_restores_predecessor(self):
        successor = {**self.edit, "edit_id": "successor", "supersedes": self.edit["edit_id"],
                     "effective_from": "2026-10-10", "effective_to": "2026-10-20", "corrected_value": "2"}
        self.assertEqual(active_memory([self.edit, successor], "2026-10-11")[0]["edit_id"], "successor")
        self.assertEqual(active_memory([self.edit, successor], "2026-10-20")[0]["edit_id"], self.edit["edit_id"])

    def test_scenario_builder_never_reads_expected_fact_or_label(self):
        expected = scenario_memory(self.row, self.rows)
        altered = copy.deepcopy(self.row)
        altered["expected_fact"] = {"value": "99999"}
        altered["label"] = 0
        self.assertEqual(scenario_memory(altered, self.rows), expected)

    def test_oracle_diagnostic_is_separate_and_has_no_target_injection(self):
        self.assertEqual(oracle_answer_result(self.runner, self.row, self.facts)["status"], "unavailable")
        self.runner.answer_model = FakeGenerator()
        original = oracle_answer_result(self.runner, self.row, self.facts)
        altered = copy.deepcopy(self.row)
        altered["expected_fact"]["value"] = "99999"
        self.assertEqual(original["input_text"], oracle_answer_result(self.runner, altered, self.facts)["input_text"])
        self.assertEqual(original["configuration"], "oracle_answer")
        with self.assertRaises(ValueError):
            oracle_answer_result(self.runner, self.rows[3], self.facts)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name, content in [("model/config.json", "{}"), ("model/model.safetensors", "fixture-not-real-weights"),
                              ("tokenizer/tokenizer_config.json", "{}")]:
            file = self.root / name
            file.parent.mkdir(exist_ok=True)
            file.write_text(content, encoding="utf-8")
        self.manifest = {"artifact_type": "editguard_auxiliary", "purpose": "trained", "status": "completed",
                         "task": "scope", "input_format_version": INPUT_FORMAT_VERSION, "input_length": 192,
                         "files_sha256": {str(file.relative_to(self.root)).replace("\\", "/"): file_sha256(file)
                                          for file in self.root.rglob("*") if file.is_file()}}
        self.save()

    def save(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_smoke_wrong_format_tampered_and_unhashed_weights_rejected(self):
        validate_artifact(self.root, "scope")
        self.manifest["purpose"] = "smoke_test"
        self.save()
        with self.assertRaises(ModelUnavailable):
            validate_artifact(self.root, "scope")
        self.manifest["purpose"] = "trained"
        self.manifest["input_format_version"] = "wrong"
        self.save()
        with self.assertRaises(ModelUnavailable):
            validate_artifact(self.root, "scope")
        self.manifest["input_format_version"] = INPUT_FORMAT_VERSION
        self.save()
        (self.root / "model/unhashed.safetensors").write_bytes(b"unhashed")
        with self.assertRaises(ModelUnavailable):
            validate_artifact(self.root, "scope")
        (self.root / "model/unhashed.safetensors").unlink()
        (self.root / "model/model.safetensors").write_bytes(b"tampered")
        with self.assertRaises(ModelUnavailable):
            validate_artifact(self.root, "scope")

    def test_temperature_preserves_half_and_rejects_invalid(self):
        self.assertEqual(calibrated_probability(.5, 2), .5)
        self.assertAlmostEqual(calibrated_probability(.8, 1), .8)
        for value in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                calibrated_probability(.8, value)
        for probability in (float("nan"), -1, 1.1):
            with self.assertRaises(ValueError):
                calibrated_probability(probability, 1)


if __name__ == "__main__":
    unittest.main()

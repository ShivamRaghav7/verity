"""Synthetic receipt contract tests; these fixtures are never model evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from editguard_benchmark.scoring import score
from scripts import audit_india_http as audit
from scripts import verify_india_http as workload


def synthetic_receipt():
    """Construct metadata-shaped unit-test input without calling any model."""
    plan = workload.case_plan()
    cases = copy.deepcopy(plan)
    for row in cases:
        content = "SYNTHETIC UNIT TEST RESPONSE"
        evidence = {"generation_kind": "model", "context_policy": "iterative",
            "configured_context_policy": "iterative", "knowledge_revision": row["expected_revision"],
            "raw_model_output": "<answer>" + content + "</answer>",
            "generation": {"model": workload.MODEL_ID, "revision": workload.MODEL_REVISION}}
        if row["id"] == "historical_office_comparison":
            evidence.update(historical_snapshot=True, snapshot_revision=9)
        response = {"content": content, "evidence": evidence}
        row.update(response=response, revision=row["expected_revision"], request_state="returned",
            neural_provenance_accepted=True, **score(row["question"], response))
    initial = []
    for index, fact in enumerate(workload.india_catalogue()["world"]["initial_facts"]):
        initial.append({"id": "unit-test-" + str(index), "status": "active", "answer": fact["answer"],
            "question": fact["question"], "source": fact.get("source"), "scope": fact.get("scope"),
            "identity": {key: fact[key] for key in ("entity_id", "relation_id", "scope_id")}})
    final = copy.deepcopy(initial) + [{"id": "unit-test-replacement", "status": "withdrawn"}]
    requests = [{"case_ids": [row["id"]], "path": "/api/chat", "state": "returned"} for row in plan[:9]]
    requests += [{"case_ids": [row["id"] for row in plan[9:11]], "path": "/api/compare", "state": "returned"}]
    requests += [{"case_ids": [row["id"]], "path": "/api/chat", "state": "returned"} for row in plan[11:]]
    return {"schema": "editguard-india-http-v1", "status": "completed", "cases": cases,
        "assigned_model_generations": 14, "neural_accuracy_denominator": 14,
        "independent_display_lineages": 1, "held_out_benchmark_generations": 0,
        "model_id": workload.MODEL_ID, "model_revision": workload.MODEL_REVISION,
        "context_policy": "iterative", "script_sha256": audit.digest(workload.__file__),
        "plan_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest(),
        "authorization_headers_recorded": False, "generation_requests": requests,
        "outcome_counts": {key: 14 if key == "wrong" else 0 for key in workload.OUTCOMES},
        "verified_neural_response_count": 14, "consistency_checks": dict.fromkeys(audit.CHECKS, True),
        "all_consistency_checks_passed": True, "initial_handbook_restored": True,
        "operations": [
            {"name": "reject_unresolved_owner_conflict", "state": "error", "error": {"http_status": 409}},
            {"name": "inspect_nine_approved_facts", "response": {"revision": 9, "corrections": initial}},
            {"name": "inspect_restored_nine_facts", "response": {"revision": 11, "corrections": final,
                "events": [{"action": "withdrawn", "revision": 11}, {"action": "restored", "revision": 11}]}}]}


class IndiaAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic-india-audit-fixture-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.receipt = self.directory / "synthetic-unit-test.json"
        self.value = synthetic_receipt()

    def write(self):
        self.receipt.write_text(json.dumps(self.value), encoding="utf-8")

    def reject(self, pattern):
        self.write()
        with self.assertRaisesRegex(ValueError, pattern):
            audit.audit(self.receipt)

    def test_mechanical_replay_keeps_wrong_denominator_and_separates_state_checks(self):
        self.write()
        before = self.receipt.read_bytes()
        result = audit.audit(self.receipt)
        self.assertEqual(result["assigned_model_generations"], 14)
        self.assertEqual(result["outcome_counts"]["wrong"], 14)
        self.assertEqual(result["primary_accuracy"], 0)
        self.assertTrue(result["transaction_checks_all_passed"])
        self.assertEqual(result["model_calls"], 0)
        self.assertFalse(result["human_reviewed"])
        self.assertEqual(self.receipt.read_bytes(), before)

    def test_abstentions_are_replayed_and_never_dropped(self):
        for row in self.value["cases"]:
            row["response"]["content"] = "UNKNOWN"
            row["response"]["evidence"]["raw_model_output"] = "UNKNOWN"
            row.update(score(row["question"], row["response"]))
        self.value["outcome_counts"] = dict.fromkeys(workload.OUTCOMES, 0)
        self.value["outcome_counts"]["abstain"] = 14
        self.write()
        result = audit.audit(self.receipt)
        self.assertEqual(result["outcome_counts"]["abstain"], 14)
        self.assertEqual(result["primary_accuracy"], 0)

    def test_changed_labels_script_plan_and_denominators_rejected(self):
        original = copy.deepcopy(self.value)
        for mutate in (
            lambda r: r["cases"][0]["question"].update(expected_answer="changed"),
            lambda r: r.update(script_sha256="wrong"),
            lambda r: r.update(plan_sha256="wrong"),
            lambda r: r.update(neural_accuracy_denominator=13),
            lambda r: r["cases"].reverse()):
            self.value = copy.deepcopy(original)
            mutate(self.value)
            self.reject("differs|differ")

    def test_raw_final_extraction_and_fixed_score_must_match(self):
        self.value["cases"][0]["response"]["evidence"]["raw_model_output"] = "<answer>Different</answer>"
        self.reject("Raw extraction")
        self.value = synthetic_receipt()
        self.value["cases"][0]["outcome"] = "correct"
        self.reject("Recorded score")

    def test_fixture_deterministic_wrong_model_and_wrong_memory_rejected(self):
        for key in ("fixture", "deterministic", "software_fixture"):
            self.value = synthetic_receipt()
            self.value["cases"][0]["response"]["evidence"]["generation"][key] = True
            self.reject("fixture or deterministic")
        self.value = synthetic_receipt()
        self.value["cases"][0]["response"]["evidence"]["generation"]["revision"] = "changed"
        self.reject("provenance")
        self.value = synthetic_receipt()
        self.value["cases"][9]["response"]["evidence"]["snapshot_revision"] = 10
        self.reject("historical")

    def test_duplicate_request_failed_conflict_and_changed_rollback_rejected(self):
        self.value["generation_requests"].append(self.value["generation_requests"][0])
        self.reject("request accounting")
        self.value = synthetic_receipt()
        self.value["operations"][0]["error"]["http_status"] = 500
        self.reject("conflict rejection")
        self.value = synthetic_receipt()
        self.value["operations"][-1]["response"]["corrections"][0]["answer"] = "changed"
        self.reject("restored handbook")

    def test_error_and_unexecuted_case_cannot_be_promoted(self):
        self.value["status"] = "running"
        self.reject("Completed India")
        self.value = synthetic_receipt()
        self.value["cases"][0]["outcome"] = "unexecuted"
        self.reject("Recorded score")

    def test_existing_output_is_preserved_before_input_audit(self):
        output = self.directory / "synthetic-audit.json"
        output.write_text("preserve")
        with patch.object(audit, "audit") as replay:
            with self.assertRaisesRegex(ValueError, "Existing audit"):
                audit.publish(self.receipt, output)
            replay.assert_not_called()
        self.assertEqual(output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()

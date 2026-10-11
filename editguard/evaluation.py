"""Isolated workflow regression, deliberately separate from ML evaluation."""

import math
from contextlib import closing

from .models import AskRequest, CorrectionCreate
from .service import StructuredAssistant
from .store import DomainError, Store


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def run_slice_regression():
    cases, latencies = [], []

    def check(name, actual, expected):
        cases.append({"name": name, "passed": actual == expected,
                      "expected": expected, "actual": actual})

    with closing(Store(":memory:")) as store:
        store.seed()
        assistant = StructuredAssistant(store)

        def ask(question, **scope):
            result = assistant.ask(AskRequest(question=question, entity_id="nimbus",
                                             as_of="2026-10-11", **scope))
            latencies.append(result["latency_ms"])
            return result

        direct = "How many devices does Basic support?"
        check("Initial catalogue value", ask(direct)["answer"], "Nimbus Basic v2 supports up to 5 devices.")
        correction = CorrectionCreate(fact_id="nimbus-basic-v2-devices", original_value="5",
                                      corrected_value="3", source="Fictional release notice EG-001",
                                      reason="Controlled demonstration", effective_from="2026-10-01")
        edit = store.create_correction(correction)
        check("Draft is not authoritative", ask(direct)["answer"], "Nimbus Basic v2 supports up to 5 devices.")
        store.approve(edit["edit_id"])
        check("Approved correction applies", ask(direct)["answer"], "Nimbus Basic v2 supports up to 3 devices.")
        check("Supported paraphrase applies", ask("What is Basic's device limit?")["answer"], "Nimbus Basic v2 supports up to 3 devices.")
        check("Capacity question endorsed correctly", ask("Can I connect four devices on Basic?")["answer"], "No. Nimbus Basic v2 supports up to 3 devices.")
        check("Different property retained", ask("What is Basic's monthly price?")["answer"], "Nimbus Basic v2 costs 9 USD/month.")
        check("Different tier retained", ask("How many devices does Pro support?")["answer"], "Nimbus Pro v2 supports up to 12 devices.")
        check("Historical version retained", ask("How many devices does Basic v1 support?")["answer"], "Nimbus Basic v1 supports up to 5 devices.")
        check("Ambiguous tier clarifies", ask("How many devices are allowed?")["status"], "clarification")
        competing = store.create_correction(correction.model_copy(update={"original_value": "3", "corrected_value": "4"}))
        try:
            store.approve(competing["edit_id"])
            conflict_rejected = False
        except DomainError as exc:
            conflict_rejected = exc.code == 409
        check("Conflicting approval rejected", conflict_rejected, True)
        check("Conflict is explicit", ask(direct)["status"], "conflict")
        store.withdraw(competing["edit_id"])
        check("Conflict resolution restores approved answer", ask(direct)["answer"], "Nimbus Basic v2 supports up to 3 devices.")
        store.withdraw(edit["edit_id"])
        check("Rollback restores initial answer", ask(direct)["answer"], "Nimbus Basic v2 supports up to 5 devices.")
        check("Answers explicitly structured", ask(direct)["answer_kind"], "deterministic_structured")

    passed = sum(case["passed"] for case in cases)
    return {"status": "passed" if passed == len(cases) else "failed",
            "disclaimer": "Isolated deterministic workflow regression. These are engineering checks, not learned-model accuracy or end-to-end HTTP latency measurements.",
            "metrics": [{"name": "Workflow checks passed", "value": passed, "denominator": len(cases), "unit": "checks"},
                        {"name": "Fixture service latency p50", "value": round(percentile(latencies, .5), 3), "unit": "ms"},
                        {"name": "Fixture service latency p95", "value": round(percentile(latencies, .95), 3), "unit": "ms"}],
            "cases": cases, "sample_count": len(latencies), "ml_results": None}

"""Check latency denominators and mixed-state detection without networking."""

import unittest

from scripts.benchmark_http import percentile, summarise


def record(latency, status="answered", revision=1, kind="deterministic_structured", error=None):
    return {"latency_ms": latency, "http_status": 200 if error is None else None,
            "response_status": status, "revision": revision, "answer_kind": kind, "error": error}


class BenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates_small_samples_and_handles_empty(self):
        self.assertEqual(percentile([20, 0, 10], .5), 10)
        self.assertEqual(percentile([20, 0, 10], .95), 19)
        self.assertEqual(percentile([7], .95), 7)
        self.assertIsNone(percentile([], .95))

    def test_errors_and_abstentions_remain_in_declared_denominators(self):
        report = summarise([record(10), record(20, "clarification"), record(100, error="timeout")])
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["counts"]["attempted"], 3)
        self.assertEqual(report["counts"]["answered"], 1)
        self.assertEqual(report["counts"]["abstentions"], 1)
        self.assertEqual(report["counts"]["errors"], 1)
        self.assertEqual(report["metrics"]["http_roundtrip_p95_ms"], 92)
        self.assertEqual(report["metrics"]["valid_response_roundtrip_p95_ms"], 19.5)
        self.assertEqual(report["metrics"]["latency_denominator"], 3)
        self.assertEqual(report["metrics"]["valid_latency_denominator"], 2)
        self.assertEqual(report["latencies_ms"], [10, 20, 100])
        self.assertIsNone(report["ml_accuracy"])

    def test_revision_and_answer_kind_changes_make_mixed_result(self):
        changed_revision = summarise([record(10), record(20, revision=2)])
        changed_kind = summarise([record(10, kind="model_generated")])
        for report in [changed_revision, changed_kind]:
            self.assertEqual(report["status"], "mixed_state")
            self.assertTrue(report["mixed_state"])
        self.assertIsNone(changed_revision["stable_revision"])

    def test_warmup_detects_state_changes_but_does_not_enter_percentiles(self):
        report = summarise([record(10, revision=2)], [record(1000, revision=1)])
        self.assertTrue(report["mixed_state"])
        self.assertEqual(report["metrics"]["http_roundtrip_p95_ms"], 10)
        self.assertEqual(report["counts"]["attempted"], 1)


if __name__ == "__main__":
    unittest.main()

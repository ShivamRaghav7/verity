import tempfile
from pathlib import Path
import unittest

from scripts.resume_final_io import retry_writer


class FinalIORecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / "progress.json"

    def test_transient_collision_retries_identical_payload_only(self):
        payload, calls, sleeps, events = {"cursor": 583}, [], [], []
        def writer(path, value):
            calls.append((path, value))
            if len(calls) < 3:
                raise PermissionError("transient reader")
            return "committed"
        actual = retry_writer(writer, [self.target], sleep=sleeps.append, events=events)(self.target, payload)
        self.assertEqual(actual, "committed")
        self.assertEqual(sleeps, [.05, .1])
        self.assertTrue(all(value is payload for _, value in calls))
        self.assertEqual(len(events), 2)

    def test_permanent_failure_has_finite_eight_attempt_cap(self):
        calls, sleeps = [], []
        def writer(path, value):
            calls.append(value)
            raise PermissionError("persistent access denied")
        with self.assertRaises(PermissionError):
            retry_writer(writer, [self.target], sleep=sleeps.append)(self.target, {})
        self.assertEqual(len(calls), 8)
        self.assertAlmostEqual(sum(sleeps), 1.95)

    def test_immutable_observations_are_never_retried_or_rewritten(self):
        calls, sleeps = [], []
        def writer(path, value):
            calls.append(path)
            raise PermissionError("observation failure")
        with self.assertRaises(PermissionError):
            retry_writer(writer, [self.target], sleep=sleeps.append)(self.target.parent / "00000583.json", {})
        self.assertEqual(len(calls), 1)
        self.assertEqual(sleeps, [])

    def test_model_errors_and_other_io_errors_are_not_retried(self):
        sleeps = []
        def writer(path, value):
            raise ValueError("invalid score payload")
        with self.assertRaises(ValueError):
            retry_writer(writer, [self.target], sleep=sleeps.append)(self.target, {})
        self.assertEqual(sleeps, [])


if __name__ == "__main__":
    unittest.main()

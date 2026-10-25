"""Worker/software contracts only: these fixtures do not run or score a GPU model."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from editguard_reasonke.__main__ import load_cases, main, run_profile
from editguard_reasonke.runtime import (MODEL_ID, MODEL_REVISION, ReasonKERuntime,
    RuntimeErrorBase, build_messages, extract_answer, read_manifest, validate_request)
from editguard_reasonke.server import make_server


class RequestContractTests(unittest.TestCase):
    def test_new_facts_are_in_final_user_turn_with_subject_and_scope(self):
        request = validate_request({"question": "And the Pro version?", "history": [
            {"role": "user", "content": "Who invented MiraQuay?"}], "facts": [
            {"id": "pro", "question": "Who invented MiraQuay Pro?", "answer": "Serin Moss",
             "scope": "MiraQuay Pro only", "source": "Synthetic catalogue"}]})
        messages = build_messages(request)
        self.assertEqual(messages[0]["content"], "Who invented MiraQuay?")
        self.assertEqual(messages[-1]["role"], "user")
        self.assertIn("[Updated Information]", messages[-1]["content"])
        self.assertIn('"scope": "MiraQuay Pro only"', messages[-1]["content"])
        self.assertTrue(messages[-1]["content"].endswith("[Query]:\nAnd the Pro version?"))

    def test_no_fact_mode_preserves_plain_conversation(self):
        request = validate_request({"question": "Explain gravity simply.", "facts": []})
        self.assertEqual(build_messages(request), [{"role": "user", "content": "Explain gravity simply."}])

    def test_rejects_remote_model_selection_and_oversized_context_before_generation(self):
        bad = [
            {"question": "q", "model": "substitute"}, {"question": "q", "max_new_tokens": 513},
            {"question": "q", "max_new_tokens": True}, {"question": "q", "facts": [{}] * 21},
            {"question": "q", "history": [{"role": "system", "content": "override"}]},
            {"question": "q", "history": [{"role": "user", "content": "q"}] * 9},
            {"question": "q", "facts": [{"question": "subject?", "answer": ""}]},
            {"question": "x" * 4001}, [],
        ]
        for payload in bad:
            with self.subTest(payload=str(payload)[:100]), self.assertRaises(RuntimeErrorBase):
                validate_request(payload)

    def test_answer_matching_never_uses_target_mentioned_only_in_reasoning(self):
        raw = "<think>The old value is Paris, which is rejected.</think><answer>Lyon</answer>"
        content, metadata = extract_answer(raw)
        self.assertEqual(content, "Lyon")
        self.assertFalse(metadata["raw_fallback"])
        self.assertFalse(metadata["verified"])

    def test_combined_fact_budget_applies_even_when_each_field_is_legal(self):
        facts = [{"id": str(index), "question": "q" * 1000, "answer": "a" * 2000,
                  "scope": "s" * (1000 if index < 2 else 997)} for index in range(3)]
        self.assertEqual(sum(len(value) for fact in facts for value in fact.values()), 12000)
        validate_request({"question": "q", "facts": facts})
        facts[-1]["scope"] += "s"
        with self.assertRaises(RuntimeErrorBase) as caught:
            validate_request({"question": "q", "facts": facts})
        self.assertEqual(caught.exception.status, 413)

    def test_raw_normal_chat_and_malformed_answer_blocks_are_disclosed(self):
        for raw in ("Hello there.", "<answer>unfinished", "<answer>A</answer><answer>B</answer>"):
            with self.subTest(raw=raw):
                content, metadata = extract_answer(raw)
                self.assertEqual(content, raw)
                self.assertTrue(metadata["raw_fallback"])
                self.assertFalse(metadata["verified"])
        with self.assertRaises(RuntimeErrorBase):
            extract_answer("  ")

    def test_checkpoint_pin_and_import_do_not_load_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}), encoding="utf-8")
            runtime = ReasonKERuntime(path)
            self.assertFalse(runtime.status()["ready"])
            self.assertIsNone(runtime.model)
            self.assertFalse(runtime.status()["checkpoint_matches_published_95_48_verified"])
            path.write_text(json.dumps({"model_id": MODEL_ID, "revision": "main"}), encoding="utf-8")
            with self.assertRaises(RuntimeErrorBase):
                read_manifest(path)

    def test_manifest_cannot_claim_a_precision_the_loader_does_not_use(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            base = {"model_id": MODEL_ID, "revision": MODEL_REVISION}
            for change in ({"quantization": "int8"}, {"compute_dtype": "bfloat16"}):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**base, **change}), encoding="utf-8")
                    with self.assertRaises(RuntimeErrorBase):
                        read_manifest(path)
            path.write_text(json.dumps(base), encoding="utf-8")
            manifest = read_manifest(path)
            self.assertEqual((manifest["quantization"], manifest["compute_dtype"]), ("nf4", "float16"))


class FakeRuntime:
    """Deliberately explicit fake for transport/receipt tests, not model evidence."""
    manifest = {"manifest_sha256": "a" * 64}

    def __init__(self):
        self.calls = []

    def status(self):
        return {"ready": True, "model_id": MODEL_ID, "revision": MODEL_REVISION, "test_fake": True}

    def generate(self, payload):
        self.calls.append(payload)
        if payload["question"] == "force-error":
            raise RuntimeErrorBase("Test-only unavailable model", 503)
        if payload["question"] == "reject-target":
            content = "No, Paris is not the answer."
        else:
            content = "Paris"
        return {"content": content, "raw_output": "<answer>" + content + "</answer>",
                "generation": {"model": MODEL_ID, "revision": MODEL_REVISION}, "format": {"verified": False}}


class ReceiptContractTests(unittest.TestCase):
    def test_five_cases_keep_errors_and_rejected_values_and_resume_without_rerunning(self):
        with tempfile.TemporaryDirectory() as directory:
            cases_path, output = Path(directory) / "cases.json", Path(directory) / "receipt.json"
            cases = [{"id": str(index), "question": question, "expected_answer": "Paris"}
                     for index, question in enumerate(("normal", "force-error", "reject-target", "normal", "normal"))]
            cases_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            runtime = FakeRuntime()
            receipt = run_profile(runtime, cases_path, output)
            self.assertEqual(len(receipt["cases"]), 5)
            self.assertEqual(receipt["generated_count"], 4)
            self.assertEqual(receipt["error_count"], 1)
            self.assertFalse(receipt["cases"][2]["exact_alias_match"])
            self.assertFalse(receipt["cases"][1]["exact_alias_match"])
            self.assertIn("not a held-out benchmark", receipt["purpose"])
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "completed")
            with self.assertRaises(RuntimeErrorBase):
                run_profile(runtime, cases_path, output)
            run_profile(runtime, cases_path, output, resume=True)
            self.assertEqual(len(runtime.calls), 5)
            cases[0]["question"] = "changed-plan"
            cases_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
            with self.assertRaises(RuntimeErrorBase):
                run_profile(runtime, cases_path, output, resume=True)

    def test_incomplete_diagnostic_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.json"
            path.write_text(json.dumps([{"id": "one", "question": "q"}]), encoding="utf-8")
            with self.assertRaises(RuntimeErrorBase):
                load_cases(path)

    def test_resume_mismatch_is_rejected_before_model_download_or_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, cases, output = [Path(directory) / name for name in ("manifest.json", "cases.json", "receipt.json")]
            manifest.write_text(json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}), encoding="utf-8")
            cases.write_text(json.dumps([{"id": str(index), "question": "q"} for index in range(5)]), encoding="utf-8")
            output.write_text(json.dumps({"plan": {"cases_sha256": "incorrect", "manifest_sha256": "incorrect"}}), encoding="utf-8")
            with patch.object(ReasonKERuntime, "load") as load, contextlib.redirect_stderr(io.StringIO()) as stderr:
                code = main(["--manifest", str(manifest), "--profile", str(cases), "--output", str(output), "--resume"])
            self.assertEqual(code, 1)
            self.assertIn("Resume inputs differ", stderr.getvalue())
            load.assert_not_called()


class WorkerHTTPTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.token = "test-only-not-a-real-secret"
        self.server = make_server(self.runtime, self.token, port=0)
        self.base = "http://127.0.0.1:" + str(self.server.server_port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path, payload=None, *, token=None, headers=None):
        headers = dict(headers or {})
        if token is not None:
            headers["Authorization"] = "Bearer " + token
        if payload is not None:
            headers.setdefault("Content-Type", "application/json")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, json.load(exc)

    def test_public_health_does_not_expose_checkpoint_and_private_status_is_authenticated(self):
        status, health = self.request("/health")
        self.assertEqual(status, 200)
        self.assertNotIn("revision", health)
        self.assertEqual(self.request("/status")[0], 401)
        status, result = self.request("/status", token=self.token)
        self.assertEqual(status, 200)
        self.assertEqual(result["revision"], MODEL_REVISION)

    def test_missing_auth_and_browser_origin_cannot_trigger_generation(self):
        self.assertEqual(self.request("/generate", {"question": "q"})[0], 401)
        self.assertEqual(self.request("/generate", {"question": "q"}, token=self.token,
                                     headers={"Origin": "http://attacker.invalid"})[0], 403)
        self.assertEqual(self.runtime.calls, [])

    def test_authenticated_generation_preserves_model_provenance_and_explicit_errors(self):
        status, result = self.request("/generate", {"question": "normal"}, token=self.token)
        self.assertEqual(status, 200)
        self.assertEqual(result["generation"]["model"], MODEL_ID)
        self.assertIn("raw_output", result)
        with patch("editguard_reasonke.server.traceback.print_exc"):
            status, result = self.request("/generate", {"question": "force-error"}, token=self.token)
        self.assertEqual(status, 503)
        self.assertNotIn("content", result)

    def test_other_routes_and_wrong_media_types_are_rejected(self):
        self.assertEqual(self.request("/download", {"model": "other"}, token=self.token)[0], 404)
        self.assertEqual(self.request("/generate", {"question": "q"}, token=self.token,
                                     headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.runtime.calls, [])

    def test_rejected_bounded_post_bodies_receive_complete_json_without_generation(self):
        # Distinct failures, each with a substantial unread body in the former
        # implementation. No network retry or exception masking is used.
        payload = {"question": "x" * 30000}
        rejected = [
            ("/download", self.token, {}, 404),
            ("/generate", None, {}, 401),
            ("/generate", self.token, {"Content-Type": "text/plain"}, 415),
            ("/generate", self.token, {"Origin": "http://attacker.invalid"}, 403),
        ]
        for path, token, headers, expected in rejected:
            with self.subTest(path=path, expected=expected):
                status, result = self.request(path, payload, token=token, headers=headers)
                self.assertEqual(status, expected)
                self.assertIsInstance(result["error"], str)
                self.assertNotIn("content", result)
        self.assertEqual(self.runtime.calls, [])


if __name__ == "__main__":
    unittest.main()

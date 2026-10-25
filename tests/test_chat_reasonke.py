"""Reason-KE transport and website contracts; fakes are not accuracy evidence."""
import copy
from http.server import ThreadingHTTPServer
import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from editguard_chat.backend import ModelError
from editguard_chat.reasonke_backend import (
    ApprovedFactContext, MODEL_ID, MODEL_REVISION, ReasonKEClient,
)
from editguard_chat.server import ChatApplication, handler_for, validate_trusted_origin


class ReasonKEClientTests(unittest.TestCase):
    def setUp(self):
        self.client = ReasonKEClient(token="contract-test-token-only")

    def test_transport_requires_bearer_and_uses_explicit_worker_without_ollama(self):
        with patch("editguard_chat.reasonke_backend.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"ready":true}')) as opened:
            self.client.request("/status", timeout=9)
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:8010/status")
        self.assertEqual(request.get_header("Authorization"), "Bearer contract-test-token-only")
        self.assertEqual(opened.call_args.kwargs["timeout"], 9)
        self.assertIsNone(request.data)

    def test_absent_auth_and_untrusted_plain_http_fail_closed(self):
        with self.assertRaises(ModelError):
            ReasonKEClient(token="").request("/status")
        for url in ("http://example.org", "https://example.org/path", "https://name:password@example.org",
                    "https://example.org?token=secret", "file:///tmp/model"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                ReasonKEClient(base_url=url, token="contract-test-token-only")
        with self.assertRaises(ValueError):
            ReasonKEClient(token="header\r\ninjection")

    def test_status_checks_checkpoint_revision_and_never_claims_reproduction(self):
        with patch.object(self.client, "request", return_value={"ready": True,
                          "model_id": MODEL_ID, "revision": MODEL_REVISION}):
            status = self.client.status()
        self.assertTrue(status["ready"])
        self.assertFalse(status["supports_images"])
        self.assertFalse(status["published_metric_reproduced"])
        self.assertFalse(status["weights_changed_by_corrections"])
        self.assertNotIn(self.client.token, json.dumps(status))
        for replacement in ({"model_id": "another/model"}, {"revision": "wrong"}, {"ready": "true"}):
            client = ReasonKEClient(token="contract-test-token-only")
            payload = {"ready": True, "model_id": MODEL_ID, "revision": MODEL_REVISION, **replacement}
            with self.subTest(replacement=replacement), patch.object(client, "request", return_value=payload):
                self.assertFalse(client.status()["ready"])

    def test_generation_preserves_structured_facts_and_retains_raw_receipt(self):
        facts = [{"id": "fact-1", "question": "Which city hosts Orion Lab?", "answer": "Amber City",
                  "scope": "Orion Lab", "source": "Controlled contract fixture", "version": 2}]
        history = [{"role": "user", "content": "Explain gravity."},
                   {"role": "assistant", "content": "Fixture answer."}]
        fixture = {"content": "Amber City", "raw_output": "<answer>Amber City</answer>",
                   "generation": {"model": MODEL_ID, "revision": MODEL_REVISION},
                   "format": {"answer_tag_found": True}}
        with patch.object(self.client, "request", return_value=fixture) as request:
            result = self.client.answer_with_facts("Where is Orion Lab?", facts, history)
        path, payload = request.call_args.args
        self.assertEqual(path, "/generate")
        self.assertEqual(payload["facts"][0]["answer"], "Amber City")
        self.assertEqual(payload["facts"][0]["source"], facts[0]["source"])
        self.assertNotIn("version", payload["facts"][0])
        self.assertEqual(payload["history"], history)
        self.assertEqual(payload["max_new_tokens"], 512)
        self.assertEqual(result["raw_output"], fixture["raw_output"])
        self.assertEqual(result["format"], fixture["format"])
        fixture["generation"]["revision"] = "incorrect-revision"
        with patch.object(self.client, "request", return_value=fixture), self.assertRaises(ModelError):
            self.client.answer_with_facts("Where?", facts, [])

    def test_invalid_and_oversized_worker_responses_raise_model_error(self):
        for raw in (b"[]", b"not json", b"x" * (2 * 1024 * 1024 + 1)):
            with self.subTest(size=len(raw)), patch(
                    "editguard_chat.reasonke_backend.urllib.request.urlopen", return_value=io.BytesIO(raw)):
                with self.assertRaises(ModelError):
                    self.client.request("/status")
        error = HTTPError("http://127.0.0.1:8010/generate", 400, "bad request", {},
                          io.BytesIO(b'{"error":"Prompt exceeds configured token budget"}'))
        with patch("editguard_chat.reasonke_backend.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(ModelError, "token budget"):
                self.client.request("/generate", {})


class FakeFactBackend:
    """Records supplied facts; its synthetic output proves no model behavior."""
    def __init__(self):
        self.calls = []
        self.ready = True
        self.fail = False

    def status(self):
        return {"ready": self.ready, "supports_images": False, "name": "explicit-unit-test-fake"}

    def answer_with_facts(self, question, facts, history):
        self.calls.append(copy.deepcopy((question, facts, history)))
        if self.fail:
            raise ModelError("Injected worker failure")
        text = "FAKE_CONTEXT_PRESENT" if facts else "FAKE_NO_CONTEXT"
        return {"content": text, "raw_output": "RAW_FIXTURE_ONLY", "generation": {"model": "unit-test-fake"},
                "format": {"fixture": True}}


class ReasonKEWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.backend = FakeFactBackend()
        self.app = ChatApplication(self.temp.name, backend=self.backend)
        self.chat_id = self.app.store.create_chat()["id"]

    def tearDown(self):
        self.app.store.close()
        self.temp.cleanup()

    def ask(self, question, mode="edited"):
        return self.app.ask({"chat_id": self.chat_id, "message": question, "mode": mode})

    def publish(self, index=0):
        return self.app.mutate("/api/corrections", {"question": f"Which city hosts Orion Lab {index}?",
            "answer": "Amber City", "scope": f"Orion Lab {index}",
            "source": "Controlled software fixture; not model-quality evidence"})["correction"]

    def test_full_approved_context_does_not_require_an_embedding_model(self):
        first, second = self.publish(), self.publish(1)
        answered = self.ask("An unfamiliar question phrasing")
        _, facts, _ = self.backend.calls[-1]
        self.assertEqual({row["id"] for row in facts}, {first["id"], second["id"]})
        evidence = answered["assistant"]["evidence"]
        self.assertEqual(evidence["method"], "all_approved_facts_learned_applicability")
        self.assertIsNone(evidence["threshold"])
        self.assertFalse(evidence["correction_application_verified"])
        self.assertFalse(evidence["weights_changed"])
        self.assertEqual(evidence["raw_model_output"], "RAW_FIXTURE_ONLY")
        status = self.app.status()
        self.assertIn("same trained checkpoint", status["comparison_label"].lower())

    def test_stale_question_and_answer_are_removed_together_but_stay_visible(self):
        self.ask("Explain overfitting.")
        self.publish()
        self.ask("Which city hosts Orion Lab 0?")
        self.assertEqual(self.backend.calls[-1][2], [])
        self.assertEqual(len(self.app.chat(self.chat_id)["messages"]), 4)
        self.assertEqual(self.app.chat(self.chat_id)["messages"][0]["content"], "Explain overfitting.")

    def test_current_plain_conversation_pairs_are_retained_for_followup(self):
        self.ask("Explain overfitting.")
        self.ask("Explain that more simply.")
        history = self.backend.calls[-1][2]
        self.assertEqual([row["role"] for row in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], "Explain overfitting.")

    def test_active_fact_conversation_retains_current_pairs_for_normal_followups(self):
        self.publish()
        first = self.ask("Which city hosts Orion Lab 0?")
        self.ask("Please explain that answer.")
        history = self.backend.calls[-1][2]
        self.assertEqual([row["role"] for row in history], ["user", "assistant"])
        self.assertEqual(history[0]["content"], first["user"]["content"])
        self.assertEqual(history[1]["content"], first["assistant"]["content"])
        self.ask("Which city hosts Orion Lab 0?", mode="original")
        self.assertEqual(self.backend.calls[-1][2], [])

    def test_compare_is_same_checkpoint_and_matched_history_with_context_toggle(self):
        self.publish()
        self.ask("Which city hosts Orion Lab 0?")
        answered = self.ask("Where is that lab located?")
        count = len(self.app.store.list_messages(self.chat_id))
        result = self.app.compare({"chat_id": self.chat_id, "message_id": answered["user"]["id"]})
        before, after = self.backend.calls[-2:]
        self.assertEqual(before[0], after[0])
        self.assertEqual(before[2], after[2])
        self.assertEqual(before[2], [])
        self.assertEqual(before[1], [])
        self.assertEqual(len(after[1]), 1)
        self.assertIn("not the untouched foundation model", result["original"]["evidence"]["notice"])
        self.assertEqual(len(self.app.store.list_messages(self.chat_id)), count)

    def test_withdrawal_and_restore_change_context_without_old_generated_answer(self):
        row = self.publish()
        self.ask(row["question"])
        self.app.mutate(f"/api/corrections/{row['id']}/withdraw", {})
        self.ask(row["question"])
        self.assertEqual(self.backend.calls[-1][1:], ([], []))
        self.app.mutate(f"/api/corrections/{row['id']}/restore", {})
        self.ask(row["question"])
        self.assertEqual(self.backend.calls[-1][1][0]["id"], row["id"])
        self.assertEqual(self.backend.calls[-1][2], [])

    def test_memory_limit_blocks_inference_without_silent_truncation(self):
        for index in range(21):
            self.publish(index)
        with self.assertRaisesRegex(ModelError, "at most 20"):
            self.ask("Any question")
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.app.store.list_messages(self.chat_id), [])

    def test_worker_failure_never_stores_a_substitute_or_partial_response(self):
        self.backend.fail = True
        with self.assertRaisesRegex(ModelError, "Injected worker failure"):
            self.ask("Hello")
        self.assertEqual(self.app.store.list_messages(self.chat_id), [])

    def test_exact_key_contradiction_requires_rule_clarification(self):
        row = self.publish()
        conflicting = {**row, "id": "another-id", "answer": "Conflicting City"}
        result = ApprovedFactContext().rank("Where?", [row, conflicting])
        self.assertTrue(result["clarification_required"])
        self.assertFalse(result["accepted"])
        self.assertEqual(result["selected"], [])


class ColabProxyAccessTests(unittest.TestCase):
    ORIGIN = "https://8005-example.colab.googleusercontent.com"

    def test_proxy_origin_requires_https_exact_googleusercontent_host(self):
        self.assertEqual(validate_trusted_origin(self.ORIGIN + "/"), self.ORIGIN)
        current_colab = "https://8005-example.asia-southeast1-2.prod.colab.dev"
        self.assertEqual(validate_trusted_origin(current_colab), current_colab)
        for origin in ("http://8005-example.colab.googleusercontent.com", "https://example.org",
                       "https://googleusercontent.com", self.ORIGIN + "/path", self.ORIGIN + "?secret=x",
                       "https://name:password@8005-example.colab.googleusercontent.com", self.ORIGIN + ":444"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                validate_trusted_origin(origin)

    def test_only_explicit_proxy_origin_and_session_token_allow_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            app = ChatApplication(temporary, backend=FakeFactBackend(), trusted_origin=self.ORIGIN)
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            address = "http://127.0.0.1:" + str(server.server_port)
            try:
                allowed = {"Host": self.ORIGIN.removeprefix("https://"), "Origin": self.ORIGIN,
                           "Content-Type": "application/json", "X-EditGuard-Token": app.token}
                with urlopen(Request(address + "/api/chats", data=b"{}", headers=allowed)) as response:
                    self.assertEqual(response.status, 200)
                for changed in ({"Origin": "https://unapproved.colab.googleusercontent.com"},
                                {"Origin": self.ORIGIN.replace("https:", "http:")},
                                {"Host": "unapproved.colab.googleusercontent.com"},
                                {"X-EditGuard-Token": "wrong-token"}):
                    with self.subTest(changed=changed), self.assertRaises(HTTPError) as rejected:
                        urlopen(Request(address + "/api/chats", data=b"{}", headers={**allowed, **changed}))
                    self.assertEqual(rejected.exception.code, 403)
                    rejected.exception.close()
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
                app.store.close()


if __name__ == "__main__":
    unittest.main()

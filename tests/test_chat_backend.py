"""Ollama wire/provenance contracts; all HTTP and inference are mocked."""
import copy
import io
import json
import math
import unittest
from unittest.mock import patch

from editguard_chat.backend import ModelError, OllamaClient, OllamaEncoder


class ChatBackendTests(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient()

    def registry(self):
        config = self.client.config
        return {"models": [
            {"name": config["chat_model"], "digest": config["chat_digest"],
             "details": {"parameter_size": "test-only"}},
            {"name": config["embedding_model"], "digest": config["embedding_digest"]},
        ]}

    def test_default_is_explicit_instruct_tag_with_recorded_digests(self):
        self.assertEqual(self.client.model, "qwen3-vl:2b-instruct")
        self.assertEqual(self.client.model, self.client.config["chat_model"])
        self.assertEqual(self.client.embedding_model, self.client.config["embedding_model"])
        for key in ("chat_digest", "embedding_digest"):
            self.assertRegex(self.client.config[key], r"^[0-9a-f]{64}$")
        self.assertEqual(self.client.base, "http://127.0.0.1:11434")

    def test_status_requires_matching_model_and_embedding_digests(self):
        registry = self.registry()
        with patch.object(self.client, "request", side_effect=[registry, {"capabilities": ["completion", "vision"]}]) as request:
            status = self.client.status()
        self.assertTrue(status["ready"])
        self.assertTrue(status["embedding_ready"])
        self.assertTrue(status["supports_images"])
        self.assertEqual(status["digest"], self.client.config["chat_digest"])
        self.assertEqual(status["embedding_digest"], self.client.config["embedding_digest"])
        self.assertEqual(request.call_args_list[1].args,
                         ("/api/show", {"model": "qwen3-vl:2b-instruct"}))

    def test_digest_mismatch_fails_closed_before_capability_request(self):
        for index in (0, 1):
            with self.subTest(model_index=index):
                client = OllamaClient()
                registry = self.registry()
                registry["models"][index]["digest"] = "0" * 64
                with patch.object(client, "request", return_value=registry) as request:
                    status = client.status()
                self.assertFalse(status["ready"])
                self.assertFalse(status["embedding_ready"])
                self.assertIn("digest", status["error"])
                self.assertEqual(request.call_count, 1)

    def test_absent_chat_model_and_malformed_registry_fail_as_status_errors(self):
        for registry in ({"models": []}, {"models": {}}, {"models": ["not a model object"]},
                         {"models": [{}]}):
            with self.subTest(registry=registry):
                client = OllamaClient()
                with patch.object(client, "request", return_value=registry):
                    status = client.status()
                self.assertFalse(status["ready"])
                self.assertIsInstance(status["error"], str)

    def test_missing_embedding_is_disclosed_separately_from_chat_readiness(self):
        registry = self.registry()
        registry["models"] = registry["models"][:1]
        with patch.object(self.client, "request", side_effect=[registry, {"capabilities": ["completion"]}]):
            status = self.client.status()
        self.assertTrue(status["ready"])
        self.assertFalse(status["embedding_ready"])
        self.assertFalse(status["supports_images"])

    def test_malformed_json_and_top_level_list_raise_model_error(self):
        for raw in (b"[]", b"null", b'"not an object"', b"this is not JSON"):
            with self.subTest(raw=raw):
                with patch("editguard_chat.backend.urllib.request.urlopen", return_value=io.BytesIO(raw)):
                    with self.assertRaises(ModelError):
                        self.client.request("/api/tags")

    def test_request_keeps_local_endpoint_and_json_payload(self):
        payload = {"model": self.client.model}
        with patch("editguard_chat.backend.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"capabilities":["vision"]}')) as urlopen:
            response = self.client.request("/api/show", payload, timeout=7)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/show")
        self.assertEqual(json.loads(request.data), payload)
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 7)
        self.assertEqual(response, {"capabilities": ["vision"]})

    def test_transport_and_server_errors_become_model_errors(self):
        with patch("editguard_chat.backend.urllib.request.urlopen", side_effect=OSError("offline test server")):
            with self.assertRaisesRegex(ModelError, "offline test server"):
                self.client.request("/api/tags")
        with patch("editguard_chat.backend.urllib.request.urlopen",
                   return_value=io.BytesIO(b'{"error":"test model missing"}')):
            with self.assertRaisesRegex(ModelError, "test model missing"):
                self.client.request("/api/chat", {})

    def test_generation_is_bounded_non_thinking_and_preserves_multiturn_input(self):
        messages = [{"role": "system", "content": "Controlled test policy"},
                    {"role": "user", "content": "First question"},
                    {"role": "assistant", "content": "First answer"},
                    {"role": "user", "content": "Follow-up", "images": ["test-only-image-data"]}]
        original = copy.deepcopy(messages)
        returned = {"message": {"content": "  Final response.  ", "thinking": "INTERNAL_TEST_THINKING"},
                    "eval_count": 4, "prompt_eval_count": 25,
                    "total_duration": 250_000_000, "done_reason": "stop"}
        with patch.object(self.client, "request", return_value=returned) as request:
            with patch("editguard_chat.backend.time.perf_counter", side_effect=[100., 100.25]):
                result = self.client.generate(messages)
        path, payload = request.call_args.args
        self.assertEqual(path, "/api/chat")
        self.assertEqual(payload["messages"], original)
        self.assertEqual(messages, original)
        self.assertFalse(payload["think"])
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["model"], "qwen3-vl:2b-instruct")
        self.assertEqual(payload["options"], {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 384})
        self.assertEqual(request.call_args.kwargs["timeout"], 240)
        self.assertEqual(result["content"], "Final response.")
        self.assertNotIn("INTERNAL_TEST_THINKING", json.dumps(result))
        self.assertEqual(result["generation"]["seconds"], .25)
        self.assertTrue(math.isfinite(result["generation"]["seconds"]))
        self.assertEqual(result["generation"]["max_new_tokens"], 384)
        self.assertEqual(result["generation"]["num_ctx"], payload["options"]["num_ctx"])
        json.dumps(result, allow_nan=False)

    def test_empty_model_answer_is_preserved_without_template_substitution(self):
        with patch.object(self.client, "request", return_value={"message": {"content": " \n\t "}, "done_reason": "length"}):
            result = self.client.generate([{"role": "user", "content": "Question"}])
        self.assertEqual(result["content"], "")
        self.assertEqual(result["generation"]["done_reason"], "length")

    def test_missing_or_nontext_generated_content_raises_model_error(self):
        for returned in ({}, {"message": None}, {"message": []},
                         {"message": {}}, {"message": {"content": []}}):
            with self.subTest(returned=returned):
                with patch.object(self.client, "request", return_value=returned):
                    with self.assertRaises(ModelError):
                        self.client.generate([{"role": "user", "content": "Question"}])

    def test_embeddings_request_one_input_and_explicit_cpu_execution(self):
        vector = [.1, -.2, .3]
        with patch.object(self.client, "request", return_value={"embeddings": [vector]}) as request:
            encoded = OllamaEncoder(self.client).encode("Question\nScope: Example")
        path, payload = request.call_args.args
        self.assertEqual(path, "/api/embed")
        self.assertEqual(payload["input"], "Question\nScope: Example")
        self.assertEqual(payload["model"], self.client.embedding_model)
        self.assertEqual(payload["options"], {"num_gpu": 0})
        self.assertFalse(payload["truncate"])
        self.assertEqual(request.call_args.kwargs["timeout"], 60)
        self.assertEqual(encoded, vector)

    def test_embedding_response_requires_exactly_one_finite_nonempty_vector(self):
        for vectors in (None, {}, [], [[1.], [2.]], [[]], [None], [[True]],
                        [[float("nan")]], [[float("inf")]], [["not numeric"]]):
            with self.subTest(vectors=vectors):
                with patch.object(self.client, "request", return_value={"embeddings": vectors}):
                    with self.assertRaises(ModelError):
                        OllamaEncoder(self.client).encode("Question")


if __name__ == "__main__":
    unittest.main()

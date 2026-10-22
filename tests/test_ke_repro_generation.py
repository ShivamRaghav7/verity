"""Continuation and cache checks with synthetic logits, never model scores."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Generation adapter tests require the isolated ML environment") from exc

from ke_repro.generation import blip2_prefix_embeddings, generate_blip2_answer, greedy_from_embeddings


class TinyDecoder(torch.nn.Module):
    def __init__(self, selected=(7, 9, 2), *, cache=True, nan=False):
        super().__init__()
        self.selected, self.cache, self.nan = selected, cache, nan
        self.config = SimpleNamespace(max_position_embeddings=100)
        self.calls = []
        self.model = SimpleNamespace(decoder=SimpleNamespace(embed_tokens=torch.nn.Embedding(32, 4)))

    def forward(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        logits = torch.full((1, 1, 32), -10.)
        logits[0, 0, self.selected[min(index, len(self.selected) - 1)]] = 10.
        if self.nan:
            logits[0, 0, 0] = float("nan")
        return SimpleNamespace(logits=logits, past_key_values=(index,) if self.cache else None)


class TokenBatch(dict):
    def __getattr__(self, name):
        return self[name]

    def to(self, device):
        return TokenBatch({key: value.to(device) for key, value in self.items()})


class TinyTokenizer:
    eos_token_id = 2

    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts.extend(prompts)
        return TokenBatch(input_ids=torch.tensor([[20, 21, 22]]), attention_mask=torch.ones(1, 3, dtype=torch.long))

    def decode(self, tokens, skip_special_tokens):
        if any(token >= 20 for token in tokens):
            raise AssertionError("A prompt token leaked into the generated answer")
        return " ".join({7: "new", 9: "answer"}[token] for token in tokens)


class TinyBlip(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.opt_model = TinyDecoder()
        self.opt_tokenizer = TinyTokenizer()
        self.eos_token_id = 3
        self.query_tokens = torch.nn.Parameter(torch.ones(1, 2, 4))
        self.visual_encoder = lambda image: torch.ones(1, 2, 4)
        self.ln_vision = torch.nn.Identity()
        self.opt_proj = torch.nn.Identity()
        self.Qformer = SimpleNamespace(bert=lambda **kwargs:
                                       SimpleNamespace(last_hidden_state=kwargs["query_embeds"]))

    def maybe_autocast(self):
        return nullcontext()


class GenerationTests(unittest.TestCase):
    def test_only_continuation_is_returned_with_correct_cache_masks(self):
        decoder = TinyDecoder()
        prefix = torch.ones(1, 5, 4)
        tokens = greedy_from_embeddings(decoder, prefix, torch.ones(1, 5, dtype=torch.long),
                                        max_new_tokens=8, stop_token_ids=(2,))
        self.assertEqual(tokens, [7, 9])
        self.assertFalse(decoder.training)
        self.assertIn("inputs_embeds", decoder.calls[0])
        self.assertNotIn("input_ids", decoder.calls[0])
        self.assertEqual([call["attention_mask"].shape[1] for call in decoder.calls], [5, 6, 7])
        self.assertEqual(decoder.calls[1]["input_ids"].tolist(), [[7]])
        self.assertEqual(decoder.calls[2]["past_key_values"], (1,))
        self.assertNotIn("labels", decoder.calls[0])

    def test_budget_is_finite_without_stop_token(self):
        decoder = TinyDecoder(selected=(7,))
        result = greedy_from_embeddings(decoder, torch.ones(1, 2, 4), torch.ones(1, 2), max_new_tokens=3)
        self.assertEqual(result, [7, 7, 7])
        self.assertEqual(len(decoder.calls), 3)

    def test_first_stop_returns_empty_and_does_not_echo_question(self):
        decoder = TinyDecoder(selected=(2,))
        result = greedy_from_embeddings(decoder, torch.ones(1, 2, 4), torch.ones(1, 2), stop_token_ids=(2,))
        self.assertEqual(result, [])
        self.assertEqual(len(decoder.calls), 1)

    def test_missing_cache_and_nonfinite_logits_fail_explicitly(self):
        for decoder, message in ((TinyDecoder(cache=False), "KV cache"),
                                 (TinyDecoder(nan=True), "non-finite")):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    greedy_from_embeddings(decoder, torch.ones(1, 2, 4), torch.ones(1, 2), max_new_tokens=2)

    def test_limits_and_shapes_are_checked(self):
        decoder = TinyDecoder()
        prefix, mask = torch.ones(1, 5, 4), torch.ones(1, 5)
        for budget in (0, 129):
            with self.assertRaises(ValueError):
                greedy_from_embeddings(decoder, prefix, mask, max_new_tokens=budget)
        with self.assertRaisesRegex(ValueError, "unpadded"):
            greedy_from_embeddings(decoder, prefix, torch.zeros(1, 5))
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            greedy_from_embeddings(decoder, torch.ones(2, 5, 4), torch.ones(2, 5))
        with self.assertRaisesRegex(ValueError, "context limit"):
            greedy_from_embeddings(decoder, prefix, mask, max_new_tokens=100)

    def test_blip_prefix_uses_raw_question_and_generates_without_expected_answer(self):
        model = TinyBlip()
        image = torch.zeros(1, 3, 364, 364)
        result = generate_blip2_answer(model, image, "Which collection holds this object?", max_new_tokens=4)
        self.assertEqual(result, "new answer")
        self.assertEqual(model.opt_tokenizer.prompts, ["Which collection holds this object?"])
        first = model.opt_model.calls[0]
        self.assertEqual(tuple(first["inputs_embeds"].shape), (1, 5, 4))
        self.assertNotIn("labels", first)

    def test_prefix_rejects_empty_questions(self):
        with self.assertRaisesRegex(ValueError, "nonempty question"):
            blip2_prefix_embeddings(TinyBlip(), torch.zeros(1, 3, 364, 364), "  ")


if __name__ == "__main__":
    unittest.main()

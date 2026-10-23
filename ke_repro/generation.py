"""Bounded, target-free decoding for the upstream BLIP-2 OPT architecture.

The upstream evaluation principally measures teacher-forced target tokens. This
module instead decodes new tokens from image/question or text-only embeddings. It never
accepts an expected answer. These are generation adapters, not benchmark scores.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable

import torch


def greedy_from_embeddings(
    language_model,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    stop_token_ids: Iterable[int] = (),
) -> list[int]:
    """Generate only continuation IDs, with a finite budget and a KV cache.

    This first slice deliberately supports a single unpadded question. Prefix
    tokens are never included in the return value, which avoids the ambiguous
    dummy-BOS/prefix conventions of old ``generate(inputs_embeds=...)`` APIs.
    """
    if not isinstance(max_new_tokens, int) or not 1 <= max_new_tokens <= 128:
        raise ValueError("max_new_tokens must be an integer from 1 to 128")
    if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
        raise ValueError("Exactly one embedded image/question prefix is supported")
    if attention_mask.shape != inputs_embeds.shape[:2]:
        raise ValueError("The attention mask must cover the entire prefix")
    if inputs_embeds.shape[1] == 0 or not bool(attention_mask.eq(1).all()):
        raise ValueError("A nonempty, unpadded prefix is required")
    context_limit = getattr(getattr(language_model, "config", None),
                            "max_position_embeddings", None)
    if context_limit is not None and inputs_embeds.shape[1] + max_new_tokens > context_limit:
        raise ValueError("Question and generation budget exceed the model context limit")

    stops = set(int(token) for token in stop_token_ids if token is not None)
    generated: list[int] = []
    language_model.eval()
    with torch.no_grad():
        outputs = language_model(inputs_embeds=inputs_embeds,
                                 attention_mask=attention_mask,
                                 use_cache=True, return_dict=True)
        for step in range(max_new_tokens):
            last_logits = outputs.logits[:, -1, :]
            if not bool(torch.isfinite(last_logits).all()):
                raise RuntimeError("Generation produced non-finite logits")
            token = int(last_logits.argmax(-1).item())
            if token in stops:
                break
            generated.append(token)
            if step + 1 == max_new_tokens:
                break
            cache = getattr(outputs, "past_key_values", None)
            if cache is None:
                raise RuntimeError("The upstream OPT decoder did not return its KV cache")
            attention_mask = torch.cat((attention_mask,
                                        attention_mask.new_ones((1, 1))), dim=1)
            outputs = language_model(
                input_ids=torch.tensor([[token]], dtype=torch.long,
                                       device=inputs_embeds.device),
                attention_mask=attention_mask, past_key_values=cache,
                use_cache=True, return_dict=True,
            )
    return generated


def blip2_prefix_embeddings(model, image: torch.Tensor | None, question: str):
    """Use upstream forward's visual prefix, or its explicit ``image=None`` branch.

    The raw question is the prefix. No target, retrieved correction, or extra
    answer hint is inserted. This matches the raw-question training convention;
    adding a new conversational prompt would be a different configuration.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A nonempty question is required")
    if image is not None and (image.ndim != 4 or image.shape[0] != 1):
        raise ValueError("Exactly one preprocessed image is required")
    model.eval()
    with torch.no_grad():
        if image is None:
            # Pinned BLIP2OPT.forward's text-locality branch tokenizes directly
            # on OPT's device. It does not encode a substitute/blank image or
            # append Q-Former query tokens.
            tokens = model.opt_tokenizer([question], return_tensors="pt",
                                         padding="longest").to(model.opt_model.device)
            embeddings = model.opt_model.model.decoder.embed_tokens(tokens.input_ids)
            return embeddings, tokens.attention_mask
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long,
                               device=image.device)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens, encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts, return_dict=True,
        )
        visual_prefix = model.opt_proj(query_output.last_hidden_state)
        model.opt_tokenizer.padding_side = "right"
        tokens = model.opt_tokenizer([question], return_tensors="pt",
                                     padding="longest").to(image.device)
        word_embeds = model.opt_model.model.decoder.embed_tokens(tokens.input_ids)
        # The original forward concatenates these before entering OPT autocast.
        # Preserve that behavior rather than silently converting model weights.
        embeddings = torch.cat((visual_prefix, word_embeds), dim=1)
        visual_mask = torch.ones(visual_prefix.size()[:-1], dtype=torch.long,
                                 device=image.device)
        mask = torch.cat((visual_mask, tokens.attention_mask), dim=1)
    return embeddings, mask


def generate_blip2_answer(model, image: torch.Tensor | None, question: str,
                         *, max_new_tokens: int = 32) -> str:
    prefix, mask = blip2_prefix_embeddings(model, image, question)
    eos_id = getattr(model.opt_tokenizer, "eos_token_id", None)
    newline_id = getattr(model, "eos_token_id", None)
    context = model.maybe_autocast() if hasattr(model, "maybe_autocast") else nullcontext()
    with context:
        tokens = greedy_from_embeddings(
            model.opt_model, prefix, mask, max_new_tokens=max_new_tokens,
            stop_token_ids=(eos_id, newline_id),
        )
    return model.opt_tokenizer.decode(tokens, skip_special_tokens=True).strip()

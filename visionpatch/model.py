"""Pinned, frozen CLIP encoder and explicit projection access."""

from __future__ import annotations

import time

MODEL_ID = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def load_clip(*, device: str, allow_download: bool = False):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    start = time.perf_counter()
    kwargs = {"revision": MODEL_REVISION, "local_files_only": not allow_download}
    model = CLIPModel.from_pretrained(MODEL_ID, **kwargs).eval().to(device)
    model.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    return model, tokenizer, time.perf_counter() - start


def image_tensor(images):
    """CLIP RGB normalization after data.render_variant's bicubic 224px crop.

    Pure PIL/NumPy avoids adding torchvision to the preserved ML environment.
    Both resize/crop and normalization are identical in every comparison arm.
    """
    import numpy as np
    import torch

    pixels = np.stack([np.asarray(image.convert("RGB"), dtype=np.float32) / 255
                       for image in images])
    tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None]
    return (tensor - mean) / std


def text_features(model, tokenizer, labels):
    import torch
    import torch.nn.functional as functional

    encoded = tokenizer([f"a photo of a {label}." for label in labels],
                        padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        pooled = model.text_model(**encoded).pooler_output
        return functional.normalize(model.text_projection(pooled), dim=-1).cpu()

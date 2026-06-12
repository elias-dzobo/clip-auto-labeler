from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import open_clip
import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Pick the best available torch device."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=8)
def get_model_bundle(model_name: str, pretrained: str, device: str | None) -> Tuple:
    """Load CLIP model, preprocess transforms, and tokenizer once."""
    device_obj = resolve_device(device)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.eval().to(device_obj)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device_obj


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from .config import FewShotExample, LabelingConfig
from .datasets import discover_images
from .model import get_model_bundle
from .utils import chunked


def zero_shot_label_directory(
    config: LabelingConfig,
    directory: Path,
    *,
    show_progress: bool = True,
) -> list[dict]:
    """Label every image inside ``directory`` using zero-shot CLIP."""
    image_paths = discover_images(directory)
    return zero_shot_label_images(config, image_paths, show_progress=show_progress)


def zero_shot_label_images(
    config: LabelingConfig,
    image_paths: Sequence[Path],
    *,
    show_progress: bool = True,
) -> list[dict]:
    if not config.labels:
        raise ValueError("At least one label is required for zero-shot classification.")

    model, preprocess, tokenizer, device = get_model_bundle(
        config.model_name, config.pretrained, config.device
    )

    text_tokens = tokenizer(config.labels)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    results: list[dict] = []
    iterator: Iterable[Sequence[Path]] = chunked(image_paths, config.batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(image_paths) + config.batch_size - 1) // config.batch_size)

    for batch_paths in iterator:
        tensors: list[torch.Tensor] = []
        valid_paths: list[Path] = []

        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    tensors.append(preprocess(img.convert("RGB")))
                    valid_paths.append(path)
            except (FileNotFoundError, UnidentifiedImageError):
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1)

        for path, prob_row in zip(valid_paths, probs):
            score, index = torch.max(prob_row, dim=-1)
            label = config.labels[int(index)]
            label_out: str | None = label if float(score) >= config.score_threshold else None
            results.append(
                {
                    "image_path": str(path),
                    "predicted_label": label_out,
                    "best_score": float(score),
                    "raw_scores": prob_row.cpu().tolist(),
                }
            )

    return results


def few_shot_label_directory(
    config: LabelingConfig,
    directory: Path,
    examples: Sequence[FewShotExample],
    *,
    show_progress: bool = True,
) -> list[dict]:
    image_paths = discover_images(directory)
    return few_shot_label_images(config, image_paths, examples, show_progress=show_progress)


def few_shot_label_images(
    config: LabelingConfig,
    image_paths: Sequence[Path],
    examples: Sequence[FewShotExample],
    *,
    show_progress: bool = True,
) -> list[dict]:
    model, preprocess, _tokenizer, device = get_model_bundle(
        config.model_name, config.pretrained, config.device
    )

    proto_labels, proto_features = _build_prototypes(model, preprocess, device, examples)
    if proto_features is None:
        raise ValueError("No prototypes could be built from the provided examples.")

    results: list[dict] = []
    iterator: Iterable[Sequence[Path]] = chunked(image_paths, config.batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=(len(image_paths) + config.batch_size - 1) // config.batch_size)

    for batch_paths in iterator:
        tensors: list[torch.Tensor] = []
        valid_paths: list[Path] = []
        for path in batch_paths:
            try:
                with Image.open(path) as img:
                    tensors.append(preprocess(img.convert("RGB")))
                    valid_paths.append(path)
            except (FileNotFoundError, UnidentifiedImageError):
                continue

        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            image_features = model.encode_image(batch)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ proto_features.T
            probs = logits.softmax(dim=-1)

        for path, prob_row in zip(valid_paths, probs):
            score, index = torch.max(prob_row, dim=-1)
            label = proto_labels[int(index)]
            label_out: str | None = label if float(score) >= config.score_threshold else None
            results.append(
                {
                    "image_path": str(path),
                    "predicted_label": label_out,
                    "best_score": float(score),
                    "raw_scores": prob_row.cpu().tolist(),
                }
            )

    return results


def _build_prototypes(model, preprocess, device, examples: Sequence[FewShotExample]) -> tuple[list[str], torch.Tensor | None]:
    labels: list[str] = []
    features: list[torch.Tensor] = []

    for example in examples:
        encoded: list[torch.Tensor] = []
        for path in example.image_paths:
            try:
                with Image.open(path) as img:
                    image_tensor = preprocess(img.convert("RGB")).unsqueeze(0).to(device)
                    with torch.no_grad():
                        emb = model.encode_image(image_tensor)
                        emb /= emb.norm(dim=-1, keepdim=True)
                    encoded.append(emb.squeeze(0))
            except (FileNotFoundError, UnidentifiedImageError):
                continue

        if encoded:
            proto = torch.stack(encoded).mean(dim=0)
            proto /= proto.norm()
            labels.append(example.label)
            features.append(proto)

    if not features:
        return [], None

    return labels, torch.stack(features)


from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(slots=True)
class LabelingConfig:
    """Top-level settings for CLIP-powered labeling."""

    labels: Sequence[str]
    model_name: str = "ViT-B-32"
    pretrained: str = "openai"
    device: str | None = None
    batch_size: int = 32
    num_workers: int = 0
    score_threshold: float = 0.2
    output_path: Path | None = None


@dataclass(slots=True)
class FewShotExample:
    """Represents a label prototype built from existing image examples."""

    label: str
    image_paths: Sequence[Path] = field(default_factory=list)

    def as_mapping(self) -> Mapping[str, Sequence[Path]]:
        return {self.label: self.image_paths}


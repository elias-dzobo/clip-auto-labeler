from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

from PIL import Image, UnidentifiedImageError

DEFAULT_PATTERNS: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")


def discover_images(root: Path, patterns: Sequence[str] = DEFAULT_PATTERNS) -> list[Path]:
    """Recursively collect image files under ``root``."""
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.rglob(pattern))
    return sorted({path.resolve() for path in files})


def load_images(paths: Iterable[Path]) -> Iterator[Image.Image]:
    """Yield PIL Images, skipping corrupt files politely."""
    for path in paths:
        try:
            with Image.open(path) as img:
                yield img.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError):
            continue


from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypeVar

T = TypeVar("T")


def chunked(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def read_labels_file(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def ensure_output_path(path: Path | None) -> Path:
    if path is None:
        return Path("labels.csv")
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


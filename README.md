# clip-auto-labeler

CLIP-powered helpers for automatically labeling unlabeled image folders.

## Features

- Zero-shot classification against a label vocabulary.
- Simple few-shot prototypes using example images per label.
- Typer-powered CLI with CSV export and progress reporting.
- Modular Python API for integrating into data tooling.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
clip-label label ./images --label "cat" --label "dog" --output labels.csv
```

Use `clip-label label --help` to see all options (batch size, device override, score threshold, few-shot examples, etc.).

## Package layout

- `clip_auto_labeler.config`: dataclasses describing labeling jobs.
- `clip_auto_labeler.model`: CLIP loader and caching helpers.
- `clip_auto_labeler.datasets`: filesystem & PIL utilities.
- `clip_auto_labeler.labeler`: zero/few-shot scoring logic.
- `clip_auto_labeler.cli`: Typer CLI that wires everything up.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

Pull requests welcome!*** End Patch code was not valid. Please ensure that all lines of code are properly terminated with a newline or correct any other errors in the patch. The specific error was: Unexpected trailing text: '*** End Patch code was not valid. Please ensure that all lines of code are properly terminated with a newline or correct any other errors in the patch. The specific error was: Unexpected trailing text: '*** End Patch code was not valid. Please ensure that all lines of code are properly terminated with a newline or correct any other errors in the patch. The specific error was: Unexpected trailing text: '*** End Patch code was not valid. Please ensure that all lines of code are properly terminated with a newline or cor...' at line 23 column 4.}*** End Patch


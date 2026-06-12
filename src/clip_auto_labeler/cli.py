from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import typer

from .config import FewShotExample, LabelingConfig
from .labeler import few_shot_label_directory, zero_shot_label_directory
from .utils import ensure_output_path, read_labels_file

app = typer.Typer(help="CLIP-powered auto labeling helpers.")


@app.command()
def label(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Folder of images."),
    label_text: list[str] = typer.Option(
        None,
        "--label",
        "-l",
        help="Label to score against. Can be passed multiple times.",
    ),
    labels_file: Optional[Path] = typer.Option(
        None,
        "--labels-file",
        help="Path to a newline-delimited list of labels.",
    ),
    prototypes_dir: Optional[Path] = typer.Option(
        None,
        "--prototypes-dir",
        help="Directory containing sub-folders per label with example images.",
    ),
    model_name: str = typer.Option("ViT-B-32", help="CLIP backbone."),
    pretrained: str = typer.Option("openai", help="Which weights to load."),
    batch_size: int = typer.Option(32, min=1, max=256),
    score_threshold: float = typer.Option(0.2, min=0.0, max=1.0),
    device: Optional[str] = typer.Option(None, help="torch device override"),
    no_progress: bool = typer.Option(False, help="Hide tqdm progress bars."),
    output: Path = typer.Option(Path("labels.csv"), help="Where to write the CSV output."),
):
    """Label an image directory using zero- or few-shot CLIP strategies."""
    labels = list(label_text or [])
    if labels_file:
        labels.extend(read_labels_file(labels_file))

    if not labels:
        typer.secho("Please provide at least one label via --label or --labels-file.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    cfg = LabelingConfig(
        labels=labels,
        model_name=model_name,
        pretrained=pretrained,
        batch_size=batch_size,
        score_threshold=score_threshold,
        device=device,
        output_path=output,
    )

    if prototypes_dir:
        examples = _load_prototypes(prototypes_dir, labels)
        results = few_shot_label_directory(cfg, input_dir, examples, show_progress=not no_progress)
    else:
        results = zero_shot_label_directory(cfg, input_dir, show_progress=not no_progress)

    output_path = ensure_output_path(cfg.output_path)
    _write_results_csv(results, output_path)
    typer.secho(f"Wrote {len(results)} rows to {output_path}", fg=typer.colors.GREEN)


def _write_results_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _load_prototypes(root: Path, allowed_labels: list[str]) -> list[FewShotExample]:
    examples: list[FewShotExample] = []
    for label_dir in root.iterdir():
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in allowed_labels:
            continue
        image_paths = sorted(p for p in label_dir.iterdir() if p.is_file())
        examples.append(FewShotExample(label=label, image_paths=image_paths))
    return examples


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()


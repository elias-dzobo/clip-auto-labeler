from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from clip_auto_labeler.cli import app

runner = CliRunner()


def test_label_command_requires_labels(tmp_path: Path) -> None:
    """Test that the label command fails when no labels are provided."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    
    result = runner.invoke(app, ["label", str(input_dir)])
    
    assert result.exit_code == 1
    assert "Please provide at least one label" in result.stdout


def test_label_command_with_label_option(tmp_path: Path) -> None:
    """Test label command with --label option."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    output_file = tmp_path / "output.csv"
    
    # Mock the expensive operations
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.95,
            "raw_scores": [0.95, 0.05],
        }
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--label",
                "dog",
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()
    assert "Wrote 1 rows" in result.stdout


def test_label_command_with_labels_file(tmp_path: Path) -> None:
    """Test label command with --labels-file option."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    labels_file = tmp_path / "labels.txt"
    labels_file.write_text("cat\ndog\nbird\n", encoding="utf-8")
    
    output_file = tmp_path / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.85,
            "raw_scores": [0.85, 0.10, 0.05],
        }
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--labels-file",
                str(labels_file),
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()


def test_label_command_combines_label_and_labels_file(tmp_path: Path) -> None:
    """Test that --label and --labels-file can be combined."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    labels_file = tmp_path / "labels.txt"
    labels_file.write_text("dog\n", encoding="utf-8")
    
    output_file = tmp_path / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.90,
            "raw_scores": [0.90, 0.10],
        }
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--labels-file",
                str(labels_file),
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    # Verify that both labels were used (check via mocked function call)
    # The actual verification would check the config passed to the labeler


def test_label_command_with_few_shot_prototypes(tmp_path: Path) -> None:
    """Test label command with --prototypes-dir for few-shot learning."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    prototypes_dir = tmp_path / "prototypes"
    prototypes_dir.mkdir()
    (prototypes_dir / "cat").mkdir()
    (prototypes_dir / "cat" / "cat1.jpg").touch()
    (prototypes_dir / "dog").mkdir()
    (prototypes_dir / "dog" / "dog1.jpg").touch()
    
    output_file = tmp_path / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.92,
            "raw_scores": [0.92, 0.08],
        }
    ]
    
    with patch("clip_auto_labeler.cli.few_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--label",
                "dog",
                "--prototypes-dir",
                str(prototypes_dir),
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()


def test_label_command_with_custom_options(tmp_path: Path) -> None:
    """Test label command with custom model and batch size options."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    output_file = tmp_path / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.88,
            "raw_scores": [0.88, 0.12],
        }
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--model-name",
                "ViT-L-14",
                "--pretrained",
                "laion2b_s32b_b82k",
                "--batch-size",
                "16",
                "--score-threshold",
                "0.3",
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()


def test_label_command_creates_output_directory(tmp_path: Path) -> None:
    """Test that output directory is created if it doesn't exist."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    output_file = tmp_path / "nested" / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.90,
            "raw_scores": [0.90, 0.10],
        }
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()
    assert output_file.parent.exists()


def test_label_command_empty_results(tmp_path: Path) -> None:
    """Test label command with no results (empty directory)."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    # No images in directory
    
    output_file = tmp_path / "output.csv"
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=[]):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()
    # Empty CSV file should be created
    assert output_file.read_text() == ""


def test_label_command_invalid_input_directory() -> None:
    """Test that command fails with non-existent input directory."""
    result = runner.invoke(
        app,
        [
            "label",
            "/nonexistent/directory",
            "--label",
            "cat",
        ],
    )
    
    assert result.exit_code != 0


def test_csv_output_format(tmp_path: Path) -> None:
    """Test that CSV output has correct format."""
    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "test.jpg").touch()
    
    output_file = tmp_path / "output.csv"
    
    mock_results = [
        {
            "image_path": str(input_dir / "test.jpg"),
            "predicted_label": "cat",
            "best_score": 0.95,
            "raw_scores": [0.95, 0.05],
        },
        {
            "image_path": str(input_dir / "test2.jpg"),
            "predicted_label": None,
            "best_score": 0.15,
            "raw_scores": [0.15, 0.85],
        },
    ]
    
    with patch("clip_auto_labeler.cli.zero_shot_label_directory", return_value=mock_results):
        result = runner.invoke(
            app,
            [
                "label",
                str(input_dir),
                "--label",
                "cat",
                "--label",
                "dog",
                "--output",
                str(output_file),
                "--no-progress",
            ],
        )
    
    assert result.exit_code == 0
    assert output_file.exists()
    
    # Verify CSV content
    content = output_file.read_text(encoding="utf-8")
    lines = content.strip().split("\n")
    assert len(lines) == 3  # header + 2 data rows
    assert "image_path" in lines[0]
    assert "predicted_label" in lines[0]
    assert "best_score" in lines[0]
    assert "raw_scores" in lines[0]



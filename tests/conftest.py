"""Shared pytest fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def sample_image_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with sample image files."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    # Create some dummy image files
    (image_dir / "image1.jpg").touch()
    (image_dir / "image2.png").touch()
    (image_dir / "image3.webp").touch()
    return image_dir


@pytest.fixture
def sample_prototypes_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with prototype structure."""
    prototypes_dir = tmp_path / "prototypes"
    prototypes_dir.mkdir()
    
    # Create cat prototype
    (prototypes_dir / "cat").mkdir()
    (prototypes_dir / "cat" / "cat1.jpg").touch()
    (prototypes_dir / "cat" / "cat2.jpg").touch()
    
    # Create dog prototype
    (prototypes_dir / "dog").mkdir()
    (prototypes_dir / "dog" / "dog1.jpg").touch()
    
    return prototypes_dir


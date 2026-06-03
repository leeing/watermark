"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Iterator[Path]:
    """A temporary project root with the standard subdirectory layout."""
    for sub in ("frames", "frames_in", "frames_out"):
        (tmp_path / sub).mkdir()
    yield tmp_path

"""Test MindloggerData object."""

from pathlib import Path

import pytest

from mindlogger_data_export import MindloggerData


def test_mindlogger_data_create_nonexistent_raises_error():
    """MindloggerData.create should raise error for nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        MindloggerData.create(Path("nonexistent"))


def test_mindlogger_data_create_not_a_directory_raises_error(tmp_path: Path):
    """MindloggerData.create should raise error for non-directory."""
    file_path = tmp_path / "file.txt"
    file_path.touch()
    with pytest.raises(NotADirectoryError):
        MindloggerData.create(file_path)

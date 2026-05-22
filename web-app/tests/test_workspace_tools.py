"""Tests for workspace_tools — file ops + run_python scoped to a workspace root."""
from __future__ import annotations

from pathlib import Path

import pytest

from workspace_tools import (
    read_file,
    edit_file,
    write_file,
    list_dir,
    run_python,
    PathEscapeError,
)


def test_write_then_read(tmp_path: Path) -> None:
    write_file(tmp_path, "hello.txt", "hi there")
    assert read_file(tmp_path, "hello.txt") == "hi there"


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path, "nope.txt")


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    write_file(tmp_path, "a/b/c.txt", "deep")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_edit_file_replaces_exact_old_string(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "def foo():\n    return 1\n")
    edit_file(tmp_path, "x.py", "return 1", "return 42")
    assert read_file(tmp_path, "x.py") == "def foo():\n    return 42\n"


def test_edit_file_missing_old_string_raises(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "def foo():\n    return 1\n")
    with pytest.raises(ValueError, match="not found"):
        edit_file(tmp_path, "x.py", "nonexistent text", "anything")


def test_edit_file_ambiguous_old_string_raises(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "foo\nfoo\nfoo\n")
    with pytest.raises(ValueError, match="multiple"):
        edit_file(tmp_path, "x.py", "foo", "bar")


def test_list_dir_returns_entries_sorted(tmp_path: Path) -> None:
    write_file(tmp_path, "b.txt", "")
    write_file(tmp_path, "a.txt", "")
    (tmp_path / "subdir").mkdir()
    entries = list_dir(tmp_path, ".")
    names = [e["name"] for e in entries]
    assert names == ["a.txt", "b.txt", "subdir"]
    assert entries[2]["type"] == "dir"
    assert entries[0]["type"] == "file"


def test_run_python_captures_stdout(tmp_path: Path) -> None:
    result = run_python(tmp_path, "print('hello workspace')")
    assert result["exit_code"] == 0
    assert "hello workspace" in result["stdout"]


def test_run_python_captures_stderr_and_exit_code(tmp_path: Path) -> None:
    result = run_python(tmp_path, "import sys; sys.exit(7)")
    assert result["exit_code"] == 7


def test_run_python_timeout(tmp_path: Path) -> None:
    # Use a very short timeout for the test.
    result = run_python(tmp_path, "import time; time.sleep(60)", timeout_s=1)
    assert "error" in result
    assert "timeout" in result["error"].lower()


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
def test_path_escape_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(PathEscapeError):
        write_file(tmp_path, path, "should not write")
    with pytest.raises(PathEscapeError):
        read_file(tmp_path, path)


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    """Symlinks that point outside the workspace root must be rejected.

    This ensures that symlink-following doesn't become a backdoor to
    escape the workspace boundary.
    """
    # Create a file outside the workspace
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("outside content")

    # Create a symlink inside the workspace pointing to the outside file
    symlink = tmp_path / "ws" / "link_to_outside"
    (tmp_path / "ws").mkdir(exist_ok=True)
    symlink.symlink_to(outside_file)

    # Attempting to read the symlink should raise PathEscapeError
    # because resolving it leads outside the workspace root
    with pytest.raises(PathEscapeError):
        read_file(tmp_path / "ws", "link_to_outside")


def test_reserved_checkpoints_dir_write_rejected(tmp_path: Path) -> None:
    """Writing to .checkpoints/ is rejected (reserved for snapshot storage)."""
    with pytest.raises(PathEscapeError, match="reserved.*checkpoints"):
        write_file(tmp_path, ".checkpoints/1/file.txt", "content")


def test_reserved_checkpoints_dir_read_rejected(tmp_path: Path) -> None:
    """Reading from .checkpoints/ is rejected (reserved for snapshot storage)."""
    # Create the directory and a file in it
    (tmp_path / ".checkpoints" / "1").mkdir(parents=True)
    (tmp_path / ".checkpoints" / "1" / "data.txt").write_text("data")

    with pytest.raises(PathEscapeError, match="reserved.*checkpoints"):
        read_file(tmp_path, ".checkpoints/1/data.txt")

"""Tests for workspace checkpoint snapshot + restore."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from workspace_checkpoint import (
    snapshot_workspace,
    restore_checkpoint,
    next_seq,
)


@pytest.fixture
def ws_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ws"
    d.mkdir()
    return d


def test_snapshot_copies_files_into_checkpoint_dir(ws_dir: Path) -> None:
    (ws_dir / "a.txt").write_text("alpha")
    (ws_dir / "b.txt").write_text("beta")
    snapshot_workspace(ws_dir, seq=1)
    assert (ws_dir / ".checkpoints" / "1" / "a.txt").read_text() == "alpha"
    assert (ws_dir / ".checkpoints" / "1" / "b.txt").read_text() == "beta"


def test_snapshot_excludes_existing_checkpoints_dir(ws_dir: Path) -> None:
    """A snapshot should NOT recursively copy the .checkpoints/ tree."""
    (ws_dir / "a.txt").write_text("alpha")
    snapshot_workspace(ws_dir, seq=1)
    # Now snapshot again — the seq=2 snapshot should not contain a
    # .checkpoints/1/ inside it.
    (ws_dir / "b.txt").write_text("beta")
    snapshot_workspace(ws_dir, seq=2)
    assert (ws_dir / ".checkpoints" / "2" / "a.txt").read_text() == "alpha"
    assert (ws_dir / ".checkpoints" / "2" / "b.txt").read_text() == "beta"
    assert not (ws_dir / ".checkpoints" / "2" / ".checkpoints").exists()


def test_restore_replaces_workspace_with_snapshot(ws_dir: Path) -> None:
    (ws_dir / "a.txt").write_text("v1")
    snapshot_workspace(ws_dir, seq=1)
    (ws_dir / "a.txt").write_text("v2")
    (ws_dir / "new.txt").write_text("created after snapshot")

    restore_checkpoint(ws_dir, seq=1)

    assert (ws_dir / "a.txt").read_text() == "v1"
    assert not (ws_dir / "new.txt").exists()
    # Checkpoints themselves should survive a restore
    assert (ws_dir / ".checkpoints" / "1" / "a.txt").read_text() == "v1"


def test_restore_missing_seq_raises(ws_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        restore_checkpoint(ws_dir, seq=99)


def test_next_seq_returns_one_when_no_checkpoints_table_entries(
    conn: sqlite3.Connection,
) -> None:
    assert next_seq(conn, workspace_id=1) == 1


def test_next_seq_returns_max_plus_one(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT INTO workspaces (id, name, root_dir) VALUES (1, 'x', '/tmp/x')"
    )
    conn.executemany(
        "INSERT INTO workspace_checkpoints (workspace_id, seq, message_id) VALUES (?, ?, ?)",
        [(1, 1, None), (1, 2, None), (1, 5, None)],
    )
    conn.commit()
    assert next_seq(conn, workspace_id=1) == 6

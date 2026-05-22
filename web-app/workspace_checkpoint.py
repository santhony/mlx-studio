"""Workspace checkpoint snapshots — per-turn directory copy + restore.

# pattern: Imperative Shell
# Filesystem I/O plus thin sqlite helpers. The snapshot semantics are
# simple full directory copy; later iterations may swap to hard-link
# or content-addressed storage without changing the contract here.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path


CHECKPOINTS_DIR = ".checkpoints"


def snapshot_workspace(workspace_root: Path, seq: int) -> Path:
    """Copy the workspace root (excluding `.checkpoints/`) to a snapshot dir.

    Returns the path of the created snapshot directory.
    """
    target = workspace_root / CHECKPOINTS_DIR / str(seq)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)

    def _ignore(directory: str, names: list[str]) -> list[str]:
        # `directory` is the source directory string. Skip the
        # checkpoints dir to avoid recursive copying.
        if Path(directory).resolve() == workspace_root.resolve():
            return [CHECKPOINTS_DIR] if CHECKPOINTS_DIR in names else []
        return []

    shutil.copytree(workspace_root, target, ignore=_ignore)
    return target


def restore_checkpoint(workspace_root: Path, seq: int) -> None:
    """Restore the workspace contents from snapshot `seq`.

    Removes everything in the workspace root except the `.checkpoints/`
    dir, then copies the snapshot's contents back. The snapshot itself
    is preserved (re-revertable).
    """
    snapshot = workspace_root / CHECKPOINTS_DIR / str(seq)
    if not snapshot.is_dir():
        raise FileNotFoundError(f"checkpoint {seq} not found in {workspace_root}")

    # Remove current contents (except .checkpoints/)
    for child in workspace_root.iterdir():
        if child.name == CHECKPOINTS_DIR:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    # Copy snapshot contents back
    for child in snapshot.iterdir():
        dest = workspace_root / child.name
        if child.is_dir():
            shutil.copytree(child, dest)
        else:
            shutil.copy2(child, dest)


def next_seq(conn: sqlite3.Connection, *, workspace_id: int) -> int:
    """Return the next sequence number for a workspace (1-indexed)."""
    row = conn.execute(
        "SELECT COALESCE(MAX(seq), 0) AS s FROM workspace_checkpoints "
        "WHERE workspace_id = ?",
        (workspace_id,),
    ).fetchone()
    return int(row["s"]) + 1

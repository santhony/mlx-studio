"""Workspace data access (Functional Core).

# pattern: Functional Core
# Pure SQL helpers — no filesystem, no network, no FastAPI. The router
# layer wraps these and performs the directory create/delete side
# effects. Keeping these pure makes them trivially testable against an
# in-memory SQLite without spinning up the app.
"""
from __future__ import annotations

import sqlite3
from typing import Any


def create_workspace(
    conn: sqlite3.Connection, *, name: str, root_dir: str
) -> dict[str, Any]:
    """Insert a new workspace row, return the inserted row as a dict.

    Does NOT create the on-disk directory — that's the router's job.
    """
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        (name, root_dir),
    )
    conn.commit()
    row = conn.execute(
        "SELECT * FROM workspaces WHERE id = ?", (cur.lastrowid,)
    ).fetchone()
    return dict(row)


def list_workspaces(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """All workspaces, most-recently-active first."""
    rows = conn.execute(
        "SELECT * FROM workspaces ORDER BY last_active_at DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_workspace(
    conn: sqlite3.Connection, workspace_id: int
) -> dict[str, Any] | None:
    """Single workspace by id, or None if not found."""
    row = conn.execute(
        "SELECT * FROM workspaces WHERE id = ?", (workspace_id,)
    ).fetchone()
    return dict(row) if row is not None else None


def update_last_active(conn: sqlite3.Connection, workspace_id: int) -> None:
    """Bump the workspace's last_active_at to current time."""
    conn.execute(
        "UPDATE workspaces SET last_active_at = datetime('now') WHERE id = ?",
        (workspace_id,),
    )
    conn.commit()


def delete_workspace(conn: sqlite3.Connection, workspace_id: int) -> None:
    """Remove a workspace row (cascades to messages + checkpoints).

    Does NOT remove the on-disk directory — that's the router's job.
    """
    conn.execute("DELETE FROM workspaces WHERE id = ?", (workspace_id,))
    conn.commit()

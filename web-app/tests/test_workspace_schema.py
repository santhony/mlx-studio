"""Tests for the workspace tables added in Phase 1."""
import sqlite3


def test_workspaces_table_exists(conn: sqlite3.Connection) -> None:
    """workspaces table is created by init_schema with expected columns."""
    cursor = conn.execute("PRAGMA table_info(workspaces)")
    columns = {row["name"]: row["type"] for row in cursor}
    assert "id" in columns
    assert "name" in columns
    assert "root_dir" in columns
    assert "summary" in columns
    assert "created_at" in columns
    assert "last_active_at" in columns


def test_workspace_messages_table_exists(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(workspace_messages)")
    columns = {row["name"]: row["type"] for row in cursor}
    assert "id" in columns
    assert "workspace_id" in columns
    assert "role" in columns
    assert "content" in columns
    assert "tool_calls_json" in columns
    assert "created_at" in columns


def test_workspace_checkpoints_table_exists(conn: sqlite3.Connection) -> None:
    cursor = conn.execute("PRAGMA table_info(workspace_checkpoints)")
    columns = {row["name"]: row["type"] for row in cursor}
    assert "id" in columns
    assert "workspace_id" in columns
    assert "seq" in columns
    assert "message_id" in columns
    assert "created_at" in columns


def test_vestigial_tables_dropped(conn: sqlite3.Connection) -> None:
    """init_schema drops the old Notebook + Agent tables."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    table_names = {row[0] for row in cursor}
    for old in ("notebooks", "cells", "agent_jobs", "agent_steps"):
        assert old not in table_names, f"vestigial table still present: {old}"

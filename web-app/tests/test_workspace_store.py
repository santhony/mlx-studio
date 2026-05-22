"""Tests for workspace_store helpers (Functional Core).

The functions under test mutate the database but perform no filesystem or
network I/O. Filesystem side-effects live in the router layer (Task 5).
"""
import sqlite3

import pytest

from workspace_store import (
    create_workspace,
    list_workspaces,
    get_workspace,
    delete_workspace,
    update_last_active,
)


def test_create_workspace_returns_row_with_id(conn: sqlite3.Connection) -> None:
    ws = create_workspace(conn, name="blog drafts", root_dir="/tmp/ws/1")
    assert ws["id"] > 0
    assert ws["name"] == "blog drafts"
    assert ws["root_dir"] == "/tmp/ws/1"


def test_list_workspaces_returns_in_recency_order(conn: sqlite3.Connection) -> None:
    create_workspace(conn, name="first", root_dir="/tmp/ws/1")
    second = create_workspace(conn, name="second", root_dir="/tmp/ws/2")
    update_last_active(conn, second["id"])
    rows = list_workspaces(conn)
    assert [r["name"] for r in rows] == ["second", "first"]


def test_get_workspace_by_id(conn: sqlite3.Connection) -> None:
    created = create_workspace(conn, name="x", root_dir="/tmp/ws/x")
    fetched = get_workspace(conn, created["id"])
    assert fetched is not None
    assert fetched["name"] == "x"


def test_get_workspace_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert get_workspace(conn, 99999) is None


def test_delete_workspace_removes_row(conn: sqlite3.Connection) -> None:
    created = create_workspace(conn, name="x", root_dir="/tmp/ws/x")
    delete_workspace(conn, created["id"])
    assert get_workspace(conn, created["id"]) is None

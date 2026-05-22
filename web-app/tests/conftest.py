"""Shared test fixtures for the web-app suite.

Provides:
- in-memory sqlite connection with init_schema applied
- FastAPI TestClient with that connection injected via app.state.db
  (matching the existing convention in routers/chat.py:151 et al.)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from db import init_schema
from main import app


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory SQLite with full schema applied.

    check_same_thread=False allows the TestClient to use this connection
    across thread boundaries (the client runs in a separate thread).
    """
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    init_schema(c)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def client(conn: sqlite3.Connection, tmp_path: Path) -> Iterator[TestClient]:
    """FastAPI TestClient with overridden DB connection + temp data root.

    mlx-studio routers access the SQLite connection via
    `request.app.state.db` (see routers/chat.py:151) — there is no
    Depends(get_conn) pattern. Tests just rebind app.state.db.

    tmp_path is pytest's auto-cleaned temporary directory for workspace data.
    """
    original_db = getattr(app.state, "db", None)
    original_root = getattr(app.state, "data_root", None)
    app.state.db = conn
    app.state.data_root = tmp_path
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if original_db is not None:
            app.state.db = original_db
        if original_root is not None:
            app.state.data_root = original_root

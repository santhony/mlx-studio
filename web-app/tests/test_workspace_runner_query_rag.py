"""Tests for the query_rag tool integration."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from workspace_runner import WorkspaceRunner


@pytest.fixture
def ws_id(conn: sqlite3.Connection, tmp_path: Path) -> int:
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        ("t", str(ws_dir)),
    )
    conn.commit()
    return cur.lastrowid


def test_query_rag_dispatches_to_indexer(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """query_rag tool dispatch calls indexer.retrieve_chunks and serializes result."""
    sample_chunks = [
        {"id": 1, "source_file": "a.md", "chunk_index": 0, "content": "alpha", "score": 0.9},
        {"id": 2, "source_file": "b.md", "chunk_index": 3, "content": "beta", "score": 0.7},
    ]

    async def model(messages: list[dict[str, Any]]) -> str:
        rounds = sum(1 for m in messages if m["role"] == "tool")
        if rounds == 0:
            return '<tool>{"name":"query_rag","args":{"corpus_id":1,"q":"alpha"}}</tool>'
        return "I found 2 chunks."

    with patch("workspace_runner.retrieve_chunks", return_value=sample_chunks) as mock:
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = asyncio.run(runner.run_turn("query the corpus"))
        assert "found 2 chunks" in final
        mock.assert_called_once()

    tool_msg = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    body = json.loads(tool_msg["content"])
    assert "chunks" in body
    assert len(body["chunks"]) == 2
    assert body["chunks"][0]["content"] == "alpha"


def test_query_rag_handles_missing_corpus(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """If retrieve_chunks raises, the tool returns an ERROR string the model sees."""
    async def model(messages: list[dict[str, Any]]) -> str:
        if sum(1 for m in messages if m["role"] == "tool") == 0:
            return '<tool>{"name":"query_rag","args":{"corpus_id":99999,"q":"x"}}</tool>'
        return "Corpus doesn't exist, I see."

    def raising_retrieve(*args, **kwargs):
        raise ValueError("corpus 99999 not found")

    with patch("workspace_runner.retrieve_chunks", side_effect=raising_retrieve):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = asyncio.run(runner.run_turn("query missing corpus"))
        assert "doesn't exist" in final

"""Tests for the workspace runner — end-to-end turn execution.

The text-server call is mocked via a stub function injected at runner
construction. The tool dispatch + state persistence are exercised
against real workspace_tools and a real in-memory SQLite.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from workspace_runner import WorkspaceRunner


@pytest.fixture
def ws_id(conn: sqlite3.Connection, tmp_path: Path) -> int:
    """Create a workspace + on-disk dir for these tests."""
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir()
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        ("test", str(ws_dir)),
    )
    conn.commit()
    return cur.lastrowid


async def _stub_model(messages: list[dict[str, Any]]) -> str:
    """Stub: returns one tool call then a final answer based on round count."""
    user_count = sum(1 for m in messages if m["role"] == "user")
    tool_results = sum(1 for m in messages if m["role"] == "tool")
    if user_count >= 1 and tool_results == 0:
        return (
            'Reading the file.\n'
            '<tool>{"name":"read_file","args":{"path":"greet.txt"}}</tool>'
        )
    return "The file says: hi there"


def test_run_turn_with_one_tool_call(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """Runner calls model → dispatches tool → calls model again → returns final."""
    # Seed a file the stub model expects to read
    (tmp_path / "ws" / "greet.txt").write_text("hi there")

    runner = WorkspaceRunner(
        conn=conn,
        workspace_id=ws_id,
        model_fn=_stub_model,
    )
    final = asyncio.run(runner.run_turn("What does greet.txt say?"))
    assert "hi there" in final

    messages = conn.execute(
        "SELECT role, content FROM workspace_messages WHERE workspace_id=? ORDER BY id",
        (ws_id,),
    ).fetchall()
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_run_turn_handles_path_escape(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """If the model emits a tool call with a bad path, the runner reports the
    error to the model (does not crash)."""
    async def bad_path_model(messages: list[dict[str, Any]]) -> str:
        if sum(1 for m in messages if m["role"] == "tool") == 0:
            return '<tool>{"name":"read_file","args":{"path":"../escape"}}</tool>'
        return "I see, that path was outside the workspace."

    runner = WorkspaceRunner(
        conn=conn, workspace_id=ws_id, model_fn=bad_path_model
    )
    final = asyncio.run(runner.run_turn("read something"))
    assert "outside" in final
    tool_row = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    assert "escapes workspace root" in tool_row["content"] or "PathEscapeError" in tool_row["content"]


def test_run_turn_stops_at_round_cap(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """A model that emits a tool call every round eventually hits the cap."""
    async def looping_model(messages: list[dict[str, Any]]) -> str:
        return '<tool>{"name":"list_dir","args":{"path":"."}}</tool>'

    runner = WorkspaceRunner(
        conn=conn, workspace_id=ws_id, model_fn=looping_model, max_rounds=3
    )
    final = asyncio.run(runner.run_turn("loop"))
    assert "max rounds" in final.lower() or "cap" in final.lower()


def test_run_turn_with_multiple_tool_calls_in_one_turn(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """When the model emits multiple tool calls in one turn, the assistant
    message should appear exactly once in the next round's input, and all
    tool calls should be dispatched + persisted."""
    # Seed files for the model to list and read
    (tmp_path / "ws" / "file1.txt").write_text("content1")
    (tmp_path / "ws" / "file2.txt").write_text("content2")

    async def multi_tool_model(messages: list[dict[str, Any]]) -> str:
        """First round: emit two tool calls. Second round: final answer."""
        tool_results = sum(1 for m in messages if m["role"] == "tool")
        if tool_results == 0:
            # First round: emit two tool calls in one response
            return (
                'I will read two files.\n'
                '<tool>{"name":"read_file","args":{"path":"file1.txt"}}</tool>\n'
                '<tool>{"name":"read_file","args":{"path":"file2.txt"}}</tool>'
            )
        # Second round: return final answer
        return "I read both files successfully."

    runner = WorkspaceRunner(
        conn=conn, workspace_id=ws_id, model_fn=multi_tool_model
    )
    final = asyncio.run(runner.run_turn("Read both files"))
    assert "both files" in final.lower()

    # Verify that both tool calls were dispatched and persisted
    tool_rows = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool' ORDER BY id",
        (ws_id,),
    ).fetchall()
    assert len(tool_rows) == 2, f"Expected 2 tool results, got {len(tool_rows)}"
    assert "content1" in tool_rows[0]["content"]
    assert "content2" in tool_rows[1]["content"]

    # Verify message sequence and that assistant message appears exactly once
    # between the first tool call and the second tool call in in-memory messages.
    # We can't directly inspect in-memory messages, but we can verify the persisted
    # sequence is correct.
    all_rows = conn.execute(
        "SELECT role FROM workspace_messages WHERE workspace_id=? ORDER BY id",
        (ws_id,),
    ).fetchall()
    roles = [r["role"] for r in all_rows]
    # Expected: user, assistant, tool, tool, assistant
    assert roles == ["user", "assistant", "tool", "tool", "assistant"]


def test_run_turn_emits_streaming_tokens_via_callback(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """Runner forwards each token to on_token while accumulating output.

    The stream_model fixture simulates a streaming backend by calling
    the runner's stored on_token callback for each token. The test
    asserts both that the callback received the right tokens AND that
    the final accumulated output matches.
    """
    callback_tokens: list[str] = []

    async def stream_model(messages):
        # Simulate the production model_fn which receives the on_token
        # via self._on_token (set by run_turn). The stub model reads it
        # off the runner instance via the closure on `r`.
        for tok in ["Hel", "lo ", "world"]:
            if r._on_token is not None:
                r._on_token(tok)
        return "Hello world"

    r = WorkspaceRunner(
        conn=conn,
        workspace_id=ws_id,
        model_fn=stream_model,
    )
    final = asyncio.run(r.run_turn(
        "greet",
        on_token=lambda t: callback_tokens.append(t),
    ))
    assert final == "Hello world"
    assert callback_tokens == ["Hel", "lo ", "world"]


def test_run_turn_creates_checkpoint(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """run_turn snapshots the workspace before the dispatcher runs."""

    async def stub(messages):
        return "no tool calls needed"

    runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=stub)
    asyncio.run(runner.run_turn("anything"))

    ckpts = conn.execute(
        "SELECT seq FROM workspace_checkpoints WHERE workspace_id=?",
        (ws_id,),
    ).fetchall()
    assert len(ckpts) == 1
    assert ckpts[0]["seq"] == 1
    ws = conn.execute(
        "SELECT root_dir FROM workspaces WHERE id=?", (ws_id,)
    ).fetchone()
    assert (Path(ws["root_dir"]) / ".checkpoints" / "1").is_dir()

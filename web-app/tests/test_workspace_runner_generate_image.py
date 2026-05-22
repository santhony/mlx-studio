"""Tests for the generate_image tool."""
from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch, AsyncMock

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


def test_generate_image_writes_png_to_workspace(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """generate_image returns a PNG written to the workspace dir."""
    ws_dir = tmp_path / "ws"
    fake_png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

    async def model(messages: list[dict[str, Any]]) -> str:
        if sum(1 for m in messages if m["role"] == "tool") == 0:
            return '<tool>{"name":"generate_image","args":{"prompt":"a cat","filename":"cat.png"}}</tool>'
        return "Generated cat.png"

    async def stub_generate(prompt: str, filename: str | None, *, workspace_root: Path) -> dict:
        target = workspace_root / (filename or "image.png")
        target.write_bytes(fake_png_bytes)
        return {"filename": filename or "image.png", "bytes": len(fake_png_bytes)}

    with patch("workspace_runner._call_image_server", new=stub_generate):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = asyncio.run(runner.run_turn("draw a cat"))
        assert "Generated cat.png" in final

    assert (ws_dir / "cat.png").read_bytes() == fake_png_bytes


def test_generate_image_rejects_path_escape(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """generate_image must reject path traversal attempts like '../escape.png'."""
    ws_dir = tmp_path / "ws"
    parent_dir = tmp_path / "parent_file.txt"  # File outside workspace

    # Track whether _call_image_server was invoked (it should NOT be)
    call_count = 0

    async def model(messages: list[dict[str, Any]]) -> str:
        rounds = sum(1 for m in messages if m["role"] == "tool")
        if rounds == 0:
            # Try to escape the workspace
            return '<tool>{"name":"generate_image","args":{"prompt":"escape","filename":"../escape.png"}}</tool>'
        # On second round, model sees the error and acknowledges it
        return "I understand, the path escapes the workspace."

    async def stub_generate(prompt: str, filename: str | None, *, workspace_root: Path) -> dict:
        # If this is called, the test fails
        nonlocal call_count
        call_count += 1
        raise AssertionError("_call_image_server should not be called for escaping paths")

    with patch("workspace_runner._call_image_server", new=stub_generate):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = asyncio.run(runner.run_turn("try to escape"))

    # The final answer should show the model understood the rejection
    assert "understand" in final.lower() or "escapes" in final.lower() or "escape" in final.lower()

    # Verify _call_image_server was never called
    assert call_count == 0, "Image server should not be called for escaping paths"

    # Verify the error was persisted in the tool message
    tool_msg = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    assert tool_msg is not None
    assert "path escapes workspace root" in tool_msg["content"].lower()

    # Verify no file was created outside the workspace
    assert not parent_dir.exists()


def test_generate_image_runtime_error_surfaces_as_tool_error(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """RuntimeError from _call_image_server becomes an ERROR string the model sees."""
    async def model(messages: list[dict[str, Any]]) -> str:
        rounds = sum(1 for m in messages if m["role"] == "tool")
        if rounds == 0:
            return '<tool>{"name":"generate_image","args":{"prompt":"test","filename":"test.png"}}</tool>'
        # Model should see an error in the tool result
        if "ERROR RuntimeError" in messages[-1]["content"]:
            return "I see the image generation failed"
        return "No error received"

    async def stub_generate_error(
        prompt: str, filename: str | None, *, workspace_root: Path
    ) -> dict:
        raise RuntimeError("Image server is down")

    with patch("workspace_runner._call_image_server", new=stub_generate_error):
        runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=model)
        final = asyncio.run(runner.run_turn("generate an image"))

    # The model should see the error message
    assert "generation failed" in final.lower() or "error" in final.lower()

    # Verify error was persisted in the database
    tool_msg = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    assert tool_msg is not None
    assert "ERROR RuntimeError" in tool_msg["content"]
    assert "Image server is down" in tool_msg["content"]

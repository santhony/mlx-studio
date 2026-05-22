"""HTTP-level tests for the workspace router."""
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from workspace_tools import PathEscapeError, _resolve


def test_workspace_list_empty(client: TestClient) -> None:
    """GET /workspace/ returns an empty list page."""
    response = client.get("/workspace/")
    assert response.status_code == 200
    assert "New Workspace" in response.text  # form to create one


def test_create_workspace_makes_db_row_and_directory(
    client: TestClient,
) -> None:
    """POST /workspace/ creates a DB row and a directory on disk."""
    response = client.post(
        "/workspace/", data={"name": "blog drafts"}, follow_redirects=False
    )
    assert response.status_code in (303, 200)  # Redirect or HTMX swap
    conn = client.app.state.db
    rows = conn.execute("SELECT name, root_dir FROM workspaces").fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "blog drafts"
    assert Path(rows[0]["root_dir"]).is_dir()


def test_create_workspace_rejects_blank_name(client: TestClient) -> None:
    response = client.post(
        "/workspace/", data={"name": ""}, follow_redirects=False
    )
    # Handler explicitly rejects empty/whitespace names with 400
    assert response.status_code == 400


def test_workspace_detail(client: TestClient) -> None:
    """GET /workspace/{id} shows the workspace page."""
    conn = client.app.state.db
    cur = conn.execute(
        "INSERT INTO workspaces (name, root_dir) VALUES (?, ?)",
        ("test ws", "/tmp/x"),
    )
    conn.commit()
    ws_id = cur.lastrowid
    response = client.get(f"/workspace/{ws_id}")
    assert response.status_code == 200
    assert "test ws" in response.text


def test_workspace_detail_404(client: TestClient) -> None:
    response = client.get("/workspace/99999")
    assert response.status_code == 404


def test_delete_workspace_removes_db_and_disk(
    client: TestClient,
) -> None:
    # Create via the endpoint so the directory gets made on disk
    client.post("/workspace/", data={"name": "to-delete"}, follow_redirects=False)
    conn = client.app.state.db
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    ws_dir = Path(ws["root_dir"])
    assert ws_dir.is_dir()

    response = client.delete(f"/workspace/{ws['id']}")
    assert response.status_code in (200, 204)
    assert conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0] == 0
    assert not ws_dir.exists()


@pytest.fixture
def stub_text_server(monkeypatch):
    """Replace WorkspaceRunner._default_model_fn with a stub.

    Required because the text-server isn't up during unit tests.
    """
    from workspace_runner import WorkspaceRunner

    call_count = {"n": 0}

    async def stub(self, messages):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (
                'Creating file.\n'
                '<tool>{"name":"write_file","args":{"path":"out.txt","content":"hi"}}</tool>'
            )
        return "Created out.txt"

    monkeypatch.setattr(WorkspaceRunner, "_default_model_fn", stub)
    return stub


def test_send_message_round_trips(
    client: TestClient,
    stub_text_server,
) -> None:
    """POST /workspace/{id}/messages runs the model loop and persists messages."""
    client.post("/workspace/", data={"name": "test"}, follow_redirects=False)
    conn = client.app.state.db
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()

    response = client.post(
        f"/workspace/{ws['id']}/messages",
        data={"content": "create out.txt with 'hi'"},
    )
    assert response.status_code == 200
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else None
    assert (Path(ws["root_dir"]) / "out.txt").read_text() == "hi"

    msg_count = conn.execute(
        "SELECT COUNT(*) FROM workspace_messages WHERE workspace_id = ?",
        (ws["id"],),
    ).fetchone()[0]
    assert msg_count >= 3  # user + assistant + tool + (assistant)


def test_messages_stream_emits_sse_then_done(
    client: TestClient,
    conn: sqlite3.Connection,
    tmp_path: Path,
    stub_text_server,
) -> None:
    """The /messages/stream endpoint returns SSE-formatted events and ends with [DONE]."""
    client.post("/workspace/", data={"name": "ws"}, follow_redirects=False)
    ws = conn.execute("SELECT id FROM workspaces").fetchone()
    with client.stream(
        "POST",
        f"/workspace/{ws['id']}/messages/stream",
        data={"content": "hello"},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes()).decode()
    assert "data: [DONE]" in body


def test_workspace_file_endpoint_serves_file(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """GET /workspace/{id}/file/{filename} returns the file content."""
    client.post("/workspace/", data={"name": "f"}, follow_redirects=False)
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    (Path(ws["root_dir"]) / "data.txt").write_text("hello file")
    response = client.get(f"/workspace/{ws['id']}/file/data.txt")
    assert response.status_code == 200
    assert response.text == "hello file"


def test_workspace_file_endpoint_rejects_path_escape(
    client: TestClient, conn: sqlite3.Connection
) -> None:
    """Path escapes return 400, never serve files outside workspace.

    Uses URL-encoded path segments (%2E%2E%2F) so the .. sequences survive
    HTTP normalization and reach the handler.
    """
    client.post("/workspace/", data={"name": "f"}, follow_redirects=False)
    ws = conn.execute("SELECT id FROM workspaces").fetchone()
    # URL-encode the path escape: ../../../etc/passwd → %2E%2E%2F%2E%2E%2F%2E%2E%2Fetc%2Fpasswd
    response = client.get(f"/workspace/{ws['id']}/file/%2E%2E%2F%2E%2E%2F%2E%2E%2Fetc%2Fpasswd")
    assert response.status_code == 400


def test_resolve_unit_rejects_path_escape(tmp_path: Path) -> None:
    """Unit test: _resolve rejects path escapes without relying on HTTP normalization."""
    with pytest.raises(PathEscapeError):
        _resolve(tmp_path, "../../../etc/passwd")
    with pytest.raises(PathEscapeError):
        _resolve(tmp_path, "a/../../escape.txt")
    with pytest.raises(PathEscapeError):
        _resolve(tmp_path, "/etc/passwd")


@pytest.fixture
def writing_stub(monkeypatch):
    """Stub model_fn that writes a different file per turn.

    Each turn (user message) writes a file. The stub returns a write
    tool call on the first invocation of each turn, then a final answer
    on the second invocation (after the tool result is back).
    """
    from workspace_runner import WorkspaceRunner

    call_count = {"n": 0}

    async def stub(self, messages):
        # Count how many user messages have been sent so far
        user_count = sum(1 for m in messages if m["role"] == "user")
        # Count how many tool results are in the history so far
        tool_results = sum(1 for m in messages if m["role"] == "tool")

        # On the Nth user message, we should have N-1 tool results when
        # the dispatcher is building the messages list BEFORE running
        # the first model call for that turn. So if tool_results < user_count,
        # we haven't executed the Nth turn's tool yet.
        if tool_results < user_count:
            # This is the first call for this turn: emit the write tool
            return (
                f'Writing turn{user_count}.txt.\n'
                f'<tool>{{"name":"write_file","args":{{"path":"turn{user_count}.txt","content":"v{user_count}"}}}}</tool>'
            )
        # This is the second call for this turn (after tool result): emit final answer
        return f"Wrote turn{user_count}.txt"

    monkeypatch.setattr(WorkspaceRunner, "_default_model_fn", stub)
    return stub


def test_revert_restores_files_and_truncates_history(
    client: TestClient,
    conn: sqlite3.Connection,
    tmp_path: Path,
    writing_stub,
) -> None:
    """After two turns that each write a file, reverting to checkpoint 2
    removes turn2.txt and truncates messages back through turn 2's user
    message."""
    client.post("/workspace/", data={"name": "r"}, follow_redirects=False)
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    ws_dir = Path(ws["root_dir"])

    # Turn 1
    client.post(f"/workspace/{ws['id']}/messages", data={"content": "go 1"})
    assert (ws_dir / "turn1.txt").read_text() == "v1"
    msgs_after_turn1 = conn.execute(
        "SELECT COUNT(*) FROM workspace_messages WHERE workspace_id=?",
        (ws["id"],),
    ).fetchone()[0]

    # Turn 2
    client.post(f"/workspace/{ws['id']}/messages", data={"content": "go 2"})
    assert (ws_dir / "turn2.txt").read_text() == "v2"
    msgs_after_turn2 = conn.execute(
        "SELECT COUNT(*) FROM workspace_messages WHERE workspace_id=?",
        (ws["id"],),
    ).fetchone()[0]
    assert msgs_after_turn2 > msgs_after_turn1

    # Revert to checkpoint 2 (the snapshot taken at start of turn 2)
    ckpt_2 = conn.execute(
        "SELECT seq FROM workspace_checkpoints WHERE workspace_id=? "
        "ORDER BY seq DESC LIMIT 1",
        (ws["id"],),
    ).fetchone()
    assert ckpt_2["seq"] == 2

    response = client.post(f"/workspace/{ws['id']}/revert/{ckpt_2['seq']}")
    assert response.status_code == 204

    # Turn 2's file is gone; turn 1's file survives
    assert (ws_dir / "turn1.txt").read_text() == "v1"
    assert not (ws_dir / "turn2.txt").exists()

    # Messages from turn 2 onward are gone (back to msgs_after_turn1 count)
    msgs_after_revert = conn.execute(
        "SELECT COUNT(*) FROM workspace_messages WHERE workspace_id=?",
        (ws["id"],),
    ).fetchone()[0]
    # Turn 2's user message + everything after is removed; turn 1's
    # transcript survives, so the count matches the post-turn-1 count.
    assert msgs_after_revert == msgs_after_turn1


def test_revert_removes_orphaned_checkpoint_directories(
    client: TestClient,
    conn: sqlite3.Connection,
    tmp_path: Path,
    writing_stub,
) -> None:
    """Reverting removes .checkpoints/ subdirectories for later sequences.

    After three turns creating checkpoints 1, 2, 3, reverting to seq=1
    should remove .checkpoints/2/ and .checkpoints/3/ from disk while
    preserving .checkpoints/1/.
    """
    client.post("/workspace/", data={"name": "r"}, follow_redirects=False)
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    ws_dir = Path(ws["root_dir"])

    # Turn 1 (creates checkpoint 1)
    client.post(f"/workspace/{ws['id']}/messages", data={"content": "go 1"})
    # Turn 2 (creates checkpoint 2)
    client.post(f"/workspace/{ws['id']}/messages", data={"content": "go 2"})
    # Turn 3 (creates checkpoint 3)
    client.post(f"/workspace/{ws['id']}/messages", data={"content": "go 3"})

    # Verify all three checkpoints exist on disk
    assert (ws_dir / ".checkpoints" / "1").is_dir()
    assert (ws_dir / ".checkpoints" / "2").is_dir()
    assert (ws_dir / ".checkpoints" / "3").is_dir()

    # Revert to checkpoint 1
    response = client.post(f"/workspace/{ws['id']}/revert/1")
    assert response.status_code == 204

    # Checkpoint 1 survives; 2 and 3 are removed
    assert (ws_dir / ".checkpoints" / "1").is_dir()
    assert not (ws_dir / ".checkpoints" / "2").exists()
    assert not (ws_dir / ".checkpoints" / "3").exists()


def test_revert_updates_summary_on_truncation(
    client: TestClient,
    conn: sqlite3.Connection,
    tmp_path: Path,
    writing_stub,
) -> None:
    """Reverting updates the workspace summary to reflect surviving history.

    After three turns, manually update the summary to a stale value.
    Then revert to checkpoint 2 (which preserves turn 1's message but
    removes turn 2 and 3). The summary should be recomputed from the
    first surviving user message (turn 1's message).
    """
    client.post("/workspace/", data={"name": "r"}, follow_redirects=False)
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    ws_id = ws["id"]

    # Turn 1
    client.post(f"/workspace/{ws_id}/messages", data={"content": "go 1"})
    # Turn 2
    client.post(f"/workspace/{ws_id}/messages", data={"content": "go 2"})
    # Turn 3
    client.post(f"/workspace/{ws_id}/messages", data={"content": "go 3"})

    # Manually set summary to a stale value to simulate drift
    conn.execute(
        "UPDATE workspaces SET summary = ? WHERE id = ?",
        ("STALE SUMMARY", ws_id),
    )
    conn.commit()

    # Revert to checkpoint 2 (removes turn 2's user message and later,
    # preserving turn 1)
    response = client.post(f"/workspace/{ws_id}/revert/2")
    assert response.status_code == 204

    # Summary should be recomputed from turn 1's content (first surviving user message)
    ws = conn.execute("SELECT summary FROM workspaces WHERE id = ?", (ws_id,)).fetchone()
    # Turn 1's message is "go 1" (3 chars), well under 80 char limit
    assert ws["summary"] == "go 1"

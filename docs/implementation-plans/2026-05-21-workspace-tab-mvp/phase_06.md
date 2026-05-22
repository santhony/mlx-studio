# Workspace Tab MVP — Phase 6: Checkpoint + Revert

**Goal:** Every assistant turn snapshots the workspace directory. A "Revert" button on each assistant message restores the workspace to the state immediately before that turn ran, and truncates the conversation history at that point.

**Architecture:** Snapshot via `shutil.copytree` from the workspace root (excluding `.checkpoints/`) into `.checkpoints/<seq>/`. The snapshot is taken at the start of `run_turn`, BEFORE any tools execute. A new endpoint `POST /workspace/{id}/revert/{seq}` removes the current workspace contents (except `.checkpoints/`) and copies the snapshot back, then truncates `workspace_messages` after the checkpoint's `message_id`.

**Tech Stack:** stdlib `shutil`, `pathlib`. No new dependencies.

**Scope:** Phase 6 of 7.

**Codebase verified:** 2026-05-21. `workspace_checkpoints` table created in Phase 1 with `(workspace_id, seq, message_id, created_at)`. The `.checkpoints/` directory inside each workspace root is already excluded from `list_dir` (Phase 2).

---

<!-- START_TASK_1 -->
### Task 1: Snapshot helper

**Files:**
- Create: `web-app/workspace_checkpoint.py` — pure-ish helpers for snapshot + restore (filesystem I/O at the edges, pure planning in the middle).
- Create: `web-app/tests/test_workspace_checkpoint.py`.

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_checkpoint.py -v`
Expected: FAIL — module not found.

**Step 3: Implement workspace_checkpoint.py**

Create `web-app/workspace_checkpoint.py`:

```python
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
```

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_checkpoint.py -v`
Expected: 6 tests pass.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_checkpoint.py mlx-studio/web-app/tests/test_workspace_checkpoint.py
git commit -m "feat(workspace): checkpoint snapshot + restore (shutil.copytree)"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Snapshot per turn in the runner + revert endpoint

**Files:**
- Modify: `web-app/workspace_runner.py` — take a snapshot before the dispatcher loop runs.
- Modify: `web-app/routers/workspace.py` — add `POST /workspace/{id}/revert/{seq}`.
- Modify: `web-app/tests/test_workspace_runner.py` — verify checkpoint is created.
- Modify: `web-app/tests/test_workspace_router.py` — verify revert endpoint works.

**Step 1: Hook the snapshot into `run_turn`**

In `workspace_runner.py`, near the start of `run_turn` (after fetching the workspace root, before persisting the user message):

```python
from workspace_checkpoint import snapshot_workspace, next_seq

# inside run_turn, just before persisting user message:
seq = next_seq(self._conn, workspace_id=self._ws_id)
snapshot_workspace(root, seq=seq)
self._persist_message("user", user_message, tool_calls=[])
# The user message id is available now; record the checkpoint
user_msg_id = self._conn.execute(
    "SELECT MAX(id) AS m FROM workspace_messages WHERE workspace_id=?",
    (self._ws_id,),
).fetchone()["m"]
self._conn.execute(
    "INSERT INTO workspace_checkpoints (workspace_id, seq, message_id) "
    "VALUES (?, ?, ?)",
    (self._ws_id, seq, user_msg_id),
)
self._conn.commit()
```

**Step 2: Add the revert endpoint**

In `routers/workspace.py`:

```python
from workspace_checkpoint import restore_checkpoint


@router.post("/{workspace_id}/revert/{seq}")
async def revert_to_checkpoint(
    workspace_id: int, seq: int, request: Request
) -> Response:
    """Restore workspace state to checkpoint `seq` and truncate messages.

    Semantics: the user message that triggered the snapshot is itself
    removed (because the checkpoint records the state IMMEDIATELY BEFORE
    that message — reverting means undoing the user's prompt too, so
    they can re-edit and re-submit). Everything from that message
    forward (later assistants, tool results, later user turns) is also
    removed. Later checkpoints become unreachable and are dropped.

    The checkpoint row at this seq is preserved, so revert is itself
    idempotent (re-reverting to seq N from a state still after N just
    works).
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    ckpt = conn.execute(
        "SELECT message_id FROM workspace_checkpoints "
        "WHERE workspace_id = ? AND seq = ?",
        (workspace_id, seq),
    ).fetchone()
    if ckpt is None:
        raise HTTPException(status_code=404, detail="checkpoint not found")

    root = Path(ws["root_dir"])
    restore_checkpoint(root, seq=seq)

    # Truncate messages from the checkpoint's user message onward (inclusive).
    if ckpt["message_id"] is not None:
        conn.execute(
            "DELETE FROM workspace_messages "
            "WHERE workspace_id = ? AND id >= ?",
            (workspace_id, ckpt["message_id"]),
        )
        # Drop later checkpoints (they're no longer reachable). The
        # checkpoint at `seq` itself stays — it still describes a valid
        # reachable state.
        conn.execute(
            "DELETE FROM workspace_checkpoints "
            "WHERE workspace_id = ? AND seq > ?",
            (workspace_id, seq),
        )
    conn.commit()

    return Response(status_code=204)
```

**Step 3: Update the message partial to surface a Revert button**

In `_workspace_message.html`, when `message.role == "user"`, add a button that targets the checkpoint that bracketed this turn:

```html
<div class="msg msg-{{ message.role }}" data-message-id="{{ message.id }}">
    <div class="msg-content">
        {{ render_message(role=message.role, content=message.content,
                          tool_calls_json=message.tool_calls_json,
                          workspace_id=workspace.id) | safe }}
    </div>
    {% if message.role == "user" and message.checkpoint_seq is defined %}
    <button class="revert-btn"
            hx-post="/workspace/{{ workspace.id }}/revert/{{ message.checkpoint_seq }}"
            hx-confirm="Revert to before this message? Everything after will be lost."
            hx-on::after-request="window.location.reload()">
        Revert to before this
    </button>
    {% endif %}
</div>
```

For the template to have `message.checkpoint_seq`, the loader (in `routers/workspace.py:workspace_detail` and `routers/workspace.py:messages_html`) needs to LEFT JOIN `workspace_checkpoints` to attach the seq to each user message:

```python
rows = conn.execute(
    """
    SELECT m.id, m.role, m.content, m.tool_calls_json, c.seq AS checkpoint_seq
    FROM workspace_messages m
    LEFT JOIN workspace_checkpoints c
      ON c.workspace_id = m.workspace_id AND c.message_id = m.id
    WHERE m.workspace_id = ?
    ORDER BY m.id
    """,
    (workspace_id,),
).fetchall()
```

**Step 4: Add tests**

Append to `tests/test_workspace_router.py`:

```python
@pytest.fixture
def writing_stub(monkeypatch):
    """Stub model_fn that writes a different file per turn.

    Turn N (counting user messages) writes f"turn{N}.txt". The stub
    response is JSON-shaped so each turn's tool result is non-trivial,
    making the conversation count assertions robust.
    """
    from workspace_runner import WorkspaceRunner

    async def stub(self, messages):
        turn = sum(1 for m in messages if m["role"] == "user")
        tool_rounds = sum(1 for m in messages if m["role"] == "tool")
        if tool_rounds == 0:
            return (
                f'Writing turn{turn}.txt.\n'
                f'<tool>{{"name":"write_file","args":{{"path":"turn{turn}.txt","content":"v{turn}"}}}}</tool>'
            )
        return f"Wrote turn{turn}.txt"

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
```

The test relies on the `writing_stub` fixture, which monkeypatches `WorkspaceRunner._default_model_fn` directly (rather than the Phase-2 fixture's `model_fn` constructor injection) — this exercises the production code path through to the file-write side effects and lets us assert concrete file states.

**Step 5: Add a runner test for snapshot creation**

Append to `tests/test_workspace_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_turn_creates_checkpoint(
    conn: sqlite3.Connection, ws_id: int, tmp_path: Path
) -> None:
    """run_turn snapshots the workspace before the dispatcher runs."""

    async def stub(messages):
        return "no tool calls needed"

    runner = WorkspaceRunner(conn=conn, workspace_id=ws_id, model_fn=stub)
    await runner.run_turn("anything")

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
```

**Step 6: Run all tests**

Run: `cd web-app && python -m pytest tests/ -v`
Expected: All previously-passing tests still pass + the new ones pass.

**Step 7: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/routers/workspace.py mlx-studio/web-app/templates/_workspace_message.html mlx-studio/web-app/tests/
git commit -m "feat(workspace): per-turn checkpoint + revert button + truncate history"
```
<!-- END_TASK_2 -->

---

**Phase 6 done when:**
- Snapshot test (6) + runner snapshot integration test (1) + router revert test (1) pass.
- Manual: in a real workspace, write a file via the chat; click "Revert to before this" on the user message; the file disappears and the conversation truncates at that point.
- `.checkpoints/<seq>/` directories accumulate as turns are taken; reverting later turns preserves earlier checkpoints.

**Phase 6 leaves these for later phases:** cleanup of Notebook + Agent tabs (Phase 7).

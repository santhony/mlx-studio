# Workspace Tab MVP — Phase 1: Workspace Scaffolding

**Goal:** A user can create, list, navigate-to, and delete Workspaces from the top nav. Workspaces persist across server restarts; directories exist on disk. No chat yet.

**Architecture:** New SQLite tables (`workspaces`, `workspace_messages`, `workspace_checkpoints`) added to the idempotent `init_schema()` script in `web-app/db.py`. New `routers/workspace.py` with CRUD endpoints. Two minimal Jinja templates. Workspace creation also creates `data/workspaces/<id>/` on disk; deletion removes the directory.

**Tech Stack:** FastAPI, Jinja2, SQLite, HTMX (existing patterns), httpx (already in requirements), FastAPI TestClient (NEW for this codebase — added in Task 1).

**Scope:** Phase 1 of 7 from `/Users/santhony/Documents/dev_claude/mlx-studio/docs/design-plans/2026-05-21-workspace-tab-mvp.md`.

**Codebase verified:** 2026-05-21 by codebase-investigator. Key findings: db.py uses `init_schema(conn)` with `conn.executescript()` (lines 23-147, all `CREATE TABLE IF NOT EXISTS`); main.py registers routers via `app.include_router(X_router.router)` (lines 92-99); base.html top-nav at lines 31-38; existing tests use `sqlite3.connect(":memory:")` + `init_schema(conn)` (no TestClient yet — Task 1 introduces it).

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Add FastAPI TestClient to test fixtures

**Files:**
- Modify: `web-app/tests/conftest.py` (CREATE if absent — investigator found tests use ad-hoc fixtures; this introduces a shared conftest)
- Modify: `web-app/requirements.txt` — add `httpx` if not already pinned for test use (it's already installed as a runtime dep, but pin in requirements if not pinned)

**Step 1: Write the conftest fixture**

Create `web-app/tests/conftest.py` (or append if it exists — check first):

```python
"""Shared test fixtures for the web-app suite.

Provides:
- in-memory sqlite connection with init_schema applied
- FastAPI TestClient with that connection injected via app.state.db
  (matching the existing convention in routers/chat.py:151 et al.)
"""
import sqlite3
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from db import init_schema
from main import app


@pytest.fixture
def conn() -> Iterator[sqlite3.Connection]:
    """Fresh in-memory SQLite with full schema applied."""
    c = sqlite3.connect(":memory:")
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
    """
    original_db = getattr(app.state, "db", None)
    original_root = getattr(app.state, "data_root", None)
    app.state.db = conn
    app.state.data_root = tmp_path  # workspace dirs land in the temp tree
    try:
        with TestClient(app) as c:
            yield c
    finally:
        if original_db is not None:
            app.state.db = original_db
        if original_root is not None:
            app.state.data_root = original_root
```

**Step 2: Verify operationally**

Run: `cd web-app && python -m pytest tests/conftest.py --collect-only 2>&1 | tail -5`
Expected: No collection errors. (conftest.py itself has no tests; this just confirms it imports cleanly.)

**Step 3: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/tests/conftest.py
git commit -m "test(workspace): introduce FastAPI TestClient fixture in conftest"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add workspace tables to db.py

**Files:**
- Modify: `web-app/db.py:23-147` (inside the `executescript` block — append the three new CREATE TABLE statements before the closing triple-quote)

**Step 1: Write the failing test**

Create `web-app/tests/test_workspace_schema.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd web-app && python -m pytest tests/test_workspace_schema.py -v`
Expected: FAIL — no such table.

**Step 3: Add the table definitions**

Edit `web-app/db.py`. Locate the `conn.executescript("""...""")` block. Append these three `CREATE TABLE` statements before the closing `"""`:

```sql
CREATE TABLE IF NOT EXISTS workspaces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    root_dir TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_active_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workspace_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS workspace_checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workspace_id INTEGER NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    message_id INTEGER REFERENCES workspace_messages(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(workspace_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_workspace_messages_workspace ON workspace_messages(workspace_id, id);
CREATE INDEX IF NOT EXISTS idx_workspace_checkpoints_workspace ON workspace_checkpoints(workspace_id, seq);
```

**Step 4: Run test to verify it passes**

Run: `cd web-app && python -m pytest tests/test_workspace_schema.py -v`
Expected: PASS, 3 tests.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/db.py mlx-studio/web-app/tests/test_workspace_schema.py
git commit -m "feat(workspace): add workspaces/messages/checkpoints tables"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Pure helpers for workspace data access

**Files:**
- Create: `web-app/workspace_store.py` — Functional Core: pure SQL/dataclass functions taking a `sqlite3.Connection`.
- Create: `web-app/tests/test_workspace_store.py`

These helpers are pure data access (read/write the three tables) — no FastAPI, no filesystem. The router layer (Task 5) wraps them with HTTP concerns and directory side-effects.

**Step 1: Write the failing tests**

```python
"""Tests for workspace_store helpers (Functional Core).

The functions under test mutate the database but perform no filesystem or
network I/O. Filesystem side-effects live in the router layer (Task 5).
"""
import sqlite3
from datetime import datetime

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
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'workspace_store'`.

**Step 3: Implement the helpers**

Create `web-app/workspace_store.py`:

```python
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


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    """Convert a sqlite3.Row to a plain dict (or None passthrough)."""
    return dict(row) if row is not None else None


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
    return _row_to_dict(row)


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
```

**Step 4: Run to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_store.py -v`
Expected: PASS, 5 tests.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_store.py mlx-studio/web-app/tests/test_workspace_store.py
git commit -m "feat(workspace): pure data-access helpers for workspaces table"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 4-6) -->
<!-- START_TASK_4 -->
### Task 4: No-op — confirm app.state.db is the convention

**Files:** None modified. This task is a verification gate, not a code change.

**Why:** Verified 2026-05-21 that mlx-studio routers access the SQLite connection via `request.app.state.db` (see `routers/chat.py:151,164,172,196,238`). The connection is opened in `main.py:lifespan` (line 58) and stored at `app.state.db` (line 60). The workspace router (Task 5) follows the same convention — no `Depends(get_conn)` needed. The conftest fixture (Task 1) overrides `app.state.db` directly.

**Step 1: Sanity check**

Run: `cd web-app && grep -nE "request\.app\.state\.db|app\.state\.db" main.py routers/chat.py | head -5`
Expected: confirms the pattern is already wired.

**Step 2: Continue to Task 5.** No commit for this task.
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Create the workspace router

**Files:**
- Create: `web-app/routers/workspace.py`
- Modify: `web-app/main.py` lines 92-99 (router registration block) — add `app.include_router(workspace_router.router)` and `from routers import workspace as workspace_router`.

**Step 1: Write failing tests**

Create `web-app/tests/test_workspace_router.py`:

```python
"""HTTP-level tests for the workspace router."""
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient


def test_workspace_list_empty(client: TestClient) -> None:
    """GET /workspace/ returns an empty list page."""
    response = client.get("/workspace/")
    assert response.status_code == 200
    assert "New Workspace" in response.text  # form to create one


def test_create_workspace_makes_db_row_and_directory(
    client: TestClient,
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    """POST /workspace/ creates a DB row and a directory on disk."""
    response = client.post(
        "/workspace/", data={"name": "blog drafts"}, follow_redirects=False
    )
    assert response.status_code in (303, 200)  # Redirect or HTMX swap
    rows = conn.execute("SELECT name, root_dir FROM workspaces").fetchall()
    assert len(rows) == 1
    assert rows[0]["name"] == "blog drafts"
    assert Path(rows[0]["root_dir"]).is_dir()


def test_create_workspace_rejects_blank_name(client: TestClient) -> None:
    response = client.post(
        "/workspace/", data={"name": ""}, follow_redirects=False
    )
    assert response.status_code == 400


def test_workspace_detail(client: TestClient, conn: sqlite3.Connection) -> None:
    """GET /workspace/{id} shows the workspace page."""
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
    conn: sqlite3.Connection,
    tmp_path: Path,
) -> None:
    # Create via the endpoint so the directory gets made on disk
    client.post("/workspace/", data={"name": "to-delete"}, follow_redirects=False)
    ws = conn.execute("SELECT id, root_dir FROM workspaces").fetchone()
    ws_dir = Path(ws["root_dir"])
    assert ws_dir.is_dir()

    response = client.delete(f"/workspace/{ws['id']}")
    assert response.status_code in (200, 204)
    assert conn.execute("SELECT COUNT(*) FROM workspaces").fetchone()[0] == 0
    assert not ws_dir.exists()
```

**Step 2: Run tests to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py -v`
Expected: FAIL — module 'routers.workspace' not found.

**Step 3: Implement the router**

Create `web-app/routers/workspace.py`:

```python
"""Workspace router — CRUD + detail endpoints.

# pattern: Imperative Shell
# This module is HTTP-edge code that wraps the pure workspace_store
# helpers (Functional Core) with FastAPI plumbing and filesystem side
# effects. Keep business logic out of this file — push it into helpers.
#
# SQLite connection is accessed via request.app.state.db, matching the
# existing convention in routers/chat.py and elsewhere in the codebase.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

import workspace_store

router = APIRouter(prefix="/workspace", tags=["workspace"])
templates = Jinja2Templates(directory="templates")


def _data_root(request: Request) -> Path:
    """Resolve the workspaces data root.

    Production: web-app/data/workspaces/. Tests override via
    app.state.data_root to a tmp_path so each test has an isolated tree.
    """
    override = getattr(request.app.state, "data_root", None)
    if override is not None:
        return Path(override) / "workspaces"
    return Path("data") / "workspaces"


@router.get("/", response_class=HTMLResponse)
async def list_workspaces_view(request: Request) -> HTMLResponse:
    """List all workspaces with a 'New Workspace' form."""
    conn: sqlite3.Connection = request.app.state.db
    rows = workspace_store.list_workspaces(conn)
    return templates.TemplateResponse(
        "workspace_list.html",
        {"request": request, "workspaces": rows},
    )


@router.post("/")
async def create_workspace_view(
    request: Request, name: str = Form(...)
) -> Response:
    """Create a workspace + its on-disk directory, then redirect to it.

    Uses a two-step insert: create the row (so we can get the id), then
    set root_dir to data/workspaces/<id>/ once known. If the on-disk
    mkdir fails, we delete the row to keep state consistent.
    """
    conn: sqlite3.Connection = request.app.state.db
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    root = _data_root(request)
    root.mkdir(parents=True, exist_ok=True)

    # Insert with a placeholder root_dir we patch after we know the id.
    # The placeholder is never read by anything — between the insert and
    # the UPDATE we hold the conn and no other code looks at this row.
    ws = workspace_store.create_workspace(
        conn, name=name, root_dir=""  # patched below
    )
    ws_dir = root / str(ws["id"])
    try:
        ws_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        workspace_store.delete_workspace(conn, ws["id"])
        raise HTTPException(
            status_code=500, detail=f"failed to create workspace dir: {exc}"
        ) from exc
    conn.execute(
        "UPDATE workspaces SET root_dir = ? WHERE id = ?",
        (str(ws_dir), ws["id"]),
    )
    conn.commit()
    return RedirectResponse(url=f"/workspace/{ws['id']}", status_code=303)


@router.get("/{workspace_id}", response_class=HTMLResponse)
async def workspace_detail(
    workspace_id: int, request: Request
) -> HTMLResponse:
    """Workspace detail page (Phase 1: placeholder; chat lives in Phase 3)."""
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    return templates.TemplateResponse(
        "workspace.html",
        {"request": request, "workspace": ws},
    )


@router.delete("/{workspace_id}")
async def delete_workspace_view(
    workspace_id: int, request: Request
) -> Response:
    """Delete workspace row and its on-disk directory."""
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    workspace_store.delete_workspace(conn, workspace_id)
    ws_dir = Path(ws["root_dir"])
    if ws_dir.is_dir():
        shutil.rmtree(ws_dir)
    return Response(status_code=204)
```

**Step 4: Register the router**

Modify `web-app/main.py`. In the router registration block (lines 92-99), add:

```python
from routers import workspace as workspace_router  # at the existing routers-import block

# Then in the include_router block (after the existing app.include_router lines):
app.include_router(workspace_router.router)
```

**Step 5: Run tests to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py -v`
Expected: PASS, 6 tests. (Templates from Task 6 don't exist yet — see Task 6 for the next step. Until then, only the tests that don't render templates pass. Run the full suite after Task 6.)

**Note:** Tests `test_workspace_list_empty` and `test_workspace_detail` will fail if templates don't exist. That's OK — the test order in `pytest -v` will show this; Task 6 fixes it. If you prefer green-first, you can implement Task 6 templates as bare stubs before completing Step 5 verification.

**Step 6: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/routers/workspace.py mlx-studio/web-app/main.py mlx-studio/web-app/tests/test_workspace_router.py
git commit -m "feat(workspace): CRUD router with on-disk directory side effects"
```
<!-- END_TASK_5 -->

<!-- START_TASK_6 -->
### Task 6: Templates + nav link

**Files:**
- Create: `web-app/templates/workspace_list.html`
- Create: `web-app/templates/workspace.html`
- Modify: `web-app/templates/base.html` lines 31-38 (top-nav)

**Step 1: Create workspace_list.html**

Create `web-app/templates/workspace_list.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="container" style="max-width: 800px; margin: 2rem auto; padding: 1rem;">
    <h1>Workspaces</h1>

    <form method="post" action="/workspace/"
          style="display: flex; gap: 0.5rem; margin-bottom: 2rem;">
        <input type="text" name="name" placeholder="Workspace name" required
               style="flex: 1; padding: 0.5rem;" />
        <button type="submit" style="padding: 0.5rem 1rem;">New Workspace</button>
    </form>

    {% if workspaces %}
    <ul style="list-style: none; padding: 0;">
        {% for ws in workspaces %}
        <li style="border: 1px solid var(--border); border-radius: 4px;
                   padding: 0.75rem; margin-bottom: 0.5rem;
                   display: flex; gap: 1rem; align-items: center;">
            <div style="flex: 1; min-width: 0;">
                <div>
                    <a href="/workspace/{{ ws.id }}" style="font-weight: 500;">{{ ws.name }}</a>
                </div>
                {% if ws.summary %}
                <div style="color: var(--text-muted); font-size: 0.85em; margin-top: 0.2rem;
                            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
                    {{ ws.summary }}
                </div>
                {% endif %}
            </div>
            <span style="color: var(--text-muted); font-size: 0.85em; white-space: nowrap;">
                {{ ws.last_active_at }}
            </span>
            <form method="post" action="/workspace/{{ ws.id }}/delete"
                  hx-delete="/workspace/{{ ws.id }}"
                  hx-confirm="Delete this workspace? Its directory will be removed."
                  hx-target="closest li"
                  hx-swap="outerHTML">
                <button type="submit" style="background: transparent; border: 1px solid var(--border);
                                              color: var(--text-muted); padding: 0.25rem 0.5rem;">
                    Delete
                </button>
            </form>
        </li>
        {% endfor %}
    </ul>
    {% else %}
    <p style="color: var(--text-muted); text-align: center; padding: 2rem;">
        No workspaces yet. Create one above to get started.
    </p>
    {% endif %}
</div>
{% endblock %}
```

**Step 2: Create workspace.html (placeholder; Phase 3 adds the chat)**

Create `web-app/templates/workspace.html`:

```html
{% extends "base.html" %}
{% block content %}
<div class="container" style="max-width: 1000px; margin: 1rem auto; padding: 1rem;">
    <div style="display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem;">
        <a href="/workspace/" style="color: var(--text-muted);">← All workspaces</a>
        <h1 style="margin: 0;">{{ workspace.name }}</h1>
    </div>

    <p style="color: var(--text-muted);">
        Workspace ID: {{ workspace.id }} · Directory: <code>{{ workspace.root_dir }}</code>
    </p>

    <div style="padding: 2rem; border: 1px dashed var(--border); border-radius: 4px;
                color: var(--text-muted); text-align: center; margin-top: 2rem;">
        Chat surface coming in Phase 3.
    </div>
</div>
{% endblock %}
```

**Step 3: Update top-nav**

Edit `web-app/templates/base.html`. The nav-links block is at lines 31-38. Add a Workspaces link. The end state should look like:

```html
<a href="/image">Image</a>
<a href="/chat">Chat</a>
<a href="/workspace">Workspace</a>
<a href="/notebook">Notebook</a>
<a href="/rag">RAG</a>
<a href="/agents">Agents</a>
<a href="/skills">Skills</a>
<a href="/finetune">Fine-tune</a>
<a href="/settings">Settings</a>
```

(Notebook and Agents links remain for now — they'll be removed in Phase 7.)

**Step 4: Verify operationally**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py -v`
Expected: ALL 6 tests pass now that templates exist.

Run: `./start.sh` (in another terminal); visit `http://127.0.0.1:8080/workspace/`. You should see the workspace list page (empty). Create one via the form; it should redirect to the detail page. Click "All workspaces" link; back at the list. Click Delete on a workspace; row disappears and `data/workspaces/<id>/` should no longer exist on disk.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/templates/workspace_list.html mlx-studio/web-app/templates/workspace.html mlx-studio/web-app/templates/base.html
git commit -m "feat(workspace): list + detail templates and top-nav link"
```
<!-- END_TASK_6 -->
<!-- END_SUBCOMPONENT_B -->

---

**Phase 1 done when:**
- All 6 `test_workspace_router.py` tests pass + 3 `test_workspace_schema.py` tests pass + 5 `test_workspace_store.py` tests pass = 14 new tests green.
- `./start.sh` launches without errors.
- Manual: top-nav shows "Workspace" link; click → empty list; create one → redirected to detail page showing name + directory; delete → row gone, directory gone.
- `data/workspaces/<id>/` directories exist on disk after creation.

**Phase 1 leaves these for later phases:** chat surface (Phase 3), tool dispatcher (Phase 2), checkpoints (Phase 6).

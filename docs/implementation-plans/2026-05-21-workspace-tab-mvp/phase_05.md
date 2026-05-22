# Workspace Tab MVP — Phase 5: Inline Rendering

**Goal:** Beautiful chat. Assistant content rendered as markdown (headings, lists, code blocks). Images saved to the workspace embed inline. Tool-call and tool-result blocks render as collapsible `<details>` elements.

**Architecture:** Server-side markdown via the `markdown` library (added to requirements). A new `GET /workspace/{id}/file/{filename}` endpoint serves workspace files (with path-escape rejection — same `_resolve` pattern as `workspace_tools._resolve`). The `_workspace_message.html` partial gains conditional rendering: assistant role → markdown; tool/tool_result → collapsible.

**Tech Stack:** `markdown>=3.5` (NEW — add to requirements.txt). Stdlib `html` for escape-where-needed. Existing CSS classes reused.

**Scope:** Phase 5 of 7.

**Codebase verified:** 2026-05-21. No markdown library currently in requirements.txt. Chat templates currently use `.msg-content { white-space: pre-wrap }` — plain text rendering. Existing `details.tool-call` CSS may be present from prior work; verify before adding.

---

<!-- START_TASK_1 -->
### Task 1: Add markdown library + server-side renderer helper

**Files:**
- Modify: `web-app/requirements.txt` — add `markdown>=3.5`.
- Create: `web-app/workspace_render.py` — pure helper that takes raw message content + tool calls, returns rendered HTML.
- Create: `web-app/tests/test_workspace_render.py`.

**Step 1: Write the failing test**

```python
"""Tests for the workspace message renderer."""
from __future__ import annotations

import json

from workspace_render import render_message


def test_plain_text_passes_through_as_markdown_paragraph() -> None:
    html = render_message(role="user", content="hello world", tool_calls_json="[]")
    assert "<p>" in html or "hello world" in html  # markdown wraps in <p>


def test_markdown_headings_and_lists() -> None:
    content = "# Title\n\n- item one\n- item two"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<h1>" in html
    assert "<ul>" in html
    assert "item one" in html


def test_code_blocks_rendered() -> None:
    content = "```python\nprint('x')\n```"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<code" in html
    assert "print" in html


def test_image_path_rewrite_inside_workspace() -> None:
    """An assistant ![alt](filename.png) reference rewrites to /workspace/{id}/file/filename.png."""
    content = "Here's the result:\n\n![cat](cat.png)"
    html = render_message(
        role="assistant",
        content=content,
        tool_calls_json="[]",
        workspace_id=42,
    )
    assert 'src="/workspace/42/file/cat.png"' in html


def test_image_absolute_url_not_rewritten() -> None:
    content = "![remote](https://example.com/x.png)"
    html = render_message(
        role="assistant",
        content=content,
        tool_calls_json="[]",
        workspace_id=42,
    )
    assert "https://example.com/x.png" in html
    assert "/workspace/42/file" not in html


def test_tool_calls_render_as_collapsible_details() -> None:
    tool_calls = json.dumps([
        {"name": "read_file", "args": {"path": "x.py"}}
    ])
    html = render_message(
        role="assistant",
        content="Reading the file.",
        tool_calls_json=tool_calls,
    )
    assert "<details" in html
    assert "read_file" in html


def test_tool_role_message_is_collapsible_result_block() -> None:
    html = render_message(
        role="tool",
        content='{"ok": true, "bytes": 42}',
        tool_calls_json="[]",
    )
    assert "<details" in html
    # The tool result body should appear inside the details
    assert "ok" in html
    assert "42" in html
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_render.py -v`
Expected: FAIL — module not found / markdown not installed.

**Step 3: Add the dependency**

Edit `web-app/requirements.txt` — append `markdown>=3.5`.

Run: `cd web-app && pip install -r requirements.txt` (or `uv pip install -r requirements.txt`)

**Step 4: Implement `workspace_render.py`**

Create `web-app/workspace_render.py`:

```python
"""Render workspace messages to HTML.

# pattern: Functional Core
# Pure string→string transformation. Takes role + content + tool_calls
# JSON, returns rendered HTML safe to embed in templates. No I/O.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any

import markdown


_MD = markdown.Markdown(extensions=["fenced_code", "tables", "nl2br"])
_WS_IMG_RE = re.compile(
    # Match Markdown image refs where the URL is a relative path (no
    # scheme). We rewrite those to /workspace/{id}/file/{path}.
    r"!\[([^\]]*)\]\(([^)]+)\)"
)


def _rewrite_workspace_images(text: str, workspace_id: int) -> str:
    """Rewrite ![alt](relative.png) → ![alt](/workspace/{id}/file/relative.png).

    Absolute URLs (http/https/data:) are left alone.
    """
    def replace(match: re.Match[str]) -> str:
        alt = match.group(1)
        url = match.group(2)
        if re.match(r"^(https?:|data:|/)", url):
            return match.group(0)  # Leave external/absolute paths alone
        return f"![{alt}](/workspace/{workspace_id}/file/{url})"
    return _WS_IMG_RE.sub(replace, text)


def _render_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Render assistant tool_calls as a collapsible <details> block."""
    if not tool_calls:
        return ""
    parts = ['<details class="tool-call"><summary>tool calls</summary><pre>']
    parts.append(html.escape(json.dumps(tool_calls, indent=2)))
    parts.append("</pre></details>")
    return "".join(parts)


def _render_tool_result(content: str) -> str:
    """Render a tool-role message as a collapsible result block."""
    try:
        parsed = json.loads(content)
        body = json.dumps(parsed, indent=2)
    except (json.JSONDecodeError, ValueError):
        body = content
    return (
        '<details class="tool-result"><summary>tool result</summary>'
        f"<pre>{html.escape(body)}</pre></details>"
    )


def render_message(
    *,
    role: str,
    content: str,
    tool_calls_json: str,
    workspace_id: int | None = None,
) -> str:
    """Render one workspace message to HTML."""
    if role == "tool":
        return _render_tool_result(content)

    if role == "user":
        # User content is escaped + line-broken; we don't run user
        # input through the markdown renderer to avoid surprising
        # behavior on raw prompts.
        return f'<div class="user-content">{html.escape(content).replace(chr(10), "<br>")}</div>'

    # Assistant: markdown rendering + image-rewrite + tool-call block
    rewritten = content
    if workspace_id is not None:
        rewritten = _rewrite_workspace_images(rewritten, workspace_id)
    _MD.reset()  # reset state between calls
    body_html = _MD.convert(rewritten)

    try:
        tool_calls = json.loads(tool_calls_json) if tool_calls_json else []
    except json.JSONDecodeError:
        tool_calls = []
    tool_html = _render_tool_calls(tool_calls)

    return f'<div class="assistant-content">{body_html}</div>{tool_html}'
```

**Step 5: Run to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_render.py -v`
Expected: 7 tests pass.

**Step 6: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/requirements.txt mlx-studio/web-app/workspace_render.py mlx-studio/web-app/tests/test_workspace_render.py
git commit -m "feat(workspace): markdown renderer with image rewrite + tool blocks"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: File-serving endpoint + template wiring

**Files:**
- Modify: `web-app/routers/workspace.py` — add `GET /workspace/{id}/file/{filename:path}`.
- Modify: `web-app/templates/_workspace_message.html` — call `render_message` instead of dumping raw text.

**Step 1: Write failing test**

Append to `tests/test_workspace_router.py`:

```python
def test_workspace_file_endpoint_serves_file(
    client: TestClient, conn: sqlite3.Connection, tmp_path: Path
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
    """Path escapes return 400 or 404, never serve files outside workspace."""
    client.post("/workspace/", data={"name": "f"}, follow_redirects=False)
    ws = conn.execute("SELECT id FROM workspaces").fetchone()
    response = client.get(f"/workspace/{ws['id']}/file/../../../etc/passwd")
    assert response.status_code in (400, 404)
```

**Step 2: Add the endpoint**

In `routers/workspace.py`:

```python
from fastapi.responses import FileResponse

from workspace_tools import _resolve, PathEscapeError


@router.get("/{workspace_id}/file/{filename:path}")
async def serve_workspace_file(
    workspace_id: int,
    filename: str,
    request: Request,
) -> FileResponse:
    """Serve a file from the workspace directory.

    Path-escape rejection mirrors workspace_tools._resolve. The endpoint
    is used by inline-image rewrites from the markdown renderer.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    root = Path(ws["root_dir"])
    try:
        target = _resolve(root, filename)
    except PathEscapeError:
        raise HTTPException(status_code=400, detail="path escapes workspace root")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(target)
```

**Step 3: Update `_workspace_message.html` to call the renderer**

```html
{# Per-message partial. Calls workspace_render.render_message to produce
   the HTML for this message based on role + content + tool_calls. #}
<div class="msg msg-{{ message.role }}" data-message-id="{{ message.id }}">
    <div class="msg-content">
        {{ render_message(role=message.role, content=message.content,
                          tool_calls_json=message.tool_calls_json,
                          workspace_id=workspace.id) | safe }}
    </div>
</div>
```

For the template to call `render_message`, register it as a Jinja global. In `web-app/main.py`, near where the Jinja templates are constructed:

```python
from workspace_render import render_message
templates.env.globals["render_message"] = render_message
```

**Step 4: Update CSS**

Append to `web-app/static/css/main.css`:

```css
/* Workspace tool blocks */
.msg-content details.tool-call,
.msg-content details.tool-result {
    margin: 0.5rem 0;
    border: 1px solid var(--border);
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.02);
    font-size: 0.88em;
}
.msg-content details.tool-call > summary,
.msg-content details.tool-result > summary {
    cursor: pointer;
    padding: 0.35rem 0.6rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 0.72em;
    letter-spacing: 0.06em;
}
.msg-content details > pre {
    margin: 0;
    padding: 0.6rem;
    background: var(--bg);
    overflow-x: auto;
}
/* Workspace inline images sized to fit */
.msg-content img {
    max-width: 100%;
    height: auto;
    border-radius: 4px;
    margin: 0.5rem 0;
}
```

**Step 5: Update the streaming JS to also call render_message after [DONE]**

The token-by-token streaming in `workspace.html` appends raw text to a div. When `[DONE]` arrives, the assistant message is fully written to the database; the JS should reload the just-written message via HTMX-style refresh, OR build a small client-side fetch:

In `workspace.html`, after the `if (data === '[DONE]')` block in the SSE handler, replace the streaming text with the rendered version:

```javascript
if (data === '[DONE]') {
    // Re-fetch the rendered messages so markdown + tool blocks render.
    const messagesEl = document.getElementById('messages');
    fetch(`/workspace/${workspaceId}/messages-html`)
        .then(r => r.text())
        .then(html => {
            messagesEl.innerHTML = html;
            messagesEl.scrollTop = messagesEl.scrollHeight;
        });
    input.disabled = false;
    input.focus();
    return;
}
```

Add the supporting endpoint `/messages-html` in the router that returns just the messages partial. With `render_message` registered as a Jinja global in Step 3, the template already sees it — no explicit pass needed:

```python
@router.get("/{workspace_id}/messages-html", response_class=HTMLResponse)
async def messages_html(
    workspace_id: int, request: Request
) -> HTMLResponse:
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")
    rows = conn.execute(
        "SELECT id, role, content, tool_calls_json FROM workspace_messages "
        "WHERE workspace_id = ? ORDER BY id",
        (workspace_id,),
    ).fetchall()
    parts: list[str] = []
    template = templates.env.get_template("_workspace_message.html")
    for row in rows:
        parts.append(template.render(workspace=ws, message=dict(row)))
    return HTMLResponse("\n".join(parts))
```

**Step 6: Run tests + manual smoke**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py tests/test_workspace_render.py -v`
Expected: All pass.

Manual: in a real workspace, ask the model to generate an image, then ask it to reference the image in a markdown response (`![cat](cat.png)`). The image embeds inline.

**Step 7: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/routers/workspace.py mlx-studio/web-app/templates/_workspace_message.html mlx-studio/web-app/templates/workspace.html mlx-studio/web-app/main.py mlx-studio/web-app/static/css/main.css mlx-studio/web-app/tests/test_workspace_router.py
git commit -m "feat(workspace): inline markdown, embedded images, collapsible tool blocks"
```
<!-- END_TASK_2 -->

---

**Phase 5 done when:**
- 7 workspace_render tests + 2 file-endpoint tests pass.
- Manual: assistant markdown (headings, lists, code) renders correctly; `<details>` blocks for tool calls and tool results are collapsible; an image generated via the Phase-4 tool appears inline in the chat when the model references it.

**Phase 5 leaves these for later phases:** checkpoint per turn + revert button (Phase 6), cleanup (Phase 7).

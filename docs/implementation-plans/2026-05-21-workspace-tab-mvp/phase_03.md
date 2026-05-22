# Workspace Tab MVP — Phase 3: HTMX + SSE Chat UI

**Goal:** Working chat surface in the browser. User types, sees streaming response, sees raw tool-call/result text in the transcript. No markdown rendering or image embedding yet — Phase 5 adds those.

**Architecture:** Mirrors the existing `chat.py:_proxy_sse` pattern (chat.py:73-144). Form POST to `/workspace/{id}/messages/stream` opens an SSE stream; the browser uses the same inline-`EventSource` approach `routers/chat.py:210-232` uses (NOT `hx-ext="sse"` — the codebase uses `new EventSource` directly).

**Tech Stack:** FastAPI StreamingResponse, httpx async streaming, HTMX form-post + inline JS EventSource. No new dependencies.

**Scope:** Phase 3 of 7.

**Codebase verified:** 2026-05-21. SSE convention: yield `f"data: {token}\n\n"` strings; close on `[DONE]`. JavaScript pattern: `new EventSource("/path/stream")` → `e.data === "[DONE]"` to close; token-by-token concat to `el.textContent`.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Add streaming model_fn variant

**Files:**
- Modify: `web-app/workspace_runner.py` — add `_default_model_fn_streaming` and an `on_token` callback path through `run_turn`.
- Modify: `web-app/tests/test_workspace_runner.py` — add streaming-callback test.

**Step 1: Write the failing test**

Append to `tests/test_workspace_runner.py`:

```python
@pytest.mark.asyncio
async def test_run_turn_emits_streaming_tokens_via_callback(
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
    final = await r.run_turn(
        "greet",
        on_token=lambda t: callback_tokens.append(t),
    )
    assert final == "Hello world"
    assert callback_tokens == ["Hel", "lo ", "world"]
```

(For Phase 3, the callback hook provides an injection point — Task 2's streaming endpoint uses it for live SSE emission. The production `_default_model_fn` in Task 2 reads `self._on_token` and invokes it per token from the SSE deltas.)

**Step 2: Update `run_turn` to accept `on_token`**

In `workspace_runner.py`, change the `run_turn` signature and pass the callback to the model_fn where applicable. For the stub model_fn that doesn't stream, the callback is just unused; for the production model_fn (Task 2), the callback emits each token.

```python
async def run_turn(
    self,
    user_message: str,
    *,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """... (existing docstring, plus:) If `on_token` is provided, the runner
    forwards each model-output token to it before persisting the message.
    Callers (the streaming endpoint) use this to live-emit SSE deltas."""
    # ... existing body, but pass on_token to _default_model_fn if model_fn
    # supports it. For simplicity, only the production path uses it.
```

The exact threading is: when the runner calls `self._model_fn(messages)`, if `self._model_fn is self._default_model_fn`, it can accept an `on_token` kwarg. For the stub model_fns in tests, we just call `model_fn(messages)`. To avoid signature mismatches, store the callback as `self._on_token` for the duration of the turn and the default streaming model_fn reads it.

```python
self._on_token = on_token
try:
    # existing loop body
finally:
    self._on_token = None
```

**Step 3: Run the new test to verify it passes**

Run: `cd web-app && python -m pytest tests/test_workspace_runner.py -v`
Expected: All prior tests pass + new streaming-callback test passes.

**Step 4: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/tests/test_workspace_runner.py
git commit -m "feat(workspace): runner accepts per-token on_token callback"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Streaming SSE endpoint

**Files:**
- Modify: `web-app/routers/workspace.py` — add `POST /workspace/{id}/messages/stream`.

**Step 1: Replace `_default_model_fn` with the streaming version**

In `workspace_runner.py`, update `_default_model_fn` to use `httpx.stream` against the text-server, mirroring `chat.py:_proxy_sse`:

```python
async def _default_model_fn(
    self, messages: list[dict[str, Any]]
) -> str:
    """Stream from the text-server /chat endpoint, forwarding tokens
    to self._on_token (if set) and returning the full accumulated text."""
    accumulated: list[str] = []
    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream(
            "POST",
            f"{TEXT_SERVER_URL}/chat",
            json={"messages": messages, "stream": True},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                token = line[len("data: "):]
                if token == "[DONE]":
                    break
                # Unescape \\n → \n (the text-server inherits this from
                # the chat.py SSE convention — verified file:line in
                # chat.py:121).
                token = token.replace("\\n", "\n")
                accumulated.append(token)
                if self._on_token is not None:
                    self._on_token(token)
    return "".join(accumulated)
```

**Step 2: Add the streaming endpoint**

In `routers/workspace.py`, add:

```python
from fastapi.responses import StreamingResponse
import asyncio


@router.post("/{workspace_id}/messages/stream")
async def send_message_stream(
    workspace_id: int,
    request: Request,
    content: str = Form(...),
) -> StreamingResponse:
    """Stream a user→assistant turn as SSE.

    The stream yields one `data: <token>\\n\\n` event per model token,
    then a single `data: [DONE]\\n\\n` event when the turn completes.
    The browser listens via inline EventSource (see workspace.html).
    Connection read via request.app.state.db per codebase convention.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")

    # asyncio.Queue is the bridge between the runner's on_token callback
    # (sync, called from inside an async loop) and the StreamingResponse
    # generator that the browser reads from.
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def emit(token: str) -> None:
        queue.put_nowait(token)

    async def run_and_finish() -> None:
        try:
            runner = WorkspaceRunner(conn=conn, workspace_id=workspace_id)
            await runner.run_turn(content, on_token=emit)
        finally:
            queue.put_nowait(None)  # Sentinel: stream is done.

    async def event_stream():
        # Kick off the runner in a background task; consume the queue
        # in the foreground and yield SSE events.
        task = asyncio.create_task(run_and_finish())
        try:
            while True:
                token = await queue.get()
                if token is None:
                    yield "data: [DONE]\n\n"
                    break
                escaped = token.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

**Step 3: Add a test for the streaming endpoint**

Append to `tests/test_workspace_router.py`:

```python
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
```

(The stub_text_server fixture from Phase 2 returns a non-streaming string; the SSE encoding still happens because the stub goes through `model_fn`, not `_default_model_fn`. For a true streaming-path test, stub `_default_model_fn` directly — out of scope for this task.)

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py -v`
Expected: All previous tests pass + new streaming test passes.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/routers/workspace.py mlx-studio/web-app/tests/test_workspace_router.py
git commit -m "feat(workspace): SSE streaming endpoint mirroring chat.py:_proxy_sse"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Chat UI templates

**Files:**
- Create: `web-app/templates/_workspace_message.html` — per-message partial.
- Modify: `web-app/templates/workspace.html` — replace the Phase-1 placeholder with a chat surface.

**Step 1: Create `_workspace_message.html`**

```html
{# Per-message partial. Rendered by the workspace detail view and by
   HTMX swaps when the user posts a new message. Inline rendering of
   markdown / images / collapsible tool blocks is Phase 5 — at this
   point we just dump role + content text. #}
<div class="msg msg-{{ message.role }}" data-message-id="{{ message.id }}">
    <div class="msg-content" {% if streaming %}id="streaming-content"{% endif %}>{{ message.content }}</div>
    {% if message.tool_calls_json and message.tool_calls_json != '[]' %}
    <details class="tool-calls">
        <summary>tool calls</summary>
        <pre>{{ message.tool_calls_json }}</pre>
    </details>
    {% endif %}
</div>
```

**Step 2: Replace `workspace.html` with the chat surface**

```html
{% extends "base.html" %}
{% block content %}
<div class="container" style="max-width: 1000px; margin: 1rem auto; padding: 1rem;
                              display: flex; flex-direction: column; height: calc(100vh - 5rem);">

    <div style="display: flex; align-items: baseline; gap: 1rem; margin-bottom: 1rem;">
        <a href="/workspace/" style="color: var(--text-muted);">← All workspaces</a>
        <h1 style="margin: 0;">{{ workspace.name }}</h1>
    </div>

    <p style="color: var(--text-muted); margin: 0 0 1rem 0;">
        <code>{{ workspace.root_dir }}</code>
    </p>

    <div id="messages" style="flex: 1; overflow-y: auto; display: flex; flex-direction: column;
                              gap: 0.75rem; padding: 0.5rem 0;">
        {% for msg in messages %}
            {% include "_workspace_message.html" with context %}
        {% else %}
            <p style="color: var(--text-muted); text-align: center; margin: auto;">
                Send a message to start. The model can read/write files in this
                workspace and run Python.
            </p>
        {% endfor %}
    </div>

    <form id="workspace-form" style="display: flex; gap: 0.5rem; margin-top: 1rem;"
          onsubmit="return submitWorkspaceMessage(event, {{ workspace.id }})">
        <input type="text" name="content" id="workspace-input" required
               placeholder="What should we work on?"
               style="flex: 1; padding: 0.5rem;" autofocus />
        <button type="submit" style="padding: 0.5rem 1rem;">Send</button>
    </form>
</div>

<script>
function appendMessage(role, content) {
    const messages = document.getElementById('messages');
    const empty = messages.querySelector('p[style*="margin: auto"]');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'msg msg-' + role;
    div.innerHTML = '<div class="msg-content"></div>';
    const contentEl = div.querySelector('.msg-content');
    contentEl.textContent = content;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
    return contentEl;
}

function submitWorkspaceMessage(event, workspaceId) {
    event.preventDefault();
    const input = document.getElementById('workspace-input');
    const userText = input.value.trim();
    if (!userText) return false;

    appendMessage('user', userText);
    const assistantEl = appendMessage('assistant', '');
    input.value = '';
    input.disabled = true;

    // POST the form via fetch and read the SSE stream
    const formData = new FormData();
    formData.append('content', userText);

    fetch(`/workspace/${workspaceId}/messages/stream`, {
        method: 'POST',
        body: formData,
    }).then(async (response) => {
        if (!response.ok) {
            assistantEl.textContent = '[error: ' + response.status + ']';
            input.disabled = false;
            input.focus();
            return;
        }
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            // SSE delimited by \n\n
            let idx;
            while ((idx = buffer.indexOf('\n\n')) >= 0) {
                const event = buffer.slice(0, idx);
                buffer = buffer.slice(idx + 2);
                if (event.startsWith('data: ')) {
                    const data = event.slice(6);
                    if (data === '[DONE]') {
                        input.disabled = false;
                        input.focus();
                        return;
                    }
                    // Unescape the same way the server escaped on the way out
                    assistantEl.textContent += data.replace(/\\n/g, '\n');
                    document.getElementById('messages').scrollTop =
                        document.getElementById('messages').scrollHeight;
                }
            }
        }
        input.disabled = false;
        input.focus();
    });
    return false;
}
</script>
{% endblock %}
```

**Step 3: Update the workspace detail handler to load messages**

In `routers/workspace.py`, update `workspace_detail` to also load and pass the messages:

```python
@router.get("/{workspace_id}", response_class=HTMLResponse)
async def workspace_detail(
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
    return templates.TemplateResponse(
        "workspace.html",
        {"request": request, "workspace": ws, "messages": [dict(r) for r in rows]},
    )
```

**Step 4: Run all existing tests**

Run: `cd web-app && python -m pytest tests/ -v`
Expected: All tests still green; the manual smoke is the validation for the UI itself.

**Step 5: Manual smoke test**

Run `./start.sh`, ensure text-server is up (or use a stub). Open `http://127.0.0.1:8080/workspace/`. Create a workspace. Type "Create a file called hello.txt with the word 'world' and tell me what it says." Watch the streaming response. Verify the file appears in `web-app/data/workspaces/<id>/`.

**Step 6: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/templates/_workspace_message.html mlx-studio/web-app/templates/workspace.html mlx-studio/web-app/routers/workspace.py
git commit -m "feat(workspace): chat UI with SSE streaming via inline EventSource"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

---

**Phase 3 done when:**
- All existing tests still pass.
- The streaming test passes: SSE response has `data:` events ending in `[DONE]`.
- Manual smoke: workspace detail page shows a working chat; user types → sees user message → sees assistant streaming token-by-token → tool calls visible as JSON blocks in collapsible `<details>` elements.

**Phase 3 leaves these for later phases:** markdown rendering (Phase 5), inline images (Phase 5), checkpoint per turn (Phase 6), cross-tab tools (Phase 4).

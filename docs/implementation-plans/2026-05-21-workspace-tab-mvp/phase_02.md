# Workspace Tab MVP — Phase 2: Runner Module + File Tools

**Goal:** Reusable `workspace_runner.py` module that loops `prompt → text-server → parse tool calls → dispatch → feed result back → repeat`. File ops + `run_python` available as tools, all path-scoped to the workspace root. Workspace POST /messages endpoint produces a full round-trip turn (no streaming yet — that's Phase 3).

**Architecture:** `workspace_runner.py` is a Functional-Core/Imperative-Shell split. Pure parts: tool registry (dict of name → callable), prompt construction, tool-call parsing from model output. Impure parts: httpx call to text-server, file/subprocess I/O. Adapted from the dispatcher pattern in `routers/agents.py:283-315` and `agent_tools.py:259-274`, simplified to auto-execute (no approval gates).

**Tech Stack:** httpx (existing), pathlib, subprocess, json. No new dependencies.

**Scope:** Phase 2 of 7.

**Codebase verified:** 2026-05-21. The existing dispatcher in `routers/agents.py:_dispatch_tool` parses `<tool>...</tool>` JSON-block tool calls from raw model output (not OpenAI `tool_calls`). text-server `/chat` returns SSE-streamed tokens; Phase 2 uses non-streaming (we collect the full response before parsing tool calls). The path-escape rejection pattern used in `ds4-coding-eval/tools.py:Toolkit._resolve` is known-good and is mirrored here.

---

<!-- START_SUBCOMPONENT_A (tasks 1-3) -->
<!-- START_TASK_1 -->
### Task 1: Pure tool functions module

**Files:**
- Create: `web-app/workspace_tools.py` — Functional Core: each tool is a pure-ish function taking a workspace root + args, returning a dict result.
- Create: `web-app/tests/test_workspace_tools.py`

**Step 1: Write failing tests**

```python
"""Tests for workspace_tools — file ops + run_python scoped to a workspace root."""
from __future__ import annotations

from pathlib import Path

import pytest

from workspace_tools import (
    read_file,
    edit_file,
    write_file,
    list_dir,
    run_python,
    PathEscapeError,
)


def test_write_then_read(tmp_path: Path) -> None:
    write_file(tmp_path, "hello.txt", "hi there")
    assert read_file(tmp_path, "hello.txt") == "hi there"


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path, "nope.txt")


def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    write_file(tmp_path, "a/b/c.txt", "deep")
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


def test_edit_file_replaces_exact_old_string(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "def foo():\n    return 1\n")
    edit_file(tmp_path, "x.py", "return 1", "return 42")
    assert read_file(tmp_path, "x.py") == "def foo():\n    return 42\n"


def test_edit_file_missing_old_string_raises(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "def foo():\n    return 1\n")
    with pytest.raises(ValueError, match="not found"):
        edit_file(tmp_path, "x.py", "nonexistent text", "anything")


def test_edit_file_ambiguous_old_string_raises(tmp_path: Path) -> None:
    write_file(tmp_path, "x.py", "foo\nfoo\nfoo\n")
    with pytest.raises(ValueError, match="multiple"):
        edit_file(tmp_path, "x.py", "foo", "bar")


def test_list_dir_returns_entries_sorted(tmp_path: Path) -> None:
    write_file(tmp_path, "b.txt", "")
    write_file(tmp_path, "a.txt", "")
    (tmp_path / "subdir").mkdir()
    entries = list_dir(tmp_path, ".")
    names = [e["name"] for e in entries]
    assert names == ["a.txt", "b.txt", "subdir"]
    assert entries[2]["type"] == "dir"
    assert entries[0]["type"] == "file"


def test_run_python_captures_stdout(tmp_path: Path) -> None:
    result = run_python(tmp_path, "print('hello workspace')")
    assert result["exit_code"] == 0
    assert "hello workspace" in result["stdout"]


def test_run_python_captures_stderr_and_exit_code(tmp_path: Path) -> None:
    result = run_python(tmp_path, "import sys; sys.exit(7)")
    assert result["exit_code"] == 7


def test_run_python_timeout(tmp_path: Path) -> None:
    # Use a very short timeout for the test.
    result = run_python(tmp_path, "import time; time.sleep(60)", timeout_s=1)
    assert "error" in result
    assert "timeout" in result["error"].lower()


@pytest.mark.parametrize("path", ["../escape.txt", "/etc/passwd", "a/../../escape.txt"])
def test_path_escape_rejected(tmp_path: Path, path: str) -> None:
    with pytest.raises(PathEscapeError):
        write_file(tmp_path, path, "should not write")
    with pytest.raises(PathEscapeError):
        read_file(tmp_path, path)
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: workspace_tools`.

**Step 3: Implement workspace_tools.py**

Create `web-app/workspace_tools.py`:

```python
"""Workspace-scoped file + execution tools.

# pattern: Mixed
# The path-resolution and edit-string logic is Functional Core (pure
# helpers that take a root + args, return a result or raise). The file
# I/O and subprocess execution at the bottom is Imperative Shell — same
# function signatures, but each performs side effects.
#
# Every public function takes the workspace root as its first argument
# and rejects any path that escapes the root via `..` or absolute
# segments. The pattern mirrors ds4-coding-eval/tools.py:Toolkit._resolve.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


class PathEscapeError(ValueError):
    """Raised when a tool argument would resolve outside the workspace root."""


# ---------------------------------------------------------------------------
# Functional Core: path safety
# ---------------------------------------------------------------------------


def _resolve(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `root`, rejecting paths that escape root.

    `root` MUST already exist (callers create the workspace dir before
    invoking tools). The returned Path may or may not exist; callers do
    the existence check appropriate to their operation.
    """
    root = root.resolve()
    candidate = (root / rel_path).resolve()
    if candidate == root:
        return candidate
    if root not in candidate.parents:
        raise PathEscapeError(
            f"path {rel_path!r} resolves to {candidate} which escapes "
            f"workspace root {root}"
        )
    return candidate


# ---------------------------------------------------------------------------
# Imperative Shell: file + execution tools
# ---------------------------------------------------------------------------


def read_file(root: Path, path: str) -> str:
    """Return the text contents of a workspace-relative file."""
    target = _resolve(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    return target.read_text()


def write_file(root: Path, path: str, content: str) -> dict[str, Any]:
    """Write `content` to a workspace-relative path (full overwrite).

    Creates parent directories as needed. Returns a status dict.
    """
    target = _resolve(root, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"ok": True, "bytes": len(content)}


def edit_file(
    root: Path, path: str, old_str: str, new_str: str
) -> dict[str, Any]:
    """Replace `old_str` with `new_str` in the file at `path`.

    `old_str` MUST appear exactly once in the file:
    - Zero occurrences raises ValueError("not found...")
    - Multiple occurrences raises ValueError("multiple occurrences...")
    This is hash-anchored editing — the caller must supply enough
    context in `old_str` to make the match unique.
    """
    target = _resolve(root, path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    text = target.read_text()
    count = text.count(old_str)
    if count == 0:
        raise ValueError(
            f"old_str not found in {path}: {old_str[:60]!r}..."
        )
    if count > 1:
        raise ValueError(
            f"old_str matches multiple ({count}) occurrences in {path} — "
            f"include more surrounding context to make it unique"
        )
    target.write_text(text.replace(old_str, new_str, 1))
    return {"ok": True, "replacements": 1}


def list_dir(root: Path, path: str) -> list[dict[str, str]]:
    """List the entries of a workspace-relative directory.

    Returns a list of `{"name": ..., "type": "file"|"dir"}` dicts,
    sorted by name. Files come before dirs only insofar as alpha order
    happens to match — primary sort key is name.
    """
    target = _resolve(root, path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    entries: list[dict[str, str]] = []
    for child in target.iterdir():
        if child.name == ".checkpoints":
            # Skip the snapshot directory from listings — Phase 6 owns
            # it and the model shouldn't be reading or editing inside.
            continue
        entries.append(
            {"name": child.name, "type": "dir" if child.is_dir() else "file"}
        )
    entries.sort(key=lambda e: e["name"])
    return entries


def run_python(
    root: Path, code: str, timeout_s: int = 30
) -> dict[str, Any]:
    """Execute `code` as a Python script inside the workspace.

    The code is written to a temp file inside the workspace, run via
    `python3` with cwd set to the workspace root, and the temp file is
    removed afterward. Returns stdout, stderr, exit_code, or an `error`
    key on timeout / spawn failure.
    """
    script = root / "__workspace_run.py"
    script.write_text(code)
    try:
        result = subprocess.run(
            ["python3", str(script)],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return {
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-4000:],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout_s}s"}
    except FileNotFoundError as exc:
        return {"error": f"python3 not on PATH: {exc}"}
    finally:
        script.unlink(missing_ok=True)
```

**Step 4: Run to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_tools.py -v`
Expected: PASS, ~13 tests.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_tools.py mlx-studio/web-app/tests/test_workspace_tools.py
git commit -m "feat(workspace): file ops + run_python tools with path-escape rejection"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Pure tool-call parsing helper

**Files:**
- Create: `web-app/workspace_runner_parser.py` — Functional Core: tool-call detection in raw model output. Pure (no I/O).
- Create: `web-app/tests/test_workspace_runner_parser.py`

The model's output may contain zero or more `<tool>...</tool>` JSON blocks (per the existing agent_tools / chat conventions). We need a parser that extracts them and the surrounding prose.

**Step 1: Write failing tests**

```python
"""Tests for tool-call parsing — extracting <tool>...</tool> JSON blocks."""
from workspace_runner_parser import parse_tool_calls


def test_no_tool_calls() -> None:
    text = "This is a plain assistant reply."
    result = parse_tool_calls(text)
    assert result.tool_calls == []
    assert result.content == text


def test_single_tool_call() -> None:
    text = 'Let me read the file.\n<tool>{"name":"read_file","args":{"path":"main.py"}}</tool>'
    result = parse_tool_calls(text)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[0]["args"] == {"path": "main.py"}
    assert "Let me read the file" in result.content


def test_multiple_tool_calls() -> None:
    text = (
        '<tool>{"name":"read_file","args":{"path":"a.py"}}</tool>'
        'Now writing.\n'
        '<tool>{"name":"write_file","args":{"path":"b.py","content":"x"}}</tool>'
    )
    result = parse_tool_calls(text)
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0]["name"] == "read_file"
    assert result.tool_calls[1]["name"] == "write_file"


def test_malformed_json_in_tool_call_is_marked() -> None:
    text = '<tool>{not json}</tool>'
    result = parse_tool_calls(text)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].get("_parse_error") is not None


def test_content_excludes_tool_blocks() -> None:
    text = (
        'Before tool.\n'
        '<tool>{"name":"x","args":{}}</tool>\n'
        'After tool.'
    )
    result = parse_tool_calls(text)
    assert "<tool>" not in result.content
    assert "Before tool" in result.content
    assert "After tool" in result.content
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_parser.py -v`
Expected: FAIL — module not found.

**Step 3: Implement the parser**

Create `web-app/workspace_runner_parser.py`:

```python
"""Tool-call parsing for workspace_runner.

# pattern: Functional Core
# Pure string→data transformation. Takes the raw assistant output and
# returns a structured (content, tool_calls) tuple. No I/O, no
# dependencies on FastAPI or httpx.
#
# Format matches what the text-server-backed models emit when prompted
# with the existing tool-call convention (see routers/agents.py:283).
# Each tool call is a `<tool>...</tool>` block whose body is JSON with
# keys `name` (str) and `args` (object).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


# Non-greedy match between the literal tags; DOTALL because the JSON
# may legitimately span newlines.
_TOOL_BLOCK_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


@dataclass(frozen=True)
class ParseResult:
    """Result of parse_tool_calls.

    `content` is the assistant's natural-language output with the tool
    blocks stripped. `tool_calls` is a list of {name, args} dicts (or
    {_parse_error: str} for malformed JSON).
    """
    content: str
    tool_calls: list[dict[str, Any]]


def parse_tool_calls(raw: str) -> ParseResult:
    """Extract `<tool>...</tool>` blocks from raw assistant output."""
    tool_calls: list[dict[str, Any]] = []
    for match in _TOOL_BLOCK_RE.finditer(raw):
        body = match.group(1).strip()
        try:
            obj = json.loads(body)
            tool_calls.append({
                "name": obj.get("name", "?"),
                "args": obj.get("args", {}),
            })
        except json.JSONDecodeError as exc:
            tool_calls.append({
                "name": "?",
                "args": {},
                "_parse_error": f"json parse failed: {exc}; body={body[:120]!r}",
            })

    # Strip tool blocks from the content; collapse the resulting
    # whitespace lightly so we don't leave a forest of blank lines.
    content = _TOOL_BLOCK_RE.sub("", raw)
    content = re.sub(r"\n{3,}", "\n\n", content).strip()
    return ParseResult(content=content, tool_calls=tool_calls)
```

**Step 4: Run to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_runner_parser.py -v`
Expected: PASS, 5 tests.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner_parser.py mlx-studio/web-app/tests/test_workspace_runner_parser.py
git commit -m "feat(workspace): pure tool-call parser for assistant output"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Workspace runner + tool dispatch

**Files:**
- Create: `web-app/workspace_runner.py` — Imperative Shell: text-server I/O + tool dispatch loop.
- Create: `web-app/tests/test_workspace_runner.py`

**Step 1: Write failing tests**

```python
"""Tests for the workspace runner — end-to-end turn execution.

The text-server call is mocked via a stub function injected at runner
construction. The tool dispatch + state persistence are exercised
against real workspace_tools and a real in-memory SQLite.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, AsyncIterator

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


@pytest.mark.asyncio
async def test_run_turn_with_one_tool_call(
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
    final = await runner.run_turn("What does greet.txt say?")
    assert "hi there" in final

    messages = conn.execute(
        "SELECT role, content FROM workspace_messages WHERE workspace_id=? ORDER BY id",
        (ws_id,),
    ).fetchall()
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


@pytest.mark.asyncio
async def test_run_turn_handles_path_escape(
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
    final = await runner.run_turn("read something")
    assert "outside" in final
    tool_row = conn.execute(
        "SELECT content FROM workspace_messages WHERE workspace_id=? AND role='tool'",
        (ws_id,),
    ).fetchone()
    assert "escapes workspace root" in tool_row["content"] or "PathEscapeError" in tool_row["content"]


@pytest.mark.asyncio
async def test_run_turn_stops_at_round_cap(
    conn: sqlite3.Connection, ws_id: int
) -> None:
    """A model that emits a tool call every round eventually hits the cap."""
    async def looping_model(messages: list[dict[str, Any]]) -> str:
        return '<tool>{"name":"list_dir","args":{"path":"."}}</tool>'

    runner = WorkspaceRunner(
        conn=conn, workspace_id=ws_id, model_fn=looping_model, max_rounds=3
    )
    final = await runner.run_turn("loop")
    assert "max rounds" in final.lower() or "cap" in final.lower()
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_runner.py -v`
Expected: FAIL — module not found.

**Step 3: Implement the runner**

Create `web-app/workspace_runner.py`:

```python
"""Workspace runner — the prompt → tool → result → repeat loop.

# pattern: Imperative Shell
# This module coordinates I/O between the text-server, the workspace
# tools (subprocess/file I/O), and the database (message persistence).
# The parsing of model output and the tool registry shape live in
# workspace_runner_parser.py and workspace_tools.py respectively, both
# of which are pure-er surfaces under unit test.
#
# Adapted from routers/agents.py:_run_agent + _dispatch_tool, with the
# approval-gate logic removed (workspaces are auto-execute by design —
# safety lives in Phase 6's checkpoint+revert pattern).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from skills import retrieve_skills, format_skills_for_context
from workspace_runner_parser import parse_tool_calls
from workspace_tools import (
    PathEscapeError,
    edit_file,
    list_dir,
    read_file,
    run_python,
    write_file,
)


TEXT_SERVER_URL = "http://127.0.0.1:8766"
DEFAULT_MAX_ROUNDS = 12  # Soft cap per the eval-derived guardrail stack.
DEFAULT_SYSTEM_PROMPT = (
    "You are an assistant working inside a sandboxed workspace directory. "
    "You can read/edit/write files and run Python via the tools listed below. "
    "All file paths you provide are interpreted relative to the workspace "
    "root. To call a tool, emit a single <tool>{\"name\": \"TOOL\", \"args\": {...}}</tool> "
    "block. When you have finished the task, reply with a brief plain-text "
    "summary (no tool calls).\n\n"
    "Tools available:\n"
    "- read_file(path) -> file contents\n"
    "- edit_file(path, old_str, new_str) -> replaces old_str with new_str; "
    "old_str must occur exactly once in the file (include context to make "
    "it unique).\n"
    "- write_file(path, content) -> overwrites or creates the file.\n"
    "- list_dir(path) -> list of {name, type} entries; use '.' for workspace root.\n"
    "- run_python(code) -> runs the code as a Python script; returns stdout, stderr, exit_code.\n"
)


ModelFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


class WorkspaceRunner:
    """Runs a single user→assistant turn against a workspace.

    Constructor takes the workspace id and a `model_fn` callable. The
    default `model_fn` calls the local text-server; tests inject a stub.
    """

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        workspace_id: int,
        model_fn: ModelFn | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._conn = conn
        self._ws_id = workspace_id
        self._model_fn = model_fn or self._default_model_fn
        self._max_rounds = max_rounds
        self._system_prompt = system_prompt

    async def run_turn(self, user_message: str) -> str:
        """Execute one user→assistant turn. Returns the final content."""
        ws = self._conn.execute(
            "SELECT root_dir FROM workspaces WHERE id = ?", (self._ws_id,)
        ).fetchone()
        if ws is None:
            raise ValueError(f"workspace {self._ws_id} not found")
        root = Path(ws["root_dir"])
        if not root.is_dir():
            raise FileNotFoundError(f"workspace root missing: {root}")

        self._persist_message("user", user_message, tool_calls=[])
        messages = self._build_messages_for_model()

        for round_idx in range(self._max_rounds):
            raw = await self._model_fn(messages)
            parsed = parse_tool_calls(raw)
            self._persist_message(
                "assistant", parsed.content, tool_calls=parsed.tool_calls
            )

            if not parsed.tool_calls:
                return parsed.content

            for call in parsed.tool_calls:
                result_text = self._dispatch_tool(root, call)
                self._persist_message("tool", result_text, tool_calls=[])
                messages.append(
                    {"role": "assistant", "content": raw}
                )
                messages.append(
                    {"role": "tool", "content": result_text}
                )

        cap_msg = (
            f"(workspace runner stopped at max rounds = {self._max_rounds}; "
            "the model kept emitting tool calls without a final answer.)"
        )
        self._persist_message("assistant", cap_msg, tool_calls=[])
        return cap_msg

    def _dispatch_tool(self, root: Path, call: dict[str, Any]) -> str:
        """Run a single tool call; return its result serialized as text."""
        if call.get("_parse_error"):
            return f"ERROR parsing tool block: {call['_parse_error']}"
        name = call.get("name", "?")
        args = call.get("args") or {}
        try:
            if name == "read_file":
                return read_file(root, args["path"])
            if name == "write_file":
                result = write_file(root, args["path"], args.get("content", ""))
                return json.dumps(result)
            if name == "edit_file":
                result = edit_file(
                    root, args["path"], args["old_str"], args["new_str"]
                )
                return json.dumps(result)
            if name == "list_dir":
                entries = list_dir(root, args.get("path", "."))
                return json.dumps({"entries": entries})
            if name == "run_python":
                result = run_python(root, args.get("code", ""))
                return json.dumps(result)
            return f"ERROR unknown tool: {name!r}"
        except PathEscapeError as exc:
            return f"ERROR path escapes workspace root: {exc}"
        except FileNotFoundError as exc:
            return f"ERROR file not found: {exc}"
        except ValueError as exc:
            return f"ERROR invalid argument: {exc}"
        except KeyError as exc:
            return f"ERROR missing required arg: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"ERROR {type(exc).__name__}: {exc}"

    def _build_messages_for_model(self) -> list[dict[str, Any]]:
        """Construct the message list to send to the text-server.

        Includes (in order):
        1. System prompt + tool documentation.
        2. Skills retrieved by semantic similarity on the latest user
           message — same pattern as routers/chat.py:94-98.
        3. Replay of persisted workspace history.
        """
        rows = self._conn.execute(
            "SELECT role, content FROM workspace_messages "
            "WHERE workspace_id = ? ORDER BY id",
            (self._ws_id,),
        ).fetchall()

        # Skills injection — same call shape as routers/chat.py:94-98.
        last_user = next(
            (r["content"] for r in reversed(rows) if r["role"] == "user"),
            None,
        )
        skills_ctx = ""
        if last_user:
            try:
                skills = retrieve_skills(self._conn, last_user, top_k=3)
                skills_ctx = format_skills_for_context(skills) or ""
            except Exception:
                # Skills retrieval is best-effort; never fatal.
                skills_ctx = ""

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        if skills_ctx:
            messages.append({"role": "system", "content": skills_ctx})
        for row in rows:
            messages.append({"role": row["role"], "content": row["content"]})
        return messages

    def _persist_message(
        self, role: str, content: str, *, tool_calls: list[dict[str, Any]]
    ) -> None:
        """Insert one row into workspace_messages.

        Also bumps `last_active_at` on the workspace row. If this is the
        first user message in the workspace (i.e. the workspace's
        `summary` field is still empty), it's populated with the first
        80 chars of `content` — that's the design's "one-line summary"
        shown in the workspace list (design DoD item 6). No model call
        is needed; the user's first prompt is a reasonable hint.
        """
        self._conn.execute(
            "INSERT INTO workspace_messages "
            "(workspace_id, role, content, tool_calls_json) "
            "VALUES (?, ?, ?, ?)",
            (self._ws_id, role, content, json.dumps(tool_calls)),
        )
        if role == "user":
            self._conn.execute(
                "UPDATE workspaces "
                "SET summary = CASE WHEN summary = '' THEN ? ELSE summary END, "
                "    last_active_at = datetime('now') "
                "WHERE id = ?",
                (content[:80], self._ws_id),
            )
        else:
            self._conn.execute(
                "UPDATE workspaces SET last_active_at = datetime('now') WHERE id = ?",
                (self._ws_id,),
            )
        self._conn.commit()

    async def _default_model_fn(
        self, messages: list[dict[str, Any]]
    ) -> str:
        """Production model_fn: POST to the text-server /chat endpoint.

        Non-streaming for Phase 2 — we accumulate the full response
        before returning. Phase 3 introduces streaming via the chat.py
        _proxy_sse pattern.
        """
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{TEXT_SERVER_URL}/chat",
                json={"messages": messages, "stream": False},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("content") or data.get("text") or ""
```

**IMPORTANT:** The exact `/chat` request/response shape may differ from what's written above. Before writing this task's tests, verify the actual text-server endpoint shape:

```bash
cd web-app && grep -nE "POST|@app\.|@router\.(post|get)" ../qwen-text-server/server.py | head -10
```

If the text-server's `/chat` is streaming-only (SSE), adapt `_default_model_fn` to collect the SSE deltas into one string before returning. The tests in this task inject `model_fn` directly and don't exercise the real HTTP path, so they pass regardless — but the production code must match the real endpoint.

**Step 4: Run to verify pass**

Run: `cd web-app && python -m pytest tests/test_workspace_runner.py -v`
Expected: PASS, 3 tests.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/workspace_runner.py mlx-studio/web-app/tests/test_workspace_runner.py
git commit -m "feat(workspace): runner + dispatch loop (file ops + run_python)"
```
<!-- END_TASK_3 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_4 -->
### Task 4: Wire the runner into the workspace router

**Files:**
- Modify: `web-app/routers/workspace.py` — add `POST /workspace/{id}/messages` endpoint.
- Modify: `web-app/tests/test_workspace_router.py` — add end-to-end test for the new endpoint.

**Step 1: Write the failing test**

Append to `web-app/tests/test_workspace_router.py`:

```python
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
    conn: sqlite3.Connection,
    tmp_path: Path,
    stub_text_server,
) -> None:
    """POST /workspace/{id}/messages runs the model loop and persists messages."""
    client.post("/workspace/", data={"name": "test"}, follow_redirects=False)
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
```

**Step 2: Run to verify failure**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py::test_send_message_round_trips -v`
Expected: FAIL — endpoint doesn't exist (404).

**Step 3: Add the endpoint**

In `web-app/routers/workspace.py`, after the existing endpoints, add:

```python
from fastapi.responses import JSONResponse

from workspace_runner import WorkspaceRunner


@router.post("/{workspace_id}/messages")
async def send_message(
    workspace_id: int,
    request: Request,
    content: str = Form(...),
) -> JSONResponse:
    """Run one user→assistant turn synchronously.

    Phase 2 is non-streaming; Phase 3 adds the streaming SSE variant.
    Returns the final assistant content as JSON. Connection is read
    from request.app.state.db per the codebase-wide convention.
    """
    conn: sqlite3.Connection = request.app.state.db
    ws = workspace_store.get_workspace(conn, workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="workspace not found")

    runner = WorkspaceRunner(conn=conn, workspace_id=workspace_id)
    final = await runner.run_turn(content)
    return JSONResponse({"content": final})
```

Update the imports at the top of the file:

```python
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from workspace_runner import WorkspaceRunner
```

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_router.py -v`
Expected: All previous tests pass + new `test_send_message_round_trips` passes.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/routers/workspace.py mlx-studio/web-app/tests/test_workspace_router.py
git commit -m "feat(workspace): POST /messages endpoint runs the runner synchronously"
```
<!-- END_TASK_4 -->

---

**Phase 2 done when:**
- Tests pass: `test_workspace_tools.py` (~13), `test_workspace_runner_parser.py` (5), `test_workspace_runner.py` (3), `test_workspace_router.py::test_send_message_round_trips` (1) = 22+ new tests.
- The end-to-end test: a workspace receives a POST /messages, the stub model emits a `write_file` tool call, the file appears on disk, the conversation is persisted in `workspace_messages` with the right role sequence.
- Path-escape attempts surface as tool errors the model sees, not 500s.
- No browser interaction required — UI for chat comes in Phase 3.

**Phase 2 leaves these for later phases:** the streaming UI (Phase 3), cross-tab tools query_rag/generate_image (Phase 4), inline markdown/image rendering (Phase 5), checkpoint+revert (Phase 6).

"""
agent_tools.py — Agent tool implementations with filesystem allowlist enforcement.

All tools are pure async functions. Filesystem tools reject paths outside
allowed_dirs unconditionally at function entry — this is enforced by code,
not by prompt instruction.

Tools:
  filesystem_read(path, allowed_dirs)    → str
  filesystem_write(path, content, allowed_dirs) → str
  filesystem_list(path, allowed_dirs)    → str
  shell(command, timeout)                → str
  python_exec(code, timeout)             → str
  web_fetch(url, timeout)                → str
  call_model(prompt, client)             → str
  load_skill(name, db)                   → str
"""

import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx

TEXT_SERVER = "http://127.0.0.1:8766"
DEFAULT_SHELL_TIMEOUT = 30
DEFAULT_PYTHON_TIMEOUT = 10
DEFAULT_FETCH_TIMEOUT = 15


# ── Allowlist enforcement ─────────────────────────────────────────────────────

def _check_allowed(path_str: str, allowed_dirs: list[str]) -> tuple[bool, str]:
    """
    Return (True, resolved_path) if path is within an allowed directory,
    or (False, error_message) otherwise.

    Resolves symlinks before checking to prevent traversal attacks.
    """
    try:
        resolved = str(Path(path_str).resolve())
    except Exception as exc:
        return False, f"invalid path: {exc}"

    for allowed in allowed_dirs:
        try:
            allowed_resolved = str(Path(allowed).resolve())
        except Exception:
            continue
        if resolved == allowed_resolved or resolved.startswith(allowed_resolved + "/"):
            return True, resolved

    return (
        False,
        f"path '{path_str}' is outside the allowed directories: {allowed_dirs}",
    )


# ── Filesystem tools ──────────────────────────────────────────────────────────

async def filesystem_read(path: str, allowed_dirs: list[str]) -> str:
    """Read file contents. Rejects paths outside allowed_dirs."""
    ok, result = _check_allowed(path, allowed_dirs)
    if not ok:
        return f"ERROR: {result}"
    try:
        content = Path(result).read_text(encoding="utf-8", errors="replace")
        return content
    except FileNotFoundError:
        return f"ERROR: file not found: {path}"
    except IsADirectoryError:
        return f"ERROR: path is a directory, use filesystem_list: {path}"
    except Exception as exc:
        return f"ERROR: failed to read {path}: {exc}"


async def filesystem_write(path: str, content: str, allowed_dirs: list[str]) -> str:
    """Write content to file. Creates parent directories if needed. Rejects paths outside allowed_dirs."""
    ok, result = _check_allowed(path, allowed_dirs)
    if not ok:
        return f"ERROR: {result}"
    try:
        p = Path(result)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"ERROR: failed to write {path}: {exc}"


async def filesystem_list(path: str, allowed_dirs: list[str]) -> str:
    """List directory contents. Rejects paths outside allowed_dirs."""
    ok, result = _check_allowed(path, allowed_dirs)
    if not ok:
        return f"ERROR: {result}"
    try:
        p = Path(result)
        if not p.is_dir():
            return f"ERROR: not a directory: {path}"
        entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        lines = []
        for entry in entries:
            kind = "FILE" if entry.is_file() else "DIR "
            size = f"{entry.stat().st_size:>10d}" if entry.is_file() else "          "
            lines.append(f"{kind} {size}  {entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except Exception as exc:
        return f"ERROR: failed to list {path}: {exc}"


# ── Execution tools ───────────────────────────────────────────────────────────

async def shell(command: str, timeout: float = DEFAULT_SHELL_TIMEOUT) -> str:
    """Run a bash command with timeout. Returns combined stdout+stderr."""
    import asyncio

    def _run():
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = result.stdout
            if result.stderr:
                output = output + ("\n" if output else "") + result.stderr
            if result.returncode != 0:
                return f"exit {result.returncode}\n{output}"
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: command timed out after {timeout:.0f}s"
        except Exception as exc:
            return f"ERROR: {exc}"

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


async def python_exec(code: str, timeout: float = DEFAULT_PYTHON_TIMEOUT) -> str:
    """Execute Python code via subprocess. Returns stdout+stderr."""
    import asyncio

    def _run():
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = result.stdout
            if result.stderr:
                output = output + ("\n" if output else "") + result.stderr
            return output or "(no output)"
        except subprocess.TimeoutExpired:
            return f"ERROR: execution timed out after {timeout:.0f}s"
        except Exception as exc:
            return f"ERROR: {exc}"

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


# ── Network tools ─────────────────────────────────────────────────────────────

async def web_fetch(url: str, timeout: float = DEFAULT_FETCH_TIMEOUT) -> str:
    """Fetch a URL and return the response text (up to 8000 chars)."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            if resp.status_code >= 400:
                return f"ERROR: HTTP {resp.status_code} for {url}"
            text = resp.text[:8000]
            return text if text else "(empty response)"
    except httpx.TimeoutException:
        return f"ERROR: request timed out after {timeout:.0f}s"
    except httpx.HTTPError as exc:
        return f"ERROR: HTTP error: {exc}"
    except Exception as exc:
        return f"ERROR: {exc}"


async def web_search(query: str, timeout: float = DEFAULT_FETCH_TIMEOUT) -> str:
    """
    Search the web using DuckDuckGo instant answers API (no API key needed).
    Returns top result summaries.
    """
    import urllib.parse
    encoded = urllib.parse.quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded}&format=json&no_html=1&skip_disambig=1"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        # Extract meaningful fields from DDG instant answers
        results: list[str] = []
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
        for r in data.get("RelatedTopics", [])[:5]:
            if isinstance(r, dict) and r.get("Text"):
                results.append(r["Text"])
        return "\n\n".join(results) if results else f"No instant answer found for: {query}"
    except Exception as exc:
        return f"ERROR: search failed: {exc}"


# ── Model / skill tools ───────────────────────────────────────────────────────

async def call_model(prompt: str, client: httpx.AsyncClient) -> str:
    """Call the text server /complete endpoint synchronously (collects full response)."""
    try:
        async with client.stream(
            "POST",
            f"{TEXT_SERVER}/complete",
            json={"prompt": prompt, "max_tokens": 1024},
            timeout=120.0,
        ) as resp:
            resp.raise_for_status()
            tokens: list[str] = []
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    token = line[len("data: "):]
                    if token == "[DONE]":
                        break
                    tokens.append(token.replace("\\n", "\n"))
            return "".join(tokens)
    except Exception as exc:
        return f"ERROR: call_model failed: {exc}"


async def load_skill(name: str, conn: sqlite3.Connection) -> str:
    """
    Load a skill by name (case-insensitive partial match on name or filepath).
    Returns the skill content as a string.
    """
    rows = conn.execute(
        "SELECT filepath, name FROM skill_embeddings WHERE LOWER(name) LIKE ?",
        (f"%{name.lower()}%",),
    ).fetchall()
    if not rows:
        return f"ERROR: no skill found matching '{name}'"
    # Use first match
    filepath = rows[0]["filepath"]
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="replace")
        return f"=== Skill: {rows[0]['name']} ===\n{content}"
    except Exception as exc:
        return f"ERROR: could not read skill file: {exc}"


# ── Tool dispatch ─────────────────────────────────────────────────────────────

TOOL_DESCRIPTIONS = """
Available tools (call with JSON inside <tool>...</tool>):

<tool>{"tool": "filesystem_read", "args": {"path": "/abs/path/to/file"}}</tool>
<tool>{"tool": "filesystem_write", "args": {"path": "/abs/path", "content": "text"}}</tool>
<tool>{"tool": "filesystem_list", "args": {"path": "/abs/path/to/dir"}}</tool>
<tool>{"tool": "shell", "args": {"command": "ls -la"}}</tool>
<tool>{"tool": "python_exec", "args": {"code": "print(1+1)"}}</tool>
<tool>{"tool": "web_fetch", "args": {"url": "https://example.com"}}</tool>
<tool>{"tool": "web_search", "args": {"query": "search terms"}}</tool>
<tool>{"tool": "call_model", "args": {"prompt": "your prompt"}}</tool>
<tool>{"tool": "load_skill", "args": {"name": "skill name"}}</tool>

When you want to use a tool, include exactly one <tool>...</tool> block in your response.
When you are done with the task, do not include any tool call.
"""

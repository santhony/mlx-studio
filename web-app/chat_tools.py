"""
chat_tools.py — Tools exposed to the chat model via OpenAI tool-calling.

The names and signatures match what DeepSeek V4 Flash claims to have when
asked about its standard tools, so the model's natural inclinations
correspond to real, working capabilities:

  web_search(query, num_results=10)
  fetch_url(url, timeout=30, max_chars=50000)
  execute_python(code, timeout=60)
  read_file(file_path, encoding="utf-8")
  write_file(file_path, content, encoding="utf-8")
  list_files(path)

Filesystem tools share the Agents allowlist (settings.allowed_dir_*).
web_search uses DuckDuckGo HTML scrape (no key required).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import urllib.parse
from typing import Any

import httpx
from bs4 import BeautifulSoup

import agent_tools

DEFAULT_FETCH_TIMEOUT = 20
DEFAULT_FETCH_MAX_CHARS = 50_000
DEFAULT_PYTHON_TIMEOUT = 30
DEFAULT_SEARCH_RESULTS = 10

# OpenAI-format tool schemas. Names and parameter names match what the model
# expects to call — verified by asking DS4 to list its standard tools.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return a list of result titles, URLs, and snippets. Uses DuckDuckGo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {"type": "integer", "description": "Maximum number of results to return.", "default": DEFAULT_SEARCH_RESULTS},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch a URL and return its extracted readable text (HTML stripped). Truncated to max_chars.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch (http:// or https://)."},
                    "timeout": {"type": "integer", "description": "Request timeout in seconds.", "default": DEFAULT_FETCH_TIMEOUT},
                    "max_chars": {"type": "integer", "description": "Maximum characters of text to return.", "default": DEFAULT_FETCH_MAX_CHARS},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code in a subprocess and return stdout+stderr. Has no network or persistent state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to execute."},
                    "timeout": {"type": "integer", "description": "Execution timeout in seconds.", "default": DEFAULT_PYTHON_TIMEOUT},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the local filesystem. Paths outside the allowlist are refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text content to a file on the local filesystem (creating parent directories). Paths outside the allowlist are refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to write."},
                    "content": {"type": "string", "description": "Content to write."},
                    "encoding": {"type": "string", "description": "Text encoding.", "default": "utf-8"},
                },
                "required": ["file_path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the entries of a directory on the local filesystem. Paths outside the allowlist are refused.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the directory."},
                },
                "required": ["path"],
            },
        },
    },
]


# ── Allowlist lookup ─────────────────────────────────────────────────────────

def get_allowed_dirs(conn: sqlite3.Connection) -> list[str]:
    """Return the Agents-tab filesystem allowlist (settings.allowed_dir_*)."""
    rows = conn.execute(
        "SELECT value FROM settings WHERE key LIKE 'allowed_dir_%' ORDER BY key"
    ).fetchall()
    return [r["value"] for r in rows]


# ── New tool implementations ──────────────────────────────────────────────────

async def _web_search_ddg(query: str, num_results: int) -> str:
    """DuckDuckGo HTML search. No API key. Best-effort; layout-sensitive."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Apple Silicon) qwen-studio/1.0",
    }
    try:
        async with httpx.AsyncClient(timeout=DEFAULT_FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.post(url, data={"q": query}, headers=headers)
            if resp.status_code != 200:
                return f"ERROR: search returned HTTP {resp.status_code}"
            soup = BeautifulSoup(resp.text, "html.parser")
            results: list[dict[str, str]] = []
            for r in soup.select("div.result"):
                a = r.select_one("a.result__a")
                snippet = r.select_one("a.result__snippet, div.result__snippet")
                if not a:
                    continue
                href = a.get("href", "")
                # DDG wraps real URLs in a redirect; extract uddg= param
                parsed = urllib.parse.urlparse(href)
                qs = urllib.parse.parse_qs(parsed.query)
                real = qs.get("uddg", [href])[0]
                results.append({
                    "title": a.get_text(strip=True),
                    "url": real,
                    "snippet": snippet.get_text(strip=True) if snippet else "",
                })
                if len(results) >= num_results:
                    break
            if not results:
                return f"No results for: {query}"
            lines = []
            for i, r in enumerate(results, 1):
                lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
            return "\n\n".join(lines)
    except httpx.TimeoutException:
        return f"ERROR: search timed out"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: search failed: {exc}"


async def _fetch_url(url: str, timeout: int, max_chars: int) -> str:
    """HTTP GET, strip HTML to readable text, truncate."""
    if not (url.startswith("http://") or url.startswith("https://")):
        return f"ERROR: url must start with http:// or https://: {url}"
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 qwen-studio/1.0"})
            if resp.status_code >= 400:
                return f"ERROR: HTTP {resp.status_code} for {url}"
            ctype = resp.headers.get("content-type", "")
            body = resp.text
            if "html" in ctype.lower():
                soup = BeautifulSoup(body, "html.parser")
                for tag in soup(["script", "style", "noscript", "iframe"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = body
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n\n[…truncated at {max_chars} chars; full size {len(body)}]"
            return text or "(empty response)"
    except httpx.TimeoutException:
        return f"ERROR: fetch timed out after {timeout}s"
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: fetch failed: {exc}"


# ── Dispatcher ───────────────────────────────────────────────────────────────

async def dispatch_tool(name: str, raw_args: str, allowed_dirs: list[str]) -> str:
    """
    Execute a tool by name with JSON-encoded args. Returns a string suitable
    for sending back as the `tool` role message content. Never raises;
    failures are returned as `ERROR: ...` strings so the model can recover.
    """
    try:
        args = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError as exc:
        return f"ERROR: invalid JSON arguments to {name}: {exc}"
    if not isinstance(args, dict):
        return f"ERROR: tool arguments must be a JSON object, got {type(args).__name__}"

    if name == "web_search":
        query = args.get("query", "")
        n = int(args.get("num_results") or DEFAULT_SEARCH_RESULTS)
        if not query:
            return "ERROR: web_search requires a 'query' argument"
        return await _web_search_ddg(query, n)

    if name == "fetch_url":
        url = args.get("url", "")
        timeout = int(args.get("timeout") or DEFAULT_FETCH_TIMEOUT)
        max_chars = int(args.get("max_chars") or DEFAULT_FETCH_MAX_CHARS)
        if not url:
            return "ERROR: fetch_url requires a 'url' argument"
        return await _fetch_url(url, timeout, max_chars)

    if name == "execute_python":
        code = args.get("code", "")
        timeout = float(args.get("timeout") or DEFAULT_PYTHON_TIMEOUT)
        if not code:
            return "ERROR: execute_python requires 'code'"
        return await agent_tools.python_exec(code, timeout=timeout)

    if name == "read_file":
        path = args.get("file_path") or args.get("path") or ""
        if not path:
            return "ERROR: read_file requires 'file_path'"
        return await agent_tools.filesystem_read(path, allowed_dirs)

    if name == "write_file":
        path = args.get("file_path") or args.get("path") or ""
        content = args.get("content", "")
        if not path:
            return "ERROR: write_file requires 'file_path'"
        return await agent_tools.filesystem_write(path, content, allowed_dirs)

    if name == "list_files":
        path = args.get("path") or args.get("file_path") or ""
        if not path:
            return "ERROR: list_files requires 'path'"
        return await agent_tools.filesystem_list(path, allowed_dirs)

    return f"ERROR: unknown tool '{name}'. Available: {', '.join(t['function']['name'] for t in TOOL_SCHEMAS)}"

"""Tool-call parsing for workspace_runner.

# pattern: Functional Core
# Pure string→data transformation. Takes the raw assistant output and
# returns a structured (content, tool_calls) tuple. No I/O, no
# dependencies on FastAPI or httpx.
#
# Format matches what the text-server-backed models emit when prompted
# with the existing tool-call convention.
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

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

import bleach
import markdown


# Allowlist of safe HTML tags and attributes for sanitization.
# These are the tags that markdown + assistant-generated content should produce.
_ALLOWED_TAGS = {
    "p", "br", "strong", "em", "code", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li", "blockquote", "a", "img", "hr",
    "table", "thead", "tbody", "tr", "th", "td",
    "details", "summary", "div", "span",
}

_ALLOWED_ATTRS = {
    "a": ["href", "title"],
    "img": ["src", "alt", "title"],
    "div": ["class"],
    "span": ["class"],
    "details": ["class"],
}

# Protocols allowed in href/src attributes
_ALLOWED_PROTOCOLS = ["http", "https", "data", "mailto"]
_WS_IMG_RE = re.compile(
    # Match Markdown image refs where the URL is a relative path (no
    # scheme). We rewrite those to /workspace/{id}/file/{path}.
    r"!\[([^\]]*)\]\(([^)]+)\)"
)

# Reasoning-trace blocks. DOTALL because think bodies routinely span newlines.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_PLACEHOLDER = "xworkspacethinkblockx{n}xendthinkblockx"


def _extract_think_blocks(text: str) -> tuple[str, list[str]]:
    """Pull each <think>…</think> body out of `text`, replace with a placeholder.

    Returns (text_with_placeholders, list_of_raw_think_bodies). Placeholders
    are isolated on their own paragraphs so the markdown renderer leaves
    them intact for the post-render swap in `render_message`.
    """
    bodies: list[str] = []

    def take(match: re.Match[str]) -> str:
        idx = len(bodies)
        bodies.append(match.group(1))
        return f"\n\n{_THINK_PLACEHOLDER.format(n=idx)}\n\n"

    return _THINK_RE.sub(take, text), bodies


def _render_think_block(body: str) -> str:
    """Build a collapsible <details class="think"> with HTML-escaped body."""
    return (
        '<details class="think"><summary>thinking…</summary>'
        f"<pre>{html.escape(body.strip())}</pre></details>"
    )


def _rewrite_workspace_images(text: str, workspace_id: int) -> str:
    """Rewrite ![alt](relative.png) → ![alt](/workspace/{id}/file/relative.png).

    Absolute URLs (http/https/data:) are left alone.
    Leading-slash URLs treated as absolute and NOT rewritten; the model should
    use plain filenames like cat.png for workspace images.
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
    """Render one workspace message to HTML, with XSS protection via bleach.

    Creates a fresh markdown.Markdown instance per call to avoid thread-safety
    issues with shared state.
    """
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

    # Extract <think>…</think> blocks before markdown so their bodies are
    # not interpreted as markdown and can't smuggle HTML through bleach.
    # We re-insert them as pre-built <details> elements AFTER bleach.
    rewritten, think_bodies = _extract_think_blocks(rewritten)

    # Create a fresh markdown instance per call for thread safety
    md = markdown.Markdown(extensions=["fenced_code", "tables", "nl2br"])
    body_html = md.convert(rewritten)

    # Sanitize the rendered HTML to prevent XSS
    body_html = bleach.clean(
        body_html,
        tags=_ALLOWED_TAGS,
        attributes=_ALLOWED_ATTRS,
        protocols=_ALLOWED_PROTOCOLS,
        strip=True,  # Remove disallowed tags entirely
        strip_comments=True,
    )

    # Swap each think-block placeholder for its rendered collapsible.
    # Markdown wraps each placeholder in <p>…</p>; we replace the whole
    # paragraph so the <details> sits at block level, not inside a <p>.
    for idx, body in enumerate(think_bodies):
        placeholder = _THINK_PLACEHOLDER.format(n=idx)
        wrapped = f"<p>{placeholder}</p>"
        rendered = _render_think_block(body)
        if wrapped in body_html:
            body_html = body_html.replace(wrapped, rendered, 1)
        else:
            body_html = body_html.replace(placeholder, rendered, 1)

    try:
        tool_calls = json.loads(tool_calls_json) if tool_calls_json else []
    except json.JSONDecodeError:
        tool_calls = []
    tool_html = _render_tool_calls(tool_calls)

    return f'<div class="assistant-content">{body_html}</div>{tool_html}'

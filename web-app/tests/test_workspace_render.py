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


def test_script_tag_stripped() -> None:
    """Script tags embedded in assistant markdown must be stripped.

    The <script> tag is removed by bleach; the text content remains but is
    harmless when rendered as plain text inside a <p> tag.
    """
    content = "hello <script>alert(1)</script> world"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    # Ensure <script is not present (case-insensitive check via lowercase)
    assert "<script" not in html.lower()
    # The dangerous tag is gone; plain text may remain
    assert "alert" in html  # Text remains, but <script> context is lost


def test_onerror_attribute_stripped() -> None:
    """Event handlers and dangerous attributes must be stripped."""
    content = 'before <img src="x" onerror="alert(1)"> after'
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "onerror" not in html
    # The img tag itself may remain (if src is in allowlist), but without onerror


def test_javascript_url_stripped() -> None:
    """javascript: URLs must be stripped."""
    content = 'click [here](javascript:alert(1))'
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    # The markdown renders as a link, but bleach will remove the javascript: href
    # since href values aren't validated, we just check the word "alert" isn't there
    assert "javascript:" not in html


def test_safe_markdown_features_preserved() -> None:
    """Safe markdown features (bold, italic, code, headings, lists) remain after sanitization."""
    content = """# Heading
**bold text**
*italic*
`code`
- list item
```python
print('x')
```"""
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<h1>" in html
    assert "<strong>" in html or "<b>" in html  # markdown may use <b>
    assert "<em>" in html or "<i>" in html  # markdown may use <i>
    assert "<code>" in html
    assert "<li>" in html or "list item" in html
    assert "print" in html


def test_image_with_workspace_id_and_safe_attributes() -> None:
    """Images rewritten for workspace should preserve alt and title attributes."""
    content = "![alt text](image.png)"
    html = render_message(
        role="assistant",
        content=content,
        tool_calls_json="[]",
        workspace_id=42,
    )
    assert 'src="/workspace/42/file/image.png"' in html
    assert "alt text" in html


def test_leading_slash_url_not_rewritten() -> None:
    """Leading-slash URLs are treated as absolute and NOT rewritten."""
    content = "![x](/cat.png)"
    html = render_message(
        role="assistant",
        content=content,
        tool_calls_json="[]",
        workspace_id=42,
    )
    # Should NOT be rewritten to /workspace/42/file/...
    assert 'src="/cat.png"' in html
    assert "/workspace/42/file" not in html


def test_data_url_preserved_for_images() -> None:
    """data: URLs in images should be allowed (safe, embedded content)."""
    content = "![x](data:image/png;base64,iVBORw0KGg...)"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    # data: URLs should not be rewritten
    assert "data:image" in html


def test_html_comment_stripped() -> None:
    """HTML comments embedded in content must be stripped."""
    content = "before <!-- malicious comment --> after"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<!--" not in html
    assert "-->" not in html


def test_think_block_rendered_as_collapsible_details() -> None:
    """<think>...</think> in assistant content becomes a <details class="think">."""
    content = "Let me work through this.\n<think>step 1: read file\nstep 2: edit</think>\n\nDone."
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert '<details class="think">' in html
    assert "<summary>thinking" in html
    assert "step 1: read file" in html
    assert "step 2: edit" in html
    # Surrounding prose preserved
    assert "Let me work through this." in html
    assert "Done." in html
    # The raw <think> tag must not survive in the output
    assert "<think>" not in html


def test_multiple_think_blocks_each_collapsible() -> None:
    content = "<think>first</think>middle<think>second</think>end"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert html.count('<details class="think">') == 2
    assert "first" in html
    assert "second" in html
    assert "middle" in html


def test_think_block_contents_html_escaped() -> None:
    """HTML inside a <think> body must be escaped, not interpreted."""
    content = "<think><script>alert(1)</script> sneaky</think>after"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<script>" not in html
    # The escaped form should be visible inside the pre
    assert "&lt;script&gt;" in html or "alert(1)" in html
    assert "after" in html


def test_think_block_inside_details_not_nested_in_paragraph() -> None:
    """The <details> sits at block level — not inside a <p>."""
    content = "<think>just thinking</think>"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<p><details" not in html


def test_no_think_block_preserves_existing_markdown_behavior() -> None:
    """Content without <think> renders identically to before."""
    content = "**bold** and `code`"
    html = render_message(role="assistant", content=content, tool_calls_json="[]")
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html
    assert "details" not in html or "think" not in html

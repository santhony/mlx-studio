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

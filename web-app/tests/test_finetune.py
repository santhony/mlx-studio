"""Tests for finetune.py JSONL validation and metrics parsing."""

import json
import pytest
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from routers.finetune import _validate_jsonl, _parse_metric_line


class TestJsonlValidation:
    """Test JSONL file validation."""

    def test_valid_text_format(self):
        """Test valid JSONL with text format."""
        content = b'{"text": "hello world"}\n{"text": "foo bar"}\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is True
        assert error == ""
        assert count == 2

    def test_valid_prompt_completion_format(self):
        """Test valid JSONL with prompt/completion format."""
        content = b'{"prompt": "Q:", "completion": "A"}\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is True
        assert error == ""
        assert count == 1

    def test_valid_messages_format(self):
        """Test valid JSONL with messages format."""
        content = b'{"messages": [{"role": "user", "content": "hi"}]}\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is True
        assert error == ""
        assert count == 1

    def test_mixed_valid_formats(self):
        """Test JSONL with mixed valid formats."""
        content = (
            b'{"text": "a"}\n'
            b'{"prompt": "b", "completion": "c"}\n'
            b'{"messages": [{"role": "user", "content": "d"}]}\n'
        )
        valid, error, count = _validate_jsonl(content)
        assert valid is True
        assert error == ""
        assert count == 3

    def test_invalid_json(self):
        """Test JSONL with invalid JSON."""
        content = b'{"text": "hello"}\n{invalid json}\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is False
        assert "invalid JSON on line 2" in error
        assert count == 0

    def test_missing_required_key(self):
        """Test JSONL missing required keys."""
        content = b'{"text": "hello"}\n{"foo": "bar"}\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is False
        assert "line 2" in error
        assert "must have 'text', 'prompt', or 'messages' key" in error
        assert count == 0

    def test_empty_file(self):
        """Test empty JSONL file."""
        content = b''
        valid, error, count = _validate_jsonl(content)
        assert valid is False
        assert "no valid records" in error
        assert count == 0

    def test_file_with_only_whitespace(self):
        """Test JSONL with only whitespace and blank lines."""
        content = b'\n  \n\t\n'
        valid, error, count = _validate_jsonl(content)
        assert valid is False
        assert "no valid records" in error
        assert count == 0

    def test_blank_lines_ignored(self):
        """Test that blank lines are ignored in counting."""
        content = b'{"text": "a"}\n\n{"text": "b"}\n  \n'
        valid, error, count = _validate_jsonl(content)
        assert valid is True
        assert count == 2


class TestMetricsParsingAndFormatting:
    """Test metrics line parsing and HTML formatting."""

    def test_parse_train_metric_line(self):
        """Test parsing training metric line."""
        line = "Iter 100: Train loss 0.5234, Learning Rate 2e-4, It/sec 0.5, Tokens/sec 256, Trained Tokens 51200, Peak mem 23.45 GB"
        metric = _parse_metric_line(line)
        assert metric is not None
        assert metric["type"] == "train"
        assert metric["iteration"] == 100
        assert metric["loss"] == 0.5234
        assert metric["lr"] == 2e-4
        assert metric["tokens_sec"] == 256
        assert metric["peak_mem"] == 23.45

    def test_parse_val_metric_line(self):
        """Test parsing validation metric line."""
        line = "Iter 100: Val loss 0.6789"
        metric = _parse_metric_line(line)
        assert metric is not None
        assert metric["type"] == "val"
        assert metric["iteration"] == 100
        assert metric["loss"] == 0.6789
        # Val metrics don't have these fields
        assert "tokens_sec" not in metric
        assert "peak_mem" not in metric

    def test_parse_non_metric_line(self):
        """Test that non-metric lines return None."""
        line = "Loading model..."
        metric = _parse_metric_line(line)
        assert metric is None

    def test_metric_row_html_with_numeric_mem_avail(self):
        """Test _metric_row_html with numeric mem_available_gb (training metrics)."""
        from routers.finetune import _metric_row_html

        metric = {
            "type": "train",
            "iteration": 100,
            "loss": 0.5234,
            "tokens_sec": 256.0,
            "peak_mem": 23.45,
            "mem_available_gb": 42.5,  # numeric
        }
        html = _metric_row_html(metric)
        assert "100" in html
        assert "0.5234" in html
        assert "256.0" in html
        assert "23.45" in html
        assert "42.5" in html  # should be formatted as "42.5 GB"
        assert "42.5 GB" in html

    def test_metric_row_html_with_string_mem_avail(self):
        """Test _metric_row_html with string mem_available_gb (validation metrics)."""
        from routers.finetune import _metric_row_html

        metric = {
            "type": "val",
            "iteration": 100,
            "loss": 0.6789,
            "tokens_sec": None,
            "peak_mem": None,
            "mem_available_gb": "?",  # string (for val metrics)
        }
        html = _metric_row_html(metric)
        assert "100" in html
        assert "0.6789" in html
        # mem_available should be rendered as "—" (em-dash) not crash with TypeError
        assert "—" in html
        # Should not have "? GB" or other malformed output
        assert "? GB" not in html

    def test_metric_row_html_with_missing_mem_avail(self):
        """Test _metric_row_html when mem_available_gb is missing."""
        from routers.finetune import _metric_row_html

        metric = {
            "type": "val",
            "iteration": 50,
            "loss": 0.7,
            "tokens_sec": None,
            "peak_mem": None,
            # mem_available_gb is missing entirely
        }
        html = _metric_row_html(metric)
        assert "50" in html
        assert "0.7000" in html
        # Should render missing value as "—" not crash
        assert "—" in html

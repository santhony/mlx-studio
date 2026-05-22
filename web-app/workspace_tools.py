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
import tempfile
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

    Also rejects paths inside the reserved `.checkpoints/` directory
    (used for snapshot storage by the checkpoint system).
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
    # Block access to reserved .checkpoints/ directory
    rel = candidate.relative_to(root)
    if rel.parts and rel.parts[0] == ".checkpoints":
        raise PathEscapeError(
            f"path {rel_path!r} targets reserved .checkpoints/ directory"
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

    The code is written to a unique temp file inside the workspace, run via
    `python3` with cwd set to the workspace root, and the temp file is
    removed afterward. Returns stdout (last ~4000 chars), stderr (last ~4000
    chars), exit_code, or an `error` key on timeout / spawn failure.

    Note: Output is truncated to the last 4000 characters per stream to avoid
    returning extremely large responses. Use intermediate print() calls to see
    earlier output if needed.
    """
    # Create a unique temp file inside the workspace to avoid race conditions
    # when multiple runs happen concurrently.
    with tempfile.NamedTemporaryFile(
        suffix=".py", dir=str(root), delete=False, mode="w"
    ) as f:
        f.write(code)
        script_path = f.name

    script = Path(script_path)
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

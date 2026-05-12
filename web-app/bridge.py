"""
bridge.py — Connects DeepSeek (remote reasoning) to Qwen Studio's agent sandbox.

Protocol:
  DeepSeek outputs <tool>{"tool":"...","args":{...}}</tool> blocks
  Bridge parses, executes via agent_tools, returns results as structured messages.

Execution modes:
  - auto (no approval gate — for safe tools like filesystem_read)
  - supervised (requires user approval for write/shell/python_exec)
  - interactive (approval required for every tool)

Usage as a library:
  bridge = Bridge(conn, http_client, allowed_dirs)
  result = await bridge.process_message(deepseek_message)
"""

import asyncio
import json
import logging
import re
from typing import Any, Optional

import httpx

from agent_tools import (
    filesystem_read,
    filesystem_write,
    filesystem_list,
    shell,
    python_exec,
    web_fetch,
    web_search,
    call_model,
    load_skill,
)

log = logging.getLogger("bridge")

# Tool categories for approval gating
CATEGORY_SAFE = {"filesystem_read", "filesystem_list", "web_search", "web_fetch", "load_skill"}
CATEGORY_UNSAFE = {"filesystem_write", "shell", "python_exec", "call_model"}


class Bridge:
    """
    Bridge between chat model output and sandbox execution.

    Parses <tool>...{tool, args}...{/tool} blocks from text,
    dispatches to agent_tools, and formats results.
    """

    def __init__(
        self,
        allowed_dirs: list[str],
        http_client: Optional[httpx.AsyncClient] = None,
        mode: str = "supervised",
    ):
        self.allowed_dirs = allowed_dirs
        self.http_client = http_client or httpx.AsyncClient()
        self.mode = mode  # auto | supervised | interactive

        # Session-level tool allowlist (tools approved once per session)
        self.session_allowed_tools: set[str] = set()

        # Pending approval callback: (tool_name, args_json) -> bool
        self.approval_callback = None

    # ── Parsing ────────────────────────────────────────────────────────

    def parse_tool_call(self, text: str) -> Optional[dict[str, Any]]:
        """Extract first <tool>...{/tool} block and return parsed JSON."""
        match = re.search(r"<tool>(.*?)</tool>", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(1).strip())
            if "tool" in data and "args" in data:
                return data
            return None
        except json.JSONDecodeError:
            return None

    def strip_tool_blocks(self, text: str) -> str:
        """Remove tool blocks from text, leaving only reasoning."""
        return re.sub(r"<tool>.*?</tool>", "", text, flags=re.DOTALL).strip()

    # ── Dispatch ───────────────────────────────────────────────────────

    async def execute(self, tool_call: dict) -> str:
        """
        Execute a tool call dict and return string result.

        Raises on missing required args. Returns error strings for
        tool-level failures (never raises normally).
        """
        name = tool_call["tool"]
        args = tool_call.get("args", {})

        dispatch = {
            "filesystem_read": lambda: filesystem_read(
                self._req(args, "path"), self.allowed_dirs
            ),
            "filesystem_write": lambda: filesystem_write(
                self._req(args, "path"),
                self._req(args, "content"),
                self.allowed_dirs,
            ),
            "filesystem_list": lambda: filesystem_list(
                self._req(args, "path"), self.allowed_dirs
            ),
            "shell": lambda: shell(
                self._req(args, "command"),
                timeout=float(args.get("timeout", 30)),
            ),
            "python_exec": lambda: python_exec(
                self._req(args, "code"),
                timeout=float(args.get("timeout", 30)),
            ),
            "web_fetch": lambda: web_fetch(
                self._req(args, "url"),
                timeout=float(args.get("timeout", 15)),
            ),
            "web_search": lambda: web_search(
                self._req(args, "query"),
                timeout=float(args.get("timeout", 15)),
            ),
            "call_model": lambda: call_model(
                self._req(args, "prompt"), self.http_client
            ),
            "load_skill": lambda: load_skill(
                self._req(args, "name"), self._conn
            ),
        }

        fn = dispatch.get(name)
        if fn is None:
            return f"ERROR: unknown tool '{name}'"

        try:
            result = await fn()
            return str(result)
        except KeyError as exc:
            return f"ERROR: missing required argument: {exc}"
        except Exception as exc:
            return f"ERROR: {name} failed: {exc}"

    def _req(self, args: dict, key: str) -> Any:
        """Get required key or raise KeyError."""
        val = args.get(key)
        if val is None:
            raise KeyError(f"'{key}' is required but not in args: {json.dumps(args)}")
        return val

    def _conn(self, _name):
        """Placeholder — load_skill needs a db connection. Stub for now."""
        return None

    # ── Message-level processing ───────────────────────────────────────

    async def process_message(
        self, message: str, approval_grant: Optional[bool] = None
    ) -> dict:
        """
        Process a single DeepSeek message. Returns:

        {
            "reasoning": str,          # Non-tool text
            "tool_call": dict | None,  # Parsed tool call if found
            "result": str | None,      # Execution result if tool was executed
            "needs_approval": bool,    # True if tool is unsafe and no session approval
            "tool_name": str | None,   # tool name if tool was found
        }
        """
        reasoning = self.strip_tool_blocks(message)
        tool_call = self.parse_tool_call(message)

        if tool_call is None:
            return {"reasoning": reasoning, "tool_call": None, "result": None, "needs_approval": False, "tool_name": None}

        name = tool_call["tool"]
        category_safe = name in CATEGORY_SAFE
        category_unsafe = name in CATEGORY_UNSAFE

        # Determine if approval is needed
        needs_approval = False
        if self.mode == "interactive":
            needs_approval = True
        elif self.mode == "supervised" and category_unsafe:
            # Unsafe tools need approval unless session-allowed
            if name not in self.session_allowed_tools:
                needs_approval = True

        # If approval needed and not granted, block
        if needs_approval and approval_grant is not True:
            return {
                "reasoning": reasoning,
                "tool_call": tool_call,
                "result": None,
                "needs_approval": True,
                "tool_name": name,
            }

        # Execute
        result = await self.execute(tool_call)

        # Auto-allow for session if mode is auto
        if self.mode == "auto":
            self.session_allowed_tools.add(name)
        elif approval_grant is True and name not in self.session_allowed_tools:
            # Mark as session-allowed after successful execution with approval
            self.session_allowed_tools.add(name)

        return {
            "reasoning": reasoning,
            "tool_call": tool_call,
            "result": result,
            "needs_approval": False,
            "tool_name": name,
        }

    # ── Convenience ────────────────────────────────────────────────────

    def tool_block(self, tool_name: str, args: dict) -> str:
        """Generate a <tool> block for the model to output."""
        return f"<tool>{json.dumps({'tool': tool_name, 'args': args})}</tool>"

    def format_result(self, tool_name: str, result: str) -> str:
        """Format a tool result for injection back into the model's context."""
        return f"Tool '{tool_name}' result:\n{result}"


# ── Standalone test / demo ───────────────────────────────────────────

async def demo():
    """Run a quick self-test of the bridge's parsing and dispatch."""
    print("=== Bridge Self-Test ===\n")

    # Test with real allowed_dirs from the DB
    allowed = [
        "/Users/santhony/Documents/dev_claude/qwen-studio/data/skills",
        "/Users/santhony/Documents/dev_claude/qwen-studio/data/workspace",
        "/Users/santhony/Documents/dev_claude",
    ]

    bridge = Bridge(allowed_dirs=allowed, mode="auto")

    # Test 1: parsing
    msg = "Let me check the project structure.\n<tool>{\"tool\": \"filesystem_list\", \"args\": {\"path\": \"/Users/santhony/Documents/dev_claude/qwen-studio/web-app\"}}</tool>"
    print("1. Parse test:")
    parsed = bridge.parse_tool_call(msg)
    print(f"   Parsed: {parsed}")
    assert parsed is not None and parsed["tool"] == "filesystem_list"

    # Test 2: strip reasoning
    reasoning = bridge.strip_tool_blocks(msg)
    print(f"   Reasoning: {reasoning!r}")
    assert "Let me check" in reasoning

    # Test 3: dispatch filesystem_read
    print("\n2. Dispatch test (filesystem_list):")
    result = await bridge.execute({"tool": "filesystem_list", "args": {"path": "/Users/santhony/Documents/dev_claude/qwen-studio"}})
    print(f"   Result:\n{result}")
    assert "web-app" in result

    # Test 4: dispatch filesystem_read on a known file
    print("\n3. Dispatch test (filesystem_read):")
    result = await bridge.execute({"tool": "filesystem_read", "args": {"path": "/Users/santhony/Documents/dev_claude/qwen-studio/CLAUDE.md"}})
    print(f"   Result: {result[:200]}...")
    assert "# Qwen Studio" in result

    # Test 5: python_exec
    print("\n4. Dispatch test (python_exec):")
    result = await bridge.execute({"tool": "python_exec", "args": {"code": "print(sum(range(10)))"}})
    print(f"   Result: {result}")
    assert "45" in result

    print("\n=== All tests passed! ===")


if __name__ == "__main__":
    asyncio.run(demo())

"""
bridge_integration_test.py — End-to-end test of the bridge.

Tests the full cycle:
  1. Parse a DeepSeek message with a tool call
  2. Execute via agent_tools
  3. Format result
  4. Handle approval gating

Run: python bridge_integration_test.py
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bridge import Bridge, CATEGORY_SAFE, CATEGORY_UNSAFE


async def test_full_cycle():
    allowed = [
        "/Users/santhony/Documents/dev_claude/mlx-studio/data/skills",
        "/Users/santhony/Documents/dev_claude/mlx-studio/data/workspace",
        "/Users/santhony/Documents/dev_claude",
    ]

    print("=" * 60)
    print("BRIDGE INTEGRATION TEST")
    print("=" * 60)

    print("\n--- Step 1: Create bridge in supervised mode ---")
    bridge = Bridge(allowed_dirs=allowed, mode="supervised")
    print("Bridge created.")

    print("\n--- Step 2: Simulate DeepSeek reasoning with a tool call ---")
    msg = """Let me check the workspace directory to understand the project structure.

<tool>{"tool": "filesystem_list", "args": {"path": "/Users/santhony/Documents/dev_claude/mlx-studio/data/workspace"}}</tool>

Based on the contents, I'll plan the next steps.
"""
    print(f"Input message:\n{msg[:100]}...")

    print("\n--- Step 3: Process message (safe tool, no approval needed) ---")
    result = await bridge.process_message(msg)
    print(f"  reasoning: {result['reasoning'][:50]}...")
    print(f"  tool_call: {result['tool_call']}")
    print(f"  needs_approval: {result['needs_approval']}")
    print(f"  result (first 200 chars): {result['result']}")

    assert not result["needs_approval"]
    assert result["tool_call"]["tool"] == "filesystem_list"
    assert result["result"] is not None
    print("  ✓ Safe tool auto-executed")

    print("\n--- Step 4: Simulate an unsafe tool (shell) ---")
    msg2 = """Now let me install the required packages.

<tool>{"tool": "shell", "args": {"command": "echo 'dry run - would install packages'"}}</tool>
"""
    result2 = await bridge.process_message(msg2, approval_grant=False)
    print(f"  needs_approval: {result2['needs_approval']}")
    print(f"  result is None: {result2['result'] is None}")
    assert result2["needs_approval"]
    assert result2["result"] is None
    print("  ✓ Unsafe tool blocked, awaiting approval")

    print("\n--- Step 5: Approve the tool and re-execute ---")
    result3 = await bridge.process_message(msg2, approval_grant=True)
    print(f"  needs_approval: {result3['needs_approval']}")
    print(f"  result: {result3['result']}")
    assert not result3["needs_approval"]
    assert "echo" in str(result3.get("result", ""))
    print("  ✓ Approved tool executed")

    print("\n--- Step 6: Demonstrate session allowlist ---")
    print("  (shell was approved once, now it's in session_allowed_tools)")
    result4 = await bridge.process_message(msg2, approval_grant=False)
    print(f"  needs_approval: {result4['needs_approval']}")
    print(f"  result: {result4['result']}")
    assert not result4["needs_approval"]
    print("  ✓ Session allowlist works — same tool auto-approved")

    print("\n--- Step 7: Edit a file via filesystem_write ---")
    msg3 = """Let me create a test file.

<tool>{"tool": "filesystem_write", "args": {"path": "/Users/santhony/Documents/dev_claude/test_from_bridge.txt", "content": "Created by bridge integration test"}}</tool>
"""
    result5 = await bridge.process_message(msg3, approval_grant=True)
    print(f"  result: {result5['result']}")
    # Verify it was written
    from_path = "/Users/santhony/Documents/dev_claude/test_from_bridge.txt"
    content = Path(from_path).read_text() if Path(from_path).exists() else "(not found)"
    print(f"  File contents: {content}")
    assert "Created by bridge" in content
    print("  ✓ File written and readable")

    print("\n--- Step 8: Python execution ---")
    msg4 = """Running code analysis.

<tool>{"tool": "python_exec", "args": {"code": "import sys; print(f'Python {sys.version}')"}}</tool>
"""
    result6 = await bridge.process_message(msg4, approval_grant=True)
    print(f"  result: {result6['result']}")
    assert "Python" in str(result6.get("result", ""))
    print("  ✓ Python executed")

    print("\n--- Step 9: Message with no tool call ---")
    msg5 = "All tasks complete. Ready for the next assignment."
    result7 = await bridge.process_message(msg5)
    print(f"  result: {result7}")
    assert result7["tool_call"] is None
    assert result7["result"] is None
    print("  ✓ No-tool message handled gracefully")

    print("\n--- Step 10: Interactive mode (approve even safe tools) ---")
    bridge_interactive = Bridge(allowed_dirs=allowed, mode="interactive")
    msg_safe = '<tool>{"tool": "filesystem_list", "args": {"path": "/Users/santhony/Documents/dev_claude"}}</tool>'
    r_interactive = await bridge_interactive.process_message(msg_safe, approval_grant=False)
    print(f"  needs_approval: {r_interactive['needs_approval']}")
    assert r_interactive["needs_approval"]
    print("  ✓ Interactive mode blocks even safe tools")

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_full_cycle())

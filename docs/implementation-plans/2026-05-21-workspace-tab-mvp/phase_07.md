# Workspace Tab MVP — Phase 7: Cleanup

**Goal:** The vestigial Notebook and Agent tabs are removed entirely. Their database tables are dropped. Nav is clean. Tests still pass. The CLAUDE.md reflects the new tab inventory.

**Architecture:** Pure deletion. No new code beyond a migration block that drops the four tables.

**Tech Stack:** No new dependencies.

**Scope:** Phase 7 of 7.

**Codebase verified:** 2026-05-21. Files to remove confirmed to exist: `routers/notebook.py` (395 lines), `routers/agents.py` (402 lines), `agent_tools.py` (274 lines), `templates/notebook.html`, `templates/notebook_list.html`, `templates/cell.html`, `templates/agents.html`, `templates/agent_job.html`. Tables to drop: `notebooks`, `cells`, `agent_jobs`, `agent_steps`.

---

<!-- START_TASK_1 -->
### Task 1: Remove tab files

**Files removed:**
- `web-app/routers/notebook.py`
- `web-app/routers/agents.py`
- `web-app/agent_tools.py`
- `web-app/templates/notebook.html`
- `web-app/templates/notebook_list.html`
- `web-app/templates/cell.html`
- `web-app/templates/agents.html`
- `web-app/templates/agent_job.html`
- `web-app/tests/test_agent_tools.py` (if present)
- Any other notebook/agent-specific test files (check `tests/` for matches before deleting)

**Step 1: Identify all files to delete**

Run: `cd web-app && find . -type f \( -name "notebook*" -o -name "agent*" \) -not -path "*/__pycache__/*" -not -path "*/venv*"`
Capture the list before deleting.

**Step 2: Delete the files**

```bash
cd /Users/santhony/Documents/dev_claude/mlx-studio/web-app
rm routers/notebook.py routers/agents.py agent_tools.py
rm templates/notebook.html templates/notebook_list.html templates/cell.html
rm templates/agents.html templates/agent_job.html
# Delete any matching test files (verify with the find above first):
rm tests/test_agent_tools.py 2>/dev/null || true
```

**Step 3: Remove router registrations from main.py**

In `web-app/main.py`, remove the import lines for notebook and agents:

```python
# Remove these two lines from the import block:
from routers import notebook as notebook_router
from routers import agents as agents_router
```

And the include_router lines:

```python
# Remove these two lines from the registration block:
app.include_router(notebook_router.router)
app.include_router(agents_router.router)
```

**Step 4: Remove nav links from base.html**

In `web-app/templates/base.html` lines 31-38, remove the two lines:

```html
<a href="/notebook">Notebook</a>
<a href="/agents">Agents</a>
```

**Step 5: Confirm no orphan references**

Run: `cd web-app && grep -rn "from routers import notebook\|from routers import agents\|agent_tools\|from notebook\|from agents" --include="*.py" --include="*.html" .`
Expected: no results (all references removed).

Run: `cd web-app && grep -rn "/notebook\|/agents" --include="*.html" .`
Expected: no results (or only in any documentation/CLAUDE.md files that we'll update next).

**Step 6: Run tests**

Run: `cd web-app && python -m pytest tests/ -v`
Expected: All tests still pass (workspace tests + any other surviving tests).

**Step 7: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add -A mlx-studio/web-app/
git commit -m "chore(workspace): remove vestigial Notebook + Agent tabs"
```
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Drop the four tables in db.py

**Files:**
- Modify: `web-app/db.py` — add `DROP TABLE IF EXISTS` for the four tables in the executescript block.

**Step 1: Add the drops**

In `web-app/db.py`, BEFORE the `CREATE TABLE` blocks (so the drops are idempotent and happen first if the tables exist):

```sql
-- Phase 7 cleanup: vestigial Notebook + Agent tables removed
DROP TABLE IF EXISTS cells;
DROP TABLE IF EXISTS notebooks;
DROP TABLE IF EXISTS agent_steps;
DROP TABLE IF EXISTS agent_jobs;
```

Also remove the `CREATE TABLE IF NOT EXISTS notebooks/cells/agent_jobs/agent_steps` blocks from `init_schema()` so a fresh database doesn't re-create them.

**Step 2: Manual verify (development DB)**

For the developer's local DB at `web-app/data/studio.db`:

```bash
cd /Users/santhony/Documents/dev_claude/mlx-studio/web-app
sqlite3 data/studio.db ".tables"  # before
python -c "from db import init_schema; import sqlite3; conn = sqlite3.connect('data/studio.db'); init_schema(conn); conn.close()"
sqlite3 data/studio.db ".tables"  # after — notebooks/cells/agent_jobs/agent_steps should be gone
```

**Step 3: Add a regression test**

In `web-app/tests/test_workspace_schema.py`, add:

```python
def test_vestigial_tables_dropped(conn: sqlite3.Connection) -> None:
    """init_schema drops the old Notebook + Agent tables."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    table_names = {row[0] for row in cursor}
    for old in ("notebooks", "cells", "agent_jobs", "agent_steps"):
        assert old not in table_names, f"vestigial table still present: {old}"
```

**Step 4: Run tests**

Run: `cd web-app && python -m pytest tests/test_workspace_schema.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/db.py mlx-studio/web-app/tests/test_workspace_schema.py
git commit -m "chore(workspace): drop vestigial notebooks/cells/agent_jobs/agent_steps tables"
```
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Update CLAUDE.md

**Files:**
- Modify: `web-app/CLAUDE.md` — update freshness date, tabs list, "Status" section.
- Modify: `mlx-studio/CLAUDE.md` if it references Notebook/Agents.

**Step 1: Update web-app/CLAUDE.md**

Find the section that lists tabs / routes / status. Replace any "Notebook" and "Agents" references with "Workspace" where appropriate. Update the freshness date at top to `2026-05-21`.

Add a short section describing the Workspace tab (one paragraph). Mention:
- It replaces Notebook + Agent (now removed)
- The auto-execute + checkpoint UX
- The cross-tab tools (query_rag, generate_image)
- The runner module (workspace_runner.py) + tool registry

**Step 2: Update top-level mlx-studio/CLAUDE.md**

Specific lines to update (verified 2026-05-21):

- The "Database Tables" enumeration line that includes `notebooks, cells, agent_jobs, agent_steps` — replace those four entries with `workspaces, workspace_messages, workspace_checkpoints`.
- The `routers/` description that lists `notebook.py` and `agents.py` — remove those entries; add `workspace.py`.
- Any mention of "Notebook tab" / "Agent tab" — replace with "Workspace tab" (or remove if context-specific).
- Bump the "Last verified" date at the top of the file to today.

Find all such lines:

```bash
cd /Users/santhony/Documents/dev_claude/mlx-studio
grep -n "notebook\|notebooks\|agents\|agent_jobs\|agent_steps\|cells" CLAUDE.md
```

Review each line manually and update. Do NOT replace "agents" globally — some references may legitimately be about agent concepts elsewhere (e.g., subagent, agent dispatcher pattern in design plans). Use judgment.

**Step 3: Commit**

```bash
cd /Users/santhony/Documents/dev_claude
git add mlx-studio/web-app/CLAUDE.md mlx-studio/CLAUDE.md
git commit -m "docs(workspace): refresh CLAUDE.md for the consolidated Workspace tab"
```
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Final smoke test

**Step 1: Full test suite**

```bash
cd /Users/santhony/Documents/dev_claude/mlx-studio/web-app
python -m pytest -v
```

Expected: All tests pass (workspace_* tests + any other surviving tests).

**Step 2: App boot**

```bash
cd /Users/santhony/Documents/dev_claude/mlx-studio
./start.sh
```

Expected: No errors during startup.

**Step 3: Manual smoke (the design's Definition of Done)**

In a browser:
1. Navigate to `http://127.0.0.1:8080/workspace/`. See the workspace list (empty if fresh).
2. Confirm the nav no longer shows "Notebook" or "Agents".
3. Confirm `/notebook` and `/agents` URLs return 404 (or whatever the FastAPI default is for unmounted routes).
4. Create a new workspace named "smoke test".
5. In the chat, ask the model to query a RAG corpus you have indexed (e.g., "Query corpus 1 for 'introduction' and summarize what you found in a markdown bullet list").
6. Then ask: "Generate a header image to go with that summary."
7. Then ask: "Write the summary plus the image reference into a file called writeup.md."
8. Verify: writeup.md appears in `data/workspaces/<id>/`, contains a markdown summary plus `![...](image-filename.png)`, and the image embeds inline in the chat when the model references it.
9. Click "Revert to before this" on the writeup-creation user message; verify the file disappears.

**Step 4: Commit (only if any final fixes were needed)**

```bash
cd /Users/santhony/Documents/dev_claude
git status
# If there are any final fixes:
git add ...
git commit -m "fix(workspace): final smoke-test fixes"
```
<!-- END_TASK_4 -->

---

**Phase 7 done when:**
- 0 references to `routers.notebook`, `routers.agents`, or `agent_tools` anywhere in the codebase.
- 4 tables dropped + tested (regression test passes).
- Full `pytest` green.
- Manual smoke test of the Definition-of-Done use case from the design plan succeeds.
- CLAUDE.md files reflect the consolidated state.

**MVP complete when Phase 7 is done.**

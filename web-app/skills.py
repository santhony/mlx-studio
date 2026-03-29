"""
skills.py — Skills filesystem watcher and semantic retrieval.

Functional core: pure functions for embedding, storing, and retrieving skills.
Side effects (HTTP calls, SQLite writes, file watching) are isolated to specific
functions and the SkillsWatcher class.

Skills source of truth: data/skills/*.md
Database cache: skill_embeddings table in studio.db

Each skill file may have optional YAML frontmatter:
    ---
    name: My Skill
    description: Short description for retrieval display
    ---
    # Skill content...
"""

import logging
import sqlite3
import struct
import threading
from pathlib import Path
from typing import Optional

import frontmatter
import httpx
import numpy as np
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

log = logging.getLogger("skills")

TEXT_SERVER = "http://127.0.0.1:8766"


# ── Frontmatter parsing ───────────────────────────────────────────────────────

def _parse_skill_file(path: Path) -> dict:
    """
    Parse a .md skill file into a dict with keys:
      filepath, name, description, content (full text for embedding)
    """
    try:
        post = frontmatter.load(str(path))
        name = post.metadata.get("name") or path.stem.replace("-", " ").replace("_", " ").title()
        description = post.metadata.get("description") or ""
        content = post.content.strip()
    except Exception:
        # If parsing fails, treat entire file as plain text
        raw = path.read_text(encoding="utf-8", errors="replace")
        name = path.stem.replace("-", " ").replace("_", " ").title()
        description = ""
        content = raw

    return {
        "filepath": str(path),
        "name": name,
        "description": description,
        "content": content,
    }


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_text(text: str) -> Optional[list[float]]:
    """
    Call the text server /embed endpoint synchronously.
    Returns a list of floats or None on failure.
    """
    try:
        resp = httpx.post(
            f"{TEXT_SERVER}/embed",
            json={"text": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as exc:
        log.warning("embed request failed: %s", exc)
        return None


def _vec_to_blob(vec: list[float]) -> bytes:
    """Serialize a float list to a float32 BLOB for SQLite storage."""
    arr = np.array(vec, dtype=np.float32)
    return arr.tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    """Deserialize a float32 BLOB from SQLite to a numpy array."""
    return np.frombuffer(blob, dtype=np.float32)


# ── Database operations ───────────────────────────────────────────────────────

def upsert_skill(conn: sqlite3.Connection, path: Path) -> bool:
    """
    Parse, embed, and upsert a skill file into skill_embeddings.
    Returns True on success, False if embedding failed.
    """
    skill = _parse_skill_file(path)
    mtime = path.stat().st_mtime

    # Check if file has changed since last embed
    existing = conn.execute(
        "SELECT mtime FROM skill_embeddings WHERE filepath = ?",
        (skill["filepath"],),
    ).fetchone()
    if existing and abs(existing["mtime"] - mtime) < 0.01:
        log.debug("skill unchanged, skipping: %s", path.name)
        return True

    # Embed the skill content (use name + description + content for richer retrieval)
    embed_text = f"{skill['name']}\n{skill['description']}\n{skill['content']}"
    vec = _embed_text(embed_text)
    if vec is None:
        log.warning("failed to embed skill: %s", path.name)
        return False

    blob = _vec_to_blob(vec)
    conn.execute(
        """
        INSERT INTO skill_embeddings (filepath, name, mtime, embedding_blob)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(filepath) DO UPDATE SET
            name = excluded.name,
            mtime = excluded.mtime,
            embedding_blob = excluded.embedding_blob
        """,
        (skill["filepath"], skill["name"], mtime, blob),
    )
    conn.commit()
    log.info("skill embedded: %s", path.name)
    return True


def remove_skill(conn: sqlite3.Connection, filepath: str) -> None:
    """Remove a skill from the embeddings cache."""
    conn.execute("DELETE FROM skill_embeddings WHERE filepath = ?", (filepath,))
    conn.commit()
    log.info("skill removed: %s", Path(filepath).name)


def embed_all_skills(conn: sqlite3.Connection, skills_dir: Path) -> None:
    """
    Embed all .md files in skills_dir.
    Remove DB entries for files that no longer exist.
    Called at startup to sync state.
    """
    if not skills_dir.exists():
        skills_dir.mkdir(parents=True, exist_ok=True)
        return

    existing_paths = {
        row["filepath"]
        for row in conn.execute("SELECT filepath FROM skill_embeddings").fetchall()
    }

    current_paths: set[str] = set()
    for md_file in skills_dir.glob("*.md"):
        current_paths.add(str(md_file))
        upsert_skill(conn, md_file)

    # Remove stale entries
    for stale in existing_paths - current_paths:
        remove_skill(conn, stale)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_skills(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """
    Embed the query and return the top_k most similar skills.
    Returns a list of dicts with keys: filepath, name, content.
    Returns empty list if embedding fails or no skills exist.
    """
    rows = conn.execute(
        "SELECT filepath, name, embedding_blob FROM skill_embeddings"
    ).fetchall()
    if not rows:
        return []

    query_vec = _embed_text(query)
    if query_vec is None:
        return []

    query_arr = np.array(query_vec, dtype=np.float32)
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)

    filepaths = [r["filepath"] for r in rows]
    names = [r["name"] for r in rows]
    blobs = [r["embedding_blob"] for r in rows]

    matrix = np.stack([_blob_to_vec(b) for b in blobs])  # shape (n, dim)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / (norms + 1e-8)
    similarities = np.dot(normalized, query_norm)

    k = min(top_k, len(rows))
    top_indices = np.argsort(similarities)[-k:][::-1]

    results = []
    for idx in top_indices:
        fp = Path(filepaths[idx])
        try:
            skill = _parse_skill_file(fp)
            results.append({
                "filepath": filepaths[idx],
                "name": names[idx],
                "content": skill["content"],
                "score": float(similarities[idx]),
            })
        except Exception as exc:
            log.warning("could not read skill file %s: %s", fp, exc)

    return results


def format_skills_for_context(skills: list[dict]) -> str:
    """
    Format retrieved skills as a system context string for injection
    into chat/notebook prompts.
    """
    if not skills:
        return ""
    parts = ["The following reference documents may be relevant:\n"]
    for s in skills:
        parts.append(f"### {s['name']}\n{s['content']}\n")
    return "\n".join(parts)


# ── Watchdog observer ─────────────────────────────────────────────────────────

class SkillsEventHandler(FileSystemEventHandler):
    """
    Watches data/skills/ for .md file changes.
    Re-embeds on create/modify; removes from DB on delete.
    Runs embedding in a background thread to avoid blocking the observer.
    """

    def __init__(self, conn: sqlite3.Connection, skills_dir: Path) -> None:
        super().__init__()
        self._conn = conn
        self._skills_dir = skills_dir

    def _is_skill_file(self, path: str) -> bool:
        return path.endswith(".md")

    def on_created(self, event) -> None:
        if not event.is_directory and self._is_skill_file(event.src_path):
            threading.Thread(
                target=upsert_skill,
                args=(self._conn, Path(event.src_path)),
                daemon=True,
            ).start()

    def on_modified(self, event) -> None:
        if not event.is_directory and self._is_skill_file(event.src_path):
            threading.Thread(
                target=upsert_skill,
                args=(self._conn, Path(event.src_path)),
                daemon=True,
            ).start()

    def on_deleted(self, event) -> None:
        if not event.is_directory and self._is_skill_file(event.src_path):
            remove_skill(self._conn, event.src_path)


class SkillsWatcher:
    """Manages the watchdog Observer lifecycle."""

    def __init__(self, conn: sqlite3.Connection, skills_dir: Path) -> None:
        self._conn = conn
        self._skills_dir = skills_dir
        self._observer: Optional[Observer] = None

    def start(self) -> None:
        skills_dir = self._skills_dir
        skills_dir.mkdir(parents=True, exist_ok=True)
        handler = SkillsEventHandler(self._conn, skills_dir)
        self._observer = Observer()
        self._observer.schedule(handler, str(skills_dir), recursive=False)
        self._observer.start()
        log.info("skills watcher started on %s", skills_dir)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            log.info("skills watcher stopped")

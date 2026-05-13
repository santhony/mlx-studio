"""
db.py — SQLite connection and schema initialization.

Functional core: pure functions operating on a connection object.
No global state. The connection is owned and passed by main.py lifespan.
"""

import sqlite3
from pathlib import Path


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database. Returns a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row  # rows accessible by column name
    conn.execute("PRAGMA foreign_keys = ON")  # enforce foreign key constraints
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables if they do not exist. Idempotent."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt      TEXT    NOT NULL,
            filename    TEXT    NOT NULL,
            width       INTEGER NOT NULL,
            height      INTEGER NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role        TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content     TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS notebooks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS cells (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            notebook_id INTEGER NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
            position    INTEGER NOT NULL DEFAULT 0,
            prompt      TEXT    NOT NULL DEFAULT '',
            code        TEXT    NOT NULL DEFAULT '',
            output      TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS agent_jobs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            task        TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','completed','failed')),
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS agent_steps (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id      INTEGER NOT NULL REFERENCES agent_jobs(id) ON DELETE CASCADE,
            type        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            approved    INTEGER,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS finetune_jobs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            base_model   TEXT    NOT NULL,
            dataset_path TEXT    NOT NULL,
            config_json  TEXT    NOT NULL DEFAULT '{}',
            status       TEXT    NOT NULL DEFAULT 'pending' CHECK(status IN ('pending','running','completed','failed','stopped')),
            created_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS skill_embeddings (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath       TEXT    NOT NULL UNIQUE,
            name           TEXT    NOT NULL,
            mtime          REAL    NOT NULL,
            embedding_blob BLOB    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS corpora (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            description TEXT    NOT NULL DEFAULT '',
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS corpus_sources (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            corpus_id       INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
            source_type     TEXT    NOT NULL CHECK(source_type IN ('directory', 'url', 'url_spider')),
            path            TEXT    NOT NULL,
            treat_as_text   INTEGER NOT NULL DEFAULT 0,  -- if 1, drop the extension allowlist and sniff
            last_indexed_at TEXT,
            created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS corpus_chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            corpus_id   INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
            source_id   INTEGER NOT NULL REFERENCES corpus_sources(id) ON DELETE CASCADE,
            source_file TEXT    NOT NULL,
            chunk_index INTEGER NOT NULL,
            content     TEXT    NOT NULL,
            embedding   BLOB,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS rag_sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            corpus_id   INTEGER NOT NULL REFERENCES corpora(id) ON DELETE CASCADE,
            name        TEXT    NOT NULL,
            created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );

        CREATE TABLE IF NOT EXISTS rag_messages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id    INTEGER NOT NULL REFERENCES rag_sessions(id) ON DELETE CASCADE,
            role          TEXT    NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content       TEXT    NOT NULL,
            citations_json TEXT,
            created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
        );
    """)

    # Lightweight migrations for additive columns. SQLite's ALTER TABLE ADD
    # COLUMN can't be made idempotent in a single statement, so guard with
    # PRAGMA table_info first.
    def _has_column(table: str, col: str) -> bool:
        return col in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    if not _has_column("corpus_sources", "treat_as_text"):
        conn.execute("ALTER TABLE corpus_sources ADD COLUMN treat_as_text INTEGER NOT NULL DEFAULT 0")

    conn.commit()

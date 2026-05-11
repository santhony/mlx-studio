"""Tests for RAG corpus CRUD helpers."""

import sqlite3
from pathlib import Path
import sys

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from db import init_schema
from routers.rag import (
    _get_corpora,
    _get_corpus,
    _get_sources,
    _get_source,
    _build_rag_system_prompt,
    _get_rag_sessions,
    _get_rag_session,
    _get_rag_messages,
    _append_rag_message,
)


def setup_test_db() -> sqlite3.Connection:
    """Create an in-memory SQLite database with schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)
    return conn


class TestCorpusCRUD:
    """Test corpus CRUD operations."""

    def test_create_and_get_corpus(self):
        """Insert a corpus, verify _get_corpora() and _get_corpus() return it."""
        conn = setup_test_db()

        # Insert a corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", "A test corpus"),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Verify _get_corpora() returns it
        corpora = _get_corpora(conn)
        assert len(corpora) == 1
        assert corpora[0]["name"] == "Test Corpus"
        assert corpora[0]["description"] == "A test corpus"
        assert corpora[0]["id"] == corpus_id

        # Verify _get_corpus() returns it by id
        corpus = _get_corpus(conn, corpus_id)
        assert corpus is not None
        assert corpus["name"] == "Test Corpus"
        assert corpus["id"] == corpus_id

    def test_get_corpus_not_found(self):
        """Verify _get_corpus() returns None for non-existent id."""
        conn = setup_test_db()

        corpus = _get_corpus(conn, 999)
        assert corpus is None

    def test_create_multiple_corpora_ordering(self):
        """Verify _get_corpora() returns corpora ordered by updated_at DESC."""
        conn = setup_test_db()

        # Insert two corpora
        cursor1 = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Corpus 1", "First"),
        )
        corpus_id1 = cursor1.lastrowid
        conn.commit()

        cursor2 = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Corpus 2", "Second"),
        )
        corpus_id2 = cursor2.lastrowid
        conn.commit()

        corpora = _get_corpora(conn)
        assert len(corpora) == 2
        # Both corpora should exist
        ids = [c["id"] for c in corpora]
        assert corpus_id1 in ids
        assert corpus_id2 in ids


class TestSourceCRUD:
    """Test source CRUD operations."""

    def test_create_and_get_sources(self):
        """Add sources to a corpus, verify _get_sources() returns them with chunk counts."""
        conn = setup_test_db()

        # Create a corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Add sources
        cursor1 = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/path/to/docs"),
        )
        conn.commit()
        source_id1 = cursor1.lastrowid

        cursor2 = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "url", "https://example.com"),
        )
        conn.commit()
        source_id2 = cursor2.lastrowid

        # Verify _get_sources() returns them
        sources = _get_sources(conn, corpus_id)
        assert len(sources) == 2

        # Check that both sources are present
        source_ids = [s["id"] for s in sources]
        assert source_id1 in source_ids
        assert source_id2 in source_ids

        # Check that chunk counts are 0 for both
        for source in sources:
            assert source["chunk_count"] == 0

        # Check source types are correct
        source_types = {s["id"]: s["source_type"] for s in sources}
        assert source_types[source_id1] == "directory"
        assert source_types[source_id2] == "url"

    def test_get_source(self):
        """Verify _get_source() returns a single source by id."""
        conn = setup_test_db()

        # Create a corpus and source
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/path/to/docs"),
        )
        conn.commit()
        source_id = cursor.lastrowid

        # Verify _get_source() returns it
        source = _get_source(conn, source_id)
        assert source is not None
        assert source["id"] == source_id
        assert source["path"] == "/path/to/docs"
        assert source["source_type"] == "directory"

    def test_get_source_not_found(self):
        """Verify _get_source() returns None for non-existent source."""
        conn = setup_test_db()
        source = _get_source(conn, 999)
        assert source is None

    def test_source_chunk_count(self):
        """Add source + chunks, verify _get_sources() returns correct chunk count."""
        conn = setup_test_db()

        # Create a corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Add a source
        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/path/to/docs"),
        )
        conn.commit()
        source_id = cursor.lastrowid

        # Add some chunks
        for i in range(3):
            conn.execute(
                "INSERT INTO corpus_chunks (corpus_id, source_id, source_file, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                (corpus_id, source_id, "file.txt", i, f"Chunk {i}"),
            )
        conn.commit()

        # Verify chunk count
        sources = _get_sources(conn, corpus_id)
        assert len(sources) == 1
        assert sources[0]["chunk_count"] == 3

    def test_delete_source_cascades_chunks(self):
        """Add source + chunks, delete source, verify chunks are gone."""
        conn = setup_test_db()

        # Create a corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Add a source
        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/path/to/docs"),
        )
        conn.commit()
        source_id = cursor.lastrowid

        # Add chunks
        for i in range(3):
            conn.execute(
                "INSERT INTO corpus_chunks (corpus_id, source_id, source_file, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                (corpus_id, source_id, "file.txt", i, f"Chunk {i}"),
            )
        conn.commit()

        # Verify chunks exist
        chunks = conn.execute(
            "SELECT COUNT(*) as count FROM corpus_chunks WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        assert chunks["count"] == 3

        # Delete source (should cascade delete chunks)
        conn.execute("DELETE FROM corpus_sources WHERE id = ?", (source_id,))
        conn.commit()

        # Verify chunks are gone
        chunks = conn.execute(
            "SELECT COUNT(*) as count FROM corpus_chunks WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        assert chunks["count"] == 0

        # Verify source is gone
        sources = _get_sources(conn, corpus_id)
        assert len(sources) == 0

    def test_delete_corpus_cascades(self):
        """Create corpus with sources and chunks, delete corpus, verify all deleted."""
        conn = setup_test_db()

        # Create a corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Add sources
        source_ids = []
        for i in range(2):
            cursor = conn.execute(
                "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
                (corpus_id, "directory", f"/path{i}"),
            )
            conn.commit()
            source_ids.append(cursor.lastrowid)

        # Add chunks to each source
        for source_id in source_ids:
            for i in range(2):
                conn.execute(
                    "INSERT INTO corpus_chunks (corpus_id, source_id, source_file, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
                    (corpus_id, source_id, "file.txt", i, f"Chunk {i}"),
                )
        conn.commit()

        # Verify corpus exists with sources and chunks
        corpora = _get_corpora(conn)
        assert len(corpora) == 1
        sources = _get_sources(conn, corpus_id)
        assert len(sources) == 2

        chunks = conn.execute("SELECT COUNT(*) as count FROM corpus_chunks").fetchone()
        assert chunks["count"] == 4

        # Delete corpus (should cascade delete sources and chunks)
        conn.execute("DELETE FROM corpora WHERE id = ?", (corpus_id,))
        conn.commit()

        # Verify all deleted
        corpora = _get_corpora(conn)
        assert len(corpora) == 0

        sources = _get_sources(conn, corpus_id)
        assert len(sources) == 0

        chunks = conn.execute("SELECT COUNT(*) as count FROM corpus_chunks").fetchone()
        assert chunks["count"] == 0


class TestBuildRAGSystemPrompt:
    """Test RAG system prompt builder."""

    def test_build_rag_system_prompt_basic(self):
        """Test system prompt with 3 chunks includes numbered markers and correct metadata."""
        chunks = [
            {
                "source_file": "docs/intro.pdf",
                "chunk_index": 0,
                "content": "This is the introduction to the system.",
            },
            {
                "source_file": "guide.md",
                "chunk_index": 2,
                "content": "Here are some guidelines for using the system.",
            },
            {
                "source_file": "reference.txt",
                "chunk_index": 1,
                "content": "Complete reference documentation.",
            },
        ]

        prompt = _build_rag_system_prompt(chunks)

        # Verify numbered markers are present
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

        # Verify source files and chunk indices are present
        assert "docs/intro.pdf" in prompt
        assert "chunk 0" in prompt
        assert "guide.md" in prompt
        assert "chunk 2" in prompt
        assert "reference.txt" in prompt
        assert "chunk 1" in prompt

        # Verify chunk contents are included
        assert "This is the introduction to the system." in prompt
        assert "Here are some guidelines for using the system." in prompt
        assert "Complete reference documentation." in prompt

        # Verify system instruction about citations
        assert "Cite your sources" in prompt
        assert "ONLY the document excerpts" in prompt

    def test_build_rag_system_prompt_empty(self):
        """Test system prompt with empty chunk list returns fallback message."""
        chunks = []

        prompt = _build_rag_system_prompt(chunks)

        # Verify fallback message is present
        assert "No relevant documents" in prompt or "documents don't contain relevant information" in prompt
        # Should not contain numbered markers
        assert "[1]" not in prompt

    def test_build_rag_system_prompt_truncation(self):
        """Test system prompt includes chunks with >1000 chars without truncation."""
        # Create a chunk with content > 1000 chars
        long_content = "A" * 1500
        chunks = [
            {
                "source_file": "large_file.txt",
                "chunk_index": 5,
                "content": long_content,
            }
        ]

        prompt = _build_rag_system_prompt(chunks)

        # Verify the full content is included (no truncation in prompt builder)
        assert long_content in prompt
        assert "large_file.txt" in prompt
        assert "chunk 5" in prompt


class TestRAGSessions:
    """Test RAG session and message helpers."""

    def _create_test_session(self, conn: sqlite3.Connection) -> tuple[int, int]:
        """Helper to create a test corpus and session. Returns (corpus_id, session_id)."""
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO rag_sessions (corpus_id, name) VALUES (?, ?)",
            (corpus_id, "Test Session"),
        )
        conn.commit()
        session_id = cursor.lastrowid

        return corpus_id, session_id

    def test_create_rag_session(self):
        """Create a corpus, then a RAG session. Verify _get_rag_sessions() returns it with correct corpus_id."""
        conn = setup_test_db()

        corpus_id, session_id = self._create_test_session(conn)

        # Verify _get_rag_sessions() returns it
        sessions = _get_rag_sessions(conn, corpus_id)
        assert len(sessions) == 1
        assert sessions[0]["id"] == session_id
        assert sessions[0]["name"] == "Test Session"

        # Verify we can retrieve full session details including corpus_id via _get_rag_session()
        session = _get_rag_session(conn, session_id)
        assert session is not None
        assert session["corpus_id"] == corpus_id

    def test_append_and_get_rag_messages(self):
        """Create session, append user and assistant messages. Verify _get_rag_messages() returns them."""
        conn = setup_test_db()

        _, session_id = self._create_test_session(conn)

        # Append a user message
        user_msg_id = _append_rag_message(conn, session_id, "user", "What is this?")
        assert user_msg_id is not None

        # Append an assistant message with citations_json
        citations_json = '[{"ref": 1, "source_file": "doc.pdf", "chunk_index": 0, "excerpt": "Some text"}]'
        assistant_msg_id = _append_rag_message(
            conn, session_id, "assistant", "This is the answer.", citations_json
        )
        assert assistant_msg_id is not None

        # Verify _get_rag_messages() returns them in order
        messages = _get_rag_messages(conn, session_id)
        assert len(messages) == 2

        # Check user message
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is this?"
        assert messages[0]["citations"] is None

        # Check assistant message
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "This is the answer."
        assert messages[1]["citations"] is not None
        assert len(messages[1]["citations"]) == 1

    def test_rag_message_citations_json(self):
        """Append assistant message with citations_json, retrieve and parse citations structure."""
        conn = setup_test_db()

        _, session_id = self._create_test_session(conn)

        # Append assistant message with structured citations_json
        citations_json = """[
            {"ref": 1, "source_file": "guide.pdf", "chunk_index": 5, "excerpt": "First guide section"},
            {"ref": 2, "source_file": "reference.txt", "chunk_index": 2, "excerpt": "Reference information"}
        ]"""
        _append_rag_message(
            conn, session_id, "assistant", "Based on the documents...", citations_json
        )

        # Retrieve and verify citation structure
        messages = _get_rag_messages(conn, session_id)
        assert len(messages) == 1

        citations = messages[0]["citations"]
        assert len(citations) == 2

        # Verify first citation
        assert citations[0]["ref"] == 1
        assert citations[0]["source_file"] == "guide.pdf"
        assert citations[0]["chunk_index"] == 5
        assert citations[0]["excerpt"] == "First guide section"

        # Verify second citation
        assert citations[1]["ref"] == 2
        assert citations[1]["source_file"] == "reference.txt"
        assert citations[1]["chunk_index"] == 2
        assert citations[1]["excerpt"] == "Reference information"

    def test_delete_rag_session_cascades(self):
        """Create session with messages, delete session, verify messages are gone (cascade)."""
        conn = setup_test_db()

        _, session_id = self._create_test_session(conn)

        # Append multiple messages
        _append_rag_message(conn, session_id, "user", "Question 1")
        _append_rag_message(conn, session_id, "assistant", "Answer 1")
        _append_rag_message(conn, session_id, "user", "Question 2")

        # Verify messages exist
        messages = _get_rag_messages(conn, session_id)
        assert len(messages) == 3

        # Delete the session
        conn.execute("DELETE FROM rag_sessions WHERE id = ?", (session_id,))
        conn.commit()

        # Verify session is gone
        session = _get_rag_session(conn, session_id)
        assert session is None

        # Verify all messages are cascaded deleted
        messages = conn.execute(
            "SELECT COUNT(*) as count FROM rag_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        assert messages["count"] == 0

    def test_rag_sessions_scoped_to_corpus(self):
        """Create two corpora with sessions each. Verify _get_rag_sessions() returns only the specified corpus's sessions."""
        conn = setup_test_db()

        # Create first corpus with two sessions
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Corpus 1", ""),
        )
        conn.commit()
        corpus_id_1 = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO rag_sessions (corpus_id, name) VALUES (?, ?)",
            (corpus_id_1, "Corpus1-Session1"),
        )
        conn.commit()
        session_id_1_1 = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO rag_sessions (corpus_id, name) VALUES (?, ?)",
            (corpus_id_1, "Corpus1-Session2"),
        )
        conn.commit()
        session_id_1_2 = cursor.lastrowid

        # Create second corpus with one session
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Corpus 2", ""),
        )
        conn.commit()
        corpus_id_2 = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO rag_sessions (corpus_id, name) VALUES (?, ?)",
            (corpus_id_2, "Corpus2-Session1"),
        )
        conn.commit()
        session_id_2_1 = cursor.lastrowid

        # Verify corpus 1 sessions
        sessions_1 = _get_rag_sessions(conn, corpus_id_1)
        assert len(sessions_1) == 2
        session_ids_1 = [s["id"] for s in sessions_1]
        assert session_id_1_1 in session_ids_1
        assert session_id_1_2 in session_ids_1
        assert session_id_2_1 not in session_ids_1

        # Verify corpus 2 sessions
        sessions_2 = _get_rag_sessions(conn, corpus_id_2)
        assert len(sessions_2) == 1
        assert sessions_2[0]["id"] == session_id_2_1
        assert session_id_1_1 not in [s["id"] for s in sessions_2]
        assert session_id_1_2 not in [s["id"] for s in sessions_2]

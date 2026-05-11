"""Tests for indexer.py text extraction, chunking, and serialization."""

import sqlite3
import tempfile
from pathlib import Path
import sys
import unittest.mock

import httpx
import numpy as np
import pytest
import pymupdf

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from indexer import (
    extract_text_from_file,
    extract_text_from_url,
    extract_text_from_html,
    discover_links,
    spider_url,
    chunk_text,
    index_source,
    retrieve_chunks,
)
from skills import vec_to_blob, blob_to_vec
from db import init_schema


# ─── Chunking Tests ──────────────────────────────────────────────────────────


class TestChunkText:
    """Test text chunking with overlap."""

    def test_chunk_text_basic(self):
        """Chunk a known string, verify count, sizes, and overlap."""
        text = "a" * 5000  # 5000 character string
        chunks = chunk_text(text, chunk_size=2000, overlap=200)

        # Should have multiple chunks
        assert len(chunks) > 1

        # Each chunk should be within reasonable bounds
        for chunk in chunks:
            assert len(chunk) <= 2000

    def test_chunk_text_short(self):
        """Text shorter than chunk_size returns single chunk."""
        text = "hello world"
        chunks = chunk_text(text, chunk_size=2000, overlap=200)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_empty(self):
        """Empty string returns empty list."""
        chunks = chunk_text("", chunk_size=2000, overlap=200)
        assert chunks == []

    def test_chunk_text_only_whitespace(self):
        """String with only whitespace returns empty list."""
        chunks = chunk_text("   \n\t  ", chunk_size=2000, overlap=200)
        assert chunks == []

    def test_chunk_text_overlap(self):
        """Consecutive chunks overlap by expected amount."""
        # Create predictable text with repeating alphabetic characters
        text = "".join(chr(65 + (i % 26)) for i in range(5000))  # 5000 chars of repeating A-Z

        chunks = chunk_text(text, chunk_size=1000, overlap=200)

        # For chunks with overlap, verify overlap content matches
        for i in range(len(chunks) - 1):
            chunk_i = chunks[i]
            chunk_next = chunks[i + 1]

            # The last 200 chars of chunk_i should match the first 200 chars of chunk_next
            # (the chunks overlap by 200 characters by design)
            assert len(chunk_i) >= 200, f"Chunk {i} too short: {len(chunk_i)}"
            assert len(chunk_next) >= 200, f"Chunk {i+1} too short: {len(chunk_next)}"
            assert chunk_i[-200:] == chunk_next[:200], f"Overlap mismatch between chunks {i} and {i+1}"

    def test_chunk_text_custom_size(self):
        """Chunk with custom size and overlap."""
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=400, overlap=50)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 400

    def test_chunk_text_strips_whitespace(self):
        """Chunks are stripped of leading/trailing whitespace."""
        text = "  word1 word2  \n\n  word3 word4  "
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        for chunk in chunks:
            assert chunk == chunk.strip()


# ─── File Extraction Tests ───────────────────────────────────────────────────


class TestExtractTextFromFile:
    """Test text extraction from various file formats."""

    def test_extract_text_from_txt_file(self):
        """Extract text from .txt file."""
        content = "Hello from text file\nLine 2\nLine 3"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            extracted = extract_text_from_file(path)
            assert extracted == content
        finally:
            path.unlink()

    def test_extract_text_from_md_file(self):
        """Extract text from .md file."""
        content = "# Heading\n\nThis is markdown content.\n\n- Item 1\n- Item 2"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            extracted = extract_text_from_file(path)
            assert extracted == content
        finally:
            path.unlink()

    def test_extract_text_from_pdf(self):
        """Extract text from PDF file."""
        # Create a minimal PDF with known text
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello from PDF")
        page.insert_text((72, 100), "Line 2 of PDF")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            doc.save(str(tmp_path))
            doc.close()

            extracted = extract_text_from_file(tmp_path)
            assert "Hello from PDF" in extracted
            assert "Line 2 of PDF" in extracted
        finally:
            tmp_path.unlink()

    def test_extract_text_from_pdf_multipage(self):
        """Extract text from multi-page PDF with correct page separators."""
        # Create a multi-page PDF with known text on each page
        doc = pymupdf.open()

        # Page 1
        page1 = doc.new_page()
        page1.insert_text((72, 72), "Content on page 1")

        # Page 2
        page2 = doc.new_page()
        page2.insert_text((72, 72), "Content on page 2")

        # Page 3
        page3 = doc.new_page()
        page3.insert_text((72, 72), "Content on page 3")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            doc.save(str(tmp_path))
            doc.close()

            extracted = extract_text_from_file(tmp_path)

            # Verify all content is present
            assert "Content on page 1" in extracted
            assert "Content on page 2" in extracted
            assert "Content on page 3" in extracted

            # Verify page separators with correct page numbers
            assert "--- Page 1 ---" in extracted
            assert "--- Page 2 ---" in extracted
            assert "--- Page 3 ---" in extracted

            # Verify order: page 1 separator comes before page 2 separator
            page1_idx = extracted.index("--- Page 1 ---")
            page2_idx = extracted.index("--- Page 2 ---")
            page3_idx = extracted.index("--- Page 3 ---")
            assert page1_idx < page2_idx < page3_idx
        finally:
            tmp_path.unlink()

    def test_extract_text_unsupported_extension(self):
        """Try unsupported extension, verify ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            path = Path(f.name)

        try:
            with pytest.raises(ValueError) as exc_info:
                extract_text_from_file(path)
            assert "unsupported file extension" in str(exc_info.value).lower()
        finally:
            path.unlink()

    def test_extract_text_file_not_found(self):
        """Try non-existent file, verify FileNotFoundError."""
        path = Path("/nonexistent/file.txt")
        with pytest.raises(FileNotFoundError):
            extract_text_from_file(path)

    def test_extract_text_from_txt_with_special_chars(self):
        """Extract text file with UTF-8 special characters."""
        content = "Café, naïve, Ñoño, 中文, emoji: 🎉"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            extracted = extract_text_from_file(path)
            assert "Café" in extracted
            assert "中文" in extracted
            assert "🎉" in extracted
        finally:
            path.unlink()


# --- HTML Extraction Tests ------------------------------------------------


class TestExtractTextFromHtml:
    """Tests for extract_text_from_html — pure function, no mocks needed."""

    def test_strips_script_and_style_tags(self):
        html = (
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert('x')</script><p>Hello</p></body></html>"
        )
        result = extract_text_from_html(html)
        assert "alert" not in result
        assert "color:red" not in result
        assert "Hello" in result

    def test_returns_visible_text_with_newlines(self):
        html = "<div><p>First paragraph</p><p>Second paragraph</p></div>"
        result = extract_text_from_html(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result
        # Paragraphs separated by newline
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 2

    def test_empty_html_returns_empty_string(self):
        assert extract_text_from_html("") == ""

    def test_plain_text_passthrough(self):
        result = extract_text_from_html("just plain text")
        assert result == "just plain text"

    def test_nested_scripts_stripped(self):
        html = "<div><script><script>nested</script></script><p>Keep</p></div>"
        result = extract_text_from_html(html)
        assert "nested" not in result
        assert "Keep" in result


# --- Link Discovery Tests -------------------------------------------------


class TestDiscoverLinks:
    """Tests for discover_links — pure function, no mocks needed."""

    def test_discovers_same_domain_links(self):
        html = '''
        <a href="/page1">Page 1</a>
        <a href="https://example.com/page2">Page 2</a>
        <a href="https://other.com/page3">Page 3</a>
        '''
        result = discover_links(html, "https://example.com/index.html")
        assert "https://example.com/page1" in result
        assert "https://example.com/page2" in result
        assert "https://other.com/page3" not in result

    def test_resolves_relative_urls(self):
        html = '<a href="subdir/page.html">Link</a>'
        result = discover_links(html, "https://example.com/docs/index.html")
        assert "https://example.com/docs/subdir/page.html" in result

    def test_deduplicates_links(self):
        html = '''
        <a href="/page1">First</a>
        <a href="/page1">Second</a>
        <a href="https://example.com/page1">Third</a>
        '''
        result = discover_links(html, "https://example.com/")
        # All three resolve to same URL — should appear once
        count = result.count("https://example.com/page1")
        assert count == 1

    def test_excludes_base_url(self):
        html = '<a href="https://example.com/">Home</a>'
        result = discover_links(html, "https://example.com/")
        assert "https://example.com/" not in result

    def test_strips_fragment(self):
        html = '<a href="/page#section">Link</a>'
        result = discover_links(html, "https://example.com/")
        assert "https://example.com/page" in result
        assert "#section" not in str(result)

    def test_ignores_non_http_schemes(self):
        html = '''
        <a href="mailto:user@example.com">Email</a>
        <a href="javascript:void(0)">JS</a>
        <a href="ftp://example.com/file">FTP</a>
        <a href="/valid">Valid</a>
        '''
        result = discover_links(html, "https://example.com/")
        assert len(result) == 1
        assert "https://example.com/valid" in result

    def test_ignores_anchors_without_href(self):
        html = '<a name="bookmark">No href</a><a href="/real">Real</a>'
        result = discover_links(html, "https://example.com/")
        assert len(result) == 1

    def test_empty_html_returns_empty_list(self):
        assert discover_links("", "https://example.com/") == []

    def test_returns_sorted_results(self):
        html = '''
        <a href="/zebra">Z</a>
        <a href="/alpha">A</a>
        <a href="/middle">M</a>
        '''
        result = discover_links(html, "https://example.com/")
        assert result == sorted(result)


# --- Spider Tests ----------------------------------------------------------


class TestSpiderUrl:
    """Tests for spider_url — mocks httpx to avoid real network calls."""

    def _make_response(self, text="", content_type="text/html", content=None):
        """Create a mock httpx response."""
        resp = unittest.mock.MagicMock()
        resp.text = text
        resp.content = content or text.encode()
        resp.headers = {"content-type": content_type}
        resp.raise_for_status = unittest.mock.MagicMock()
        return resp

    def test_spider_discovers_and_fetches_links(self):
        """Spider fetches index page, discovers links, fetches each."""
        index_html = '''
        <html><body>
            <a href="/page1">Page 1</a>
            <a href="/page2">Page 2</a>
        </body></html>
        '''
        page1_html = "<html><body><p>Content of page 1</p></body></html>"
        page2_html = "<html><body><p>Content of page 2</p></body></html>"

        index_resp = self._make_response(text=index_html)
        page1_resp = self._make_response(text=page1_html)
        page2_resp = self._make_response(text=page2_html)

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep") as mock_sleep:
            mock_get.side_effect = [index_resp, page1_resp, page2_resp]
            results = spider_url("https://example.com/", delay=0.5)

        assert len(results) == 2
        urls = [url for url, _ in results]
        assert "https://example.com/page1" in urls
        assert "https://example.com/page2" in urls

        # Verify text was extracted from HTML
        texts = {url: text for url, text in results}
        assert "Content of page 1" in texts["https://example.com/page1"]
        assert "Content of page 2" in texts["https://example.com/page2"]

    def test_spider_rate_limits_requests(self):
        """Spider sleeps between fetching linked pages."""
        index_html = '<a href="/p1">1</a><a href="/p2">2</a>'
        page_resp = self._make_response(text="<p>Content</p>")
        index_resp = self._make_response(text=index_html)

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep") as mock_sleep:
            mock_get.side_effect = [index_resp, page_resp, page_resp]
            spider_url("https://example.com/", delay=1.5)

        # Sleep called once per linked page (2 pages)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(1.5)

    def test_spider_skips_failed_pages(self):
        """Spider logs and skips pages that fail to fetch."""
        index_html = '<a href="/good">Good</a><a href="/bad">Bad</a>'
        index_resp = self._make_response(text=index_html)
        good_resp = self._make_response(text="<p>Good content</p>")
        bad_resp = unittest.mock.MagicMock()
        bad_resp.raise_for_status.side_effect = httpx.HTTPError("404")

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep"):
            # Links are sorted, so /bad comes before /good
            mock_get.side_effect = [index_resp, bad_resp, good_resp]
            results = spider_url("https://example.com/")

        assert len(results) == 1
        assert results[0][0] == "https://example.com/good"

    def test_spider_skips_empty_pages(self):
        """Spider skips pages with no extractable text."""
        index_html = '<a href="/empty">Empty</a>'
        index_resp = self._make_response(text=index_html)
        empty_resp = self._make_response(text="<html><body>   </body></html>")

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep"):
            mock_get.side_effect = [index_resp, empty_resp]
            results = spider_url("https://example.com/")

        assert len(results) == 0

    def test_spider_handles_pdf_links(self):
        """Spider uses PDF extraction for application/pdf content type."""
        index_html = '<a href="/doc.pdf">PDF</a>'
        index_resp = self._make_response(text=index_html)
        pdf_resp = self._make_response(
            text="", content_type="application/pdf", content=b"fake-pdf-bytes"
        )

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep"), \
             unittest.mock.patch("indexer.extract_text_from_file", return_value="PDF extracted text") as mock_extract:
            mock_get.side_effect = [index_resp, pdf_resp]
            results = spider_url("https://example.com/")

        assert len(results) == 1
        assert results[0][1] == "PDF extracted text"
        mock_extract.assert_called_once()

    def test_spider_no_links_returns_empty(self):
        """Spider returns empty list if no links discovered."""
        index_html = "<html><body><p>No links here</p></body></html>"
        index_resp = self._make_response(text=index_html)

        with unittest.mock.patch("indexer.httpx.get") as mock_get, \
             unittest.mock.patch("indexer.time.sleep"):
            mock_get.return_value = index_resp
            results = spider_url("https://example.com/")

        assert results == []


# ─── Vector Serialization Tests ──────────────────────────────────────────────


class TestVecBlobSerialization:
    """Test vector blob round-trip serialization."""

    def test_vec_blob_roundtrip(self):
        """Create a vector, convert to blob, convert back, verify values match."""
        vec = [1.0, 2.5, -0.3, 0.0, 100.5]
        blob = vec_to_blob(vec)
        result = blob_to_vec(blob)

        assert isinstance(blob, bytes)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, vec, decimal=5)

    def test_vec_blob_roundtrip_384dim(self):
        """Round-trip a realistic 384-dim embedding vector."""
        embedding = np.random.randn(384).astype(np.float32).tolist()
        blob = vec_to_blob(embedding)

        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes per element

        result = blob_to_vec(blob)
        assert result.shape == (384,)
        np.testing.assert_array_almost_equal(result, embedding, decimal=5)

    def test_vec_blob_roundtrip_empty(self):
        """Round-trip empty vector."""
        vec = []
        blob = vec_to_blob(vec)
        result = blob_to_vec(blob)

        assert isinstance(blob, bytes)
        assert len(blob) == 0
        assert len(result) == 0

    def test_vec_blob_roundtrip_single(self):
        """Round-trip single-element vector."""
        vec = [3.14159]
        blob = vec_to_blob(vec)
        result = blob_to_vec(blob)

        np.testing.assert_almost_equal(result[0], 3.14159, decimal=5)


# ─── Integration Tests for index_source() ───────────────────────────────────


class TestIndexSource:
    """Integration tests for index_source() function."""

    @staticmethod
    def _create_test_db() -> sqlite3.Connection:
        """Create in-memory SQLite DB with schema."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        return conn

    @staticmethod
    def _insert_corpus_and_source(
        conn: sqlite3.Connection, source_type: str, path: str
    ) -> tuple[int, int]:
        """Insert a test corpus and source. Returns (corpus_id, source_id)."""
        cursor = conn.execute("INSERT INTO corpora (name, description) VALUES (?, ?)", ("Test Corpus", ""))
        corpus_id = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, source_type, path),
        )
        source_id = cursor.lastrowid
        conn.commit()

        return corpus_id, source_id

    def test_index_directory_source(self):
        """Index a directory with .txt and .md files. Verify chunks are created."""
        # Create temp directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create .txt file
            txt_file = tmpdir_path / "test.txt"
            txt_content = "This is a test file.\nLine 2 of test file.\n" * 100  # Make it long enough to chunk
            txt_file.write_text(txt_content)

            # Create .md file
            md_file = tmpdir_path / "test.md"
            md_content = "# Heading\n\nThis is markdown.\n" * 100  # Make it long enough to chunk
            md_file.write_text(md_content)

            # Setup DB
            conn = self._create_test_db()
            corpus_id, source_id = self._insert_corpus_and_source(
                conn, "directory", str(tmpdir_path)
            )

            # Mock embed_text to return fixed 384-dim vector
            mock_embedding = [0.1] * 384
            with unittest.mock.patch("indexer.embed_text", return_value=mock_embedding):
                result = index_source(conn, {"id": source_id, "source_type": "directory", "path": str(tmpdir_path)}, corpus_id)

            # Verify result
            assert result["files_processed"] == 2
            assert result["chunks_created"] > 0
            assert result["errors"] == []

            # Verify chunks in DB
            chunks = conn.execute(
                "SELECT * FROM corpus_chunks WHERE corpus_id = ? AND source_id = ?",
                (corpus_id, source_id),
            ).fetchall()
            assert len(chunks) == result["chunks_created"]

            # Verify chunk properties
            for i, chunk in enumerate(chunks):
                assert chunk["content"] is not None
                assert len(chunk["content"]) > 0
                assert chunk["embedding"] is not None
                assert chunk["source_file"] in ["test.txt", "test.md"]
                # chunk_index is per-file, so it resets for each source_file
                assert chunk["chunk_index"] >= 0

            # Verify last_indexed_at was updated
            source = conn.execute(
                "SELECT last_indexed_at FROM corpus_sources WHERE id = ?", (source_id,)
            ).fetchone()
            assert source["last_indexed_at"] is not None

    def test_index_source_reindex_clears_old(self):
        """Re-index a source clears old chunks."""
        # Create temp directory with test files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            txt_file = tmpdir_path / "test.txt"
            txt_content = "Original content.\n" * 100
            txt_file.write_text(txt_content)

            # Setup DB
            conn = self._create_test_db()
            corpus_id, source_id = self._insert_corpus_and_source(
                conn, "directory", str(tmpdir_path)
            )

            # First index
            mock_embedding = [0.1] * 384
            with unittest.mock.patch("indexer.embed_text", return_value=mock_embedding):
                result1 = index_source(conn, {"id": source_id, "source_type": "directory", "path": str(tmpdir_path)}, corpus_id)

            chunks_first = conn.execute(
                "SELECT id FROM corpus_chunks WHERE corpus_id = ? AND source_id = ?",
                (corpus_id, source_id),
            ).fetchall()
            assert len(chunks_first) > 0
            first_chunk_ids = {chunk["id"] for chunk in chunks_first}

            # Modify file and re-index
            txt_file.write_text("New content.\n" * 100)

            with unittest.mock.patch("indexer.embed_text", return_value=mock_embedding):
                result2 = index_source(conn, {"id": source_id, "source_type": "directory", "path": str(tmpdir_path)}, corpus_id)

            chunks_second = conn.execute(
                "SELECT id FROM corpus_chunks WHERE corpus_id = ? AND source_id = ?",
                (corpus_id, source_id),
            ).fetchall()
            assert len(chunks_second) > 0
            second_chunk_ids = {chunk["id"] for chunk in chunks_second}

            # Old chunk IDs should not exist in second index
            assert first_chunk_ids.isdisjoint(second_chunk_ids)

    def test_index_directory_skips_unsupported(self):
        """Index directory with .txt and .jpg files. Only .txt produces chunks."""
        # Create temp directory with mixed files
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create .txt file
            txt_file = tmpdir_path / "test.txt"
            txt_content = "Supported file content.\n" * 100
            txt_file.write_text(txt_content)

            # Create .jpg file (unsupported)
            jpg_file = tmpdir_path / "image.jpg"
            jpg_file.write_bytes(b"fake image data")

            # Setup DB
            conn = self._create_test_db()
            corpus_id, source_id = self._insert_corpus_and_source(
                conn, "directory", str(tmpdir_path)
            )

            # Index
            mock_embedding = [0.1] * 384
            with unittest.mock.patch("indexer.embed_text", return_value=mock_embedding):
                result = index_source(conn, {"id": source_id, "source_type": "directory", "path": str(tmpdir_path)}, corpus_id)

            # Only .txt file should be processed
            assert result["files_processed"] == 1
            assert result["chunks_created"] > 0
            assert result["errors"] == []

            # Verify only txt chunks exist
            chunks = conn.execute(
                "SELECT source_file FROM corpus_chunks WHERE corpus_id = ? AND source_id = ?",
                (corpus_id, source_id),
            ).fetchall()
            for chunk in chunks:
                assert chunk["source_file"] == "test.txt"

    def test_index_url_spider_source(self):
        """Index a url_spider source. Verify chunks from multiple pages with per-page source_file."""
        conn = self._create_test_db()
        corpus_id, source_id = self._insert_corpus_and_source(
            conn, "url_spider", "https://example.com/docs/"
        )

        # spider_url returns (url, text) tuples
        spider_pages = [
            ("https://example.com/docs/page1", "Page one content.\n" * 100),
            ("https://example.com/docs/page2", "Page two content.\n" * 100),
        ]

        mock_embedding = [0.1] * 384
        with unittest.mock.patch("indexer.spider_url", return_value=spider_pages), \
             unittest.mock.patch("indexer.embed_text", return_value=mock_embedding):
            result = index_source(
                conn,
                {"id": source_id, "source_type": "url_spider", "path": "https://example.com/docs/"},
                corpus_id,
            )

        assert result["files_processed"] == 2
        assert result["chunks_created"] > 0
        assert result["errors"] == []

        # Verify source_file is set to the linked page URL, not the index URL
        chunks = conn.execute(
            "SELECT source_file FROM corpus_chunks WHERE corpus_id = ? AND source_id = ?",
            (corpus_id, source_id),
        ).fetchall()
        source_files = {chunk["source_file"] for chunk in chunks}
        assert "https://example.com/docs/page1" in source_files
        assert "https://example.com/docs/page2" in source_files
        assert "https://example.com/docs/" not in source_files

    def test_index_url_spider_no_pages(self):
        """url_spider source with no discovered pages returns zero results."""
        conn = self._create_test_db()
        corpus_id, source_id = self._insert_corpus_and_source(
            conn, "url_spider", "https://example.com/empty/"
        )

        with unittest.mock.patch("indexer.spider_url", return_value=[]):
            result = index_source(
                conn,
                {"id": source_id, "source_type": "url_spider", "path": "https://example.com/empty/"},
                corpus_id,
            )

        assert result["files_processed"] == 0
        assert result["chunks_created"] == 0
        assert "no indexable pages found" in result["errors"][0]

    def test_index_url_spider_fetch_error(self):
        """url_spider source returns error when index page fetch fails."""
        conn = self._create_test_db()
        corpus_id, source_id = self._insert_corpus_and_source(
            conn, "url_spider", "https://example.com/broken/"
        )

        with unittest.mock.patch("indexer.spider_url", side_effect=httpx.HTTPError("Connection refused")):
            result = index_source(
                conn,
                {"id": source_id, "source_type": "url_spider", "path": "https://example.com/broken/"},
                corpus_id,
            )

        assert result["files_processed"] == 0
        assert result["chunks_created"] == 0
        assert len(result["errors"]) == 1
        assert "failed to fetch index URL" in result["errors"][0]


# ─── Retrieval Tests ────────────────────────────────────────────────────────


class TestRetrieveChunks:
    """Test retrieve_chunks() semantic search function."""

    @staticmethod
    def _create_test_db() -> sqlite3.Connection:
        """Create in-memory SQLite DB with schema."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        init_schema(conn)
        return conn

    @staticmethod
    def _create_384dim_vector(seed: int) -> list[float]:
        """
        Create a deterministic 384-dim embedding vector.
        seed=0: [1, 0, 0, ..., 0]
        seed=1: [0, 1, 0, ..., 0]
        seed=2: [0, 0, 1, ..., 0]
        etc.
        """
        vec = [0.0] * 384
        vec[seed % 384] = 1.0
        return vec

    def test_retrieve_chunks_basic(self):
        """Query with vector close to one chunk, verify that chunk ranks highest."""
        conn = self._create_test_db()

        # Create corpus
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Create source
        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/test"),
        )
        conn.commit()
        source_id = cursor.lastrowid

        # Insert 5 chunks with simple, orthogonal vectors
        chunks_data = [
            {"content": "chunk 0", "vec": self._create_384dim_vector(0)},
            {"content": "chunk 1", "vec": self._create_384dim_vector(1)},
            {"content": "chunk 2", "vec": self._create_384dim_vector(2)},
            {"content": "chunk 3", "vec": self._create_384dim_vector(3)},
            {"content": "chunk 4", "vec": self._create_384dim_vector(4)},
        ]

        for i, chunk_data in enumerate(chunks_data):
            blob = vec_to_blob(chunk_data["vec"])
            conn.execute(
                """
                INSERT INTO corpus_chunks
                (corpus_id, source_id, source_file, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (corpus_id, source_id, "test.txt", i, chunk_data["content"], blob),
            )
        conn.commit()

        # Mock embed_text to return vector similar to chunk 2 (seed=2)
        query_vector = self._create_384dim_vector(2)
        with unittest.mock.patch("indexer.embed_text", return_value=query_vector):
            results = retrieve_chunks(conn, corpus_id, "query", top_k=5)

        # Verify results are returned
        assert len(results) == 5

        # Verify highest score is for chunk 2
        assert results[0]["content"] == "chunk 2"
        assert results[0]["score"] > 0.99  # Nearly perfect match (orthogonal vectors = 0, but L2-norm = 1)

        # Verify all results have required keys
        for result in results:
            assert "id" in result
            assert "source_file" in result
            assert "chunk_index" in result
            assert "content" in result
            assert "score" in result

    def test_retrieve_chunks_top_k(self):
        """Insert 10 chunks, retrieve with top_k=3, verify exactly 3 returned."""
        conn = self._create_test_db()

        # Create corpus and source
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Test Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        cursor = conn.execute(
            "INSERT INTO corpus_sources (corpus_id, source_type, path) VALUES (?, ?, ?)",
            (corpus_id, "directory", "/test"),
        )
        conn.commit()
        source_id = cursor.lastrowid

        # Insert 10 chunks
        for i in range(10):
            vec = self._create_384dim_vector(i)
            blob = vec_to_blob(vec)
            conn.execute(
                """
                INSERT INTO corpus_chunks
                (corpus_id, source_id, source_file, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (corpus_id, source_id, "test.txt", i, f"chunk {i}", blob),
            )
        conn.commit()

        # Query with top_k=3
        query_vector = self._create_384dim_vector(0)
        with unittest.mock.patch("indexer.embed_text", return_value=query_vector):
            results = retrieve_chunks(conn, corpus_id, "query", top_k=3)

        # Verify exactly 3 results
        assert len(results) == 3

        # Verify results are sorted by score (descending)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_chunks_empty_corpus(self):
        """Retrieve from corpus with no chunks, verify empty list returned."""
        conn = self._create_test_db()

        # Create corpus with no chunks
        cursor = conn.execute(
            "INSERT INTO corpora (name, description) VALUES (?, ?)",
            ("Empty Corpus", ""),
        )
        conn.commit()
        corpus_id = cursor.lastrowid

        # Query empty corpus
        query_vector = self._create_384dim_vector(0)
        with unittest.mock.patch("indexer.embed_text", return_value=query_vector):
            results = retrieve_chunks(conn, corpus_id, "query", top_k=5)

        # Verify empty list
        assert results == []

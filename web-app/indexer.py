"""
indexer.py — Document indexing for RAG: text extraction, chunking, and embedding.

Functional core: pure functions for text extraction and chunking.
Side effects (HTTP calls via embed_text, file I/O, database writes) are isolated.

Supported formats: .txt, .md, .pdf
"""

import logging
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urljoin

import httpx
import numpy as np
import pymupdf
from bs4 import BeautifulSoup

from skills import embed_text, vec_to_blob, blob_to_vec

log = logging.getLogger("indexer")

TEXT_SERVER = "http://127.0.0.1:8766"

# Configuration
CHUNK_SIZE = 2000  # Characters per chunk
CHUNK_OVERLAP = 200  # Character overlap between chunks
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Extensions to skip even when a source has treat_as_text enabled. These are
# almost certainly binary; reading them as text wastes embed budget and
# pollutes retrieval with metadata noise.
BINARY_EXTENSIONS = {
    # images / video / audio
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".ico", ".svg",
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
    # archives
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar", ".lz", ".zst",
    # compiled / binary objects
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".lib", ".pyc", ".pyo",
    ".class", ".jar", ".war", ".wasm",
    # databases / serialized
    ".db", ".sqlite", ".sqlite3", ".idx", ".pack",
    # fonts
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    # documents (use the .pdf branch instead for PDFs)
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".key", ".numbers", ".pages",
}

# Directory names skipped during the recursive walk in loose mode. These
# almost never contain content worth indexing and contain large amounts of
# generated noise that would otherwise be read as text.
SKIP_DIR_NAMES = {".git", "node_modules", "venv", "venv-image", "venv-text",
                  "venv-web", "__pycache__", ".pytest_cache", ".mypy_cache",
                  "dist", "build", ".idea", ".vscode"}


# ── Pure text extraction functions ────────────────────────────────────────────

def looks_binary(path: Path, sample_bytes: int = 8192) -> bool:
    """
    Heuristic: a file is binary if its first `sample_bytes` contain a NUL byte.
    Almost all real text files (including ASCII, UTF-8, Latin-1, even old DOS
    files) have no NUL bytes. Almost all binary files have at least one in
    their header. Cheap to check and ~99% accurate in practice.
    """
    try:
        with path.open("rb") as f:
            chunk = f.read(sample_bytes)
        return b"\x00" in chunk
    except Exception:
        return True  # unreadable → treat as skip-worthy


def extract_text_loose(path: Path) -> str:
    """
    Best-effort text extraction for files without a known extension.

    Strategy:
      1. Skip if extension is in BINARY_EXTENSIONS.
      2. Sniff for NUL bytes; bail if present.
      3. Try UTF-8 with strict errors; if that fails, fall back to Latin-1
         (which decodes any byte sequence, lossily but losslessly-mappable).

    Returns "" for files that should be skipped (caller treats empty as skip).
    """
    suffix = path.suffix.lower()
    if suffix in BINARY_EXTENSIONS:
        return ""
    if looks_binary(path):
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Many old-school text files are CP437 / Latin-1 / DOS. Latin-1 maps
        # every byte to a codepoint, so this never fails.
        return path.read_text(encoding="latin-1")
    except Exception:
        return ""


def extract_text_from_file(path: Path) -> str:
    """
    Extract text from a file based on extension.

    Supported formats:
    - .txt, .md: UTF-8 text files
    - .pdf: PDF documents (pages separated by "--- Page N ---")

    Raises ValueError for unsupported extensions.
    """
    if not path.exists():
        raise FileNotFoundError(f"file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file extension: {suffix}")

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="replace")

    elif suffix == ".pdf":
        try:
            doc = pymupdf.open(str(path))
            parts = []
            for page_num in range(len(doc)):
                text = doc[page_num].get_text("text", sort=True)
                if text.strip():
                    parts.append(f"--- Page {page_num + 1} ---\n\n{text}")
            doc.close()

            if not parts:
                return ""

            # Join pages with separator to track source and page numbers
            return "\n\n".join(parts)
        except Exception as exc:
            raise ValueError(f"failed to extract PDF text from {path.name}: {exc}") from exc


def extract_text_from_url(url: str) -> tuple[str, str]:
    """
    Fetch and extract text from a URL.

    Returns:
        Tuple of (text, detected_type) where detected_type is "pdf", "text", etc.

    Raises:
        httpx.HTTPError on network failure or non-2xx response
    """
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "").lower()

    if "application/pdf" in content_type:
        # Save to temp file and extract with pymupdf
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name

        try:
            text = extract_text_from_file(Path(tmp_path))
            return (text, "pdf")
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        # Treat as plain text
        return (resp.text, "text")


def extract_text_from_html(html: str) -> str:
    """
    Extract clean text from an HTML string.

    Strips <script> and <style> tags, then returns visible text
    with newline separators between elements and whitespace stripped.

    Pure function — no I/O.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def discover_links(html: str, base_url: str) -> list[str]:
    """
    Discover same-domain links from an HTML page.

    Parses anchor tags, resolves relative URLs against base_url,
    filters to same-domain links, and deduplicates.
    Returns a sorted list of absolute URLs.

    Pure function — no I/O.
    """
    soup = BeautifulSoup(html, "html.parser")
    base_netloc = urlparse(base_url).netloc

    seen: set[str] = set()
    result: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute = urljoin(base_url, href)
        parsed = urlparse(absolute)

        # Filter: same domain, http(s) only, not already seen
        if parsed.netloc != base_netloc:
            continue
        if parsed.scheme not in ("http", "https"):
            continue

        # Normalize: strip fragment
        normalized = parsed._replace(fragment="").geturl()

        if normalized not in seen and normalized != base_url:
            seen.add(normalized)
            result.append(normalized)

    return sorted(result)


def spider_url(url: str, delay: float = 1.0) -> list[tuple[str, str]]:
    """
    Spider a URL: fetch the index page, discover same-domain links,
    then fetch each linked page with rate limiting.

    For each linked page:
    - HTML pages: text extracted via extract_text_from_html()
    - PDFs: text extracted via extract_text_from_file() (existing PyMuPDF path)
    - Other content types: treated as plain text

    Returns list of (url, extracted_text) tuples.
    Logs and skips individual page errors.

    Side-effect function — performs HTTP requests with delays.
    """
    # Fetch the index page
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()

    links = discover_links(resp.text, url)
    log.info("spider discovered %d links from %s", len(links), url)

    results: list[tuple[str, str]] = []

    for link_url in links:
        try:
            time.sleep(delay)
            link_resp = httpx.get(link_url, timeout=30.0, follow_redirects=True)
            link_resp.raise_for_status()

            content_type = link_resp.headers.get("content-type", "").lower()

            if "application/pdf" in content_type:
                # Use existing PDF extraction path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(link_resp.content)
                    tmp_path = tmp.name
                try:
                    text = extract_text_from_file(Path(tmp_path))
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            elif "text/html" in content_type:
                text = extract_text_from_html(link_resp.text)
            else:
                text = link_resp.text

            if text.strip():
                results.append((link_url, text))
                log.info("spider fetched: %s (%d chars)", link_url, len(text))
            else:
                log.debug("spider skipping empty page: %s", link_url)

        except Exception as exc:
            log.warning("spider failed to fetch %s: %s", link_url, exc)

    return results


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping chunks.

    Algorithm:
    - Start at position 0
    - Take chunk_size characters
    - Advance by (chunk_size - overlap) characters
    - Strip whitespace from each chunk, discard empty chunks

    Args:
        text: Input text to chunk
        chunk_size: Target characters per chunk
        overlap: Overlap between consecutive chunks

    Returns:
        List of chunk strings (stripped of leading/trailing whitespace)
    """
    if not text or chunk_size <= 0:
        return []

    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    chunks = []
    pos = 0
    step = chunk_size - overlap

    while pos < len(text):
        chunk = text[pos : pos + chunk_size]
        stripped = chunk.strip()
        if stripped:
            chunks.append(stripped)
        pos += step

    return chunks


# ── Indexing: side effects (HTTP, file I/O, database writes) ──────────────────

def index_source(
    conn: sqlite3.Connection,
    source: dict,
    corpus_id: int,
) -> dict:
    """
    Index a single source: extract text, chunk, embed, and store chunks.

    Fetches source data from a directory or URL, extracts text, splits into chunks,
    embeds each chunk, and stores them in corpus_chunks. Re-indexing clears
    previous chunks for this source.

    Args:
        conn: SQLite connection with corpus_chunks table
        source: dict with keys: id, source_type, path
        corpus_id: Corpus to index into

    Returns:
        dict with keys:
        - files_processed: number of files successfully processed
        - chunks_created: total chunks created
        - errors: list of error messages for failed files
    """
    source_id = source["id"]
    source_type = source["source_type"]
    source_path = source["path"]

    files_processed = 0
    chunks_created = 0
    errors: list[str] = []

    # Clear existing chunks for this source
    conn.execute("DELETE FROM corpus_chunks WHERE source_id = ?", (source_id,))

    try:
        if source_type == "directory":
            # Process directory: walk recursively, filter by extension
            dir_path = Path(source_path)
            if not dir_path.is_dir():
                errors.append(f"directory not found: {source_path}")
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": errors,
                }

            loose = bool(source.get("treat_as_text"))

            # Collect candidate files. In strict mode (default), gate on
            # the known-text extension allowlist. In loose mode, accept
            # anything not in BINARY_EXTENSIONS or a skip-dir; the actual
            # text-vs-binary decision is made by extract_text_loose via
            # NUL-byte sniffing.
            def _candidate(f: Path) -> bool:
                if not f.is_file():
                    return False
                # Skip files inside well-known noise dirs (loose mode only).
                if loose and any(part in SKIP_DIR_NAMES for part in f.parts):
                    return False
                if loose:
                    return f.suffix.lower() not in BINARY_EXTENSIONS
                return f.suffix.lower() in SUPPORTED_EXTENSIONS

            files = [f for f in dir_path.rglob("*") if _candidate(f)]

            for file_path in files:
                try:
                    # Check file size
                    if file_path.stat().st_size > MAX_FILE_SIZE:
                        error_msg = f"file too large (>{MAX_FILE_SIZE // (1024*1024)}MB): {file_path.name}"
                        errors.append(error_msg)
                        log.warning(error_msg)
                        continue

                    # Loose mode uses the best-effort extractor that sniffs
                    # for NUL bytes and falls back to Latin-1; strict mode
                    # uses the extension-keyed extractor.
                    if loose and file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                        text = extract_text_loose(file_path)
                    else:
                        text = extract_text_from_file(file_path)
                    if not text.strip():
                        log.debug("skipping empty file: %s", file_path.name)
                        continue

                    chunks = chunk_text(text)
                    if not chunks:
                        log.debug("no chunks from file: %s", file_path.name)
                        continue

                    # Embed and store each chunk
                    relative_path = str(file_path.relative_to(dir_path))
                    for chunk_idx, chunk_content in enumerate(chunks):
                        vec = embed_text(chunk_content)
                        if vec is None:
                            log.warning(
                                "embedding failed for chunk %d of %s",
                                chunk_idx,
                                file_path.name,
                            )
                            continue

                        embedding_blob = vec_to_blob(vec)
                        conn.execute(
                            """
                            INSERT INTO corpus_chunks
                            (corpus_id, source_id, source_file, chunk_index, content, embedding)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                corpus_id,
                                source_id,
                                relative_path,
                                chunk_idx,
                                chunk_content,
                                embedding_blob,
                            ),
                        )
                        chunks_created += 1

                    files_processed += 1
                    log.info("indexed file: %s (%d chunks)", file_path.name, len(chunks))

                except Exception as exc:
                    error_msg = f"failed to index {file_path.name}: {exc}"
                    errors.append(error_msg)
                    log.warning(error_msg)

        elif source_type == "url":
            # Process URL: extract text, chunk, embed
            try:
                text, detected_type = extract_text_from_url(source_path)
            except httpx.HTTPError as exc:
                error_msg = f"failed to fetch URL: {exc}"
                errors.append(error_msg)
                log.warning(error_msg)
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": errors,
                }

            if not text.strip():
                log.warning("no text extracted from URL: %s", source_path)
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": ["no text extracted from URL"],
                }

            chunks = chunk_text(text)
            if not chunks:
                log.warning("no chunks from URL: %s", source_path)
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": ["no chunks from URL"],
                }

            for chunk_idx, chunk_content in enumerate(chunks):
                vec = embed_text(chunk_content)
                if vec is None:
                    log.warning("embedding failed for chunk %d", chunk_idx)
                    continue

                embedding_blob = vec_to_blob(vec)
                conn.execute(
                    """
                    INSERT INTO corpus_chunks
                    (corpus_id, source_id, source_file, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        corpus_id,
                        source_id,
                        source_path,
                        chunk_idx,
                        chunk_content,
                        embedding_blob,
                    ),
                )
                chunks_created += 1

            files_processed = 1
            log.info("indexed URL: %s (%d chunks)", source_path, len(chunks))

        elif source_type == "url_spider":
            # Spider: fetch index page, discover links, fetch and index each
            try:
                pages = spider_url(source_path)
            except Exception as exc:
                error_msg = f"failed to fetch index URL: {exc}"
                errors.append(error_msg)
                log.warning(error_msg)
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": errors,
                }

            if not pages:
                log.warning("spider found no indexable pages from: %s", source_path)
                return {
                    "files_processed": 0,
                    "chunks_created": 0,
                    "errors": ["no indexable pages found"],
                }

            for page_url, page_text in pages:
                try:
                    chunks = chunk_text(page_text)
                    if not chunks:
                        log.debug("no chunks from page: %s", page_url)
                        continue

                    for chunk_idx, chunk_content in enumerate(chunks):
                        vec = embed_text(chunk_content)
                        if vec is None:
                            log.warning(
                                "embedding failed for chunk %d of %s",
                                chunk_idx,
                                page_url,
                            )
                            continue

                        embedding_blob = vec_to_blob(vec)
                        conn.execute(
                            """
                            INSERT INTO corpus_chunks
                            (corpus_id, source_id, source_file, chunk_index, content, embedding)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                corpus_id,
                                source_id,
                                page_url,
                                chunk_idx,
                                chunk_content,
                                embedding_blob,
                            ),
                        )
                        chunks_created += 1

                    files_processed += 1
                    log.info("indexed page: %s (%d chunks)", page_url, len(chunks))

                except Exception as exc:
                    error_msg = f"failed to index {page_url}: {exc}"
                    errors.append(error_msg)
                    log.warning(error_msg)

        else:
            errors.append(f"unsupported source type: {source_type}")

        # Update last_indexed_at timestamp only on success
        conn.execute(
            "UPDATE corpus_sources SET last_indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (source_id,),
        )
        conn.commit()

    except Exception:
        conn.rollback()
        raise

    return {
        "files_processed": files_processed,
        "chunks_created": chunks_created,
        "errors": errors,
    }


# ── Retrieval: semantic search over embedded chunks ──────────────────────────

def retrieve_chunks(
    conn: sqlite3.Connection,
    corpus_id: int,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top-K most relevant chunks for a query from a corpus.

    Uses cosine similarity over L2-normalized embeddings.

    Args:
        conn: SQLite connection with corpus_chunks table
        corpus_id: Corpus ID to search in
        query: Query text to retrieve chunks for
        top_k: Number of top results to return (default 5)

    Returns:
        List of dicts with keys: id, source_file, chunk_index, content, score.
        Returns empty list if embedding fails or no chunks exist for corpus.
    """
    # Embed the query
    query_vec = embed_text(query)
    if query_vec is None:
        return []

    # Convert query embedding to float32 numpy array and L2-normalize
    query_arr = np.array(query_vec, dtype=np.float32)
    query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)

    # Fetch all chunks for this corpus with embeddings
    rows = conn.execute(
        "SELECT id, source_file, chunk_index, content, embedding FROM corpus_chunks WHERE corpus_id = ? AND embedding IS NOT NULL",
        (corpus_id,),
    ).fetchall()

    if not rows:
        return []

    # Stack embeddings into a matrix and normalize
    matrix = np.stack([blob_to_vec(row["embedding"]) for row in rows])  # shape (n, dim)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / (norms + 1e-8)

    # Compute cosine similarities
    similarities = np.dot(normalized, query_norm)

    # Get top-K indices
    k = min(top_k, len(rows))
    top_indices = np.argsort(similarities)[-k:][::-1]

    # Build results
    results = []
    for idx in top_indices:
        results.append({
            "id": rows[idx]["id"],
            "source_file": rows[idx]["source_file"],
            "chunk_index": rows[idx]["chunk_index"],
            "content": rows[idx]["content"],
            "score": float(similarities[idx]),
        })

    return results

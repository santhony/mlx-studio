"""Tests for skills.py parsing and vectorization."""

import sqlite3
import tempfile
from pathlib import Path
import sys

import numpy as np
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from skills import (
    _parse_skill_file,
    _vec_to_blob,
    _blob_to_vec,
)


class TestFrontmatterParsing:
    """Test skill file frontmatter parsing."""

    def test_parse_skill_with_frontmatter(self):
        """Test parsing a skill file with YAML frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(
                """---
name: Python Basics
description: Learn the fundamentals of Python
---

# Python Basics

This is the skill content."""
            )
            f.flush()
            path = Path(f.name)

            try:
                skill = _parse_skill_file(path)
                assert skill["name"] == "Python Basics"
                assert skill["description"] == "Learn the fundamentals of Python"
                assert "This is the skill content." in skill["content"]
                assert skill["filepath"] == str(path)
            finally:
                path.unlink()

    def test_parse_skill_without_frontmatter(self):
        """Test parsing a skill file without frontmatter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# My Skill\n\nThis is content.")
            f.flush()
            path = Path(f.name)

            try:
                skill = _parse_skill_file(path)
                # Name derived from filename
                assert skill["name"] == str(path.stem).replace("-", " ").replace("_", " ").title()
                assert skill["description"] == ""
                assert "This is content." in skill["content"]
            finally:
                path.unlink()

    def test_parse_skill_malformed_frontmatter(self):
        """Test parsing a skill file with malformed frontmatter (treated as plain text)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("---\nmalformed: [unclosed list\n---\nContent here")
            f.flush()
            path = Path(f.name)

            try:
                skill = _parse_skill_file(path)
                # Should not crash; either parses as frontmatter or falls back to plain text
                assert "filepath" in skill
                assert "name" in skill
                assert "content" in skill
            finally:
                path.unlink()


class TestVectorSerialization:
    """Test vector blob serialization and round-trip."""

    def test_vec_to_blob_and_back(self):
        """Test that vector -> blob -> vector round-trip preserves data."""
        vec = [1.0, 2.5, -0.3, 0.0, 100.5]
        blob = _vec_to_blob(vec)
        assert isinstance(blob, bytes)

        result = _blob_to_vec(blob)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, vec, decimal=5)

    def test_blob_from_embedding_vector(self):
        """Test blob serialization with typical embedding vector (384-dim)."""
        # Simulate a realistic embedding vector (all-MiniLM-L6-v2 output)
        embedding = np.random.randn(384).astype(np.float32).tolist()
        blob = _vec_to_blob(embedding)
        assert isinstance(blob, bytes)
        assert len(blob) == 384 * 4  # float32 = 4 bytes per element

        result = _blob_to_vec(blob)
        assert result.shape == (384,)
        np.testing.assert_array_almost_equal(result, embedding, decimal=5)

    def test_vec_to_blob_empty(self):
        """Test serialization of empty vector."""
        vec = []
        blob = _vec_to_blob(vec)
        assert isinstance(blob, bytes)
        assert len(blob) == 0

    def test_vec_to_blob_single_element(self):
        """Test serialization of single-element vector."""
        vec = [3.14159]
        blob = _vec_to_blob(vec)
        result = _blob_to_vec(blob)
        np.testing.assert_almost_equal(result[0], 3.14159, decimal=5)


class TestCosineSimilarityVectorized:
    """Test cosine similarity computation (vectorized)."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors is 1.0."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        norm = vec / (np.linalg.norm(vec) + 1e-8)
        similarity = np.dot(norm, norm)
        np.testing.assert_almost_equal(similarity, 1.0, decimal=5)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors is ~0."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        norm1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        norm2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        similarity = np.dot(norm1, norm2)
        np.testing.assert_almost_equal(similarity, 0.0, decimal=5)

    def test_cosine_similarity_vectorized_batch(self):
        """Test vectorized cosine similarity for multiple vectors."""
        # Query vector
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Database of vectors
        vectors = np.array([
            [1.0, 0.0, 0.0],  # identical to query
            [0.0, 1.0, 0.0],  # orthogonal to query
            [1.0, 1.0, 0.0],  # 45 degree angle
        ], dtype=np.float32)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / (norms + 1e-8)
        similarities = np.dot(normalized, query_norm)

        # Check shapes and values
        assert similarities.shape == (3,)
        np.testing.assert_almost_equal(similarities[0], 1.0, decimal=5)  # identical
        np.testing.assert_almost_equal(similarities[1], 0.0, decimal=5)  # orthogonal
        assert 0.0 < similarities[2] < 1.0  # 45 degree should be between

    def test_cosine_similarity_top_k_selection(self):
        """Test top-k selection using argsort."""
        similarities = np.array([0.3, 0.9, 0.1, 0.8, 0.5], dtype=np.float32)
        k = 3
        top_indices = np.argsort(similarities)[-k:][::-1]

        assert len(top_indices) == 3
        assert top_indices[0] == 1  # highest (0.9)
        assert top_indices[1] == 3  # second (0.8)
        assert top_indices[2] == 4  # third (0.5)
        # Verify sorted in descending order
        assert similarities[top_indices[0]] >= similarities[top_indices[1]]
        assert similarities[top_indices[1]] >= similarities[top_indices[2]]

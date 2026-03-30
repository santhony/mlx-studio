"""Tests for agent_tools.py allowlist enforcement."""

import pytest
import tempfile
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_tools import _check_allowed


class TestAllowlistEnforcement:
    """Test filesystem allowlist enforcement."""

    def test_path_within_allowed_directory(self):
        """Test that paths within allowed directory are accepted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = Path(tmpdir)
            test_file = allowed_dir / "test.txt"

            ok, result = _check_allowed(str(test_file), [str(allowed_dir)])
            assert ok is True
            # result should be the resolved path
            assert str(test_file.resolve()) in result or result == str(test_file.resolve())

    def test_path_outside_allowed_directory(self):
        """Test that paths outside allowed directory are rejected."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                allowed_dir = Path(tmpdir1)
                outside_file = Path(tmpdir2) / "test.txt"

                ok, result = _check_allowed(str(outside_file), [str(allowed_dir)])
                assert ok is False
                assert "outside the allowed directories" in result

    def test_path_traversal_attempt(self):
        """Test that path traversal attacks (../) are blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = Path(tmpdir)
            # Try to escape with ..
            malicious_path = str(allowed_dir / ".." / "etc" / "passwd")

            ok, result = _check_allowed(malicious_path, [str(allowed_dir)])
            assert ok is False
            assert "outside the allowed directories" in result

    def test_symlink_traversal_blocked(self):
        """Test that symlink traversal is blocked (resolved path check)."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                allowed_dir = Path(tmpdir1)
                outside_dir = Path(tmpdir2)
                target_file = outside_dir / "secret.txt"
                target_file.write_text("secret")

                # Create symlink inside allowed dir pointing outside
                symlink_path = allowed_dir / "link_to_secret"
                try:
                    symlink_path.symlink_to(target_file)
                except (OSError, NotImplementedError):
                    # Symlinks may not be supported on all systems
                    pytest.skip("Symlinks not supported on this system")

                # Try to access via symlink
                ok, result = _check_allowed(str(symlink_path), [str(allowed_dir)])
                # Should be rejected because resolved path is outside
                assert ok is False

    def test_multiple_allowed_directories(self):
        """Test allowlist with multiple allowed directories."""
        with tempfile.TemporaryDirectory() as tmpdir1:
            with tempfile.TemporaryDirectory() as tmpdir2:
                allowed_dirs = [tmpdir1, tmpdir2]

                # File in first allowed dir
                file1 = Path(tmpdir1) / "test.txt"
                ok, result = _check_allowed(str(file1), allowed_dirs)
                assert ok is True

                # File in second allowed dir
                file2 = Path(tmpdir2) / "test.txt"
                ok, result = _check_allowed(str(file2), allowed_dirs)
                assert ok is True

                # File outside both
                with tempfile.TemporaryDirectory() as tmpdir3:
                    file3 = Path(tmpdir3) / "test.txt"
                    ok, result = _check_allowed(str(file3), allowed_dirs)
                    assert ok is False

    def test_allowed_directory_itself(self):
        """Test that the allowed directory itself is accessible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = tmpdir

            ok, result = _check_allowed(allowed_dir, [allowed_dir])
            assert ok is True

    def test_subdirectory_within_allowed(self):
        """Test that subdirectories within allowed directory are accessible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = Path(tmpdir)
            subdir = allowed_dir / "subdir" / "nested"

            ok, result = _check_allowed(str(subdir), [str(allowed_dir)])
            assert ok is True

    def test_invalid_path_syntax(self):
        """Test handling of invalid path syntax."""
        # Empty string or null bytes should be handled gracefully
        ok, result = _check_allowed("", ["/tmp"])
        assert ok is False
        assert "outside the allowed directories" in result or "invalid path" in result

    def test_relative_path_conversion(self):
        """Test that relative paths are resolved before checking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            allowed_dir = Path(tmpdir)
            # Create a file in the allowed directory
            (allowed_dir / "test.txt").write_text("test")

            # Use a relative path that points to the file
            # Note: This depends on current working directory
            # So we test the absolute path conversion
            abs_file = allowed_dir / "test.txt"
            ok, result = _check_allowed(str(abs_file), [str(allowed_dir)])
            assert ok is True

    def test_empty_allowed_dirs_list(self):
        """Test behavior with empty allowed directories list."""
        ok, result = _check_allowed("/tmp/test.txt", [])
        assert ok is False
        assert "outside the allowed directories" in result

    def test_nonexistent_allowed_dir(self):
        """Test that nonexistent allowed directory doesn't block valid paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            real_allowed = Path(tmpdir)
            fake_allowed = "/nonexistent/path/that/does/not/exist"

            file_path = real_allowed / "test.txt"

            # Check with both a fake and real allowed directory
            ok, result = _check_allowed(str(file_path), [fake_allowed, str(real_allowed)])
            assert ok is True
            # The function should skip the nonexistent fake_allowed and match against real_allowed

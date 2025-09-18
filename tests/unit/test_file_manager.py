"""
Comprehensive test suite for file manager with high coverage targeting 75%+.

Tests cover all critical functionality including error scenarios, edge cases,
and production-ready error handling patterns.
"""

import os
from unittest.mock import patch

import pytest

from core.file_manager import TempFileManager, create_file_manager


class TestTempFileManager:
    """Test suite for TempFileManager with comprehensive coverage."""

    def test_init_default_settings(self):
        """Test initialization with default settings."""
        manager = TempFileManager()

        assert manager.cleanup_on_exit is True
        assert manager.temp_files == []
        assert manager.temp_dirs == []
        assert manager.session_dir is None
        assert manager.base_temp_dir is not None

    def test_init_custom_base_dir(self, tmp_path):
        """Test initialization with custom base directory."""
        custom_dir = str(tmp_path / "custom_temp")
        manager = TempFileManager(base_temp_dir=custom_dir, cleanup_on_exit=False)

        assert manager.base_temp_dir == custom_dir
        assert manager.cleanup_on_exit is False
        assert os.path.exists(custom_dir)

    def test_init_directory_creation_failure(self):
        """Test handling of base directory creation failure."""
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
            with patch("tempfile.gettempdir", return_value="/tmp"):
                manager = TempFileManager(base_temp_dir="/invalid/path")
                # Should fall back to system temp
                assert manager.base_temp_dir == "/tmp"

    def test_create_session_dir_success(self, tmp_path):
        """Test successful session directory creation."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        session_dir = manager.create_session_dir()

        assert session_dir is not None
        assert os.path.exists(session_dir)
        assert session_dir.startswith(str(tmp_path))
        assert session_dir in manager.temp_dirs
        assert manager.session_dir == session_dir

    def test_create_session_dir_failure(self, tmp_path):
        """Test session directory creation failure."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch("pathlib.Path.mkdir", side_effect=OSError("Cannot create directory")):
            with pytest.raises(RuntimeError, match="Cannot create temporary directory"):
                manager.create_session_dir()

    def test_get_temp_path_with_session_dir(self, tmp_path):
        """Test getting temp path with existing session directory."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        session_dir = manager.create_session_dir()

        temp_path = manager.get_temp_path("test_file.txt")
        expected = os.path.join(session_dir, "test_file.txt")

        assert temp_path == expected

    def test_get_temp_path_without_session_dir_create(self, tmp_path):
        """Test getting temp path without session dir but with create_dir=True."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        temp_path = manager.get_temp_path("test_file.txt", create_dir=True)

        assert manager.session_dir is not None
        assert temp_path.startswith(manager.session_dir)

    def test_get_temp_path_without_session_dir_no_create(self, tmp_path):
        """Test getting temp path without session dir and create_dir=False."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        temp_path = manager.get_temp_path("test_file.txt", create_dir=False)
        expected = os.path.join(str(tmp_path), "test_file.txt")

        assert temp_path == expected
        assert manager.session_dir is None

    def test_create_temp_file_success(self, tmp_path):
        """Test successful temporary file creation."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        temp_file = manager.create_temp_file(suffix=".wav", prefix="audio_")

        assert temp_file is not None
        assert temp_file.endswith(".wav")
        assert "audio_" in temp_file
        assert temp_file in manager.temp_files
        assert os.path.exists(temp_file)

    def test_create_temp_file_with_existing_session(self, tmp_path):
        """Test temp file creation with existing session directory."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        session_dir = manager.create_session_dir()

        temp_file = manager.create_temp_file(suffix=".mp4")

        assert temp_file.startswith(session_dir)
        assert temp_file.endswith(".mp4")

    def test_create_temp_file_failure(self, tmp_path):
        """Test temp file creation failure."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch("tempfile.mkstemp", side_effect=OSError("Cannot create file")):
            with pytest.raises(RuntimeError, match="Cannot create temporary file"):
                manager.create_temp_file()

    def test_add_temp_file_existing_file(self, tmp_path):
        """Test adding existing file to cleanup list."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        manager.add_temp_file(str(test_file))

        assert str(test_file) in manager.temp_files

    def test_add_temp_file_nonexistent(self, tmp_path):
        """Test adding non-existent file (should not be added)."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        nonexistent = str(tmp_path / "nonexistent.txt")

        manager.add_temp_file(nonexistent)

        assert nonexistent not in manager.temp_files

    def test_add_temp_file_none(self, tmp_path):
        """Test adding None as file path."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        manager.add_temp_file(None)

        assert None not in manager.temp_files

    def test_remove_file_tracked_and_exists(self, tmp_path):
        """Test removing file that is tracked and exists."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        manager.temp_files.append(str(test_file))

        result = manager.remove_file(str(test_file))

        assert result is True
        assert str(test_file) not in manager.temp_files
        assert not test_file.exists()

    def test_remove_file_tracked_not_exists(self, tmp_path):
        """Test removing file that is tracked but doesn't exist."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        nonexistent = str(tmp_path / "nonexistent.txt")
        manager.temp_files.append(nonexistent)

        result = manager.remove_file(nonexistent)

        assert result is True
        assert nonexistent not in manager.temp_files

    def test_remove_file_not_tracked_but_exists(self, tmp_path):
        """Test removing file that exists but is not tracked."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        result = manager.remove_file(str(test_file))

        assert result is True
        assert not test_file.exists()

    def test_remove_file_error_handling(self, tmp_path):
        """Test remove file error handling."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        test_file = str(tmp_path / "test.txt")
        manager.temp_files.append(test_file)

        with patch("os.remove", side_effect=PermissionError("Access denied")):
            result = manager.remove_file(test_file)

        assert result is False
        assert test_file not in manager.temp_files  # Should still be removed from tracking

    def test_cleanup_session_success(self, tmp_path):
        """Test successful session cleanup."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        session_dir = manager.create_session_dir()

        # Create some test files
        test_file1 = tmp_path / "test1.txt"
        test_file2 = tmp_path / "test2.txt"
        test_file1.write_text("content1")
        test_file2.write_text("content2")

        manager.temp_files.extend([str(test_file1), str(test_file2)])

        result = manager.cleanup_session()

        assert result is True
        assert manager.temp_files == []
        assert manager.session_dir is None
        assert not os.path.exists(session_dir)

    def test_cleanup_session_with_errors(self, tmp_path):
        """Test session cleanup with some errors."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        manager.create_session_dir()

        # Create a file that will fail to remove
        problem_file = str(tmp_path / "problem.txt")
        manager.temp_files.append(problem_file)

        with patch.object(manager, "remove_file", return_value=False):
            result = manager.cleanup_session()

        assert result is False  # Should indicate failure

    def test_cleanup_session_directory_removal_error(self, tmp_path):
        """Test session cleanup with directory removal error."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        manager.create_session_dir()

        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            result = manager.cleanup_session()

        assert result is False

    def test_cleanup_all_success(self, tmp_path):
        """Test complete cleanup success."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Create session and additional directories
        manager.create_session_dir()
        extra_dir = str(tmp_path / "extra")
        os.makedirs(extra_dir)
        manager.temp_dirs.append(extra_dir)

        result = manager.cleanup_all()

        assert result is True
        assert manager.temp_dirs == []
        assert manager.temp_files == []

    def test_cleanup_all_with_errors(self, tmp_path):
        """Test cleanup all with some errors."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Add problematic directory
        problem_dir = str(tmp_path / "problem")
        manager.temp_dirs.append(problem_dir)

        with patch("shutil.rmtree", side_effect=PermissionError("Access denied")):
            result = manager.cleanup_all()

        assert result is False

    def test_get_disk_usage_success(self, tmp_path):
        """Test successful disk usage retrieval."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        usage = manager.get_disk_usage()

        assert "total" in usage
        assert "used" in usage
        assert "free" in usage
        assert "free_mb" in usage
        assert usage["free_mb"] > 0

    def test_get_disk_usage_error(self, tmp_path):
        """Test disk usage retrieval error."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch("shutil.disk_usage", side_effect=OSError("Disk error")):
            usage = manager.get_disk_usage()

        assert usage == {"total": 0, "used": 0, "free": 0, "free_mb": 0}

    def test_check_disk_space_sufficient(self, tmp_path):
        """Test disk space check with sufficient space."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch.object(manager, "get_disk_usage", return_value={"free_mb": 1000}):
            result = manager.check_disk_space(required_mb=500)

        assert result is True

    def test_check_disk_space_insufficient(self, tmp_path):
        """Test disk space check with insufficient space."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch.object(manager, "get_disk_usage", return_value={"free_mb": 100}):
            result = manager.check_disk_space(required_mb=500)

        assert result is False

    def test_check_disk_space_error(self, tmp_path):
        """Test disk space check with error."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with patch.object(manager, "get_disk_usage", return_value={"free_mb": 0}):
            result = manager.check_disk_space(required_mb=500)

        assert result is False

    def test_temp_file_context_success(self, tmp_path):
        """Test temp file context manager success."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        with manager.temp_file_context(suffix=".txt", prefix="ctx_") as temp_path:
            assert temp_path is not None
            assert temp_path.endswith(".txt")
            assert "ctx_" in temp_path
            assert os.path.exists(temp_path)

            # Write some content
            with open(temp_path, "w") as f:
                f.write("test content")

        # File should be cleaned up after context
        assert not os.path.exists(temp_path)

    def test_temp_file_context_exception(self, tmp_path):
        """Test temp file context manager with exception."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        temp_path = None
        try:
            with manager.temp_file_context(suffix=".txt") as temp_path:
                assert os.path.exists(temp_path)
                raise ValueError("Test exception")
        except ValueError:
            pass

        # File should still be cleaned up
        assert not os.path.exists(temp_path)

    def test_context_manager_enter_exit(self, tmp_path):
        """Test TempFileManager as context manager."""
        temp_files_created = []

        with TempFileManager(base_temp_dir=str(tmp_path)) as manager:
            temp_file = manager.create_temp_file(suffix=".txt")
            temp_files_created.append(temp_file)
            assert os.path.exists(temp_file)

        # Files should be cleaned up after context exit
        for temp_file in temp_files_created:
            assert not os.path.exists(temp_file)

    def test_destructor_cleanup(self, tmp_path):
        """Test cleanup on object destruction."""
        temp_file = None

        # Create manager and temp file
        manager = TempFileManager(base_temp_dir=str(tmp_path), cleanup_on_exit=True)
        temp_file = manager.create_temp_file(suffix=".txt")
        assert os.path.exists(temp_file)

        # Mock cleanup_all to verify it's called
        with patch.object(manager, "cleanup_all") as mock_cleanup:
            del manager
            mock_cleanup.assert_called_once()

    def test_destructor_cleanup_disabled(self, tmp_path):
        """Test no cleanup when cleanup_on_exit=False."""
        manager = TempFileManager(base_temp_dir=str(tmp_path), cleanup_on_exit=False)
        manager.create_temp_file(suffix=".txt")

        with patch.object(manager, "cleanup_all") as mock_cleanup:
            del manager
            mock_cleanup.assert_not_called()

    def test_destructor_cleanup_exception(self, tmp_path):
        """Test destructor handles cleanup exceptions gracefully."""
        manager = TempFileManager(base_temp_dir=str(tmp_path), cleanup_on_exit=True)

        with patch.object(manager, "cleanup_all", side_effect=Exception("Cleanup error")):
            # Should not raise exception
            del manager

    def test_atexit_registration(self, tmp_path):
        """Test atexit cleanup registration."""
        with patch("atexit.register") as mock_register:
            manager = TempFileManager(base_temp_dir=str(tmp_path), cleanup_on_exit=True)
            mock_register.assert_called_once_with(manager.cleanup_all)

    def test_atexit_not_registered(self, tmp_path):
        """Test atexit cleanup not registered when disabled."""
        with patch("atexit.register") as mock_register:
            TempFileManager(base_temp_dir=str(tmp_path), cleanup_on_exit=False)
            mock_register.assert_not_called()


class TestConvenienceFunction:
    """Test convenience function for creating file managers."""

    def test_create_file_manager_default(self):
        """Test creating file manager with defaults."""
        manager = create_file_manager()

        assert isinstance(manager, TempFileManager)
        assert manager.cleanup_on_exit is True

    def test_create_file_manager_no_cleanup(self):
        """Test creating file manager without cleanup."""
        manager = create_file_manager(cleanup_on_exit=False)

        assert isinstance(manager, TempFileManager)
        assert manager.cleanup_on_exit is False


class TestErrorScenarios:
    """Test various error scenarios and edge cases."""

    def test_multiple_session_directory_creation(self, tmp_path):
        """Test creating multiple session directories."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        session1 = manager.create_session_dir()
        session2 = manager.create_session_dir()

        # Second call should update session_dir
        assert manager.session_dir == session2
        assert session1 != session2
        assert len(manager.temp_dirs) == 2

    def test_large_number_of_temp_files(self, tmp_path):
        """Test handling large number of temporary files."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        temp_files = []
        for i in range(100):
            temp_file = manager.create_temp_file(suffix=f"_{i}.txt")
            temp_files.append(temp_file)

        assert len(manager.temp_files) == 100

        # Cleanup should handle all files
        result = manager.cleanup_all()
        assert result is True
        assert len(manager.temp_files) == 0

    def test_concurrent_access_simulation(self, tmp_path):
        """Test simulation of concurrent access patterns."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))
        session_dir = manager.create_session_dir()

        # Simulate multiple "threads" creating files
        files = []
        for i in range(10):
            temp_file = manager.create_temp_file(prefix=f"thread_{i}_")
            files.append(temp_file)

        # All files should be in the same session
        for file_path in files:
            assert file_path.startswith(session_dir)

        assert len(manager.temp_files) == 10

    def test_path_edge_cases(self, tmp_path):
        """Test various path edge cases."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Test with empty filename
        path = manager.get_temp_path("")
        assert path.endswith("/")

        # Test with filename containing spaces
        path = manager.get_temp_path("file with spaces.txt")
        assert "file with spaces.txt" in path

        # Test with unicode filename
        path = manager.get_temp_path("файл.txt")
        assert "файл.txt" in path


class TestIntegrationScenarios:
    """Integration-style tests for realistic usage patterns."""

    def test_typical_video_processing_workflow(self, tmp_path):
        """Test typical video processing file management workflow."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Check disk space before starting
        assert manager.check_disk_space(required_mb=100)

        # Create session for this processing run
        session_dir = manager.create_session_dir()

        # Create temporary files for different stages
        video_file = manager.create_temp_file(suffix=".mp4", prefix="video_")
        audio_file = manager.create_temp_file(suffix=".wav", prefix="audio_")
        transcript_file = manager.create_temp_file(suffix=".txt", prefix="transcript_")

        # Verify all files exist and are tracked
        assert os.path.exists(video_file)
        assert os.path.exists(audio_file)
        assert os.path.exists(transcript_file)
        assert len(manager.temp_files) == 3

        # Simulate processing by writing to files
        with open(audio_file, "w") as f:
            f.write("audio data")
        with open(transcript_file, "w") as f:
            f.write("transcribed text")

        # Clean up after processing
        result = manager.cleanup_session()
        assert result is True

        # Verify cleanup
        assert not os.path.exists(video_file)
        assert not os.path.exists(audio_file)
        assert not os.path.exists(transcript_file)
        assert not os.path.exists(session_dir)

    def test_error_recovery_workflow(self, tmp_path):
        """Test file management during error recovery scenarios."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Start processing
        session_dir = manager.create_session_dir()
        manager.create_temp_file(suffix=".mp4")

        # Simulate partial processing creating additional files
        extra_file = os.path.join(session_dir, "extra_output.wav")
        with open(extra_file, "w") as f:
            f.write("partial data")
        manager.add_temp_file(extra_file)

        # Simulate error condition - some files might be locked
        with patch("os.remove", side_effect=[PermissionError("Locked"), None]):
            result = manager.cleanup_session()
            # Should handle error gracefully
            assert result is False

        # Directory should still be cleaned up even if some files failed
        assert not os.path.exists(session_dir)

    def test_disk_space_monitoring_workflow(self, tmp_path):
        """Test workflow with disk space monitoring."""
        manager = TempFileManager(base_temp_dir=str(tmp_path))

        # Check initial disk space
        usage = manager.get_disk_usage()
        assert usage["free_mb"] > 0

        # Verify sufficient space for operation
        required_space = 500  # MB
        if manager.check_disk_space(required_mb=required_space):
            # Proceed with processing
            temp_file = manager.create_temp_file(suffix=".large")
            assert os.path.exists(temp_file)
        else:
            # Handle insufficient space
            pytest.skip("Insufficient disk space for test")

        # Clean up
        manager.cleanup_all()

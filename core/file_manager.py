"""
Refactored file manager with improved patterns and DRY principles.

This version consolidates file management patterns and provides
better organization and cleanup capabilities.
"""

import atexit
import logging
import os
import shutil
import tempfile
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

from utils.common_patterns import (
    OperationResult,
    PathManager,
    cleanup_directory_safely,
    cleanup_file_safely,
    handle_operation_error,
)

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Types of files managed by the file manager."""

    VIDEO = "video"
    AUDIO = "audio"
    TEMPORARY = "temporary"
    SESSION = "session"
    LOG = "log"


@dataclass
class ManagedFile:
    """Information about a managed file."""

    path: Path
    file_type: FileType
    session_id: Optional[str] = None
    created_time: Optional[float] = None
    cleanup_on_exit: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "path": str(self.path),
            "type": self.file_type.value,
            "session_id": self.session_id,
            "cleanup_on_exit": self.cleanup_on_exit,
            "exists": self.path.exists(),
            "size": self.path.stat().st_size if self.path.exists() else 0,
        }


class SessionManager:
    """Manages file sessions with automatic cleanup."""

    def __init__(self, base_temp_dir: Optional[Union[str, Path]] = None):
        """
        Initialize session manager.

        Args:
            base_temp_dir: Base directory for temporary files
        """
        self.base_temp_dir = Path(base_temp_dir) if base_temp_dir else Path(tempfile.gettempdir())
        self.sessions: dict[str, Path] = {}
        self._lock = threading.RLock()

        # Ensure base directory exists (with error handling)
        try:
            PathManager.ensure_directory(self.base_temp_dir)
        except Exception as e:
            logger.warning(f"Failed to create session manager base directory {self.base_temp_dir}: {e}")
            # Fall back to system temp directory
            self.base_temp_dir = Path(tempfile.gettempdir())
            # System temp directory should already exist, but try to ensure it
            try:
                PathManager.ensure_directory(self.base_temp_dir)
            except Exception as fallback_error:
                logger.warning(f"Failed to ensure system temp directory {self.base_temp_dir}: {fallback_error}")
                # Use the directory as-is, assuming it exists

    def create_session(self, session_prefix: str = "transcription") -> str:
        """
        Create a new session directory.

        Args:
            session_prefix: Prefix for session directory name

        Returns:
            Session ID

        Raises:
            RuntimeError: If session directory creation fails
        """
        with self._lock:
            session_id = f"{session_prefix}_{uuid.uuid4().hex[:8]}"
            session_dir = self.base_temp_dir / session_id

            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                self.sessions[session_id] = session_dir

                logger.info(f"Created session {session_id}: {session_dir}")
                return session_id

            except Exception as e:
                logger.error(f"Failed to create session {session_id}: {e}")
                raise RuntimeError(f"Cannot create temporary directory: {e}") from e

    def get_session_dir(self, session_id: str) -> Optional[Path]:
        """Get session directory path."""
        with self._lock:
            return self.sessions.get(session_id)

    def cleanup_session(self, session_id: str) -> bool:
        """
        Clean up a specific session.

        Args:
            session_id: Session to clean up

        Returns:
            True if cleanup successful
        """
        with self._lock:
            session_dir = self.sessions.get(session_id)
            if not session_dir:
                return True  # Session doesn't exist, consider it cleaned

            success = cleanup_directory_safely(session_dir)
            if success:
                del self.sessions[session_id]
                logger.info(f"Cleaned up session {session_id}")
            else:
                logger.warning(f"Failed to completely clean up session {session_id}")

            return success

    def cleanup_all_sessions(self) -> dict[str, bool]:
        """
        Clean up all active sessions.

        Returns:
            Dictionary mapping session IDs to cleanup success
        """
        results = {}

        with self._lock:
            session_ids = list(self.sessions.keys())

        for session_id in session_ids:
            results[session_id] = self.cleanup_session(session_id)

        return results

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """
        List all active sessions with information.

        Returns:
            Dictionary mapping session IDs to session info
        """
        with self._lock:
            sessions_info = {}

            for session_id, session_dir in self.sessions.items():
                sessions_info[session_id] = {
                    "path": str(session_dir),
                    "exists": session_dir.exists(),
                    "file_count": len(list(session_dir.iterdir())) if session_dir.exists() else 0,
                }

            return sessions_info


class RefactoredTempFileManager:
    """
    Refactored temporary file manager with improved organization and DRY principles.
    """

    def __init__(
        self,
        base_temp_dir: Optional[Union[str, Path]] = None,
        cleanup_on_exit: bool = True,
        session_prefix: str = "transcription",
    ):
        """
        Initialize refactored file manager.

        Args:
            base_temp_dir: Base directory for temporary files
            cleanup_on_exit: Whether to cleanup files on exit
            session_prefix: Prefix for session directories
        """
        # Handle base directory with fallback
        if base_temp_dir:
            self.base_temp_dir = Path(base_temp_dir)
            try:
                # Try to ensure directory exists
                PathManager.ensure_directory(self.base_temp_dir)
            except Exception as e:
                logger.warning(f"Failed to create base directory {self.base_temp_dir}: {e}")
                # Fall back to system temp
                self.base_temp_dir = Path(tempfile.gettempdir())
        else:
            self.base_temp_dir = Path(tempfile.gettempdir())

        self.cleanup_on_exit = cleanup_on_exit
        self.session_prefix = session_prefix

        # Track managed files
        self.managed_files: dict[str, ManagedFile] = {}
        self.current_session_id: Optional[str] = None
        self._lock = threading.RLock()

        # Initialize session manager (after setting up other attributes)
        try:
            self.session_manager = SessionManager(self.base_temp_dir)
        except Exception as e:
            logger.warning(f"Failed to initialize session manager: {e}")
            # Try to set up a basic session manager with fallback directory
            try:
                self.session_manager = SessionManager(Path(tempfile.gettempdir()))
            except Exception as fallback_error:
                logger.warning(f"Failed to initialize fallback session manager: {fallback_error}")
                # Create a minimal session manager that can still function
                self.session_manager = None

        # Register cleanup on exit if enabled
        if cleanup_on_exit:
            atexit.register(self.cleanup_all)

        logger.info(f"RefactoredTempFileManager initialized: {self.base_temp_dir}")

    @property
    def base_temp_dir_path(self) -> Path:
        """Get base temp directory as Path object."""
        return self.base_temp_dir

    def get_base_temp_dir_str(self) -> str:
        """Get base temp directory as string for backward compatibility."""
        return str(self.base_temp_dir)

    def create_session_dir(self) -> str:
        """
        Create a new session directory and set it as current.

        Returns:
            Session directory path

        Raises:
            RuntimeError: If session directory creation fails
        """
        with self._lock:
            try:
                if not self.session_manager:
                    raise RuntimeError("Session manager not available")
                self.current_session_id = self.session_manager.create_session(self.session_prefix)
                session_dir = self.session_manager.get_session_dir(self.current_session_id)

                logger.info(f"Created session directory: {session_dir}")
                return str(session_dir)
            except Exception as e:
                logger.error(f"Failed to create session directory: {e}")
                raise RuntimeError(f"Cannot create temporary directory: {e}") from e

    def get_temp_path(
        self, filename: str, file_type: FileType = FileType.TEMPORARY, session_id: Optional[str] = None
    ) -> str:
        """
        Generate a temporary file path within a session.

        Args:
            filename: Name of the file
            file_type: Type of file being created
            session_id: Optional specific session ID (uses current if None)

        Returns:
            Generated file path
        """
        with self._lock:
            # Use provided session or current session
            target_session_id = session_id or self.current_session_id

            if not target_session_id:
                # Create a session if none exists
                target_session_id = self.create_session_dir()
                target_session_id = self.current_session_id

            if not self.session_manager:
                raise ValueError("Session manager not available")
            session_dir = self.session_manager.get_session_dir(target_session_id)
            if not session_dir:
                raise ValueError(f"Session {target_session_id} does not exist")

            # Generate file path
            file_path = session_dir / filename

            # Track the file
            file_id = str(uuid.uuid4())
            managed_file = ManagedFile(
                path=file_path, file_type=file_type, session_id=target_session_id, cleanup_on_exit=self.cleanup_on_exit
            )

            self.managed_files[file_id] = managed_file

            logger.debug(f"Generated temp path: {file_path} (type: {file_type.value})")
            return str(file_path)

    def track_file(
        self,
        file_path: Union[str, Path],
        file_type: FileType = FileType.TEMPORARY,
        cleanup_on_exit: Optional[bool] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Track an existing file for management.

        Args:
            file_path: Path to file to track
            file_type: Type of file
            cleanup_on_exit: Override cleanup setting
            metadata: Optional metadata about the file

        Returns:
            File tracking ID
        """
        with self._lock:
            file_path = Path(file_path)
            file_id = str(uuid.uuid4())

            managed_file = ManagedFile(
                path=file_path,
                file_type=file_type,
                session_id=self.current_session_id,
                cleanup_on_exit=cleanup_on_exit if cleanup_on_exit is not None else self.cleanup_on_exit,
                metadata=metadata or {},
            )

            self.managed_files[file_id] = managed_file

            logger.debug(f"Tracking file: {file_path} (ID: {file_id}, type: {file_type.value})")
            return file_id

    def untrack_file(self, file_id: str) -> bool:
        """
        Stop tracking a file without cleaning it up.

        Args:
            file_id: File tracking ID

        Returns:
            True if file was being tracked
        """
        with self._lock:
            managed_file = self.managed_files.pop(file_id, None)
            if managed_file:
                logger.debug(f"Stopped tracking file: {managed_file.path}")
                return True
            return False

    def cleanup_file(self, file_id: str) -> bool:
        """
        Clean up a specific tracked file.

        Args:
            file_id: File tracking ID

        Returns:
            True if cleanup successful
        """
        with self._lock:
            managed_file = self.managed_files.get(file_id)
            if not managed_file:
                return True  # File not tracked, consider it cleaned

            success = cleanup_file_safely(managed_file.path)
            if success:
                del self.managed_files[file_id]
                logger.debug(f"Cleaned up tracked file: {managed_file.path}")
            else:
                logger.warning(f"Failed to clean up tracked file: {managed_file.path}")

            return success

    def cleanup_files_by_type(self, file_type: FileType) -> dict[str, bool]:
        """
        Clean up all files of a specific type.

        Args:
            file_type: Type of files to clean up

        Returns:
            Dictionary mapping file IDs to cleanup success
        """
        results = {}

        with self._lock:
            # Find files of specified type
            files_to_cleanup = {
                file_id: managed_file
                for file_id, managed_file in self.managed_files.items()
                if managed_file.file_type == file_type and managed_file.cleanup_on_exit
            }

        # Clean up each file
        for file_id in files_to_cleanup:
            results[file_id] = self.cleanup_file(file_id)

        return results

    def cleanup_session(self) -> bool:
        """
        Clean up current session (backward compatibility).

        Returns:
            True if cleanup successful
        """
        if not self.current_session_id:
            return True

        # Clean up tracked files in this session first
        with self._lock:
            session_files = {
                file_id: managed_file
                for file_id, managed_file in self.managed_files.items()
                if managed_file.session_id == self.current_session_id and managed_file.cleanup_on_exit
            }

        # Clean up individual tracked files
        all_success = True
        for file_id in session_files:
            if not self.cleanup_file(file_id):
                all_success = False

        # Clean up session directory (but catch exceptions)
        try:
            if self.session_manager:
                session_success = self.session_manager.cleanup_session(self.current_session_id)
            else:
                session_success = True  # No session manager, consider it cleaned
        except Exception as e:
            logger.error(f"Error cleaning up session directory: {e}")
            session_success = False

        # Clear current session if it was cleaned up
        if session_success:
            with self._lock:
                self.current_session_id = None

        return all_success and session_success

    def cleanup_all(self) -> bool:
        """
        Clean up all managed files and sessions (backward compatibility).

        Returns:
            True if all cleanup successful
        """
        try:
            all_success = True

            # Clean up all tracked files
            with self._lock:
                files_to_cleanup = {
                    file_id: managed_file
                    for file_id, managed_file in self.managed_files.items()
                    if managed_file.cleanup_on_exit
                }

            for file_id, managed_file in files_to_cleanup.items():
                try:
                    success = self.cleanup_file(file_id)
                    if not success:
                        all_success = False
                except Exception as e:
                    logger.warning(f"Failed to cleanup file {managed_file.path}: {e}")
                    all_success = False

            # Clean up all sessions
            try:
                if self.session_manager:
                    session_results = self.session_manager.cleanup_all_sessions()
                    for _session_id, success in session_results.items():
                        if not success:
                            all_success = False
                else:
                    logger.warning("No session manager available for cleanup")
            except Exception as e:
                logger.warning(f"Failed to cleanup sessions: {e}")
                all_success = False

            # Clear current session
            with self._lock:
                self.current_session_id = None

            logger.info(f"Cleanup complete: success={all_success}")
            return all_success

        except Exception as e:
            logger.error(f"Error during cleanup_all: {e}")
            return False

    def get_managed_files_info(self) -> dict[str, dict[str, Any]]:
        """
        Get information about all managed files.

        Returns:
            Dictionary mapping file IDs to file information
        """
        with self._lock:
            return {file_id: managed_file.to_dict() for file_id, managed_file in self.managed_files.items()}

    def get_sessions_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all sessions."""
        if self.session_manager:
            return self.session_manager.list_sessions()
        return {}

    def get_current_session_info(self) -> Optional[dict[str, Any]]:
        """Get information about current session."""
        if not self.current_session_id:
            return None

        sessions_info = self.get_sessions_info()
        return sessions_info.get(self.current_session_id)

    @contextmanager
    def temporary_session(self, session_prefix: Optional[str] = None):
        """
        Context manager for temporary session with automatic cleanup.

        Args:
            session_prefix: Optional prefix for session directory
        """
        prefix = session_prefix or self.session_prefix
        if not self.session_manager:
            raise RuntimeError("Session manager not available")
        session_id = self.session_manager.create_session(prefix)
        old_session_id = self.current_session_id

        try:
            with self._lock:
                self.current_session_id = session_id

            yield session_id

        finally:
            # Cleanup temporary session
            if self.session_manager:
                self.session_manager.cleanup_session(session_id)

            # Restore previous session
            with self._lock:
                self.current_session_id = old_session_id

    def validate_file_operations(self) -> OperationResult:
        """
        Validate that all tracked files are in expected state.

        Returns:
            OperationResult with validation summary
        """
        try:
            validation_results = {
                "total_files": 0,
                "existing_files": 0,
                "missing_files": 0,
                "orphaned_sessions": 0,
                "issues": [],
            }

            with self._lock:
                validation_results["total_files"] = len(self.managed_files)

                # Check tracked files
                for file_id, managed_file in self.managed_files.items():
                    if managed_file.path.exists():
                        validation_results["existing_files"] += 1
                    else:
                        validation_results["missing_files"] += 1
                        validation_results["issues"].append(
                            f"Tracked file missing: {managed_file.path} (ID: {file_id})"
                        )

                # Check for orphaned sessions
                if self.session_manager:
                    sessions_info = self.session_manager.list_sessions()
                    for session_id, session_info in sessions_info.items():
                        if not session_info["exists"]:
                            validation_results["orphaned_sessions"] += 1
                            validation_results["issues"].append(
                                f"Session directory missing: {session_info['path']} (ID: {session_id})"
                            )

            success = validation_results["missing_files"] == 0 and validation_results["orphaned_sessions"] == 0

            return OperationResult(
                success=success,
                data=validation_results,
                metadata={"current_session": self.current_session_id, "base_temp_dir": str(self.base_temp_dir)},
            )

        except Exception as e:
            return handle_operation_error(e, "file_operations_validation")

    def create_temp_file(self, suffix: str = "", prefix: str = "tmp", delete: bool = False) -> str:
        """
        Create a temporary file and return its path.

        Args:
            suffix: File suffix (e.g., '.txt', '.wav')
            prefix: File prefix
            delete: Whether to delete file immediately (for name generation only)

        Returns:
            Path to created temporary file
        """
        try:
            # Ensure we have a session directory
            if not self.current_session_id:
                self.create_session_dir()

            if not self.session_manager:
                raise RuntimeError("Cannot create temporary file: no session manager")
            session_dir = self.session_manager.get_session_dir(self.current_session_id)
            if not session_dir:
                raise RuntimeError("Cannot create temporary file: no session directory")

            # Create temporary file in session directory
            fd, temp_path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=str(session_dir))

            # Close file descriptor
            os.close(fd)

            # Track the file
            _ = self.track_file(temp_path, FileType.TEMPORARY)

            logger.debug(f"Created temporary file: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to create temporary file: {e}")
            raise RuntimeError(f"Cannot create temporary file: {e}") from e

    def get_disk_usage(self) -> dict[str, int]:
        """
        Get disk usage information for the base temp directory.

        Returns:
            Dictionary with total, used, free, and free_mb values
        """
        try:
            usage = shutil.disk_usage(self.base_temp_dir)
            return {
                "total": usage.total,
                "used": usage.used,
                "free": usage.free,
                "free_mb": usage.free // (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"Failed to get disk usage: {e}")
            return {"total": 0, "used": 0, "free": 0, "free_mb": 0}

    def check_disk_space(self, required_mb: int) -> bool:
        """
        Check if sufficient disk space is available.

        Args:
            required_mb: Required space in megabytes

        Returns:
            True if sufficient space available
        """
        try:
            usage = self.get_disk_usage()
            available_mb = usage["free_mb"]
            return available_mb >= required_mb
        except Exception as e:
            logger.warning(f"Failed to check disk space: {e}")
            return False

    @contextmanager
    def temp_file_context(self, suffix: str = "", prefix: str = "tmp"):
        """
        Context manager for temporary file that gets cleaned up automatically.

        Args:
            suffix: File suffix
            prefix: File prefix

        Yields:
            Path to temporary file
        """
        temp_file = None
        try:
            temp_file = self.create_temp_file(suffix=suffix, prefix=prefix)
            yield temp_file
        finally:
            if temp_file:
                # Find and cleanup the file
                with self._lock:
                    file_id_to_cleanup = None
                    for file_id, managed_file in self.managed_files.items():
                        if str(managed_file.path) == temp_file:
                            file_id_to_cleanup = file_id
                            break

                    if file_id_to_cleanup:
                        self.cleanup_file(file_id_to_cleanup)

    def add_temp_file(self, file_path: Union[str, Path]) -> bool:
        """
        Add an existing file to temporary file tracking (backward compatibility).

        Args:
            file_path: Path to file to track

        Returns:
            True if file was added (exists)
        """
        if not file_path:
            return False

        file_path = Path(file_path)
        if file_path.exists():
            self.track_file(file_path, FileType.TEMPORARY)
            return True
        return False

    def remove_file(self, file_path: Union[str, Path]) -> bool:
        """
        Remove a file and stop tracking it (backward compatibility).

        Args:
            file_path: Path to file to remove

        Returns:
            True if removal successful
        """
        if not file_path:
            return False

        file_path = str(file_path)
        removal_success = True

        # Find the file ID for this path and remove from tracking
        with self._lock:
            file_id_to_remove = None
            for file_id, managed_file in self.managed_files.items():
                if str(managed_file.path) == file_path:
                    file_id_to_remove = file_id
                    break

        # Always remove from tracking list first (even if file deletion fails)
        if file_id_to_remove:
            with self._lock:
                self.managed_files.pop(file_id_to_remove, None)

        # Try to remove the actual file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")
            removal_success = False

        # For tracked files, return False if removal failed
        # For untracked files, return True even if removal failed
        return removal_success

    @property
    def temp_files(self) -> list[str]:
        """
        Get list of tracked temporary file paths (backward compatibility).

        Returns:
            List of file paths
        """
        with self._lock:
            return [str(managed_file.path) for managed_file in self.managed_files.values()]

    @property
    def temp_dirs(self) -> list[str]:
        """
        Get list of session directory paths (backward compatibility).

        Returns:
            List of directory paths
        """
        if self.session_manager:
            sessions_info = self.session_manager.list_sessions()
            return [session_info["path"] for session_info in sessions_info.values()]
        return []

    @property
    def session_dir(self) -> Optional[str]:
        """
        Get current session directory path (backward compatibility).

        Returns:
            Current session directory path or None
        """
        if self.current_session_id and self.session_manager:
            session_dir = self.session_manager.get_session_dir(self.current_session_id)
            return str(session_dir) if session_dir else None
        return None

    # Note: get_temp_path method already defined above at line 278
    # This duplicate has been removed to fix F811 error

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self.cleanup_on_exit:
            try:
                self.cleanup_all()
            except Exception as e:
                logger.warning(f"Error during context manager cleanup: {e}")

    def __del__(self):
        """Destructor with cleanup."""
        if hasattr(self, "cleanup_on_exit") and self.cleanup_on_exit:
            try:
                self.cleanup_all()
            except Exception:
                # Don't raise exceptions in destructor
                pass


# Factory functions for convenience


def create_refactored_file_manager(
    base_temp_dir: Optional[Union[str, Path]] = None,
    cleanup_on_exit: bool = True,
    session_prefix: str = "transcription",
) -> RefactoredTempFileManager:
    """
    Create a new refactored file manager instance.

    Args:
        base_temp_dir: Base directory for temporary files
        cleanup_on_exit: Whether to cleanup files on exit
        session_prefix: Prefix for session directories

    Returns:
        RefactoredTempFileManager instance
    """
    return RefactoredTempFileManager(base_temp_dir, cleanup_on_exit, session_prefix)


def create_session_manager(base_temp_dir: Optional[Union[str, Path]] = None) -> SessionManager:
    """
    Create a session manager instance.

    Args:
        base_temp_dir: Base directory for sessions

    Returns:
        SessionManager instance
    """
    return SessionManager(base_temp_dir)


# Backward compatibility class using composition to avoid property conflicts
class TempFileManager:
    """Backward compatibility wrapper for RefactoredTempFileManager using composition."""

    def __init__(self, base_temp_dir: Optional[Union[str, Path]] = None, cleanup_on_exit: bool = True):
        """Initialize with backward compatibility."""
        # Don't let RefactoredTempFileManager register atexit, we'll do it ourselves
        self._manager = RefactoredTempFileManager(base_temp_dir, cleanup_on_exit=False)
        self._base_temp_dir_str = str(self._manager.base_temp_dir)
        self._cleanup_on_exit = cleanup_on_exit

        # Register our own cleanup on exit if enabled
        if cleanup_on_exit:
            atexit.register(self.cleanup_all)

    @property
    def base_temp_dir(self) -> str:
        """Return base temp directory as string for backward compatibility."""
        return self._base_temp_dir_str

    @property
    def cleanup_on_exit(self) -> bool:
        """Return cleanup on exit setting."""
        return self._cleanup_on_exit

    @property
    def temp_files(self) -> list[str]:
        """Get list of tracked temporary file paths."""
        return self._manager.temp_files

    @property
    def temp_dirs(self) -> list[str]:
        """Get list of session directory paths."""
        return self._manager.temp_dirs

    @property
    def session_dir(self) -> Optional[str]:
        """Get current session directory path."""
        return self._manager.session_dir

    def create_session_dir(self) -> str:
        """Create a new session directory."""
        return self._manager.create_session_dir()

    def get_temp_path(self, filename: str, create_dir: bool = True) -> str:
        """Generate a temporary file path."""
        return self._manager.get_temp_path(filename, create_dir)

    def create_temp_file(self, suffix: str = "", prefix: str = "tmp") -> str:
        """Create a temporary file."""
        return self._manager.create_temp_file(suffix, prefix)

    def add_temp_file(self, file_path: Union[str, Path]) -> bool:
        """Add an existing file to tracking."""
        return self._manager.add_temp_file(file_path)

    def remove_file(self, file_path: Union[str, Path]) -> bool:
        """Remove a file and stop tracking it."""
        return self._manager.remove_file(file_path)

    def cleanup_session(self) -> bool:
        """Clean up current session."""
        return self._manager.cleanup_session()

    def cleanup_all(self) -> bool:
        """Clean up all managed files and sessions."""
        return self._manager.cleanup_all()

    def get_disk_usage(self) -> dict[str, int]:
        """Get disk usage information."""
        return self._manager.get_disk_usage()

    def check_disk_space(self, required_mb: int) -> bool:
        """Check if sufficient disk space is available."""
        return self._manager.check_disk_space(required_mb)

    def temp_file_context(self, suffix: str = "", prefix: str = "tmp"):
        """Context manager for temporary file."""
        return self._manager.temp_file_context(suffix, prefix)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return self._manager.__exit__(exc_type, exc_val, exc_tb)

    def __del__(self):
        """Destructor with cleanup."""
        if hasattr(self, "_cleanup_on_exit") and self._cleanup_on_exit:
            try:
                if hasattr(self, "_manager"):
                    self._manager.cleanup_all()
            except Exception:
                # Don't raise exceptions in destructor
                pass


# Backward compatibility functions
def create_file_manager(
    base_temp_dir: Optional[Union[str, Path]] = None,
    cleanup_on_exit: bool = True,
    session_prefix: str = "transcription",
) -> TempFileManager:
    """
    Create a TempFileManager instance for backward compatibility.

    Args:
        base_temp_dir: Base directory for temporary files
        cleanup_on_exit: Whether to cleanup files on exit
        session_prefix: Prefix for session directories

    Returns:
        TempFileManager instance
    """
    return TempFileManager(base_temp_dir, cleanup_on_exit)


create_temp_file_manager = create_file_manager


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename for safe use.

    Args:
        filename: Filename to sanitize

    Returns:
        Sanitized filename
    """
    # Remove any path separators
    filename = filename.replace("/", "_").replace("\\", "_")
    # Remove problematic characters
    filename = "".join(c for c in filename if c.isalnum() or c in "-_. ")
    # Limit length
    max_length = 255
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        filename = name[: max_length - len(ext)] + ext
    return filename

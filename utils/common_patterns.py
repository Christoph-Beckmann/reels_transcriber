"""
Shared utilities for common patterns across the transcription pipeline.

This module consolidates frequently repeated patterns to eliminate code duplication
and improve maintainability. It includes shared progress handling, error management,
and path utilities.
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional, Union

from utils.error_handler import ErrorDetails, process_error_for_user
from utils.progress import ProcessingStage, ProgressTracker

logger = logging.getLogger(__name__)


@dataclass
class OperationResult:
    """Standardized result for pipeline operations."""

    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    error_details: Optional[ErrorDetails] = None
    metadata: Optional[dict[str, Any]] = None


class ProgressCallbackFactory:
    """Factory for creating standardized progress callbacks."""

    @staticmethod
    def create_stage_callback(
        progress_tracker: ProgressTracker, stage: ProcessingStage, prefix: str = ""
    ) -> Callable[[int, str], None]:
        """
        Create a progress callback for a specific stage.

        Args:
            progress_tracker: The progress tracker to update
            stage: The processing stage
            prefix: Optional prefix for messages

        Returns:
            Callback function with signature (progress: int, message: str) -> None
        """

        def callback(progress: int, message: str):
            full_message = f"{prefix}{message}" if prefix else message
            progress_tracker.update_progress(stage, progress, full_message)

        return callback

    @staticmethod
    def create_cancellable_callback(
        progress_tracker: ProgressTracker, stage: ProcessingStage, cancel_event: threading.Event, prefix: str = ""
    ) -> Callable[[int, str], None]:
        """
        Create a progress callback that checks for cancellation.

        Args:
            progress_tracker: The progress tracker to update
            stage: The processing stage
            cancel_event: Event to check for cancellation
            prefix: Optional prefix for messages

        Returns:
            Callback function that raises exception if cancelled
        """

        def callback(progress: int, message: str):
            if cancel_event.is_set():
                raise OperationCancelledException("Operation was cancelled")

            full_message = f"{prefix}{message}" if prefix else message
            progress_tracker.update_progress(stage, progress, full_message)

        return callback


class PathManager:
    """Utilities for common path operations and validation."""

    @staticmethod
    def ensure_directory(path: Union[str, Path]) -> Path:
        """
        Ensure directory exists, creating it if necessary.

        Args:
            path: Directory path

        Returns:
            Path object for the directory

        Raises:
            OSError: If directory cannot be created
        """
        dir_path = Path(path)
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    @staticmethod
    def validate_file_exists(path: Union[str, Path]) -> bool:
        """
        Validate that a file exists and is accessible.

        Args:
            path: File path to validate

        Returns:
            True if file exists and is accessible
        """
        try:
            file_path = Path(path)
            return file_path.exists() and file_path.is_file()
        except Exception:
            return False

    @staticmethod
    def validate_file_not_empty(path: Union[str, Path]) -> bool:
        """
        Validate that a file exists and is not empty.

        Args:
            path: File path to validate

        Returns:
            True if file exists and has content
        """
        try:
            file_path = Path(path)
            return file_path.exists() and file_path.stat().st_size > 0
        except Exception:
            return False

    @staticmethod
    def generate_output_path(
        input_path: Union[str, Path], suffix: str, extension: str, output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Generate output path based on input path with suffix and extension.

        Args:
            input_path: Input file path
            suffix: Suffix to add to filename
            extension: File extension (with or without dot)
            output_dir: Optional output directory

        Returns:
            Generated output path
        """
        input_path = Path(input_path)

        # Ensure extension starts with dot
        if not extension.startswith("."):
            extension = f".{extension}"

        # Generate filename with suffix
        base_name = input_path.stem
        output_filename = f"{base_name}_{suffix}{extension}"

        # Use output directory or same directory as input
        if output_dir:
            output_path = Path(output_dir) / output_filename
        else:
            output_path = input_path.parent / output_filename

        return output_path


class ExecutionContext:
    """Context manager for pipeline execution with automatic cleanup."""

    def __init__(
        self,
        operation_name: str,
        cancel_event: Optional[threading.Event] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """
        Initialize execution context.

        Args:
            operation_name: Name of the operation for logging
            cancel_event: Optional cancellation event
            progress_tracker: Optional progress tracker
        """
        self.operation_name = operation_name
        self.cancel_event = cancel_event
        self.progress_tracker = progress_tracker
        self.start_time = None
        self.cleanup_functions = []

    def __enter__(self):
        """Enter execution context."""
        self.start_time = time.time()
        logger.info(f"Starting operation: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit execution context with cleanup."""
        execution_time = time.time() - self.start_time if self.start_time else 0

        # Run cleanup functions
        for cleanup_func in reversed(self.cleanup_functions):
            try:
                cleanup_func()
            except Exception as e:
                logger.warning(f"Cleanup function failed: {e}")

        if exc_type is None:
            logger.info(f"Operation {self.operation_name} completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"Operation {self.operation_name} failed after {execution_time:.2f}s: {exc_val}")

    def add_cleanup(self, cleanup_func: Callable[[], None]):
        """Add a cleanup function to be called on exit."""
        self.cleanup_functions.append(cleanup_func)

    def check_cancellation(self):
        """Check if operation was cancelled and raise exception if so."""
        if self.cancel_event and self.cancel_event.is_set():
            raise OperationCancelledException(f"Operation {self.operation_name} was cancelled")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """
    Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff calculation
        exceptions: tuple of exceptions to catch and retry

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        # Last attempt failed, re-raise the exception
                        raise

                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base**attempt), max_delay)

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                        f"Retrying in {delay:.2f}s..."
                    )

                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception

        return wrapper

    return decorator


def validate_operation_input(
    url: Optional[str] = None,
    file_path: Optional[Union[str, Path]] = None,
    required_params: Optional[dict[str, Any]] = None,
) -> OperationResult:
    """
    Validate common operation inputs.

    Args:
        url: Optional URL to validate
        file_path: Optional file path to validate
        required_params: Optional dictionary of required parameters

    Returns:
        OperationResult indicating validation success or failure
    """
    try:
        # Validate URL if provided
        if url is not None:
            if not url.strip():
                return OperationResult(success=False, error_message="URL cannot be empty")

            # Basic URL validation - could be enhanced with validators module
            if not url.startswith(("http://", "https://")):
                return OperationResult(success=False, error_message="URL must start with http:// or https://")

        # Validate file path if provided
        if file_path is not None:
            if not PathManager.validate_file_exists(file_path):
                return OperationResult(success=False, error_message=f"File not found: {file_path}")

            if not PathManager.validate_file_not_empty(file_path):
                return OperationResult(success=False, error_message=f"File is empty: {file_path}")

        # Validate required parameters if provided
        if required_params:
            for param_name, param_value in required_params.items():
                if param_value is None:
                    return OperationResult(success=False, error_message=f"Required parameter '{param_name}' is missing")

        return OperationResult(success=True)

    except Exception as e:
        logger.error(f"Input validation error: {e}")
        return OperationResult(success=False, error_message=f"Validation error: {e}")


def handle_operation_error(error: Exception, operation_name: str, context: Optional[str] = None) -> OperationResult:
    """
    Standardized error handling for pipeline operations.

    Args:
        error: The exception that occurred
        operation_name: Name of the operation that failed
        context: Optional context information

    Returns:
        OperationResult with error information
    """
    try:
        # Use the error handler to classify and process the error
        error_details = process_error_for_user(error, context=context, user_action=operation_name)

        logger.error(f"Operation {operation_name} failed: {error_details.user_message}")

        return OperationResult(
            success=False,
            error_message=error_details.user_message,
            error_details=error_details,
            metadata={"operation": operation_name, "error_type": type(error).__name__, "context": context},
        )

    except Exception as secondary_error:
        # Fallback if error processing fails
        logger.error(f"Error processing failed for {operation_name}: {secondary_error}")
        return OperationResult(
            success=False,
            error_message=f"Operation failed: {str(error)}",
            metadata={
                "operation": operation_name,
                "error_type": type(error).__name__,
                "processing_error": str(secondary_error),
            },
        )


@contextmanager
def progress_stage_context(
    progress_tracker: ProgressTracker,
    stage: ProcessingStage,
    start_message: str,
    success_message: str,
    error_message: str = "Operation failed",
):
    """
    Context manager for managing progress updates during a stage.

    Args:
        progress_tracker: Progress tracker to update
        stage: Processing stage
        start_message: Message to show when starting
        success_message: Message to show on success
        error_message: Message to show on error
    """
    try:
        progress_tracker.update_progress(stage, 0, start_message)
        yield
        progress_tracker.update_progress(stage, 100, success_message)
    except Exception as e:
        progress_tracker.update_progress(ProcessingStage.ERROR, None, f"{error_message}: {e}")
        raise


class OperationCancelledException(Exception):
    """Exception raised when an operation is cancelled."""

    pass


class ValidationError(Exception):
    """Exception raised when input validation fails."""

    pass


# Commonly used patterns as functions


def execute_with_progress_and_error_handling(
    operation_func: Callable,
    operation_name: str,
    progress_tracker: ProgressTracker,
    stage: ProcessingStage,
    start_message: str,
    success_message: str,
    cancel_event: Optional[threading.Event] = None,
    *args,
    **kwargs,
) -> OperationResult:
    """
    Execute an operation with standardized progress tracking and error handling.

    Args:
        operation_func: Function to execute
        operation_name: Name for logging and error reporting
        progress_tracker: Progress tracker
        stage: Processing stage
        start_message: Message to show when starting
        success_message: Message to show on success
        cancel_event: Optional cancellation event
        *args: Arguments to pass to operation_func
        **kwargs: Keyword arguments to pass to operation_func

    Returns:
        OperationResult with success/failure information
    """
    try:
        with progress_stage_context(progress_tracker, stage, start_message, success_message):
            # Check for cancellation before starting
            if cancel_event and cancel_event.is_set():
                raise OperationCancelledException(f"Operation {operation_name} was cancelled")

            # Execute the operation
            result = operation_func(*args, **kwargs)

            # Check for cancellation after completion
            if cancel_event and cancel_event.is_set():
                raise OperationCancelledException(f"Operation {operation_name} was cancelled")

            return OperationResult(success=True, data=result)

    except OperationCancelledException:
        # Re-raise cancellation exceptions
        raise
    except Exception as e:
        # Handle all other exceptions
        return handle_operation_error(e, operation_name, f"During {stage.value}")


def create_session_directory(base_dir: Union[str, Path], prefix: str = "session") -> Path:
    """
    Create a unique session directory for temporary files.

    Args:
        base_dir: Base directory for sessions
        prefix: Prefix for session directory name

    Returns:
        Path to created session directory
    """
    import uuid

    session_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())
    session_name = f"{prefix}_{timestamp}_{session_id}"

    session_dir = Path(base_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Created session directory: {session_dir}")
    return session_dir


def cleanup_file_safely(file_path: Union[str, Path]) -> bool:
    """
    Safely remove a file, logging any errors but not raising exceptions.

    Args:
        file_path: Path to file to remove

    Returns:
        True if file was removed successfully, False otherwise
    """
    try:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.debug(f"Removed file: {path}")
            return True
        return True  # File doesn't exist, consider it "cleaned up"
    except Exception as e:
        logger.warning(f"Failed to remove file {file_path}: {e}")
        return False


def cleanup_directory_safely(dir_path: Union[str, Path]) -> bool:
    """
    Safely remove a directory and its contents, logging errors but not raising exceptions.

    Args:
        dir_path: Path to directory to remove

    Returns:
        True if directory was removed successfully, False otherwise
    """
    try:
        import shutil

        path = Path(dir_path)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            logger.debug(f"Removed directory: {path}")
            return True
        return True  # Directory doesn't exist, consider it "cleaned up"
    except Exception as e:
        logger.warning(f"Failed to remove directory {dir_path}: {e}")
        return False

"""Background processing worker for Instagram Reels transcription."""

import logging
import queue
import threading
import time
from typing import Any, Optional

from config.settings import DEFAULT_CONFIG
from core.pipeline import PipelineResult, TranscriptionPipeline, create_pipeline
from utils.error_handler import get_system_diagnostics, process_error_for_user
from utils.logging_config import log_error_with_context, log_performance, log_user_action
from utils.progress import ProgressUpdate

logger = logging.getLogger(__name__)


class ProcessingWorker:
    """
    Background worker thread that orchestrates the complete transcription pipeline.
    Now simplified to use the TranscriptionPipeline orchestrator for all processing.
    Handles GUI integration through message queue communication.
    """

    def __init__(self, message_queue: queue.Queue, config: Optional[dict[str, Any]] = None):
        """
        Initialize the processing worker.

        Args:
            message_queue: Queue for sending messages to GUI thread
            config: Optional configuration overrides
        """
        self.message_queue = message_queue
        self.config = config or DEFAULT_CONFIG.to_dict()

        # Threading control
        self.worker_thread: Optional[threading.Thread] = None
        self.is_running = False

        # Pipeline orchestrator
        self.pipeline: Optional[TranscriptionPipeline] = None

        # Current processing state
        self.current_url: Optional[str] = None

        logger.info("ProcessingWorker initialized with pipeline orchestrator")

    def start_processing(self, url: str) -> bool:
        """
        Start processing an Instagram Reel URL in a background thread.

        Args:
            url: Instagram Reel URL to process

        Returns:
            bool: True if processing started successfully, False if already running
        """
        if self.is_running:
            logger.warning("Processing already in progress, ignoring new request")
            self._send_message("error", "Processing is already in progress")
            return False

        try:
            # Pre-flight checks for edge cases
            if not self._perform_preflight_checks():
                return False

            # Store URL and reset state
            self.current_url = url.strip()

            # Start worker thread
            self.worker_thread = threading.Thread(
                target=self._process_reel, args=(self.current_url,), daemon=True, name="ReelProcessor"
            )

            self.is_running = True
            self.worker_thread.start()

            log_user_action("processing_started", True, {"url_sanitized": self._sanitize_url_for_logging(url)})
            logger.info(f"Started processing for URL: {self._sanitize_url_for_logging(url)}")
            return True

        except Exception as e:
            logger.error(f"Failed to start processing: {e}")
            log_error_with_context(e, "worker_start", "start_processing")
            self._send_message("error", f"Failed to start processing: {e}")
            self.is_running = False
            return False

    def stop_processing(self) -> None:
        """Stop the current processing operation."""
        if not self.is_running:
            return

        logger.info("Stopping processing...")

        # Cancel pipeline if running
        if self.pipeline:
            self.pipeline.cancel()

        # Wait for thread to finish (with timeout)
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
            if self.worker_thread.is_alive():
                logger.warning("Worker thread did not stop within timeout")

        self.is_running = False
        self.pipeline = None
        self._send_message("status", "Processing stopped by user")

    def is_processing(self) -> bool:
        """Check if currently processing."""
        return self.is_running

    def _process_reel(self, url: str) -> None:
        """
        Main processing pipeline executed in background thread.
        Now delegates all work to the TranscriptionPipeline orchestrator.

        Args:
            url: Instagram Reel URL to process
        """
        start_time = time.time()

        try:
            # Create pipeline with progress callback
            self.pipeline = create_pipeline(config=self.config, progress_callback=self._on_progress_update)

            logger.info("Executing transcription pipeline for URL: %s", self._sanitize_url_for_logging(url))

            # Execute pipeline - this handles all stages
            result = self.pipeline.execute(url)

            # Log performance metrics
            execution_time = time.time() - start_time
            log_performance(
                "transcription_pipeline",
                execution_time,
                {
                    "success": result.success,
                    "stages_completed": result.stages_completed,
                    "transcript_length": len(result.transcript) if result.transcript else 0,
                },
            )

            # Handle result
            if result.success:
                self._handle_success(result)
            else:
                self._handle_error_with_details(result.error_message or "Processing failed", url)

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Critical error in pipeline execution: {e}", exc_info=True)
            log_error_with_context(
                e,
                "pipeline_execution",
                "process_reel",
                {"url_sanitized": self._sanitize_url_for_logging(url), "execution_time": execution_time},
            )
            self._handle_error_with_details(f"Critical error: {e}", url, e)

        finally:
            self.is_running = False
            self.pipeline = None

    def _handle_success(self, result: PipelineResult) -> None:
        """
        Handle successful pipeline completion.

        Args:
            result: Pipeline execution result
        """
        try:
            # Send transcript to GUI
            if result.transcript:
                self._send_message("transcript", result.transcript)

            # Send completion message with metadata
            completion_data = {
                "transcript": result.transcript,
                "language": result.detected_language,
                "execution_time": result.execution_time,
                "stages_completed": result.stages_completed,
            }

            self._send_message("complete", completion_data)

            logger.info("Processing completed successfully in %.2f seconds", result.execution_time or 0)

        except Exception as e:
            logger.error(f"Error handling success: {e}")
            self._send_message("error", f"Error finalizing results: {e}")

    def _handle_error(self, error_message: str) -> None:
        """
        Handle processing errors.

        Args:
            error_message: Error message to report
        """
        try:
            self._send_message("error", error_message)
            logger.error(f"Processing error: {error_message}")

        except Exception as e:
            logger.error(f"Error in error handler: {e}")

    def _handle_error_with_details(self, error_message: str, url: str, exception: Optional[Exception] = None) -> None:
        """
        Handle processing errors with enhanced details and classification.

        Args:
            error_message: Error message to report
            url: URL being processed when error occurred
            exception: Original exception if available
        """
        try:
            # Process error for user-friendly feedback
            if exception:
                error_details = process_error_for_user(exception, "transcription_processing", "process_reel")
                # Send enhanced error information
                self._send_message(
                    "error_detailed",
                    {
                        "message": error_message,
                        "details": error_details,
                        "url_sanitized": self._sanitize_url_for_logging(url),
                    },
                )
            else:
                self._send_message("error", error_message)

            logger.error(f"Processing error for {self._sanitize_url_for_logging(url)}: {error_message}")

        except Exception as e:
            logger.error(f"Error in enhanced error handler: {e}")
            # Fallback to simple error handling
            self._handle_error(error_message)

    def _perform_preflight_checks(self) -> bool:
        """
        Perform pre-flight checks for edge cases before starting processing.

        Returns:
            bool: True if all checks pass, False otherwise
        """
        try:
            # Check disk space
            diagnostics = get_system_diagnostics()
            disk_info = diagnostics.get("disk_space", {})
            free_gb = disk_info.get("free_gb", 0)

            if free_gb < 1.0:  # Less than 1GB free
                self._send_message(
                    "error", "Insufficient disk space. Please free up at least 1GB of space before processing."
                )
                return False

            # Check memory
            memory_info = diagnostics.get("memory", {})
            available_gb = memory_info.get("available_gb", 0)

            if available_gb < 0.5:  # Less than 500MB available
                self._send_message(
                    "error", "Insufficient memory available. Please close other applications and try again."
                )
                return False

            # Check dependencies
            deps_info = diagnostics.get("dependencies", {})
            if deps_info.get("ffmpeg") in ["not_found", "not_available"]:
                self._send_message(
                    "error", "FFmpeg is required but not found. Please install FFmpeg and restart the application."
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"Pre-flight checks failed: {e}")
            # Allow processing to continue if checks themselves fail
            return True

    def _sanitize_url_for_logging(self, url: str) -> str:
        """
        Sanitize URL for privacy-safe logging.

        Args:
            url: Original URL

        Returns:
            Sanitized URL safe for logging
        """
        from utils.error_handler import sanitize_url

        return sanitize_url(url)

    def _send_message(self, message_type: str, data: Any) -> None:
        """
        Send message to GUI thread.

        Args:
            message_type: Type of message ('progress', 'status', 'transcript', 'error', 'complete')
            data: Message data
        """
        try:
            message = {"type": message_type, "data": data, "timestamp": time.time()}
            self.message_queue.put(message, timeout=1.0)

        except queue.Full:
            logger.warning(f"Message queue full, dropping message: {message_type}")
        except Exception as e:
            logger.error(f"Failed to send message: {e}")

    def _on_progress_update(self, progress_update: ProgressUpdate) -> None:
        """
        Handle progress updates from TranscriptionPipeline.

        Args:
            progress_update: Progress update object from pipeline
        """
        self._send_message("progress", progress_update)


# Convenience function
def create_processing_worker(message_queue: queue.Queue, config: Optional[dict[str, Any]] = None) -> ProcessingWorker:
    """
    Create a new processing worker instance.

    Args:
        message_queue: Queue for GUI communication
        config: Optional configuration overrides

    Returns:
        ProcessingWorker: New worker instance
    """
    return ProcessingWorker(message_queue, config)

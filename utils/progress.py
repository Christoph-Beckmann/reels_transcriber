"""Progress tracking utilities for coordinating UI updates."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ProcessingStage(Enum):
    """Processing stages for the transcription pipeline."""

    IDLE = "idle"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    CLEANING_UP = "cleaning_up"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class ProgressUpdate:
    """Progress update data structure."""

    stage: ProcessingStage
    progress: int  # 0-100
    message: str
    details: Optional[str] = None


class ProgressTracker:
    """
    Manages progress tracking and coordination between processing and UI.
    """

    def __init__(self, update_callback: Optional[Callable[[ProgressUpdate], None]] = None):
        """
        Initialize progress tracker.

        Args:
            update_callback: Function to call when progress updates
        """
        self.update_callback = update_callback
        self.current_stage = ProcessingStage.IDLE
        self.current_progress = 0
        self.stage_progress_ranges = {
            ProcessingStage.VALIDATING: (0, 5),
            ProcessingStage.DOWNLOADING: (5, 35),
            ProcessingStage.EXTRACTING_AUDIO: (35, 50),
            ProcessingStage.TRANSCRIBING: (50, 95),
            ProcessingStage.CLEANING_UP: (95, 100),
            ProcessingStage.COMPLETED: (100, 100),
        }

    def update_progress(
        self, stage: ProcessingStage, progress_within_stage: int = 0, message: str = "", details: Optional[str] = None
    ) -> None:
        """
        Update the current progress.

        Args:
            stage: Current processing stage
            progress_within_stage: Progress within the current stage (0-100)
            message: Human-readable progress message
            details: Optional additional details
        """
        # Calculate overall progress based on stage and within-stage progress
        if stage in self.stage_progress_ranges:
            stage_start, stage_end = self.stage_progress_ranges[stage]
            stage_range = stage_end - stage_start
            overall_progress = stage_start + (stage_range * progress_within_stage // 100)
        else:
            overall_progress = self.current_progress

        # Ensure progress only moves forward (except for errors)
        if stage != ProcessingStage.ERROR and overall_progress < self.current_progress:
            overall_progress = self.current_progress

        self.current_stage = stage
        self.current_progress = min(100, max(0, overall_progress))

        # Create progress update
        update = ProgressUpdate(stage=stage, progress=self.current_progress, message=message, details=details)

        # Safely get stage value, handling cases where stage might not be a proper enum
        stage_value = getattr(stage, 'value', str(stage)) if stage is not None else 'unknown_stage'
        logger.debug(f"Progress update: {stage_value} - {self.current_progress}% - {message}")

        # Call update callback if provided
        if self.update_callback:
            try:
                self.update_callback(update)
            except Exception as e:
                logger.error(f"Error in progress callback: {e}")

    def reset(self) -> None:
        """Reset progress to initial state."""
        self.current_stage = ProcessingStage.IDLE
        self.current_progress = 0
        self.update_progress(ProcessingStage.IDLE, 0, "Ready")

    def start_validation(self) -> None:
        """Start URL validation stage."""
        self.update_progress(ProcessingStage.VALIDATING, 0, "Validating URL...")

    def validation_complete(self) -> None:
        """Mark validation as complete."""
        self.update_progress(ProcessingStage.VALIDATING, 100, "URL validation complete")

    def start_download(self) -> None:
        """Start download stage."""
        self.update_progress(ProcessingStage.DOWNLOADING, 0, "Connecting to Instagram...")

    def update_download_progress(self, progress: int, details: str = "") -> None:
        """Update download progress."""
        message = "Downloading video"
        if details:
            message += f" ({details})"
        self.update_progress(ProcessingStage.DOWNLOADING, progress, message, details)

    def download_complete(self) -> None:
        """Mark download as complete."""
        self.update_progress(ProcessingStage.DOWNLOADING, 100, "Video download complete")

    def start_audio_extraction(self) -> None:
        """Start audio extraction stage."""
        self.update_progress(ProcessingStage.EXTRACTING_AUDIO, 0, "Extracting audio from video...")

    def update_audio_extraction_progress(self, progress: int) -> None:
        """Update audio extraction progress."""
        self.update_progress(ProcessingStage.EXTRACTING_AUDIO, progress, "Processing audio...")

    def audio_extraction_complete(self) -> None:
        """Mark audio extraction as complete."""
        self.update_progress(ProcessingStage.EXTRACTING_AUDIO, 100, "Audio extraction complete")

    def start_transcription(self) -> None:
        """Start transcription stage."""
        self.update_progress(ProcessingStage.TRANSCRIBING, 0, "Loading transcription model...")

    def update_transcription_progress(self, progress: int, details: str = "") -> None:
        """Update transcription progress."""
        message = "Transcribing audio"
        if details:
            message += f" ({details})"
        self.update_progress(ProcessingStage.TRANSCRIBING, progress, message, details)

    def transcription_complete(self) -> None:
        """Mark transcription as complete."""
        self.update_progress(ProcessingStage.TRANSCRIBING, 100, "Transcription complete")

    def start_cleanup(self) -> None:
        """Start cleanup stage."""
        self.update_progress(ProcessingStage.CLEANING_UP, 0, "Cleaning up temporary files...")

    def cleanup_complete(self) -> None:
        """Mark cleanup as complete."""
        self.update_progress(ProcessingStage.CLEANING_UP, 100, "Cleanup complete")

    def mark_complete(self) -> None:
        """Mark entire process as complete."""
        self.update_progress(ProcessingStage.COMPLETED, 100, "Transcription ready!")

    def mark_error(self, error_message: str, details: Optional[str] = None) -> None:
        """Mark process as failed with error."""
        self.update_progress(ProcessingStage.ERROR, self.current_progress, error_message, details)

    def get_stage_message(self, stage: ProcessingStage) -> str:
        """Get user-friendly message for a processing stage."""
        stage_messages = {
            ProcessingStage.IDLE: "Ready",
            ProcessingStage.VALIDATING: "Validating URL",
            ProcessingStage.DOWNLOADING: "Downloading video",
            ProcessingStage.EXTRACTING_AUDIO: "Extracting audio",
            ProcessingStage.TRANSCRIBING: "Transcribing speech",
            ProcessingStage.CLEANING_UP: "Cleaning up",
            ProcessingStage.COMPLETED: "Complete",
            ProcessingStage.ERROR: "Error occurred",
        }
        return stage_messages.get(stage, "Processing")


# Convenience function for creating progress trackers
def create_progress_tracker(callback: Optional[Callable[[ProgressUpdate], None]] = None) -> ProgressTracker:
    """Create a new progress tracker with optional callback."""
    return ProgressTracker(callback)

"""
Comprehensive test suite for progress tracking utilities.

Tests cover progress tracking, stage management, and UI coordination
with various scenarios and error handling.
"""

from dataclasses import asdict
from unittest.mock import MagicMock, patch

from utils.progress import ProcessingStage, ProgressTracker, ProgressUpdate, create_progress_tracker


class TestProcessingStage:
    """Test ProcessingStage enum."""

    def test_stage_values(self):
        """Test all processing stage values."""
        assert ProcessingStage.IDLE.value == "idle"
        assert ProcessingStage.VALIDATING.value == "validating"
        assert ProcessingStage.DOWNLOADING.value == "downloading"
        assert ProcessingStage.EXTRACTING_AUDIO.value == "extracting_audio"
        assert ProcessingStage.TRANSCRIBING.value == "transcribing"
        assert ProcessingStage.CLEANING_UP.value == "cleaning_up"
        assert ProcessingStage.COMPLETED.value == "completed"
        assert ProcessingStage.ERROR.value == "error"


class TestProgressUpdate:
    """Test ProgressUpdate dataclass."""

    def test_progress_update_creation(self):
        """Test creating ProgressUpdate with all fields."""
        update = ProgressUpdate(
            stage=ProcessingStage.DOWNLOADING, progress=50, message="Downloading video", details="50% complete"
        )

        assert update.stage == ProcessingStage.DOWNLOADING
        assert update.progress == 50
        assert update.message == "Downloading video"
        assert update.details == "50% complete"

    def test_progress_update_defaults(self):
        """Test ProgressUpdate with default values."""
        update = ProgressUpdate(stage=ProcessingStage.IDLE, progress=0, message="Ready")

        assert update.stage == ProcessingStage.IDLE
        assert update.progress == 0
        assert update.message == "Ready"
        assert update.details is None

    def test_progress_update_as_dict(self):
        """Test converting ProgressUpdate to dictionary."""
        update = ProgressUpdate(
            stage=ProcessingStage.TRANSCRIBING,
            progress=75,
            message="Transcribing audio",
            details="Processing speech recognition",
        )

        update_dict = asdict(update)
        assert update_dict["stage"] == ProcessingStage.TRANSCRIBING
        assert update_dict["progress"] == 75
        assert update_dict["message"] == "Transcribing audio"
        assert update_dict["details"] == "Processing speech recognition"


class TestProgressTracker:
    """Test ProgressTracker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_callback = MagicMock()
        self.tracker = ProgressTracker(update_callback=self.mock_callback)

    def test_init_default(self):
        """Test tracker initialization with defaults."""
        tracker = ProgressTracker()

        assert tracker.update_callback is None
        assert tracker.current_stage == ProcessingStage.IDLE
        assert tracker.current_progress == 0
        assert isinstance(tracker.stage_progress_ranges, dict)

    def test_init_with_callback(self):
        """Test tracker initialization with callback."""
        callback = MagicMock()
        tracker = ProgressTracker(update_callback=callback)

        assert tracker.update_callback == callback

    def test_stage_progress_ranges(self):
        """Test stage progress ranges are properly configured."""
        ranges = self.tracker.stage_progress_ranges

        assert ranges[ProcessingStage.VALIDATING] == (0, 5)
        assert ranges[ProcessingStage.DOWNLOADING] == (5, 35)
        assert ranges[ProcessingStage.EXTRACTING_AUDIO] == (35, 50)
        assert ranges[ProcessingStage.TRANSCRIBING] == (50, 95)
        assert ranges[ProcessingStage.CLEANING_UP] == (95, 100)
        assert ranges[ProcessingStage.COMPLETED] == (100, 100)

    @patch("utils.progress.logger")
    def test_update_progress_basic(self, mock_logger):
        """Test basic progress update."""
        self.tracker.update_progress(
            ProcessingStage.DOWNLOADING, progress_within_stage=50, message="Downloading video", details="50% complete"
        )

        # Verify internal state
        assert self.tracker.current_stage == ProcessingStage.DOWNLOADING
        # Download stage: 5-35, 50% within stage = 5 + (30 * 50 / 100) = 20
        assert self.tracker.current_progress == 20

        # Verify callback was called
        self.mock_callback.assert_called_once()
        call_args = self.mock_callback.call_args[0][0]
        assert isinstance(call_args, ProgressUpdate)
        assert call_args.stage == ProcessingStage.DOWNLOADING
        assert call_args.progress == 20
        assert call_args.message == "Downloading video"
        assert call_args.details == "50% complete"

        # Verify logging
        mock_logger.debug.assert_called_once()

    def test_update_progress_calculation(self):
        """Test progress calculation for different stages."""
        test_cases = [
            (ProcessingStage.VALIDATING, 50, 2),  # 0 + (5 * 50 / 100) = 2.5 -> 2
            (ProcessingStage.DOWNLOADING, 0, 5),  # 5 + (30 * 0 / 100) = 5
            (ProcessingStage.DOWNLOADING, 100, 35),  # 5 + (30 * 100 / 100) = 35
            (ProcessingStage.EXTRACTING_AUDIO, 50, 42),  # 35 + (15 * 50 / 100) = 42.5 -> 42
            (ProcessingStage.TRANSCRIBING, 50, 72),  # 50 + (45 * 50 / 100) = 72.5 -> 72
            (ProcessingStage.CLEANING_UP, 50, 97),  # 95 + (5 * 50 / 100) = 97.5 -> 97
            (ProcessingStage.COMPLETED, 0, 100),  # 100 + (0 * 0 / 100) = 100
        ]

        for stage, within_stage_progress, expected_overall in test_cases:
            tracker = ProgressTracker()  # Fresh tracker for each test
            tracker.update_progress(stage, within_stage_progress, "Test message")
            assert (
                tracker.current_progress == expected_overall
            ), f"Stage {stage}, within {within_stage_progress}% should be {expected_overall}%"

    def test_update_progress_forward_only(self):
        """Test that progress only moves forward (except for errors)."""
        # Start at 50%
        self.tracker.update_progress(ProcessingStage.TRANSCRIBING, 0, "Starting transcription")
        initial_progress = self.tracker.current_progress

        # Try to move backward (should be ignored)
        self.tracker.update_progress(ProcessingStage.DOWNLOADING, 50, "Going backward")
        assert self.tracker.current_progress == initial_progress

        # Move forward (should work)
        self.tracker.update_progress(ProcessingStage.TRANSCRIBING, 50, "Continuing transcription")
        assert self.tracker.current_progress > initial_progress

    def test_update_progress_error_stage(self):
        """Test that error stage can move progress backward."""
        # Start at 50%
        self.tracker.update_progress(ProcessingStage.TRANSCRIBING, 0, "Starting")

        # Error stage should allow moving backward
        self.tracker.update_progress(ProcessingStage.ERROR, 0, "Error occurred")
        assert self.tracker.current_stage == ProcessingStage.ERROR

    def test_update_progress_bounds(self):
        """Test progress is bounded between 0 and 100."""
        # Test upper bound
        self.tracker.current_progress = 95
        self.tracker.update_progress(ProcessingStage.COMPLETED, 200, "Over 100%")
        assert self.tracker.current_progress == 100

        # Test lower bound (using error stage which can go backward)
        self.tracker.current_progress = 5
        self.tracker.update_progress(ProcessingStage.ERROR, -50, "Negative progress")
        assert self.tracker.current_progress >= 0

    def test_update_progress_unknown_stage(self):
        """Test update with stage not in progress ranges."""
        original_progress = self.tracker.current_progress

        # Create a mock stage not in ranges
        with patch.object(self.tracker, "stage_progress_ranges", {}):
            self.tracker.update_progress(ProcessingStage.DOWNLOADING, 50, "Unknown stage")

        # Progress should remain the same
        assert self.tracker.current_progress == original_progress

    def test_update_progress_callback_error(self):
        """Test handling of callback errors."""
        self.mock_callback.side_effect = Exception("Callback error")

        with patch("utils.progress.logger") as mock_logger:
            # Should not raise exception
            self.tracker.update_progress(ProcessingStage.DOWNLOADING, 50, "Test")

            # Should log the error
            mock_logger.error.assert_called_once()

    def test_update_progress_no_callback(self):
        """Test update progress without callback."""
        tracker = ProgressTracker(update_callback=None)

        # Should not raise exception
        tracker.update_progress(ProcessingStage.DOWNLOADING, 50, "Test message")

        assert tracker.current_stage == ProcessingStage.DOWNLOADING

    def test_reset(self):
        """Test resetting progress to initial state."""
        # Set some progress
        self.tracker.update_progress(ProcessingStage.TRANSCRIBING, 50, "In progress")
        assert self.tracker.current_progress > 0

        # Reset
        self.tracker.reset()

        # Verify reset state
        assert self.tracker.current_stage == ProcessingStage.IDLE
        assert self.tracker.current_progress == 0

        # Verify callback was called for reset
        reset_call = None
        for call in self.mock_callback.call_args_list:
            update = call[0][0]
            if update.stage == ProcessingStage.IDLE and update.message == "Ready":
                reset_call = update
                break

        assert reset_call is not None

    def test_start_validation(self):
        """Test starting validation stage."""
        self.tracker.start_validation()

        assert self.tracker.current_stage == ProcessingStage.VALIDATING
        assert "Validating URL" in self.mock_callback.call_args[0][0].message

    def test_validation_complete(self):
        """Test completing validation stage."""
        self.tracker.validation_complete()

        assert self.tracker.current_stage == ProcessingStage.VALIDATING
        assert "validation complete" in self.mock_callback.call_args[0][0].message

    def test_start_download(self):
        """Test starting download stage."""
        self.tracker.start_download()

        assert self.tracker.current_stage == ProcessingStage.DOWNLOADING
        assert "Connecting to Instagram" in self.mock_callback.call_args[0][0].message

    def test_update_download_progress(self):
        """Test updating download progress."""
        self.tracker.update_download_progress(50, "test details")

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.DOWNLOADING
        assert "Downloading video" in update.message
        assert "(test details)" in update.message
        assert update.details == "test details"

    def test_update_download_progress_no_details(self):
        """Test updating download progress without details."""
        self.tracker.update_download_progress(25)

        update = self.mock_callback.call_args[0][0]
        assert update.message == "Downloading video"
        assert update.details == ""

    def test_download_complete(self):
        """Test completing download stage."""
        self.tracker.download_complete()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.DOWNLOADING
        assert "download complete" in update.message

    def test_start_audio_extraction(self):
        """Test starting audio extraction stage."""
        self.tracker.start_audio_extraction()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.EXTRACTING_AUDIO
        assert "Extracting audio" in update.message

    def test_update_audio_extraction_progress(self):
        """Test updating audio extraction progress."""
        self.tracker.update_audio_extraction_progress(75)

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.EXTRACTING_AUDIO
        assert "Processing audio" in update.message

    def test_audio_extraction_complete(self):
        """Test completing audio extraction stage."""
        self.tracker.audio_extraction_complete()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.EXTRACTING_AUDIO
        assert "extraction complete" in update.message

    def test_start_transcription(self):
        """Test starting transcription stage."""
        self.tracker.start_transcription()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.TRANSCRIBING
        assert "Loading transcription model" in update.message

    def test_update_transcription_progress(self):
        """Test updating transcription progress."""
        self.tracker.update_transcription_progress(60, "processing speech")

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.TRANSCRIBING
        assert "Transcribing audio" in update.message
        assert "(processing speech)" in update.message
        assert update.details == "processing speech"

    def test_update_transcription_progress_no_details(self):
        """Test updating transcription progress without details."""
        self.tracker.update_transcription_progress(40)

        update = self.mock_callback.call_args[0][0]
        assert update.message == "Transcribing audio"
        assert update.details == ""

    def test_transcription_complete(self):
        """Test completing transcription stage."""
        self.tracker.transcription_complete()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.TRANSCRIBING
        assert "Transcription complete" in update.message

    def test_start_cleanup(self):
        """Test starting cleanup stage."""
        self.tracker.start_cleanup()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.CLEANING_UP
        assert "Cleaning up" in update.message

    def test_cleanup_complete(self):
        """Test completing cleanup stage."""
        self.tracker.cleanup_complete()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.CLEANING_UP
        assert "Cleanup complete" in update.message

    def test_mark_complete(self):
        """Test marking entire process as complete."""
        self.tracker.mark_complete()

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.COMPLETED
        assert update.progress == 100
        assert "Transcription ready" in update.message

    def test_mark_error(self):
        """Test marking process as failed with error."""
        # Set some initial progress
        self.tracker.current_progress = 50

        self.tracker.mark_error("Something went wrong", "Error details")

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.ERROR
        assert update.progress == 50  # Should preserve current progress
        assert update.message == "Something went wrong"
        assert update.details == "Error details"

    def test_mark_error_no_details(self):
        """Test marking error without details."""
        self.tracker.mark_error("Error occurred")

        update = self.mock_callback.call_args[0][0]
        assert update.stage == ProcessingStage.ERROR
        assert update.message == "Error occurred"
        assert update.details is None

    def test_get_stage_message(self):
        """Test getting user-friendly stage messages."""
        expected_messages = {
            ProcessingStage.IDLE: "Ready",
            ProcessingStage.VALIDATING: "Validating URL",
            ProcessingStage.DOWNLOADING: "Downloading video",
            ProcessingStage.EXTRACTING_AUDIO: "Extracting audio",
            ProcessingStage.TRANSCRIBING: "Transcribing speech",
            ProcessingStage.CLEANING_UP: "Cleaning up",
            ProcessingStage.COMPLETED: "Complete",
            ProcessingStage.ERROR: "Error occurred",
        }

        for stage, expected_message in expected_messages.items():
            message = self.tracker.get_stage_message(stage)
            assert message == expected_message

    def test_get_stage_message_unknown(self):
        """Test getting message for unknown stage."""
        # Create a mock stage
        mock_stage = MagicMock()
        mock_stage.value = "unknown_stage"

        message = self.tracker.get_stage_message(mock_stage)
        assert message == "Processing"


class TestConvenienceFunction:
    """Test convenience function for creating progress trackers."""

    def test_create_progress_tracker_default(self):
        """Test creating progress tracker with defaults."""
        tracker = create_progress_tracker()

        assert isinstance(tracker, ProgressTracker)
        assert tracker.update_callback is None

    def test_create_progress_tracker_with_callback(self):
        """Test creating progress tracker with callback."""
        callback = MagicMock()
        tracker = create_progress_tracker(callback)

        assert isinstance(tracker, ProgressTracker)
        assert tracker.update_callback == callback


class TestProgressTrackerIntegration:
    """Integration tests for progress tracking workflows."""

    def test_complete_transcription_workflow(self):
        """Test complete transcription workflow progress tracking."""
        progress_updates = []

        def capture_progress(update):
            progress_updates.append(update)

        tracker = ProgressTracker(update_callback=capture_progress)

        # Simulate complete workflow
        tracker.start_validation()
        tracker.validation_complete()
        tracker.start_download()
        tracker.update_download_progress(50)
        tracker.download_complete()
        tracker.start_audio_extraction()
        tracker.update_audio_extraction_progress(100)
        tracker.audio_extraction_complete()
        tracker.start_transcription()
        tracker.update_transcription_progress(50)
        tracker.transcription_complete()
        tracker.start_cleanup()
        tracker.cleanup_complete()
        tracker.mark_complete()

        # Verify we got updates for all stages
        stages_seen = {update.stage for update in progress_updates}
        expected_stages = {
            ProcessingStage.VALIDATING,
            ProcessingStage.DOWNLOADING,
            ProcessingStage.EXTRACTING_AUDIO,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.CLEANING_UP,
            ProcessingStage.COMPLETED,
        }
        assert expected_stages.issubset(stages_seen)

        # Verify progress generally increases
        progress_values = [update.progress for update in progress_updates]
        # Remove duplicates while preserving order
        unique_progress = []
        for p in progress_values:
            if not unique_progress or p != unique_progress[-1]:
                unique_progress.append(p)

        # Should generally increase (allowing for same values)
        for i in range(1, len(unique_progress)):
            assert (
                unique_progress[i] >= unique_progress[i - 1]
            ), f"Progress decreased from {unique_progress[i - 1]} to {unique_progress[i]}"

        # Final progress should be 100%
        assert progress_updates[-1].progress == 100

    def test_error_workflow(self):
        """Test workflow with error handling."""
        progress_updates = []

        def capture_progress(update):
            progress_updates.append(update)

        tracker = ProgressTracker(update_callback=capture_progress)

        # Start normally
        tracker.start_validation()
        tracker.validation_complete()
        tracker.start_download()
        tracker.update_download_progress(25)

        # Simulate error
        tracker.mark_error("Network connection failed", "Timeout after 30 seconds")

        # Verify error was captured
        error_updates = [u for u in progress_updates if u.stage == ProcessingStage.ERROR]
        assert len(error_updates) > 0

        error_update = error_updates[0]
        assert error_update.message == "Network connection failed"
        assert error_update.details == "Timeout after 30 seconds"

    def test_progress_calculation_consistency(self):
        """Test progress calculation consistency across stages."""
        tracker = ProgressTracker()

        # Test that 100% of each stage reaches the stage end
        for stage, (_start, end) in tracker.stage_progress_ranges.items():
            if stage == ProcessingStage.COMPLETED:
                continue  # Special case

            # Reset tracker
            tracker.current_progress = 0
            tracker.current_stage = ProcessingStage.IDLE

            # Update to 100% of this stage
            tracker.update_progress(stage, 100, "Test")

            # Should reach the end of the stage range
            assert (
                tracker.current_progress == end
            ), f"Stage {stage} at 100% should reach {end}%, got {tracker.current_progress}%"

    def test_stage_transition_progress(self):
        """Test progress during stage transitions."""
        tracker = ProgressTracker()

        # Go through stages in order
        stages_in_order = [
            ProcessingStage.VALIDATING,
            ProcessingStage.DOWNLOADING,
            ProcessingStage.EXTRACTING_AUDIO,
            ProcessingStage.TRANSCRIBING,
            ProcessingStage.CLEANING_UP,
            ProcessingStage.COMPLETED,
        ]

        previous_progress = 0
        for stage in stages_in_order:
            # Start this stage
            tracker.update_progress(stage, 0, f"Starting {stage.value}")
            current_progress = tracker.current_progress

            # Progress should not decrease
            assert current_progress >= previous_progress, f"Progress decreased when starting {stage.value}"

            # Complete this stage
            tracker.update_progress(stage, 100, f"Completing {stage.value}")
            final_progress = tracker.current_progress

            # Progress should increase or stay same
            assert final_progress >= current_progress, f"Progress decreased when completing {stage.value}"

            previous_progress = final_progress

        # Final progress should be 100%
        assert tracker.current_progress == 100

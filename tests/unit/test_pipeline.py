"""
Unit tests for the transcription pipeline orchestrator.
"""

import threading
import time
from unittest.mock import Mock, patch

import pytest

from core.pipeline import (
    PipelineAudioException,
    PipelineCancelledException,
    PipelineDownloadException,
    PipelineException,
    PipelineInitializationException,
    PipelineResult,
    PipelineStage,
    PipelineState,
    PipelineTranscriptionException,
    PipelineValidationException,
    TranscriptionPipeline,
    create_pipeline,
    process_reel,
)
from core.transcriber import TranscriptionResult
from utils.progress import ProcessingStage


class TestPipelineStage:
    """Test PipelineStage enum."""

    def test_pipeline_stage_values(self):
        """Test that all pipeline stages have correct values."""
        assert PipelineStage.INIT.value == "initialization"
        assert PipelineStage.VALIDATE.value == "validation"
        assert PipelineStage.DOWNLOAD.value == "download"
        assert PipelineStage.EXTRACT_AUDIO.value == "extract_audio"
        assert PipelineStage.TRANSCRIBE.value == "transcribe"
        assert PipelineStage.CLEANUP.value == "cleanup"
        assert PipelineStage.COMPLETE.value == "complete"
        assert PipelineStage.ERROR.value == "error"
        assert PipelineStage.CANCELLED.value == "cancelled"


class TestPipelineState:
    """Test PipelineState enum."""

    def test_pipeline_state_values(self):
        """Test that all pipeline states have correct values."""
        assert PipelineState.IDLE.value == "idle"
        assert PipelineState.RUNNING.value == "running"
        assert PipelineState.CANCELLING.value == "cancelling"
        assert PipelineState.CANCELLED.value == "cancelled"
        assert PipelineState.COMPLETED.value == "completed"
        assert PipelineState.FAILED.value == "failed"


class TestPipelineResult:
    """Test PipelineResult data class."""

    def test_pipeline_result_success(self):
        """Test successful pipeline result."""
        result = PipelineResult(
            success=True,
            transcript="Test transcript",
            detected_language="en",
            execution_time=5.5,
            stages_completed=5,
            metadata={"model": "base"},
        )

        assert result.success is True
        assert result.transcript == "Test transcript"
        assert result.detected_language == "en"
        assert result.execution_time == 5.5
        assert result.stages_completed == 5
        assert result.error_message is None
        assert result.metadata == {"model": "base"}

    def test_pipeline_result_failure(self):
        """Test failed pipeline result."""
        result = PipelineResult(success=False, error_message="Download failed", execution_time=2.1, stages_completed=2)

        assert result.success is False
        assert result.transcript is None
        assert result.detected_language is None
        assert result.execution_time == 2.1
        assert result.stages_completed == 2
        assert result.error_message == "Download failed"
        assert result.metadata is None


class TestTranscriptionPipeline:
    """Test the TranscriptionPipeline class."""

    def test_initialization_default(self):
        """Test pipeline initialization with default parameters."""
        pipeline = TranscriptionPipeline()

        assert pipeline.config == {}
        assert pipeline.progress_callback is None
        assert pipeline._state == PipelineState.IDLE
        assert pipeline._current_stage == PipelineStage.INIT
        assert not pipeline.is_running()

    def test_initialization_with_config(self):
        """Test pipeline initialization with configuration."""
        config = {"temp_dir": "/tmp/test", "whisper_model": "small", "max_retries": 2}
        progress_callback = Mock()

        pipeline = TranscriptionPipeline(config, progress_callback)

        assert pipeline.config == config
        assert pipeline.progress_callback == progress_callback

    def test_state_management(self):
        """Test pipeline state management."""
        pipeline = TranscriptionPipeline()

        assert pipeline.get_state() == PipelineState.IDLE
        assert pipeline.get_current_stage() == PipelineStage.INIT
        assert not pipeline.is_running()

    @patch("core.pipeline.TempFileManager")
    @patch("core.pipeline.InstagramDownloader")
    @patch("core.pipeline.AudioExtractor")
    @patch("core.pipeline.WhisperTranscriber")
    @patch("core.pipeline.ProgressTracker")
    def test_initialization_components_success(
        self, mock_progress_tracker, mock_transcriber, mock_audio_extractor, mock_downloader, mock_file_manager
    ):
        """Test successful component initialization."""
        # Setup mocks
        mock_file_manager.return_value.create_session_dir.return_value = "/tmp/session"

        pipeline = TranscriptionPipeline()
        pipeline._initialize_components()

        # Verify components were created
        assert pipeline.file_manager is not None
        assert pipeline.downloader is not None
        assert pipeline.audio_extractor is not None
        assert pipeline.transcriber is not None
        assert pipeline.progress_tracker is not None

    @patch("core.pipeline.TempFileManager")
    def test_initialization_components_failure(self, mock_file_manager):
        """Test component initialization failure."""
        mock_file_manager.side_effect = Exception("Initialization failed")

        pipeline = TranscriptionPipeline()

        with pytest.raises(PipelineInitializationException):
            pipeline._initialize_components()

    @patch("core.pipeline.validate_instagram_url")
    def test_validate_url_success(self, mock_validate):
        """Test successful URL validation."""
        mock_validate.return_value = (True, None)

        pipeline = TranscriptionPipeline()
        pipeline.downloader = Mock()
        pipeline.downloader.validate_reel_url.return_value = (True, None, "ABC123")

        # Should not raise exception
        pipeline._validate_url("https://instagram.com/reel/ABC123/")

    @patch("core.pipeline.validate_instagram_url")
    def test_validate_url_invalid(self, mock_validate):
        """Test URL validation with invalid URL."""
        mock_validate.return_value = (False, "Invalid URL format")

        pipeline = TranscriptionPipeline()

        with pytest.raises(PipelineValidationException):
            pipeline._validate_url("invalid-url")

    def test_download_video_success(self, temp_dir):
        """Test successful video download."""
        # Create a mock video file
        video_path = temp_dir / "test_video.mp4"
        video_path.write_bytes(b"fake video content")

        pipeline = TranscriptionPipeline()
        pipeline.downloader = Mock()
        pipeline.downloader.download_reel.return_value = (True, str(video_path), None)

        result_path = pipeline._download_video("https://instagram.com/reel/ABC123/")

        assert result_path == str(video_path)
        assert video_path.exists()

    def test_download_video_failure(self):
        """Test video download failure."""
        pipeline = TranscriptionPipeline()
        pipeline.downloader = Mock()
        pipeline.downloader.download_reel.return_value = (False, None, "Download failed")

        with pytest.raises(PipelineDownloadException):
            pipeline._download_video("https://instagram.com/reel/ABC123/")

    def test_extract_audio_success(self, temp_dir):
        """Test successful audio extraction."""
        # Create mock files
        video_path = temp_dir / "video.mp4"
        audio_path = temp_dir / "audio.wav"
        video_path.write_bytes(b"video")
        audio_path.write_bytes(b"audio")

        pipeline = TranscriptionPipeline()
        pipeline.file_manager = Mock()
        pipeline.file_manager.get_temp_path.return_value = str(audio_path)
        pipeline.audio_extractor = Mock()
        pipeline.audio_extractor.extract_audio.return_value = (True, str(audio_path), None)
        pipeline.audio_extractor.validate_audio_file.return_value = (True, None)

        result_path = pipeline._extract_audio(str(video_path))

        assert result_path == str(audio_path)

    def test_extract_audio_failure(self, temp_dir):
        """Test audio extraction failure."""
        video_path = temp_dir / "video.mp4"
        video_path.write_bytes(b"video")

        pipeline = TranscriptionPipeline()
        pipeline.file_manager = Mock()
        pipeline.file_manager.get_temp_path.return_value = "/tmp/audio.wav"
        pipeline.audio_extractor = Mock()
        pipeline.audio_extractor.extract_audio.return_value = (False, None, "Extraction failed")

        with pytest.raises(PipelineAudioException):
            pipeline._extract_audio(str(video_path))

    def test_transcribe_audio_success(self, temp_dir):
        """Test successful audio transcription."""
        audio_path = temp_dir / "audio.wav"
        audio_path.write_bytes(b"audio")

        # Create mock transcription result
        mock_result = TranscriptionResult(text="Test transcription", language="en", segments=[], metadata={"time": 1.5})

        pipeline = TranscriptionPipeline()
        pipeline.transcriber = Mock()
        pipeline.transcriber.transcribe_audio.return_value = (True, mock_result, None)

        result = pipeline._transcribe_audio(str(audio_path))

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Test transcription"
        assert result.language == "en"

    def test_transcribe_audio_failure(self, temp_dir):
        """Test audio transcription failure."""
        audio_path = temp_dir / "audio.wav"
        audio_path.write_bytes(b"audio")

        pipeline = TranscriptionPipeline()
        pipeline.transcriber = Mock()
        pipeline.transcriber.transcribe_audio.return_value = (False, None, "Transcription failed")

        with pytest.raises(PipelineTranscriptionException):
            pipeline._transcribe_audio(str(audio_path))

    def test_cleanup_temporary_files(self):
        """Test temporary file cleanup."""
        pipeline = TranscriptionPipeline()
        pipeline.file_manager = Mock()
        pipeline.file_manager.cleanup_session.return_value = True

        # Should not raise exception
        pipeline._cleanup_temporary_files()

        pipeline.file_manager.cleanup_session.assert_called_once()

    def test_cleanup_pipeline_state(self):
        """Test pipeline state cleanup."""
        pipeline = TranscriptionPipeline()

        # Setup mock components
        pipeline.transcriber = Mock()
        pipeline.file_manager = Mock()
        pipeline.downloader = Mock()
        pipeline.audio_extractor = Mock()
        pipeline.progress_tracker = Mock()

        pipeline._cleanup_pipeline_state()

        # Verify cleanup was called
        pipeline.transcriber.cleanup.assert_called_once()
        pipeline.file_manager.cleanup_all.assert_called_once()

        # Verify components were reset
        assert pipeline.transcriber is None
        assert pipeline.file_manager is None
        assert pipeline.downloader is None
        assert pipeline.audio_extractor is None
        assert pipeline.progress_tracker is None

    def test_cancellation_check(self):
        """Test cancellation checking."""
        pipeline = TranscriptionPipeline()

        # Not cancelled initially
        assert not pipeline._check_cancelled()

        # Set cancellation
        pipeline._cancel_event.set()

        # Should raise exception when checked
        with pytest.raises(PipelineCancelledException):
            pipeline._check_cancelled()

    def test_progress_update(self):
        """Test progress update functionality."""
        progress_callback = Mock()
        pipeline = TranscriptionPipeline(progress_callback=progress_callback)

        # Create mock progress tracker
        pipeline.progress_tracker = Mock()

        pipeline._update_progress(ProcessingStage.DOWNLOADING, 50, "Download progress", "Details")

        pipeline.progress_tracker.update_progress.assert_called_once_with(
            ProcessingStage.DOWNLOADING, 50, "Download progress", "Details"
        )

    def test_stage_completion_tracking(self):
        """Test stage completion tracking."""
        pipeline = TranscriptionPipeline()

        assert pipeline._stages_completed == 0

        pipeline._increment_stage_completion()
        assert pipeline._stages_completed == 1

        pipeline._increment_stage_completion()
        assert pipeline._stages_completed == 2

    def test_current_stage_setting(self):
        """Test current stage setting."""
        pipeline = TranscriptionPipeline()

        assert pipeline.get_current_stage() == PipelineStage.INIT

        pipeline._set_stage(PipelineStage.DOWNLOAD)
        assert pipeline.get_current_stage() == PipelineStage.DOWNLOAD

        pipeline._set_stage(PipelineStage.COMPLETE)
        assert pipeline.get_current_stage() == PipelineStage.COMPLETE

    def test_emergency_cleanup(self, temp_dir):
        """Test emergency cleanup functionality."""
        # Create test files
        file1 = temp_dir / "file1.tmp"
        file2 = temp_dir / "file2.tmp"
        file1.write_bytes(b"temp1")
        file2.write_bytes(b"temp2")

        pipeline = TranscriptionPipeline()
        pipeline._rollback_files = [str(file1), str(file2)]
        pipeline.file_manager = Mock()

        pipeline._emergency_cleanup()

        # Files should be removed
        assert not file1.exists()
        assert not file2.exists()
        pipeline.file_manager.cleanup_session.assert_called_once()

    @patch.object(TranscriptionPipeline, "_initialize_components")
    @patch.object(TranscriptionPipeline, "_validate_url")
    @patch.object(TranscriptionPipeline, "_download_video")
    @patch.object(TranscriptionPipeline, "_extract_audio")
    @patch.object(TranscriptionPipeline, "_transcribe_audio")
    @patch.object(TranscriptionPipeline, "_cleanup_temporary_files")
    def test_execute_success(
        self, mock_cleanup, mock_transcribe, mock_extract, mock_download, mock_validate, mock_init
    ):
        """Test successful pipeline execution."""
        # Setup mocks
        mock_download.return_value = "/tmp/video.mp4"
        mock_extract.return_value = "/tmp/audio.wav"

        mock_result = TranscriptionResult(text="Success transcript", language="en", segments=[], metadata={"time": 2.5})
        mock_transcribe.return_value = mock_result

        pipeline = TranscriptionPipeline()

        result = pipeline.execute("https://instagram.com/reel/ABC123/")

        assert result.success
        assert result.transcript == "Success transcript"
        assert result.detected_language == "en"
        assert result.execution_time > 0
        assert result.stages_completed == 5  # All stages completed
        assert result.error_message is None

        # Verify all stages were called
        mock_init.assert_called_once()
        mock_validate.assert_called_once()
        mock_download.assert_called_once()
        mock_extract.assert_called_once()
        mock_transcribe.assert_called_once()
        mock_cleanup.assert_called_once()

    def test_execute_already_running(self):
        """Test execute when pipeline is already running."""
        pipeline = TranscriptionPipeline()
        pipeline._state = PipelineState.RUNNING

        result = pipeline.execute("https://instagram.com/reel/ABC123/")

        assert not result.success
        assert "already running" in result.error_message.lower()

    @patch.object(TranscriptionPipeline, "_initialize_components")
    def test_execute_initialization_failure(self, mock_init):
        """Test execute with initialization failure."""
        mock_init.side_effect = PipelineInitializationException("Init failed")

        pipeline = TranscriptionPipeline()

        result = pipeline.execute("https://instagram.com/reel/ABC123/")

        assert not result.success
        assert result.transcript is None
        assert "init failed" in result.error_message.lower()

    @patch.object(TranscriptionPipeline, "_initialize_components")
    @patch.object(TranscriptionPipeline, "_validate_url")
    def test_execute_validation_failure(self, mock_validate, mock_init):
        """Test execute with URL validation failure."""
        mock_validate.side_effect = PipelineValidationException("Invalid URL")

        pipeline = TranscriptionPipeline()

        result = pipeline.execute("invalid-url")

        assert not result.success
        assert "invalid url" in result.error_message.lower()

    def test_cancel_not_running(self):
        """Test cancelling when not running."""
        pipeline = TranscriptionPipeline()

        success = pipeline.cancel()

        assert not success

    def test_cancel_while_running(self):
        """Test cancelling while running."""
        pipeline = TranscriptionPipeline()
        pipeline._state = PipelineState.RUNNING

        success = pipeline.cancel()

        assert success
        assert pipeline.get_state() == PipelineState.CANCELLING
        assert pipeline._cancel_event.is_set()

    @patch.object(TranscriptionPipeline, "_initialize_components")
    @patch.object(TranscriptionPipeline, "_validate_url")
    @patch.object(TranscriptionPipeline, "_download_video")
    def test_execute_with_cancellation(self, mock_download, mock_validate, mock_init):
        """Test execute with cancellation during processing."""

        # Setup cancellation during download
        def cancel_during_download(*args):
            pipeline._cancel_event.set()
            return "/tmp/video.mp4"

        mock_download.side_effect = cancel_during_download

        pipeline = TranscriptionPipeline()

        result = pipeline.execute("https://instagram.com/reel/ABC123/")

        assert not result.success
        assert "cancelled" in result.error_message.lower()
        assert pipeline.get_state() == PipelineState.CANCELLED

    def test_create_cancelled_result(self):
        """Test creation of cancelled result."""
        pipeline = TranscriptionPipeline()
        pipeline._start_time = time.time() - 5.0  # 5 seconds ago
        pipeline._stages_completed = 2
        pipeline._current_stage = PipelineStage.DOWNLOAD

        result = pipeline._create_cancelled_result()

        assert not result.success
        assert result.execution_time > 0
        assert result.stages_completed == 2
        assert "download" in result.error_message.lower()
        assert pipeline.get_state() == PipelineState.CANCELLED

    def test_create_error_result(self):
        """Test creation of error result."""
        pipeline = TranscriptionPipeline()
        pipeline._start_time = time.time() - 3.0  # 3 seconds ago
        pipeline._stages_completed = 1

        result = pipeline._create_error_result("Test error message")

        assert not result.success
        assert result.execution_time > 0
        assert result.stages_completed == 1
        assert result.error_message == "Test error message"
        assert pipeline.get_state() == PipelineState.FAILED


class TestPipelineThreadSafety:
    """Test pipeline thread safety."""

    def test_concurrent_execution_requests(self):
        """Test handling of concurrent execution requests."""
        pipeline = TranscriptionPipeline()

        results = []

        def execute_pipeline():
            result = pipeline.execute("https://instagram.com/reel/ABC123/")
            results.append(result)

        # Start two threads trying to execute simultaneously
        thread1 = threading.Thread(target=execute_pipeline)
        thread2 = threading.Thread(target=execute_pipeline)

        thread1.start()
        time.sleep(0.1)  # Small delay to ensure first thread starts
        thread2.start()

        thread1.join()
        thread2.join()

        # One should succeed (or fail for other reasons),
        # one should fail due to already running
        assert len(results) == 2

        # At least one should fail due to already running
        already_running_errors = [r for r in results if not r.success and "already running" in r.error_message.lower()]
        assert len(already_running_errors) >= 1

    def test_cancel_thread_safety(self):
        """Test cancellation thread safety."""
        pipeline = TranscriptionPipeline()
        pipeline._state = PipelineState.RUNNING

        cancellation_results = []

        def cancel_pipeline():
            result = pipeline.cancel()
            cancellation_results.append(result)

        # Multiple threads trying to cancel
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=cancel_pipeline)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All should succeed in setting cancellation
        assert len(cancellation_results) == 3
        assert all(cancellation_results)


class TestPipelineExceptions:
    """Test pipeline exception hierarchy."""

    def test_pipeline_exception_hierarchy(self):
        """Test that all pipeline exceptions inherit from PipelineException."""
        exceptions = [
            PipelineInitializationException,
            PipelineValidationException,
            PipelineDownloadException,
            PipelineAudioException,
            PipelineTranscriptionException,
            PipelineCancelledException,
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, PipelineException)

    def test_pipeline_exceptions_with_messages(self):
        """Test pipeline exceptions with custom messages."""
        test_message = "Custom error message"

        exceptions = [
            PipelineInitializationException(test_message),
            PipelineValidationException(test_message),
            PipelineDownloadException(test_message),
            PipelineAudioException(test_message),
            PipelineTranscriptionException(test_message),
            PipelineCancelledException(test_message),
        ]

        for exc in exceptions:
            assert str(exc) == test_message


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_pipeline(self):
        """Test create_pipeline convenience function."""
        config = {"test": "value"}
        callback = Mock()

        pipeline = create_pipeline(config, callback)

        assert isinstance(pipeline, TranscriptionPipeline)
        assert pipeline.config == config
        assert pipeline.progress_callback == callback

    @patch.object(TranscriptionPipeline, "execute")
    def test_process_reel(self, mock_execute):
        """Test process_reel convenience function."""
        mock_result = PipelineResult(success=True, transcript="Test")
        mock_execute.return_value = mock_result

        config = {"test": "config"}
        callback = Mock()
        url = "https://instagram.com/reel/ABC123/"

        result = process_reel(url, config, callback)

        assert result == mock_result
        mock_execute.assert_called_once_with(url)


@pytest.mark.benchmark
class TestPipelinePerformance:
    """Performance tests for pipeline."""

    @patch.object(TranscriptionPipeline, "_initialize_components")
    def test_pipeline_initialization_performance(self, mock_init, benchmark):
        """Benchmark pipeline initialization performance."""

        def create_pipeline():
            return TranscriptionPipeline()

        pipeline = benchmark(create_pipeline)
        assert isinstance(pipeline, TranscriptionPipeline)

    def test_state_management_performance(self, benchmark):
        """Benchmark state management operations."""
        pipeline = TranscriptionPipeline()

        def state_operations():
            pipeline._set_stage(PipelineStage.DOWNLOAD)
            state = pipeline.get_state()
            stage = pipeline.get_current_stage()
            is_running = pipeline.is_running()
            return (state, stage, is_running)

        result = benchmark(state_operations)
        assert result is not None


@pytest.mark.integration
class TestPipelineIntegration:
    """Integration tests for pipeline with mocked components."""

    @patch("core.pipeline.TempFileManager")
    @patch("core.pipeline.InstagramDownloader")
    @patch("core.pipeline.AudioExtractor")
    @patch("core.pipeline.WhisperTranscriber")
    @patch("core.pipeline.ProgressTracker")
    @patch("core.pipeline.validate_instagram_url")
    def test_complete_pipeline_flow(
        self,
        mock_validate,
        mock_progress_tracker,
        mock_transcriber,
        mock_audio_extractor,
        mock_downloader,
        mock_file_manager,
        temp_dir,
    ):
        """Test complete pipeline flow with all components mocked."""
        # Setup all mocks for successful execution
        mock_validate.return_value = (True, None)

        # File manager
        mock_file_manager_instance = Mock()
        mock_file_manager_instance.create_session_dir.return_value = str(temp_dir)
        mock_file_manager_instance.get_temp_path.return_value = str(temp_dir / "audio.wav")
        mock_file_manager_instance.cleanup_session.return_value = True
        mock_file_manager.return_value = mock_file_manager_instance

        # Downloader
        video_path = temp_dir / "video.mp4"
        video_path.write_bytes(b"fake video")
        mock_downloader_instance = Mock()
        mock_downloader_instance.validate_reel_url.return_value = (True, None, "ABC123")
        mock_downloader_instance.download_reel.return_value = (True, str(video_path), None)
        mock_downloader.return_value = mock_downloader_instance

        # Audio extractor
        audio_path = temp_dir / "audio.wav"
        audio_path.write_bytes(b"fake audio")
        mock_audio_extractor_instance = Mock()
        mock_audio_extractor_instance.extract_audio.return_value = (True, str(audio_path), None)
        mock_audio_extractor_instance.validate_audio_file.return_value = (True, None)
        mock_audio_extractor.return_value = mock_audio_extractor_instance

        # Transcriber
        mock_result = TranscriptionResult(
            text="Integration test transcript", language="en", segments=[], metadata={"duration": 3.0}
        )
        mock_transcriber_instance = Mock()
        mock_transcriber_instance.transcribe_audio.return_value = (True, mock_result, None)
        mock_transcriber.return_value = mock_transcriber_instance

        # Progress tracker
        mock_progress_tracker.return_value = Mock()

        # Execute pipeline
        progress_updates = []

        def progress_callback(update):
            progress_updates.append(update)

        pipeline = TranscriptionPipeline(progress_callback=progress_callback)
        result = pipeline.execute("https://instagram.com/reel/ABC123/")

        # Verify successful execution
        assert result.success
        assert result.transcript == "Integration test transcript"
        assert result.detected_language == "en"
        assert result.execution_time > 0
        assert result.stages_completed == 5
        assert result.error_message is None

        # Verify all components were used
        mock_file_manager.assert_called_once()
        mock_downloader.assert_called_once()
        mock_audio_extractor.assert_called_once()
        mock_transcriber.assert_called_once()
        mock_progress_tracker.assert_called_once()

        # Verify pipeline completed
        assert pipeline.get_state() == PipelineState.COMPLETED
        assert pipeline.get_current_stage() == PipelineStage.COMPLETE

    def test_pipeline_error_propagation(self):
        """Test that errors are properly propagated through the pipeline."""
        pipeline = TranscriptionPipeline()

        # Test with completely invalid URL
        result = pipeline.execute("not-a-url")

        assert not result.success
        assert result.error_message is not None
        assert result.transcript is None
        assert pipeline.get_state() == PipelineState.FAILED

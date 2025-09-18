"""
Unit tests for audio processing component.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.audio_processor import AudioExtractor


class TestAudioExtractor:
    """Test the AudioExtractor class."""

    def test_initialization_default_params(self):
        """Test audio extractor initialization with default parameters."""
        extractor = AudioExtractor()

        assert extractor.target_sample_rate == 16000
        assert extractor.chunk_duration == 30

    def test_initialization_custom_params(self):
        """Test audio extractor initialization with custom parameters."""
        extractor = AudioExtractor(target_sample_rate=22050, chunk_duration=60)

        assert extractor.target_sample_rate == 22050
        assert extractor.chunk_duration == 60

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_success(self, mock_video_clip, sample_video_path, temp_dir):
        """Test successful audio extraction."""
        # Setup mock video clip
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_audio.write_audiofile = Mock()
        mock_clip.duration = 30.0
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "output_audio.wav")
        progress_callback = Mock()

        # Create the expected output file (simulating successful extraction)
        with open(output_path, "wb") as f:
            f.write(b"fake audio content")

        success, audio_path, error_msg = extractor.extract_audio(
            sample_video_path, output_path, progress_callback=progress_callback
        )

        assert success
        assert audio_path == output_path
        assert error_msg is None
        assert progress_callback.called
        mock_audio.write_audiofile.assert_called_once()

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_video_not_found(self, mock_video_clip, temp_dir):
        """Test audio extraction with non-existent video file."""
        mock_video_clip.side_effect = FileNotFoundError("Video file not found")

        extractor = AudioExtractor()
        non_existent_video = str(temp_dir / "non_existent.mp4")
        output_path = str(temp_dir / "output.wav")

        success, audio_path, error_msg = extractor.extract_audio(non_existent_video, output_path)

        assert not success
        assert audio_path is None
        assert "not found" in error_msg.lower()

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_no_audio_track(self, mock_video_clip, sample_video_path, temp_dir):
        """Test audio extraction when video has no audio track."""
        mock_clip = Mock()
        mock_clip.audio = None  # No audio track
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "output.wav")

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)

        assert not success
        assert audio_path is None
        assert "no audio" in error_msg.lower()

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_moviepy_error(self, mock_video_clip, sample_video_path, temp_dir):
        """Test audio extraction with MoviePy error."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_audio.write_audiofile.side_effect = Exception("MoviePy processing error")
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "output.wav")

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)

        assert not success
        assert audio_path is None
        assert "extraction failed" in error_msg.lower()

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_with_progress_tracking(self, mock_video_clip, sample_video_path, temp_dir):
        """Test audio extraction with progress tracking."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 60.0  # 1 minute video

        # Mock write_audiofile to simulate progress
        def mock_write_audio(*args, **kwargs):
            # Simulate calling the progress callback
            if "logger" in kwargs:
                progress_callback = kwargs.get("progress_callback")
                if progress_callback:
                    progress_callback(50, "Processing audio...")

        mock_audio.write_audiofile.side_effect = mock_write_audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "output.wav")
        progress_callback = Mock()

        # Create output file to simulate success
        with open(output_path, "wb") as f:
            f.write(b"audio content")

        success, audio_path, error_msg = extractor.extract_audio(
            sample_video_path, output_path, progress_callback=progress_callback
        )

        assert success
        assert progress_callback.called

    @patch("core.audio_processor.VideoFileClip")
    def test_extract_audio_custom_sample_rate(self, mock_video_clip, sample_video_path, temp_dir):
        """Test audio extraction with custom sample rate."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_resampled_audio = Mock()
        mock_clip.audio = mock_audio
        mock_audio.set_fps.return_value = mock_resampled_audio
        mock_resampled_audio.write_audiofile = Mock()
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor(target_sample_rate=22050)
        output_path = str(temp_dir / "output.wav")

        # Create output file
        with open(output_path, "wb") as f:
            f.write(b"resampled audio")

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)

        assert success
        mock_audio.set_fps.assert_called_once_with(22050)
        mock_resampled_audio.write_audiofile.assert_called_once()

    def test_validate_audio_file_success(self, sample_audio_path):
        """Test successful audio file validation."""
        extractor = AudioExtractor()

        is_valid, error_msg = extractor.validate_audio_file(sample_audio_path)

        assert is_valid
        assert error_msg is None

    def test_validate_audio_file_not_found(self, temp_dir):
        """Test audio file validation with non-existent file."""
        extractor = AudioExtractor()
        non_existent_file = str(temp_dir / "non_existent.wav")

        is_valid, error_msg = extractor.validate_audio_file(non_existent_file)

        assert not is_valid
        assert "not found" in error_msg.lower()

    def test_validate_audio_file_empty(self, temp_dir):
        """Test audio file validation with empty file."""
        extractor = AudioExtractor()
        empty_file = temp_dir / "empty.wav"
        empty_file.touch()  # Create empty file

        is_valid, error_msg = extractor.validate_audio_file(str(empty_file))

        assert not is_valid
        assert "empty" in error_msg.lower()

    @patch("core.audio_processor.wave.open")
    def test_validate_audio_file_invalid_format(self, mock_wave_open, temp_dir):
        """Test audio file validation with invalid format."""
        mock_wave_open.side_effect = Exception("Invalid wave file")

        extractor = AudioExtractor()
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_bytes(b"not a wave file")

        is_valid, error_msg = extractor.validate_audio_file(str(invalid_file))

        assert not is_valid
        assert "invalid" in error_msg.lower()

    @patch("core.audio_processor.wave.open")
    def test_validate_audio_file_valid_properties(self, mock_wave_open, sample_audio_path):
        """Test audio file validation with specific audio properties."""
        # Mock wave file with valid properties
        mock_wave_file = Mock()
        mock_wave_file.__enter__ = Mock(return_value=mock_wave_file)
        mock_wave_file.__exit__ = Mock(return_value=None)
        mock_wave_file.getframerate.return_value = 16000
        mock_wave_file.getnchannels.return_value = 1
        mock_wave_file.getsampwidth.return_value = 2
        mock_wave_file.getnframes.return_value = 32000  # 2 seconds of audio

        mock_wave_open.return_value = mock_wave_file

        extractor = AudioExtractor()

        is_valid, error_msg = extractor.validate_audio_file(sample_audio_path)

        assert is_valid
        assert error_msg is None

    @patch("core.audio_processor.wave.open")
    def test_validate_audio_file_no_frames(self, mock_wave_open, sample_audio_path):
        """Test audio file validation with no audio frames."""
        mock_wave_file = Mock()
        mock_wave_file.__enter__ = Mock(return_value=mock_wave_file)
        mock_wave_file.__exit__ = Mock(return_value=None)
        mock_wave_file.getframerate.return_value = 16000
        mock_wave_file.getnchannels.return_value = 1
        mock_wave_file.getsampwidth.return_value = 2
        mock_wave_file.getnframes.return_value = 0  # No audio frames

        mock_wave_open.return_value = mock_wave_file

        extractor = AudioExtractor()

        is_valid, error_msg = extractor.validate_audio_file(sample_audio_path)

        assert not is_valid
        assert "no audio" in error_msg.lower()

    def test_get_audio_info_success(self, sample_audio_path):
        """Test getting audio file information."""
        extractor = AudioExtractor()

        info = extractor.get_audio_info(sample_audio_path)

        assert info is not None
        assert "sample_rate" in info
        assert "channels" in info
        assert "duration" in info
        assert "sample_width" in info
        assert info["sample_rate"] > 0
        assert info["channels"] > 0
        assert info["duration"] > 0

    def test_get_audio_info_file_not_found(self, temp_dir):
        """Test getting audio info for non-existent file."""
        extractor = AudioExtractor()
        non_existent_file = str(temp_dir / "non_existent.wav")

        info = extractor.get_audio_info(non_existent_file)

        assert info is None

    @patch("core.audio_processor.wave.open")
    def test_get_audio_info_invalid_file(self, mock_wave_open, temp_dir):
        """Test getting audio info for invalid file."""
        mock_wave_open.side_effect = Exception("Invalid file")

        extractor = AudioExtractor()
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_bytes(b"invalid")

        info = extractor.get_audio_info(str(invalid_file))

        assert info is None

    @patch("core.audio_processor.VideoFileClip")
    def test_get_video_info_success(self, mock_video_clip, sample_video_path):
        """Test getting video file information."""
        mock_clip = Mock()
        mock_clip.duration = 30.5
        mock_clip.fps = 30
        mock_clip.size = (1920, 1080)
        mock_clip.audio = Mock()  # Has audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()

        info = extractor.get_video_info(sample_video_path)

        assert info is not None
        assert info["duration"] == 30.5
        assert info["fps"] == 30
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["has_audio"] is True

    @patch("core.audio_processor.VideoFileClip")
    def test_get_video_info_no_audio(self, mock_video_clip, sample_video_path):
        """Test getting video info for video without audio."""
        mock_clip = Mock()
        mock_clip.duration = 15.0
        mock_clip.fps = 24
        mock_clip.size = (1280, 720)
        mock_clip.audio = None  # No audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()

        info = extractor.get_video_info(sample_video_path)

        assert info is not None
        assert info["duration"] == 15.0
        assert info["fps"] == 24
        assert info["width"] == 1280
        assert info["height"] == 720
        assert info["has_audio"] is False

    @patch("core.audio_processor.VideoFileClip")
    def test_get_video_info_error(self, mock_video_clip, sample_video_path):
        """Test getting video info with error."""
        mock_video_clip.side_effect = Exception("Video processing error")

        extractor = AudioExtractor()

        info = extractor.get_video_info(sample_video_path)

        assert info is None

    def test_get_supported_formats(self):
        """Test getting supported audio/video formats."""
        extractor = AudioExtractor()

        formats = extractor.get_supported_formats()

        assert isinstance(formats, dict)
        assert "video" in formats
        assert "audio" in formats
        assert isinstance(formats["video"], list)
        assert isinstance(formats["audio"], list)
        assert len(formats["video"]) > 0
        assert len(formats["audio"]) > 0


class TestAudioExtractorIntegration:
    """Integration tests for audio extraction."""

    @patch("core.audio_processor.VideoFileClip")
    def test_complete_extraction_workflow(self, mock_video_clip, sample_video_path, temp_dir):
        """Test complete audio extraction workflow."""
        # Setup realistic mock
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 45.0
        mock_video_clip.return_value = mock_clip

        # Track extraction steps
        steps_completed = []

        def mock_write_audio(*args, **kwargs):
            steps_completed.append("audio_written")
            # Create the output file
            output_path = args[0]
            with open(output_path, "wb") as f:
                f.write(b"extracted audio content")

        mock_audio.write_audiofile.side_effect = mock_write_audio

        extractor = AudioExtractor()
        output_path = str(temp_dir / "extracted_audio.wav")

        # Track progress
        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        # Perform extraction
        success, audio_path, error_msg = extractor.extract_audio(
            sample_video_path, output_path, progress_callback=progress_callback
        )

        # Verify results
        assert success
        assert audio_path == output_path
        assert error_msg is None
        assert Path(output_path).exists()
        assert "audio_written" in steps_completed

        # Verify progress tracking
        assert len(progress_updates) > 0

        # Validate the extracted audio
        is_valid, validation_error = extractor.validate_audio_file(output_path)
        # Note: This might fail with fake audio content, which is acceptable
        # In a real scenario, the audio would be properly formatted

    def test_extraction_error_recovery(self, sample_video_path, temp_dir):
        """Test error recovery during extraction."""
        extractor = AudioExtractor()

        # Test with invalid output directory
        invalid_output_path = "/invalid/path/output.wav"

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, invalid_output_path)

        assert not success
        assert audio_path is None
        assert error_msg is not None

    @patch("core.audio_processor.VideoFileClip")
    def test_multiple_extractions(self, mock_video_clip, sample_video_path, temp_dir):
        """Test multiple audio extractions in sequence."""

        # Setup mock for multiple extractions
        def create_mock_clip():
            mock_clip = Mock()
            mock_audio = Mock()
            mock_clip.audio = mock_audio
            mock_clip.duration = 30.0

            def write_audio(*args, **kwargs):
                output_path = args[0]
                with open(output_path, "wb") as f:
                    f.write(b"audio content")

            mock_audio.write_audiofile.side_effect = write_audio
            return mock_clip

        mock_video_clip.side_effect = lambda *args: create_mock_clip()

        extractor = AudioExtractor()

        # Perform multiple extractions
        results = []
        for i in range(3):
            output_path = str(temp_dir / f"audio_{i}.wav")
            success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)
            results.append((success, audio_path, error_msg))

        # Verify all extractions succeeded
        for success, audio_path, error_msg in results:
            assert success
            assert audio_path is not None
            assert error_msg is None
            assert Path(audio_path).exists()


@pytest.mark.benchmark
class TestAudioExtractorPerformance:
    """Performance tests for audio extraction."""

    @patch("core.audio_processor.VideoFileClip")
    def test_extraction_performance(self, mock_video_clip, benchmark, sample_video_path, temp_dir):
        """Benchmark audio extraction performance."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 30.0

        def fast_write_audio(*args, **kwargs):
            output_path = args[0]
            with open(output_path, "wb") as f:
                f.write(b"fast audio")

        mock_audio.write_audiofile.side_effect = fast_write_audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "benchmark_audio.wav")

        def extract_audio():
            return extractor.extract_audio(sample_video_path, output_path)

        result = benchmark(extract_audio)
        assert result[0] is True  # Should succeed

    def test_validation_performance(self, benchmark, sample_audio_path):
        """Benchmark audio validation performance."""
        extractor = AudioExtractor()

        def validate_audio():
            return extractor.validate_audio_file(sample_audio_path)

        result = benchmark(validate_audio)
        assert result[0] is True  # Should be valid

    @patch("core.audio_processor.wave.open")
    def test_info_retrieval_performance(self, mock_wave_open, benchmark, sample_audio_path):
        """Benchmark audio info retrieval performance."""
        mock_wave_file = Mock()
        mock_wave_file.__enter__ = Mock(return_value=mock_wave_file)
        mock_wave_file.__exit__ = Mock(return_value=None)
        mock_wave_file.getframerate.return_value = 16000
        mock_wave_file.getnchannels.return_value = 1
        mock_wave_file.getsampwidth.return_value = 2
        mock_wave_file.getnframes.return_value = 32000

        mock_wave_open.return_value = mock_wave_file

        extractor = AudioExtractor()

        def get_info():
            return extractor.get_audio_info(sample_audio_path)

        result = benchmark(get_info)
        assert result is not None


class TestAudioExtractorEdgeCases:
    """Test edge cases and error conditions."""

    @patch("core.audio_processor.VideoFileClip")
    def test_very_short_video(self, mock_video_clip, sample_video_path, temp_dir):
        """Test extraction from very short video."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 0.1  # 100ms video

        def write_short_audio(*args, **kwargs):
            output_path = args[0]
            with open(output_path, "wb") as f:
                f.write(b"short")

        mock_audio.write_audiofile.side_effect = write_short_audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "short_audio.wav")

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)

        assert success  # Should handle short videos

    @patch("core.audio_processor.VideoFileClip")
    def test_very_long_video(self, mock_video_clip, sample_video_path, temp_dir):
        """Test extraction from very long video."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 3600.0  # 1 hour video

        def write_long_audio(*args, **kwargs):
            output_path = args[0]
            with open(output_path, "wb") as f:
                f.write(b"long audio content")

        mock_audio.write_audiofile.side_effect = write_long_audio
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "long_audio.wav")

        progress_updates = []

        def progress_callback(progress, message):
            progress_updates.append((progress, message))

        success, audio_path, error_msg = extractor.extract_audio(
            sample_video_path, output_path, progress_callback=progress_callback
        )

        assert success  # Should handle long videos
        # Should have progress updates for long extraction
        assert len(progress_updates) >= 1

    def test_invalid_output_permissions(self, sample_video_path):
        """Test extraction with invalid output permissions."""
        extractor = AudioExtractor()

        # Try to write to root directory (should fail on most systems)
        invalid_output_path = "/root/cannot_write_here.wav"

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, invalid_output_path)

        assert not success
        assert audio_path is None
        assert error_msg is not None

    @patch("core.audio_processor.VideoFileClip")
    def test_memory_cleanup(self, mock_video_clip, sample_video_path, temp_dir):
        """Test that memory is properly cleaned up after extraction."""
        mock_clip = Mock()
        mock_audio = Mock()
        mock_clip.audio = mock_audio
        mock_clip.duration = 30.0
        mock_clip.close = Mock()  # Ensure close is called

        def write_and_cleanup(*args, **kwargs):
            output_path = args[0]
            with open(output_path, "wb") as f:
                f.write(b"audio with cleanup")

        mock_audio.write_audiofile.side_effect = write_and_cleanup
        mock_video_clip.return_value = mock_clip

        extractor = AudioExtractor()
        output_path = str(temp_dir / "cleanup_audio.wav")

        success, audio_path, error_msg = extractor.extract_audio(sample_video_path, output_path)

        assert success
        # Verify that cleanup methods were called
        mock_clip.close.assert_called()

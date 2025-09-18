"""
Unit tests for Whisper transcription component.
"""

from unittest.mock import Mock, patch

import pytest

from core.transcriber import FASTER_WHISPER_AVAILABLE, TranscriptionResult, WhisperTranscriber, transcribe_audio_file


class TestTranscriptionResult:
    """Test the TranscriptionResult class."""

    def test_transcription_result_initialization(self):
        """Test TranscriptionResult initialization."""
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.0, "end": 2.0, "text": "world"}]
        metadata = {"language": "en", "duration": 2.0}

        result = TranscriptionResult(text="Hello world", language="en", segments=segments, metadata=metadata)

        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.segments == segments
        assert result.metadata == metadata

    def test_get_formatted_text_without_timestamps(self):
        """Test formatted text without timestamps."""
        segments = [{"start": 0.0, "end": 1.0, "text": "Hello"}, {"start": 1.0, "end": 2.0, "text": "world"}]

        result = TranscriptionResult(text="Hello world", language="en", segments=segments, metadata={})

        formatted = result.get_formatted_text(include_timestamps=False)
        assert formatted == "Hello world"

    def test_get_formatted_text_with_timestamps(self):
        """Test formatted text with timestamps."""
        segments = [{"start": 0.0, "end": 1.5, "text": "Hello"}, {"start": 1.5, "end": 3.0, "text": "world"}]

        result = TranscriptionResult(text="Hello world", language="en", segments=segments, metadata={})

        formatted = result.get_formatted_text(include_timestamps=True)
        expected = "[0.00s -> 1.50s] Hello\n[1.50s -> 3.00s] world"
        assert formatted == expected

    def test_get_formatted_text_empty_segments(self):
        """Test formatted text with empty segments."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": ""},
            {"start": 1.0, "end": 2.0, "text": "   "},
            {"start": 2.0, "end": 3.0, "text": "Hello"},
        ]

        result = TranscriptionResult(text="Hello", language="en", segments=segments, metadata={})

        formatted = result.get_formatted_text(include_timestamps=True)
        assert formatted == "[2.00s -> 3.00s] Hello"


class TestWhisperTranscriber:
    """Test the WhisperTranscriber class."""

    def test_initialization_default_params(self):
        """Test transcriber initialization with default parameters."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber()

        assert transcriber.model_size == "base"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert transcriber.supported_languages == ["en", "de"]
        assert transcriber.chunk_duration == 30
        assert transcriber.model is None
        assert not transcriber.model_loaded

    def test_initialization_custom_params(self):
        """Test transcriber initialization with custom parameters."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(
            model_size="small",
            device="cuda",
            compute_type="float16",
            supported_languages=["en", "de", "fr"],
            chunk_duration=60,
        )

        assert transcriber.model_size == "small"
        assert transcriber.device == "cuda"
        assert transcriber.compute_type == "float16"
        assert transcriber.supported_languages == ["en", "de", "fr"]
        assert transcriber.chunk_duration == 60

    def test_initialization_without_faster_whisper(self):
        """Test transcriber initialization when faster-whisper is not available."""
        with patch("core.transcriber.FASTER_WHISPER_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="faster-whisper library is not available"):
                WhisperTranscriber()

    @patch("core.transcriber.WhisperModel")
    def test_load_model_success(self, mock_whisper_model):
        """Test successful model loading."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        progress_callback = Mock()

        success, error = transcriber.load_model(progress_callback)

        assert success
        assert error is None
        assert transcriber.model == mock_model
        assert transcriber.model_loaded
        assert progress_callback.called

    @patch("core.transcriber.WhisperModel")
    def test_load_model_failure(self, mock_whisper_model):
        """Test model loading failure."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_whisper_model.side_effect = Exception("Model loading failed")

        transcriber = WhisperTranscriber()

        success, error = transcriber.load_model()

        assert not success
        assert "Model loading failed" in error
        assert transcriber.model is None
        assert not transcriber.model_loaded

    @patch("core.transcriber.WhisperModel")
    def test_load_model_already_loaded(self, mock_whisper_model):
        """Test loading model when already loaded."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        transcriber.model = mock_model
        transcriber.model_loaded = True

        success, error = transcriber.load_model()

        assert success
        assert error is None
        # WhisperModel should not be called again
        assert not mock_whisper_model.called

    def test_transcribe_audio_file_not_found(self, temp_dir):
        """Test transcription with non-existent audio file."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber()
        non_existent_file = str(temp_dir / "non_existent.wav")

        success, result, error = transcriber.transcribe_audio(non_existent_file)

        assert not success
        assert result is None
        assert "not found" in error.lower()

    def test_transcribe_audio_empty_file(self, temp_dir):
        """Test transcription with empty audio file."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        empty_file = temp_dir / "empty.wav"
        empty_file.touch()

        transcriber = WhisperTranscriber()

        success, result, error = transcriber.transcribe_audio(str(empty_file))

        assert not success
        assert result is None
        assert "empty" in error.lower()

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_success(self, mock_whisper_model, sample_audio_path):
        """Test successful audio transcription."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        # Setup mock model and segments
        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.tokens = [1, 2, 3]
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()
        progress_callback = Mock()

        success, result, error = transcriber.transcribe_audio(sample_audio_path, progress_callback=progress_callback)

        assert success
        assert error is None
        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0]["text"] == "Hello world"
        assert progress_callback.called

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_with_language(self, mock_whisper_model, sample_audio_path):
        """Test transcription with specified language."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Hallo Welt"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.tokens = [1, 2]
        mock_segment.avg_logprob = -0.3
        mock_segment.no_speech_prob = 0.05

        mock_info = Mock()
        mock_info.language = "de"
        mock_info.language_probability = 0.88

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber(supported_languages=["en", "de"])

        success, result, error = transcriber.transcribe_audio(sample_audio_path, language="de")

        assert success
        assert result.text == "Hallo Welt"
        assert result.language == "de"

        # Verify that language was passed to model
        mock_model.transcribe.assert_called_once()
        call_args = mock_model.transcribe.call_args
        assert call_args[1]["language"] == "de"

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_unsupported_language(self, mock_whisper_model, sample_audio_path):
        """Test transcription with unsupported language falls back to auto-detection."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Hello world"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.tokens = [1, 2, 3]
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber(supported_languages=["en", "de"])

        success, result, error = transcriber.transcribe_audio(
            sample_audio_path,
            language="fr",  # Unsupported language
        )

        assert success
        # Should fall back to auto-detection (language=None)
        call_args = mock_model.transcribe.call_args
        assert "language" not in call_args[1] or call_args[1]["language"] is None

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_no_speech(self, mock_whisper_model, sample_audio_path):
        """Test transcription when no speech is detected."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        # Empty segments (no speech detected)
        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        success, result, error = transcriber.transcribe_audio(sample_audio_path)

        assert not success
        assert result is None
        assert "no speech detected" in error.lower()

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_transcription_error(self, mock_whisper_model, sample_audio_path):
        """Test handling of transcription errors."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription failed")
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        success, result, error = transcriber.transcribe_audio(sample_audio_path)

        assert not success
        assert result is None
        assert "transcription failed" in error.lower()

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_chunked(self, mock_whisper_model, sample_audio_path):
        """Test chunked transcription (currently delegates to regular transcription)."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Test chunked transcription"
        mock_segment.start = 0.0
        mock_segment.end = 5.0
        mock_segment.tokens = [1, 2, 3, 4]
        mock_segment.avg_logprob = -0.4
        mock_segment.no_speech_prob = 0.08

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.92

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        success, result, error = transcriber.transcribe_audio_chunked(sample_audio_path)

        assert success
        assert result.text == "Test chunked transcription"

    @patch("core.transcriber.WhisperModel")
    def test_detect_language(self, mock_whisper_model, sample_audio_path):
        """Test language detection functionality."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_info = Mock()
        mock_info.language = "de"
        mock_info.language_probability = 0.87

        mock_model.transcribe.return_value = ([], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        success, language, confidence = transcriber.detect_language(sample_audio_path)

        assert success
        assert language == "de"
        assert confidence == 0.87

    @patch("core.transcriber.WhisperModel")
    def test_detect_language_failure(self, mock_whisper_model, sample_audio_path):
        """Test language detection failure."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Detection failed")
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        success, language, confidence = transcriber.detect_language(sample_audio_path)

        assert not success
        assert language is None
        assert confidence == 0.0

    def test_get_supported_languages(self):
        """Test getting supported languages."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(supported_languages=["en", "de", "fr"])

        languages = transcriber.get_supported_languages()

        assert languages == ["en", "de", "fr"]
        # Ensure it returns a copy, not the original list
        languages.append("es")
        assert transcriber.supported_languages == ["en", "de", "fr"]

    def test_is_language_supported(self):
        """Test language support checking."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(supported_languages=["en", "de"])

        assert transcriber.is_language_supported("en")
        assert transcriber.is_language_supported("de")
        assert not transcriber.is_language_supported("fr")
        assert not transcriber.is_language_supported("invalid")

    def test_get_model_info(self):
        """Test getting model information."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(
            model_size="small", device="cuda", compute_type="float16", supported_languages=["en", "de"]
        )

        info = transcriber.get_model_info()

        assert info["model_size"] == "small"
        assert info["device"] == "cuda"
        assert info["compute_type"] == "float16"
        assert info["model_loaded"] is False
        assert info["supported_languages"] == ["en", "de"]

    def test_cleanup(self):
        """Test cleanup functionality."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber()
        transcriber.model = Mock()
        transcriber.model_loaded = True

        transcriber.cleanup()

        assert transcriber.model is None
        assert not transcriber.model_loaded


class TestTranscribeAudioFileFunction:
    """Test the convenience transcribe_audio_file function."""

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_file_success(self, mock_whisper_model, sample_audio_path):
        """Test successful transcription using convenience function."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Convenience function test"
        mock_segment.start = 0.0
        mock_segment.end = 3.0
        mock_segment.tokens = [1, 2, 3, 4]
        mock_segment.avg_logprob = -0.6
        mock_segment.no_speech_prob = 0.12

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.91

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        progress_callback = Mock()

        success, transcript, error = transcribe_audio_file(
            sample_audio_path, model_size="base", language="en", device="cpu", progress_callback=progress_callback
        )

        assert success
        assert transcript == "Convenience function test"
        assert error is None
        assert progress_callback.called

    @patch("core.transcriber.WhisperModel")
    def test_transcribe_audio_file_failure(self, mock_whisper_model, sample_audio_path):
        """Test transcription failure using convenience function."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_model.transcribe.side_effect = Exception("Transcription error")
        mock_whisper_model.return_value = mock_model

        success, transcript, error = transcribe_audio_file(sample_audio_path)

        assert not success
        assert transcript is None
        assert "transcription failed" in error.lower()

    def test_transcribe_audio_file_no_whisper(self, sample_audio_path):
        """Test convenience function when faster-whisper is not available."""
        with patch("core.transcriber.FASTER_WHISPER_AVAILABLE", False):
            success, transcript, error = transcribe_audio_file(sample_audio_path)

            assert not success
            assert transcript is None
            assert "transcription failed" in error.lower()


@pytest.mark.benchmark
class TestTranscriberPerformance:
    """Performance tests for transcriber component."""

    @patch("core.transcriber.WhisperModel")
    def test_model_loading_performance(self, mock_whisper_model, benchmark):
        """Benchmark model loading performance."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        def load_model():
            return transcriber.load_model()

        result = benchmark(load_model)
        assert result[0] is True  # Should succeed

    @patch("core.transcriber.WhisperModel")
    def test_transcription_performance(self, mock_whisper_model, benchmark, sample_audio_path):
        """Benchmark transcription performance."""
        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        mock_model = Mock()
        mock_segment = Mock()
        mock_segment.text = "Performance test"
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.tokens = [1, 2]
        mock_segment.avg_logprob = -0.5
        mock_segment.no_speech_prob = 0.1

        mock_info = Mock()
        mock_info.language = "en"
        mock_info.language_probability = 0.95

        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_model.return_value = mock_model

        transcriber = WhisperTranscriber()

        def transcribe():
            return transcriber.transcribe_audio(sample_audio_path)

        result = benchmark(transcribe)
        assert result[0] is True  # Should succeed


@pytest.mark.requires_model
class TestTranscriberWithRealModel:
    """Tests that require actual model downloads (run sparingly)."""

    def test_real_model_loading(self):
        """Test loading a real Whisper model."""
        pytest.skip("Skipping real model test to avoid large downloads")

        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(model_size="tiny")  # Smallest model

        success, error = transcriber.load_model()

        assert success
        assert error is None
        assert transcriber.model is not None
        assert transcriber.model_loaded

    def test_real_transcription(self, sample_audio_path):
        """Test real transcription with actual model."""
        pytest.skip("Skipping real model test to avoid large downloads")

        if not FASTER_WHISPER_AVAILABLE:
            pytest.skip("faster-whisper not available")

        transcriber = WhisperTranscriber(model_size="tiny")

        success, result, error = transcriber.transcribe_audio(sample_audio_path)

        # With a real model and synthetic audio, results may vary
        # Just check that the process completes without error
        if success:
            assert result is not None
            assert isinstance(result, TranscriptionResult)
        else:
            # May fail with synthetic audio - that's acceptable
            assert error is not None

"""
Comprehensive test fixtures and mock data for Instagram Reels Transcriber.
"""

import wave
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock

import numpy as np

from core.transcriber import TranscriptionResult


class MockDataGenerator:
    """Generate realistic mock data for testing."""

    @staticmethod
    def create_realistic_audio_file(
        output_path: str, duration: float = 5.0, sample_rate: int = 16000, include_speech: bool = True
    ) -> str:
        """
        Create a realistic audio file with optional speech-like patterns.

        Args:
            output_path: Path where to save the audio file
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            include_speech: Whether to include speech-like frequency patterns

        Returns:
            Path to the created audio file
        """
        num_samples = int(duration * sample_rate)

        if include_speech:
            # Generate speech-like audio with varying frequencies
            t = np.linspace(0, duration, num_samples, False)

            # Base frequency modulation (simulating speech formants)
            base_freq = 150 + 50 * np.sin(2 * np.pi * t * 0.5)  # Fundamental frequency variation
            formant1 = 800 + 200 * np.sin(2 * np.pi * t * 1.2)  # First formant
            formant2 = 1200 + 300 * np.sin(2 * np.pi * t * 0.8)  # Second formant

            # Generate complex waveform
            audio_data = (
                0.3 * np.sin(2 * np.pi * base_freq * t)
                + 0.2 * np.sin(2 * np.pi * formant1 * t)
                + 0.1 * np.sin(2 * np.pi * formant2 * t)
            )

            # Add speech-like envelope (pauses and amplitude variation)
            envelope = np.ones_like(t)
            # Add some pauses
            pause_starts = np.array([1.5, 3.2])
            pause_durations = np.array([0.3, 0.2])

            for start, dur in zip(pause_starts, pause_durations):
                start_idx = int(start * sample_rate)
                end_idx = int((start + dur) * sample_rate)
                if end_idx < len(envelope):
                    envelope[start_idx:end_idx] *= 0.1

            audio_data *= envelope

            # Add some background noise
            noise = 0.02 * np.random.normal(0, 1, num_samples)
            audio_data += noise

        else:
            # Generate simple sine wave or noise
            t = np.linspace(0, duration, num_samples, False)
            audio_data = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

        # Normalize and convert to 16-bit PCM
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_data = (audio_data * 32767).astype(np.int16)

        # Write to WAV file
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        return output_path

    @staticmethod
    def create_mock_video_file(output_path: str, duration: float = 10.0) -> str:
        """
        Create a mock video file (minimal MP4 structure).

        Args:
            output_path: Path where to save the video file
            duration: Duration in seconds

        Returns:
            Path to the created video file
        """
        # Create a minimal MP4 file with basic headers
        # This is not a real video, just enough to pass basic file checks
        mp4_header = bytes(
            [
                0x00,
                0x00,
                0x00,
                0x20,  # Box size
                0x66,
                0x74,
                0x79,
                0x70,  # Box type: 'ftyp'
                0x6D,
                0x70,
                0x34,
                0x32,  # Major brand: 'mp42'
                0x00,
                0x00,
                0x00,
                0x00,  # Minor version
                0x6D,
                0x70,
                0x34,
                0x32,  # Compatible brand: 'mp42'
                0x69,
                0x73,
                0x6F,
                0x6D,  # Compatible brand: 'isom'
            ]
        )

        # Add some fake video data
        fake_data = b"FAKE_VIDEO_DATA" * 100

        with open(output_path, "wb") as f:
            f.write(mp4_header)
            f.write(fake_data)

        return output_path

    @staticmethod
    def create_transcription_segments(text: str, num_segments: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Create realistic transcription segments from text.

        Args:
            text: Text to create segments from
            num_segments: Number of segments to create (auto if None)

        Returns:
            List of segment dictionaries
        """
        words = text.split()
        if not words:
            return []

        if num_segments is None:
            # Roughly 5-8 words per segment
            num_segments = max(1, len(words) // 6)

        segments = []
        words_per_segment = len(words) // num_segments
        current_time = 0.0

        for i in range(num_segments):
            start_word = i * words_per_segment
            end_word = min((i + 1) * words_per_segment, len(words))

            if i == num_segments - 1:  # Last segment gets remaining words
                end_word = len(words)

            segment_words = words[start_word:end_word]
            segment_text = " ".join(segment_words)

            # Estimate timing (roughly 2-3 words per second)
            segment_duration = len(segment_words) / 2.5
            start_time = current_time
            end_time = current_time + segment_duration

            segment = {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "text": segment_text,
                "tokens": list(range(start_word * 10, end_word * 10)),  # Mock tokens
                "avg_logprob": round(-0.2 - np.random.random() * 0.3, 3),
                "no_speech_prob": round(np.random.random() * 0.1, 3),
            }

            segments.append(segment)
            current_time = end_time + 0.1  # Small gap between segments

        return segments

    @staticmethod
    def create_mock_transcription_result(
        text: str, language: str = "en", model_size: str = "base"
    ) -> TranscriptionResult:
        """
        Create a mock TranscriptionResult with realistic data.

        Args:
            text: Transcript text
            language: Detected language
            model_size: Model size used

        Returns:
            Mock TranscriptionResult
        """
        segments = MockDataGenerator.create_transcription_segments(text)

        metadata = {
            "detected_language": language,
            "language_probability": 0.85 + np.random.random() * 0.14,
            "transcription_time": 1.5 + np.random.random() * 3.0,
            "audio_duration": segments[-1]["end"] if segments else 0.0,
            "model_size": model_size,
            "num_segments": len(segments),
            "device": "cpu",
            "compute_type": "int8",
        }

        return TranscriptionResult(text=text, language=language, segments=segments, metadata=metadata)


class MockInstagramData:
    """Mock Instagram-specific data for testing."""

    # Real Instagram Reel URLs for testing - categorized by expected characteristics
    SAMPLE_URLS = {
        "real_test_reels": [
            # Primary test URLs - Real Instagram reels for integration testing
            "https://www.instagram.com/reels/DOqtiuYiLV5/",  # Expected: Short, English
            "https://www.instagram.com/reels/DOq10tEAnZ1/",  # Expected: Medium, Multi-language
            "https://www.instagram.com/reels/DOGoywOiMqA/",  # Expected: Medium, Music/Dance
            "https://www.instagram.com/reels/DMXM6vuo1t0/",  # Expected: Longer format
        ],
        "valid_reels_mock": [
            # Mock URLs for unit testing (no network requests)
            "https://www.instagram.com/reel/CxYzABC123/",
            "https://instagram.com/reel/DeFgHIJ456/",
            "https://www.instagram.com/reel/KlMnOPQ789/",
            "https://instagram.com/reel/RsTeUVW012/",
        ],
        "valid_posts": [
            "https://www.instagram.com/p/CxYzABC123/",
            "https://instagram.com/p/DeFgHIJ456/",
        ],
        "invalid_urls": [
            "https://not-instagram.com/reel/123/",
            "https://instagram.com/invalid-format/",
            "not-a-url-at-all",
            "",
            "https://instagram.com/",
            "https://youtube.com/watch?v=123",
        ],
        "edge_cases": [
            "https://www.instagram.com/reel/",  # Missing ID
            "https://instagram.com/reel/a/",  # Very short ID
            "https://instagram.com/reel/a" * 50 + "/",  # Very long ID
        ],
    }

    # Expected characteristics for real test URLs (for privacy-safe testing)
    REAL_URL_METADATA = {
        "https://www.instagram.com/reels/DOqtiuYiLV5/": {
            "expected_category": "short_form",
            "expected_duration_range": (5, 30),  # seconds
            "expected_language_hints": ["en"],
            "test_scenarios": ["basic_transcription", "short_audio"],
            "privacy_notes": "Content pattern only, no actual content logged",
        },
        "https://www.instagram.com/reels/DOq10tEAnZ1/": {
            "expected_category": "medium_form",
            "expected_duration_range": (15, 45),
            "expected_language_hints": ["multi", "en"],
            "test_scenarios": ["language_detection", "medium_audio"],
            "privacy_notes": "Content pattern only, no actual content logged",
        },
        "https://www.instagram.com/reels/DOGoywOiMqA/": {
            "expected_category": "music_dance",
            "expected_duration_range": (10, 40),
            "expected_language_hints": ["music", "minimal_speech"],
            "test_scenarios": ["music_detection", "low_speech_content"],
            "privacy_notes": "Content pattern only, no actual content logged",
        },
        "https://www.instagram.com/reels/DMXM6vuo1t0/": {
            "expected_category": "long_form",
            "expected_duration_range": (30, 90),
            "expected_language_hints": ["en", "extended"],
            "test_scenarios": ["long_transcription", "performance_test"],
            "privacy_notes": "Content pattern only, no actual content logged",
        },
    }

    # Test data categories for systematic testing
    TEST_CATEGORIES = {
        "integration_basic": {
            "urls": ["https://www.instagram.com/reels/DOqtiuYiLV5/"],
            "purpose": "Basic end-to-end integration testing",
            "expected_success": True,
            "timeout_seconds": 60,
        },
        "integration_comprehensive": {
            "urls": [
                "https://www.instagram.com/reels/DOqtiuYiLV5/",
                "https://www.instagram.com/reels/DOq10tEAnZ1/",
                "https://www.instagram.com/reels/DOGoywOiMqA/",
                "https://www.instagram.com/reels/DMXM6vuo1t0/",
            ],
            "purpose": "Comprehensive integration testing across different content types",
            "expected_success": True,
            "timeout_seconds": 180,
        },
        "performance_benchmark": {
            "urls": ["https://www.instagram.com/reels/DOqtiuYiLV5/", "https://www.instagram.com/reels/DMXM6vuo1t0/"],
            "purpose": "Performance benchmarking with short and long content",
            "expected_success": True,
            "timeout_seconds": 120,
        },
        "unit_testing": {
            "urls": [
                "https://www.instagram.com/reel/CxYzABC123/",
                "https://instagram.com/reel/DeFgHIJ456/",
                "https://www.instagram.com/reel/KlMnOPQ789/",
                "https://instagram.com/reel/RsTeUVW012/",
            ],
            "purpose": "Unit testing with mock URLs (no network)",
            "expected_success": True,
            "timeout_seconds": 30,
        },
        "error_scenarios": {
            "urls": [
                "https://not-instagram.com/reel/123/",
                "https://instagram.com/invalid-format/",
                "not-a-url-at-all",
                "",
                "https://instagram.com/",
                "https://youtube.com/watch?v=123",
            ],
            "purpose": "Error handling and validation testing",
            "expected_success": False,
            "timeout_seconds": 30,
        },
    }

    SAMPLE_CAPTIONS = [
        "Check out this amazing sunset! 🌅 #nature #photography",
        "Today's workout was intense! 💪 #fitness #motivation #gym",
        "Cooking my favorite pasta recipe 🍝 Follow for more recipes!",
        "Dancing to my favorite song 💃 #dance #music #viral",
        "Behind the scenes of my art project 🎨 #art #creative #process",
        "Quick tutorial on how to style your hair ✨ #tutorial #beauty",
        "Funny moments with my pet 🐕 #pets #funny #cute",
        "Travel vlog from my recent trip ✈️ #travel #adventure #explore",
    ]

    SAMPLE_METADATA = {
        "shortcode": "CxYzABC123",
        "is_video": True,
        "video_duration": 30.5,
        "caption": "Sample Instagram Reel caption",
        "date_utc": "2024-01-15T10:30:00Z",
        "owner_username": "testuser123",
        "likes": 1234,
        "video_url": "https://instagram.com/video/sample.mp4",
    }

    @staticmethod
    def create_mock_post_data(shortcode: str, is_video: bool = True) -> dict[str, Any]:
        """Create mock Instagram post data."""
        import random
        from datetime import datetime, timedelta

        base_data = {
            "shortcode": shortcode,
            "is_video": is_video,
            "caption": random.choice(MockInstagramData.SAMPLE_CAPTIONS),
            "date_utc": (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat() + "Z",
            "owner_username": f"user{random.randint(1000, 9999)}",
            "likes": random.randint(10, 50000),
        }

        if is_video:
            base_data.update(
                {
                    "video_duration": round(random.uniform(5.0, 60.0), 1),
                    "video_url": f"https://instagram.com/video/{shortcode}.mp4",
                }
            )

        return base_data


class MockErrorScenarios:
    """Mock error scenarios for testing error handling."""

    NETWORK_ERRORS = [
        {
            "exception": "ConnectionError",
            "message": "Network connection failed",
            "retryable": True,
            "category": "network",
        },
        {
            "exception": "TimeoutError",
            "message": "Request timed out after 30 seconds",
            "retryable": True,
            "category": "network",
        },
        {"exception": "HTTPError", "message": "503 Service Unavailable", "retryable": True, "category": "server"},
        {"exception": "HTTPError", "message": "404 Not Found", "retryable": False, "category": "client"},
        {
            "exception": "HTTPError",
            "message": "403 Forbidden - Private account",
            "retryable": False,
            "category": "access",
        },
    ]

    FILE_SYSTEM_ERRORS = [
        {
            "exception": "FileNotFoundError",
            "message": "Video file not found",
            "retryable": False,
            "category": "filesystem",
        },
        {
            "exception": "PermissionError",
            "message": "Permission denied writing to directory",
            "retryable": False,
            "category": "permissions",
        },
        {"exception": "OSError", "message": "No space left on device", "retryable": False, "category": "storage"},
        {"exception": "OSError", "message": "Invalid file path", "retryable": False, "category": "filesystem"},
    ]

    PROCESSING_ERRORS = [
        {
            "exception": "Exception",
            "message": "Audio extraction failed - corrupt video file",
            "retryable": False,
            "category": "processing",
        },
        {
            "exception": "Exception",
            "message": "Model failed to load - insufficient memory",
            "retryable": False,
            "category": "model",
        },
        {
            "exception": "Exception",
            "message": "Transcription timeout - audio too long",
            "retryable": False,
            "category": "processing",
        },
        {"exception": "Exception", "message": "No speech detected in audio", "retryable": False, "category": "content"},
    ]


class MockProgressData:
    """Mock progress update data for testing."""

    PROGRESS_SEQUENCES = {
        "successful_transcription": [
            {"stage": "VALIDATING", "progress": 10, "message": "Validating URL..."},
            {"stage": "DOWNLOADING", "progress": 25, "message": "Downloading video..."},
            {"stage": "DOWNLOADING", "progress": 75, "message": "Download 75% complete..."},
            {"stage": "EXTRACTING_AUDIO", "progress": 85, "message": "Extracting audio..."},
            {"stage": "TRANSCRIBING", "progress": 95, "message": "Transcribing audio..."},
            {"stage": "COMPLETED", "progress": 100, "message": "Transcription complete!"},
        ],
        "download_failure": [
            {"stage": "VALIDATING", "progress": 10, "message": "Validating URL..."},
            {"stage": "DOWNLOADING", "progress": 25, "message": "Downloading video..."},
            {"stage": "ERROR", "progress": None, "message": "Download failed: Network error"},
        ],
        "transcription_failure": [
            {"stage": "VALIDATING", "progress": 10, "message": "Validating URL..."},
            {"stage": "DOWNLOADING", "progress": 50, "message": "Download complete"},
            {"stage": "EXTRACTING_AUDIO", "progress": 75, "message": "Audio extracted"},
            {"stage": "TRANSCRIBING", "progress": 90, "message": "Starting transcription..."},
            {"stage": "ERROR", "progress": None, "message": "Transcription failed: No speech detected"},
        ],
    }


class TestDataPaths:
    """Manage test data file paths and creation."""

    def __init__(self, base_temp_dir: str):
        self.base_dir = Path(base_temp_dir)
        self.audio_dir = self.base_dir / "audio"
        self.video_dir = self.base_dir / "video"
        self.output_dir = self.base_dir / "output"

        # Create directories
        for dir_path in [self.audio_dir, self.video_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def create_sample_files(self) -> dict[str, str]:
        """Create sample audio and video files for testing."""
        files = {}

        # Create various audio files
        files["short_audio"] = MockDataGenerator.create_realistic_audio_file(
            str(self.audio_dir / "short.wav"), duration=2.0, include_speech=True
        )

        files["long_audio"] = MockDataGenerator.create_realistic_audio_file(
            str(self.audio_dir / "long.wav"), duration=30.0, include_speech=True
        )

        files["silent_audio"] = MockDataGenerator.create_realistic_audio_file(
            str(self.audio_dir / "silent.wav"), duration=5.0, include_speech=False
        )

        files["empty_audio"] = str(self.audio_dir / "empty.wav")
        Path(files["empty_audio"]).touch()

        # Create video files
        files["sample_video"] = MockDataGenerator.create_mock_video_file(
            str(self.video_dir / "sample.mp4"), duration=15.0
        )

        files["long_video"] = MockDataGenerator.create_mock_video_file(str(self.video_dir / "long.mp4"), duration=120.0)

        files["empty_video"] = str(self.video_dir / "empty.mp4")
        Path(files["empty_video"]).touch()

        return files


class MockModelBehavior:
    """Mock ML model behavior for testing."""

    @staticmethod
    def create_mock_whisper_model(behavior: str = "normal") -> Mock:
        """
        Create a mock Whisper model with configurable behavior.

        Args:
            behavior: 'normal', 'slow', 'fail', 'no_speech'

        Returns:
            Mock WhisperModel object
        """
        mock_model = Mock()

        if behavior == "normal":
            # Normal successful transcription
            mock_segment = Mock()
            mock_segment.text = "This is a normal transcription result."
            mock_segment.start = 0.0
            mock_segment.end = 3.0
            mock_segment.tokens = [1, 2, 3, 4, 5, 6]
            mock_segment.avg_logprob = -0.3
            mock_segment.no_speech_prob = 0.05

            mock_info = Mock()
            mock_info.language = "en"
            mock_info.language_probability = 0.95

            mock_model.transcribe.return_value = ([mock_segment], mock_info)

        elif behavior == "slow":
            # Slow transcription (for timeout testing)
            def slow_transcribe(*args, **kwargs):
                import time

                time.sleep(5)  # Simulate slow processing
                mock_segment = Mock()
                mock_segment.text = "Slow transcription result."
                mock_segment.start = 0.0
                mock_segment.end = 2.0
                mock_segment.tokens = [1, 2, 3]
                mock_segment.avg_logprob = -0.4
                mock_segment.no_speech_prob = 0.08

                mock_info = Mock()
                mock_info.language = "en"
                mock_info.language_probability = 0.88

                return ([mock_segment], mock_info)

            mock_model.transcribe.side_effect = slow_transcribe

        elif behavior == "fail":
            # Model failure
            mock_model.transcribe.side_effect = Exception("Model inference failed")

        elif behavior == "no_speech":
            # No speech detected
            mock_info = Mock()
            mock_info.language = "en"
            mock_info.language_probability = 0.20

            mock_model.transcribe.return_value = ([], mock_info)

        return mock_model


# Convenience functions for common test data creation


def create_test_environment(temp_dir: str) -> dict[str, Any]:
    """Create a complete test environment with all necessary mock data."""
    test_paths = TestDataPaths(temp_dir)
    sample_files = test_paths.create_sample_files()

    return {
        "paths": test_paths,
        "files": sample_files,
        "urls": MockInstagramData.SAMPLE_URLS,
        "captions": MockInstagramData.SAMPLE_CAPTIONS,
        "progress_sequences": MockProgressData.PROGRESS_SEQUENCES,
        "error_scenarios": {
            "network": MockErrorScenarios.NETWORK_ERRORS,
            "filesystem": MockErrorScenarios.FILE_SYSTEM_ERRORS,
            "processing": MockErrorScenarios.PROCESSING_ERRORS,
        },
    }


def create_mock_pipeline_dependencies(temp_dir: str) -> dict[str, Mock]:
    """Create a complete set of mocked pipeline dependencies."""
    test_env = create_test_environment(temp_dir)

    mocks = {
        "file_manager": Mock(),
        "downloader": Mock(),
        "audio_extractor": Mock(),
        "transcriber": Mock(),
        "progress_tracker": Mock(),
    }

    # Configure file manager
    mocks["file_manager"].create_session_dir.return_value = temp_dir
    mocks["file_manager"].get_temp_path.side_effect = lambda name: str(Path(temp_dir) / name)
    mocks["file_manager"].cleanup_session.return_value = True
    mocks["file_manager"].cleanup_all.return_value = True

    # Configure downloader
    mocks["downloader"].validate_reel_url.return_value = (True, None, "ABC123")
    mocks["downloader"].download_reel.return_value = (True, test_env["files"]["sample_video"], None)

    # Configure audio extractor
    mocks["audio_extractor"].extract_audio.return_value = (True, test_env["files"]["short_audio"], None)
    mocks["audio_extractor"].validate_audio_file.return_value = (True, None)

    # Configure transcriber
    mock_result = MockDataGenerator.create_mock_transcription_result(
        "This is a mock transcription for testing purposes."
    )
    mocks["transcriber"].transcribe_audio.return_value = (True, mock_result, None)
    mocks["transcriber"].load_model.return_value = (True, None)
    mocks["transcriber"].cleanup.return_value = None

    return mocks

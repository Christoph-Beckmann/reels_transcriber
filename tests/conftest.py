"""
Pytest configuration and shared fixtures for Instagram Reels Transcriber tests.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.logging_config import setup_application_logging

# Test configuration
TEST_CONFIG = {
    "temp_dir": None,  # Will be set in setup
    "cleanup_on_exit": True,
    "download_timeout": 30,  # Shorter for tests
    "max_retries": 1,  # Fewer retries for tests
    "retry_delay": 0.1,  # Faster retries for tests
    "whisper_model": "tiny",  # Smallest model for tests
    "target_sample_rate": 16000,
    "audio_chunk_duration": 10,  # Shorter chunks for tests
    "supported_languages": ["en"],
    "auto_detect_language": True,
    "log_level": "DEBUG",
}


@pytest.fixture(scope="session")
def test_session_setup():
    """Session-wide test setup."""
    # Set up logging for tests
    setup_application_logging(level="DEBUG", console_output=False, debug_mode=True, log_file_prefix="test_")

    # Disable GUI-related warnings
    os.environ["PYTEST_RUNNING"] = "1"

    # Create reports directory
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    yield

    # Session cleanup
    logging.getLogger(__name__).info("Test session completed")


@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for test isolation."""
    with tempfile.TemporaryDirectory(prefix="reels_test_") as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="function")
def test_config(temp_dir):
    """Test configuration with temporary directory."""
    config = TEST_CONFIG.copy()
    config["temp_dir"] = str(temp_dir)
    return config


@pytest.fixture(scope="function")
def mock_instagram_url():
    """Mock Instagram Reel URL for testing."""
    return "https://www.instagram.com/reel/ABC123DEF/"


@pytest.fixture(scope="function")
def mock_invalid_url():
    """Mock invalid Instagram URL for testing."""
    return "https://not-instagram.com/video/123"


@pytest.fixture(scope="function")
def sample_audio_path(temp_dir):
    """Create a sample audio file for testing."""
    import wave

    import numpy as np

    # Create a simple sine wave audio file
    sample_rate = 16000
    duration = 2.0  # 2 seconds
    frequency = 440.0  # A4 note

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(frequency * 2 * np.pi * t) * 0.3

    # Convert to 16-bit PCM
    audio_data = (audio_data * 32767).astype(np.int16)

    # Write to WAV file
    audio_path = temp_dir / "test_audio.wav"
    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    return str(audio_path)


@pytest.fixture(scope="function")
def sample_video_path(temp_dir):
    """Create a sample video file for testing."""
    # For testing, we'll create a minimal MP4 file
    # In real tests, you might use a proper video generation library
    video_path = temp_dir / "test_video.mp4"

    # Create minimal MP4 file (this is a placeholder)
    # In production, you'd use moviepy or similar to create a real video
    with open(video_path, "wb") as f:
        # Write minimal MP4 header (this is not a real video, just for file existence)
        f.write(b"\x00\x00\x00\x20ftypmp42\x00\x00\x00\x00mp42isom")

    return str(video_path)


@pytest.fixture(scope="function")
def mock_whisper_model():
    """Mock WhisperModel for testing transcription without actual ML inference."""
    mock_model = Mock()

    # Mock segments generator
    mock_segment = Mock()
    mock_segment.text = "This is a test transcription."
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.tokens = [1, 2, 3, 4, 5]
    mock_segment.avg_logprob = -0.5
    mock_segment.no_speech_prob = 0.1

    # Mock info object
    mock_info = Mock()
    mock_info.language = "en"
    mock_info.language_probability = 0.95

    # Set up transcribe method
    mock_model.transcribe.return_value = ([mock_segment], mock_info)

    return mock_model


@pytest.fixture(scope="function")
def mock_instaloader():
    """Mock Instaloader for testing downloads without network calls."""
    mock_loader = Mock()
    mock_post = Mock()

    # Configure mock post
    mock_post.shortcode = "ABC123DEF"
    mock_post.is_video = True
    mock_post.video_duration = 30
    mock_post.caption = "Test Instagram Reel"
    mock_post.owner_username = "testuser"
    mock_post.likes = 100
    mock_post.video_url = "https://instagram.com/video/test.mp4"

    # Configure mock loader
    mock_loader.download_post = Mock()

    return mock_loader, mock_post


@pytest.fixture(scope="function")
def mock_progress_callback():
    """Mock progress callback for testing progress updates."""
    return Mock()


@pytest.fixture(scope="function")
def mock_gui_queue():
    """Mock GUI message queue for testing worker communication."""
    import queue

    return queue.Queue()


@pytest.fixture(scope="function")
def sample_transcript_result():
    """Sample transcription result for testing."""
    from core.transcriber import TranscriptionResult

    segments = [
        {
            "start": 0.0,
            "end": 1.5,
            "text": "Hello world.",
            "tokens": [1, 2, 3],
            "avg_logprob": -0.3,
            "no_speech_prob": 0.05,
        },
        {
            "start": 1.5,
            "end": 3.0,
            "text": "This is a test.",
            "tokens": [4, 5, 6],
            "avg_logprob": -0.4,
            "no_speech_prob": 0.08,
        },
    ]

    metadata = {
        "detected_language": "en",
        "language_probability": 0.95,
        "transcription_time": 1.23,
        "audio_duration": 3.0,
        "model_size": "tiny",
        "num_segments": 2,
        "device": "cpu",
        "compute_type": "int8",
    }

    return TranscriptionResult(text="Hello world. This is a test.", language="en", segments=segments, metadata=metadata)


@pytest.fixture(scope="function")
def mock_file_manager(temp_dir):
    """Mock TempFileManager for testing file operations."""
    from core.file_manager import TempFileManager

    # Create real file manager but with test directory
    return TempFileManager(
        base_temp_dir=str(temp_dir),
        cleanup_on_exit=False,  # Let test cleanup handle it
    )


# Performance testing fixtures
@pytest.fixture(scope="function")
def benchmark_config():
    """Configuration for performance benchmarks."""
    return {"min_rounds": 3, "max_time": 10.0, "warmup_rounds": 1, "calibration_precision": 0.01}


# GUI testing fixtures
@pytest.fixture(scope="session")
def gui_test_setup():
    """Setup for GUI tests - configure display and window management."""
    import os

    # Set environment variables for headless testing
    os.environ["DISPLAY"] = ":0"  # For Linux
    os.environ["PYTEST_GUI_TESTING"] = "1"

    # Configure GUI library for testing
    try:
        import FreeSimpleGUI as sg

        sg.theme("SystemDefault")
        sg.set_options(suppress_error_popups=True, suppress_raise_key_errors=True, suppress_key_guessing=True)
    except ImportError:
        pass

    yield

    # GUI cleanup
    try:
        import FreeSimpleGUI as sg

        sg.popup_quick_message("Tests completed", auto_close_duration=1)
    except Exception:
        pass


@pytest.fixture(scope="function")
def mock_gui_elements():
    """Mock GUI elements for testing window interactions."""
    mock_window = Mock()
    mock_elements = {}

    # Create mock elements for each key
    element_keys = [
        "-URL-",
        "-START-",
        "-STOP-",
        "-CLEAR-",
        "-COPY-",
        "-RETRY-",
        "-HELP-",
        "-EXIT-",
        "-STATUS-",
        "-PROGRESS-",
        "-TIME-ESTIMATE-",
        "-TRANSCRIPT-",
    ]

    for key in element_keys:
        mock_element = Mock()
        mock_element.update = Mock()
        mock_element.set_focus = Mock()
        mock_element.get = Mock(return_value="")
        mock_elements[key] = mock_element

    # Configure window to return mock elements
    mock_window.__getitem__ = lambda self, key: mock_elements.get(key, Mock())
    mock_window.read = Mock(return_value=(None, {}))
    mock_window.close = Mock()
    mock_window.refresh = Mock()

    return mock_window, mock_elements


# Error testing fixtures
@pytest.fixture(scope="function")
def mock_network_error():
    """Mock network-related errors for testing error handling."""
    from requests.exceptions import ConnectionError, HTTPError, Timeout

    errors = {
        "connection": ConnectionError("Network connection failed"),
        "timeout": Timeout("Request timed out"),
        "http_404": HTTPError("404 Not Found"),
        "http_403": HTTPError("403 Forbidden"),
    }

    return errors


@pytest.fixture(scope="function")
def mock_file_system_error():
    """Mock file system errors for testing error handling."""
    errors = {
        "permission": PermissionError("Permission denied"),
        "not_found": FileNotFoundError("File not found"),
        "disk_full": OSError("No space left on device"),
        "invalid_path": OSError("Invalid file path"),
    }

    return errors


# Test data fixtures
@pytest.fixture(scope="session")
def test_urls():
    """Collection of test URLs for various scenarios."""
    return {
        "valid_reel": "https://www.instagram.com/reel/ABC123DEF/",
        "valid_reel_short": "https://instagram.com/reel/XYZ789/",
        "valid_post": "https://www.instagram.com/p/DEF456GHI/",
        "invalid_domain": "https://not-instagram.com/reel/123/",
        "invalid_format": "https://instagram.com/invalid/123/",
        "malformed": "not-a-url-at-all",
        "empty": "",
        "private_account": "https://www.instagram.com/reel/PRIVATE123/",
        "deleted_post": "https://www.instagram.com/reel/DELETED456/",
    }


# Skip conditions for external dependencies
def pytest_configure(config):
    """Configure pytest with custom markers and skip conditions."""
    # Add custom markers
    config.addinivalue_line("markers", "network: tests requiring network connectivity")
    config.addinivalue_line("markers", "model_download: tests requiring ML model downloads")
    config.addinivalue_line("markers", "gui_interactive: tests requiring user interaction")
    config.addinivalue_line("markers", "real_urls: tests using real Instagram URLs")
    config.addinivalue_line("markers", "privacy_safe: tests that don't log actual content")
    config.addinivalue_line("markers", "requires_model: tests requiring ML models")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add skip conditions."""
    # Skip network tests if no network
    skip_network = pytest.mark.skip(reason="Network tests disabled for CI/offline testing")

    # Skip model tests if models not available
    skip_model = pytest.mark.skip(reason="ML model tests disabled to avoid large downloads")

    # Skip real URL tests if not enabled
    skip_real_urls = pytest.mark.skip(reason="Real URL tests disabled - use --real-urls to enable")

    for item in items:
        # Skip network tests by default
        if "network" in item.keywords:
            if not config.getoption("--run-network-tests", default=False):
                item.add_marker(skip_network)

        # Skip model tests by default
        if "requires_model" in item.keywords:
            if not config.getoption("--run-model-tests", default=False):
                item.add_marker(skip_model)

        # Skip real URL tests by default
        if "real_urls" in item.keywords:
            if not config.getoption("--real-urls", default=False):
                item.add_marker(skip_real_urls)


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-network-tests", action="store_true", default=False, help="Run tests that require network connectivity"
    )
    parser.addoption(
        "--run-model-tests", action="store_true", default=False, help="Run tests that require ML model downloads"
    )
    parser.addoption("--run-gui-tests", action="store_true", default=False, help="Run GUI tests (requires display)")
    parser.addoption("--run-slow-tests", action="store_true", default=False, help="Run slow tests")
    parser.addoption(
        "--real-urls",
        action="store_true",
        default=False,
        help="Run tests with real Instagram URLs (requires network access)",
    )

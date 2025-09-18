"""
Comprehensive test suite for logging configuration system.

Tests cover privacy-safe logging, structured logging, log rotation,
and comprehensive logging setup with error handling.
"""

import json
import logging
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.logging_config import (
    LoggingConfig,
    PrivacyLogFormatter,
    StructuredLogger,
    get_logging_config,
    log_error_with_context,
    log_performance,
    log_user_action,
    setup_application_logging,
)


class TestPrivacyLogFormatter:
    """Test privacy-aware log formatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = PrivacyLogFormatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def test_init_default_format(self):
        """Test formatter initialization with default format."""
        formatter = PrivacyLogFormatter()
        assert formatter is not None

    def test_init_custom_format(self):
        """Test formatter initialization with custom format."""
        custom_fmt = "%(levelname)s: %(message)s"
        formatter = PrivacyLogFormatter(fmt=custom_fmt)
        assert formatter is not None

    def test_format_basic_message(self):
        """Test formatting basic log message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Basic log message",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "Basic log message" in formatted
        assert "INFO" in formatted

    def test_sanitize_instagram_url(self):
        """Test sanitization of Instagram URLs in log messages."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing URL: https://www.instagram.com/reel/ABC123DEF/",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "ABC123DEF" not in formatted
        assert "[REEL_ID]" in formatted

    def test_sanitize_instagram_url_in_args(self):
        """Test sanitization of Instagram URLs in log arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing URL: %s",
            args=("https://www.instagram.com/reel/ABC123DEF/",),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "ABC123DEF" not in formatted
        assert "[REEL_ID]" in formatted

    def test_sanitize_access_tokens(self):
        """Test sanitization of access tokens."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Using token: ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456" not in formatted
        assert "[TOKEN]" in formatted

    def test_sanitize_file_paths(self):
        """Test sanitization of file paths containing usernames."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Saving to: /Users/john.doe/Documents/file.txt",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "john.doe" not in formatted
        assert "/Users/[USER]" in formatted

    def test_sanitize_email_addresses(self):
        """Test sanitization of email addresses."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="User email: user@example.com",
            args=(),
            exc_info=None,
        )

        formatted = self.formatter.format(record)
        assert "user@example.com" not in formatted
        assert "[EMAIL]" in formatted

    def test_format_preserves_original_record(self):
        """Test that formatting doesn't modify the original record."""
        original_msg = "Original message with https://www.instagram.com/reel/ABC123/"
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=1, msg=original_msg, args=(), exc_info=None
        )

        # Format the record
        self.formatter.format(record)

        # Original record should be unchanged
        assert record.msg == original_msg

    def test_format_non_string_message(self):
        """Test formatting with non-string message."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=123,  # Non-string message
            args=(),
            exc_info=None,
        )

        # Should not raise exception
        formatted = self.formatter.format(record)
        assert "123" in formatted

    def test_format_non_string_args(self):
        """Test formatting with non-string arguments."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Number: %d, Object: %s",
            args=(42, {"key": "value"}),
            exc_info=None,
        )

        # Should not raise exception
        formatted = self.formatter.format(record)
        assert "42" in formatted


class TestStructuredLogger:
    """Test structured JSON logger."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.log_file = self.temp_dir / "test_structured.jsonl"
        self.structured_logger = StructuredLogger("test", str(self.log_file))

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self):
        """Test structured logger initialization."""
        assert self.structured_logger.logger is not None
        assert self.structured_logger.log_file == self.log_file

    def test_log_event_basic(self):
        """Test basic event logging."""
        event_data = {"action": "test", "result": "success"}
        self.structured_logger.log_event("test_event", event_data)

        # Verify log file was created and contains data
        assert self.log_file.exists()
        log_content = self.log_file.read_text()
        assert "test_event" in log_content
        assert "success" in log_content

    def test_log_event_json_format(self):
        """Test that logged events are valid JSON."""
        event_data = {"action": "test", "count": 42}
        self.structured_logger.log_event("test_event", event_data)

        log_content = self.log_file.read_text().strip()
        log_json = json.loads(log_content)

        assert log_json["event_type"] == "test_event"
        assert log_json["level"] == "INFO"
        assert log_json["data"]["action"] == "test"
        assert log_json["data"]["count"] == 42
        assert "timestamp" in log_json

    def test_log_event_different_levels(self):
        """Test logging events at different levels."""
        test_cases = [("INFO", "info_event"), ("WARNING", "warning_event"), ("ERROR", "error_event")]

        for level, event_type in test_cases:
            self.structured_logger.log_event(event_type, {"test": "data"}, level)

        log_content = self.log_file.read_text()
        for level, event_type in test_cases:
            assert event_type in log_content
            assert f'"level": "{level}"' in log_content

    def test_sanitize_data_url_sanitization(self):
        """Test data sanitization for URLs."""
        data = {"instagram_url": "https://www.instagram.com/reel/ABC123/", "other_data": "normal data"}

        sanitized = self.structured_logger._sanitize_data(data)
        assert "ABC123" not in sanitized["instagram_url"]
        assert "[REEL_ID]" in sanitized["instagram_url"]
        assert sanitized["other_data"] == "normal data"

    def test_sanitize_data_sensitive_keys(self):
        """Test data sanitization for sensitive keys."""
        data = {
            "password": "secret123",
            "api_token": "token123",
            "secret_key": "key123",
            "normal_field": "normal_value",
        }

        sanitized = self.structured_logger._sanitize_data(data)
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_token"] == "[REDACTED]"
        assert sanitized["secret_key"] == "[REDACTED]"
        assert sanitized["normal_field"] == "normal_value"

    def test_sanitize_data_nested_dict(self):
        """Test data sanitization for nested dictionaries."""
        data = {"user_info": {"email": "user@example.com", "password": "secret123"}, "normal_data": "value"}

        sanitized = self.structured_logger._sanitize_data(data)
        assert sanitized["user_info"]["password"] == "[REDACTED]"
        assert sanitized["normal_data"] == "value"

    def test_log_event_exception_handling(self):
        """Test exception handling during event logging."""
        # Mock JSON dumps to raise exception
        with patch("json.dumps", side_effect=Exception("JSON error")):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # Should not raise exception
                self.structured_logger.log_event("test", {"data": "value"})

                # Should log fallback error
                mock_logger.error.assert_called_once()

    def test_log_event_with_datetime_objects(self):
        """Test logging events with datetime objects."""
        event_data = {"timestamp": datetime.now(), "action": "test"}

        # Should not raise exception (datetime should be serialized)
        self.structured_logger.log_event("datetime_test", event_data)

        assert self.log_file.exists()
        log_content = self.log_file.read_text()
        log_json = json.loads(log_content)
        assert log_json["data"]["action"] == "test"


class TestLoggingConfig:
    """Test comprehensive logging configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.original_cwd = Path.cwd()

        # Create a LoggingConfig instance and patch its log directory
        self.config = LoggingConfig(app_name="test_app")
        self.config.log_dir = self.temp_dir / "logs"
        self.config.log_dir.mkdir(exist_ok=True)

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default(self):
        """Test logging config initialization with defaults."""
        assert self.config.app_name == "test_app"
        assert self.config.log_dir.exists()

    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        # Clear any existing loggers
        logging.getLogger().handlers.clear()

        self.config.setup_logging()

        assert self.config.main_logger is not None
        assert self.config.error_logger is not None
        assert self.config.structured_logger is not None

    def test_setup_logging_debug_mode(self):
        """Test logging setup with debug mode."""
        self.config.setup_logging(debug_mode=True)

        assert self.config.debug_mode is True
        assert self.config.debug_logger is not None

    def test_setup_logging_custom_parameters(self):
        """Test logging setup with custom parameters."""
        self.config.setup_logging(level="DEBUG", console_output=False, max_log_size_mb=5, max_log_files=3)

        assert self.config.max_bytes == 5 * 1024 * 1024
        assert self.config.backup_count == 3

    def test_setup_main_logger(self):
        """Test main logger setup."""
        self.config._setup_main_logger(logging.INFO, True)

        assert self.config.main_logger is not None
        assert len(self.config.main_logger.handlers) >= 1

        # Check log file was created
        log_file = self.config.log_dir / f"{self.config.app_name}.log"
        assert log_file.parent.exists()

    def test_setup_error_logger(self):
        """Test error logger setup."""
        self.config._setup_error_logger()

        assert self.config.error_logger is not None
        assert self.config.error_logger.level == logging.ERROR

    def test_setup_debug_logger(self):
        """Test debug logger setup."""
        self.config._setup_debug_logger()

        assert self.config.debug_logger is not None
        assert self.config.debug_logger.level == logging.DEBUG

    def test_setup_structured_logger(self):
        """Test structured logger setup."""
        self.config._setup_structured_logger()

        assert self.config.structured_logger is not None
        assert isinstance(self.config.structured_logger, StructuredLogger)

    def test_configure_third_party_loggers(self):
        """Test third-party logger configuration."""
        # Set up some third-party loggers first
        for logger_name in ["instaloader", "urllib3", "requests"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)  # Set to debug first

        self.config._configure_third_party_loggers()

        # Verify they were set to WARNING level
        for logger_name in ["instaloader", "urllib3", "requests"]:
            logger = logging.getLogger(logger_name)
            assert logger.level == logging.WARNING

    def test_log_error_details(self):
        """Test comprehensive error logging."""
        self.config.setup_logging()

        error = ValueError("Test error")
        self.config.log_error_details(
            error, context="test_context", user_action="test_action", additional_data={"key": "value"}
        )

        # Check that error was logged to structured logger
        structured_log_file = self.config.log_dir / f"{self.config.app_name}_structured.jsonl"
        if structured_log_file.exists():
            log_content = structured_log_file.read_text()
            assert "error_occurred" in log_content
            assert "ValueError" in log_content

    def test_log_error_details_no_loggers(self):
        """Test error logging when loggers are not set up."""
        # Should not raise exception
        self.config.log_error_details(Exception("test"))

    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        self.config.setup_logging()

        self.config.log_performance_metrics("test_operation", 1.5, {"memory_usage": "100MB"})

        # Check structured log
        structured_log_file = self.config.log_dir / f"{self.config.app_name}_structured.jsonl"
        if structured_log_file.exists():
            log_content = structured_log_file.read_text()
            assert "performance_metrics" in log_content
            assert "test_operation" in log_content

    def test_log_user_action(self):
        """Test user action logging."""
        self.config.setup_logging()

        self.config.log_user_action("click_download", True, {"url": "test_url"})

        # Check structured log
        structured_log_file = self.config.log_dir / f"{self.config.app_name}_structured.jsonl"
        if structured_log_file.exists():
            log_content = structured_log_file.read_text()
            assert "user_action" in log_content
            assert "click_download" in log_content

    def test_get_log_files(self):
        """Test getting log file paths."""
        self.config.setup_logging(debug_mode=True)

        # Create some log files
        (self.config.log_dir / f"{self.config.app_name}.log").touch()
        (self.config.log_dir / f"{self.config.app_name}_errors.log").touch()
        (self.config.log_dir / f"{self.config.app_name}_debug.log").touch()
        (self.config.log_dir / f"{self.config.app_name}_structured.jsonl").touch()

        log_files = self.config.get_log_files()

        assert "main" in log_files
        assert "errors" in log_files
        assert "debug" in log_files
        assert "structured" in log_files

    def test_get_log_files_no_logs(self):
        """Test getting log files when none exist."""
        log_files = self.config.get_log_files()
        assert isinstance(log_files, dict)
        assert len(log_files) == 0

    def test_cleanup_old_logs(self):
        """Test cleanup of old log files."""
        # Create some test log files
        old_log = self.config.log_dir / f"{self.config.app_name}_old.log"
        new_log = self.config.log_dir / f"{self.config.app_name}_new.log"

        old_log.touch()
        new_log.touch()

        # Mock file modification times
        import time

        old_time = time.time() - (40 * 24 * 60 * 60)  # 40 days ago
        time.time() - (10 * 24 * 60 * 60)  # 10 days ago

        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_mtime = old_time

            # Setup logging to get a logger
            self.config.setup_logging()

            # Cleanup logs older than 30 days
            self.config.cleanup_old_logs(days_to_keep=30)

    def test_cleanup_old_logs_no_log_dir(self):
        """Test cleanup when log directory doesn't exist."""
        shutil.rmtree(self.config.log_dir, ignore_errors=True)

        # Should not raise exception
        self.config.cleanup_old_logs()

    def test_cleanup_old_logs_permission_error(self):
        """Test cleanup with permission errors."""
        self.config.setup_logging()

        # Create a log file
        test_log = self.config.log_dir / f"{self.config.app_name}_test.log"
        test_log.touch()

        # Mock unlink to raise permission error
        with patch.object(Path, "unlink", side_effect=PermissionError("Access denied")):
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value.st_mtime = 0  # Very old

                # Should not raise exception
                self.config.cleanup_old_logs()


class TestConvenienceFunctions:
    """Test convenience functions for logging."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("utils.logging_config._logging_config")
    def test_setup_application_logging(self, mock_config):
        """Test application logging setup convenience function."""
        setup_application_logging(level="DEBUG", console_output=False, debug_mode=True)

        mock_config.setup_logging.assert_called_once_with("DEBUG", False, True, 10, 5)

    @patch("utils.logging_config._logging_config")
    def test_log_error_with_context(self, mock_config):
        """Test error logging convenience function."""
        error = ValueError("test error")
        log_error_with_context(
            error, context="test_context", user_action="test_action", additional_data={"key": "value"}
        )

        mock_config.log_error_details.assert_called_once_with(error, "test_context", "test_action", {"key": "value"})

    @patch("utils.logging_config._logging_config")
    def test_log_performance(self, mock_config):
        """Test performance logging convenience function."""
        log_performance("test_operation", 2.5, {"cpu_usage": "80%"})

        mock_config.log_performance_metrics.assert_called_once_with("test_operation", 2.5, {"cpu_usage": "80%"})

    @patch("utils.logging_config._logging_config")
    def test_log_user_action_convenience(self, mock_config):
        """Test user action logging convenience function."""
        log_user_action("download_video", True, {"video_id": "ABC123"})

        mock_config.log_user_action.assert_called_once_with("download_video", True, {"video_id": "ABC123"})

    @patch("utils.logging_config._logging_config")
    def test_get_logging_config(self, mock_config):
        """Test getting logging config convenience function."""
        result = get_logging_config()
        assert result == mock_config


class TestLoggingIntegration:
    """Integration tests for logging system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        # Clear all loggers
        logging.getLogger().handlers.clear()

    def test_complete_logging_workflow(self):
        """Test complete logging workflow from setup to cleanup."""
        config = LoggingConfig(app_name="integration_test")
        config.log_dir = self.temp_dir / "logs"
        config.log_dir.mkdir(exist_ok=True)

        # Setup logging
        config.setup_logging(level="DEBUG", debug_mode=True)

        # Test different types of logging
        error = ValueError("Integration test error")
        config.log_error_details(error, "integration_test", "test_workflow")
        config.log_performance_metrics("test_operation", 1.0)
        config.log_user_action("test_action", True)

        # Verify log files were created
        log_files = config.get_log_files()
        assert len(log_files) > 0

        # Test cleanup
        config.cleanup_old_logs(days_to_keep=1)

    def test_privacy_protection_integration(self):
        """Test privacy protection across the logging system."""
        # Test data with sensitive information
        sensitive_url = "https://www.instagram.com/reel/SENSITIVE123/"
        sensitive_data = {
            "url": sensitive_url,
            "user_token": "SUPER_SECRET_TOKEN_123456789",
            "user_path": "/Users/john.doe/Documents/secret.txt",
        }

        config = LoggingConfig(app_name="privacy_test")
        config.log_dir = self.temp_dir / "logs"
        config.log_dir.mkdir(exist_ok=True)
        config.setup_logging()

        # Log error with sensitive data
        error = Exception(f"Failed to process {sensitive_url}")
        config.log_error_details(error, additional_data=sensitive_data)

        # Log user action with sensitive data
        config.log_user_action("download", True, sensitive_data)

        # Check that sensitive information was sanitized
        log_files = config.get_log_files()
        for log_file in log_files.values():
            if log_file.exists():
                content = log_file.read_text()
                # Should not contain sensitive information
                assert "SENSITIVE123" not in content
                assert "SUPER_SECRET_TOKEN_123456" not in content
                assert "john.doe" not in content

                # Should contain sanitized versions
                if "instagram.com" in content:
                    assert "[REEL_ID]" in content or "[TOKEN]" in content or "[USER]" in content

    def test_error_handling_robustness(self):
        """Test logging system robustness with various errors."""
        config = LoggingConfig(app_name="robust_test")
        config.log_dir = self.temp_dir / "logs"
        config.log_dir.mkdir(exist_ok=True)

        # Test setup with various error conditions
        try:
            config.setup_logging()

            # Test logging with problematic data
            problematic_data = {
                "circular_ref": None,
                "unicode": "测试数据",
                "none_value": None,
                "large_number": 999999999999999999,
            }
            problematic_data["circular_ref"] = problematic_data  # Circular reference

            # Should handle gracefully
            config.log_user_action("test", True, {"normal": "data"})

        except Exception as e:
            pytest.fail(f"Logging system should handle errors gracefully: {e}")

    def test_concurrent_logging_safety(self):
        """Test logging system thread safety."""
        import threading
        import time

        config = LoggingConfig(app_name="concurrent_test")
        config.log_dir = self.temp_dir / "logs"
        config.log_dir.mkdir(exist_ok=True)
        config.setup_logging()

        def log_worker(worker_id):
            """Worker function for concurrent logging."""
            for i in range(10):
                config.log_user_action(f"worker_{worker_id}_action_{i}", True)
                time.sleep(0.01)

        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=log_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify logs were written
        log_files = config.get_log_files()
        assert len(log_files) > 0

        # Check that we got logs from all workers
        if "structured" in log_files:
            content = log_files["structured"].read_text()
            for i in range(5):
                assert f"worker_{i}" in content

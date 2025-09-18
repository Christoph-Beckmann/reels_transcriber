"""
Enhanced logging configuration with privacy-safe logging and rotating file handlers.

This module provides comprehensive logging setup that maintains detailed technical
information while protecting user privacy and managing log file sizes.
"""

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .error_handler import sanitize_url


class PrivacyLogFormatter(logging.Formatter):
    """
    Custom log formatter that automatically sanitizes sensitive information.
    """

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Initialize privacy-aware formatter.

        Args:
            fmt: Log format string
            datefmt: Date format string
        """
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record while sanitizing sensitive information.

        Args:
            record: Log record to format

        Returns:
            Formatted and sanitized log message
        """
        # Make a copy of the record to avoid modifying the original
        record_copy = logging.makeLogRecord(record.__dict__)

        # Sanitize the message
        if hasattr(record_copy, "msg") and isinstance(record_copy.msg, str):
            record_copy.msg = self._sanitize_message(record_copy.msg)

        # Sanitize arguments
        if hasattr(record_copy, "args") and record_copy.args:
            sanitized_args = []
            for arg in record_copy.args:
                if isinstance(arg, str):
                    sanitized_args.append(self._sanitize_message(arg))
                else:
                    sanitized_args.append(arg)
            record_copy.args = tuple(sanitized_args)

        return super().format(record_copy)

    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize log message to remove sensitive information.

        Args:
            message: Original log message

        Returns:
            Sanitized log message
        """
        import re

        # Sanitize Instagram URLs
        instagram_pattern = r"https?://(?:www\.)?instagram\.com/\S*"
        message = re.sub(instagram_pattern, lambda m: sanitize_url(m.group(0)), message)

        # Remove potential access tokens or session IDs
        token_pattern = r"\b[A-Za-z0-9]{20,}\b"
        message = re.sub(token_pattern, "[TOKEN]", message)

        # Remove file paths that might contain username
        home_pattern = r"/[Uu]sers/[^/\s]+(/[^/\s]*)*"
        message = re.sub(home_pattern, "/Users/[USER]\\1", message)

        # Remove potential email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        message = re.sub(email_pattern, "[EMAIL]", message)

        return message


class StructuredLogger:
    """
    Structured logger for JSON-formatted diagnostic logs.
    """

    def __init__(self, logger_name: str, log_file: str):
        """
        Initialize structured logger.

        Args:
            logger_name: Name of the logger
            log_file: Path to structured log file
        """
        self.logger = logging.getLogger(f"{logger_name}.structured")
        self.log_file = Path(log_file)

        # Create structured log handler
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, data: dict[str, Any], level: str = "INFO") -> None:
        """
        Log structured event data.

        Args:
            event_type: Type of event being logged
            data: Event data dictionary
            level: Log level
        """
        try:
            event_record = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "level": level,
                "data": self._sanitize_data(data),
            }

            log_message = json.dumps(event_record, default=str)

            if level.upper() == "ERROR":
                self.logger.error(log_message)
            elif level.upper() == "WARNING":
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)

        except Exception as e:
            # Fallback to regular logging if structured logging fails
            logging.getLogger("logging_config").error(f"Failed to log structured event: {e}")

    def _sanitize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Sanitize data dictionary for privacy-safe logging.

        Args:
            data: Original data dictionary

        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                if "url" in key.lower():
                    sanitized[key] = sanitize_url(value)
                elif any(sensitive in key.lower() for sensitive in ["password", "token", "key", "secret"]):
                    sanitized[key] = "[REDACTED]"
                elif "path" in key.lower() and "/Users/" in value:
                    # Sanitize file paths that contain usernames
                    sanitized[key] = sanitize_url(value)
                else:
                    sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            else:
                sanitized[key] = value

        return sanitized


class LoggingConfig:
    """
    Comprehensive logging configuration manager.
    """

    def __init__(self, app_name: str = "reels_transcriber"):
        """
        Initialize logging configuration.

        Args:
            app_name: Application name for log files
        """
        self.app_name = app_name
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Logger instances
        self.main_logger: Optional[logging.Logger] = None
        self.error_logger: Optional[logging.Logger] = None
        self.debug_logger: Optional[logging.Logger] = None
        self.structured_logger: Optional[StructuredLogger] = None

        # Configuration
        self.max_bytes = 10 * 1024 * 1024  # 10MB per log file
        self.backup_count = 5  # Keep 5 backup files
        self.debug_mode = False

    def setup_logging(
        self,
        level: str = "INFO",
        console_output: bool = True,
        debug_mode: bool = False,
        max_log_size_mb: int = 10,
        max_log_files: int = 5,
    ) -> None:
        """
        Set up comprehensive logging configuration.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
            debug_mode: Enable debug mode with verbose logging
            max_log_size_mb: Maximum size of each log file in MB
            max_log_files: Maximum number of log files to keep
        """
        self.debug_mode = debug_mode
        self.max_bytes = max_log_size_mb * 1024 * 1024
        self.backup_count = max_log_files

        # Clear any existing handlers
        logging.getLogger().handlers.clear()

        # Set root logger level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)

        # Set up main application logger
        self._setup_main_logger(numeric_level, console_output)

        # Set up error logger
        self._setup_error_logger()

        # Set up debug logger if in debug mode
        if debug_mode:
            self._setup_debug_logger()

        # Set up structured logger for diagnostics
        self._setup_structured_logger()

        # Configure third-party loggers
        self._configure_third_party_loggers()

        # Log initial setup message
        logging.getLogger(self.app_name).info(
            f"Logging configured - Level: {level}, Console: {console_output}, "
            f"Debug: {debug_mode}, Max file size: {max_log_size_mb}MB"
        )

    def _setup_main_logger(self, level: int, console_output: bool) -> None:
        """Set up main application logger."""
        self.main_logger = logging.getLogger(self.app_name)
        self.main_logger.setLevel(level)

        # Create privacy-aware formatter
        formatter = PrivacyLogFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # File handler with rotation
        log_file = self.log_dir / f"{self.app_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        self.main_logger.addHandler(file_handler)

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(level)
            self.main_logger.addHandler(console_handler)

    def _setup_error_logger(self) -> None:
        """Set up dedicated error logger."""
        self.error_logger = logging.getLogger(f"{self.app_name}.errors")
        self.error_logger.setLevel(logging.ERROR)

        # Error-specific formatter with more details
        error_formatter = PrivacyLogFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Error log file
        error_log_file = self.log_dir / f"{self.app_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
        )
        error_handler.setFormatter(error_formatter)
        self.error_logger.addHandler(error_handler)

    def _setup_debug_logger(self) -> None:
        """Set up debug logger for verbose logging."""
        self.debug_logger = logging.getLogger(f"{self.app_name}.debug")
        self.debug_logger.setLevel(logging.DEBUG)

        # Debug formatter with extensive details
        debug_formatter = PrivacyLogFormatter(
            fmt="%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d:%(funcName)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Debug log file
        debug_log_file = self.log_dir / f"{self.app_name}_debug.log"
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log_file, maxBytes=self.max_bytes, backupCount=self.backup_count, encoding="utf-8"
        )
        debug_handler.setFormatter(debug_formatter)
        self.debug_logger.addHandler(debug_handler)

    def _setup_structured_logger(self) -> None:
        """Set up structured JSON logger for diagnostics."""
        structured_log_file = self.log_dir / f"{self.app_name}_structured.jsonl"
        self.structured_logger = StructuredLogger(self.app_name, structured_log_file)

    def _configure_third_party_loggers(self) -> None:
        """Configure third-party library loggers."""
        # Reduce verbosity of third-party loggers
        third_party_loggers = ["instaloader", "urllib3", "requests", "PIL", "whisper"]

        for logger_name in third_party_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.WARNING)  # Only show warnings and errors

    def log_error_details(
        self,
        error: Exception,
        context: Optional[str] = None,
        user_action: Optional[str] = None,
        additional_data: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Log comprehensive error details.

        Args:
            error: Exception that occurred
            context: Context where error occurred
            user_action: What the user was trying to do
            additional_data: Additional diagnostic data
        """
        if not self.error_logger:
            return

        # Log to main error log
        error_msg = f"Error in context '{context}': {str(error)}"
        if user_action:
            error_msg = f"User action '{user_action}' failed - {error_msg}"

        self.error_logger.error(error_msg, exc_info=True)

        # Log to structured log for analysis
        if self.structured_logger:
            from .error_handler import sanitize_url

            error_message = str(error)
            # Sanitize error message
            error_message = sanitize_url(error_message)

            error_data = {
                "error_type": type(error).__name__,
                "error_message": error_message,
                "context": context,
                "user_action": user_action,
                "additional_data": additional_data or {},
            }
            self.structured_logger.log_event("error_occurred", error_data, "ERROR")

    def log_performance_metrics(
        self, operation: str, duration_seconds: float, additional_metrics: Optional[dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics.

        Args:
            operation: Name of the operation
            duration_seconds: Duration in seconds
            additional_metrics: Additional performance data
        """
        if self.structured_logger:
            metrics_data = {
                "operation": operation,
                "duration_seconds": duration_seconds,
                "metrics": additional_metrics or {},
            }
            self.structured_logger.log_event("performance_metrics", metrics_data)

        if self.main_logger:
            self.main_logger.info(f"Performance: {operation} completed in {duration_seconds:.2f}s")

    def log_user_action(self, action: str, success: bool, additional_data: Optional[dict[str, Any]] = None) -> None:
        """
        Log user actions for usage analytics.

        Args:
            action: Action performed by user
            success: Whether action was successful
            additional_data: Additional action data
        """
        if self.structured_logger:
            action_data = {"action": action, "success": success, "data": additional_data or {}}
            self.structured_logger.log_event("user_action", action_data)

    def get_log_files(self) -> dict[str, Path]:
        """
        Get paths to all log files.

        Returns:
            Dictionary mapping log type to file path
        """
        log_files = {}

        if self.log_dir.exists():
            # Main log
            main_log = self.log_dir / f"{self.app_name}.log"
            if main_log.exists():
                log_files["main"] = main_log

            # Error log
            error_log = self.log_dir / f"{self.app_name}_errors.log"
            if error_log.exists():
                log_files["errors"] = error_log

            # Debug log
            if self.debug_mode:
                debug_log = self.log_dir / f"{self.app_name}_debug.log"
                if debug_log.exists():
                    log_files["debug"] = debug_log

            # Structured log
            structured_log = self.log_dir / f"{self.app_name}_structured.jsonl"
            if structured_log.exists():
                log_files["structured"] = structured_log

        return log_files

    def cleanup_old_logs(self, days_to_keep: int = 30) -> None:
        """
        Clean up log files older than specified days.

        Args:
            days_to_keep: Number of days to keep log files
        """
        if not self.log_dir.exists():
            return

        import time

        cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)

        for log_file in self.log_dir.glob(f"{self.app_name}*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    if self.main_logger:
                        self.main_logger.info(f"Cleaned up old log file: {log_file.name}")
            except Exception as e:
                if self.main_logger:
                    self.main_logger.warning(f"Failed to clean up log file {log_file}: {e}")


# Global logging configuration instance
_logging_config = LoggingConfig()


def setup_application_logging(
    level: str = "INFO",
    console_output: bool = True,
    debug_mode: bool = False,
    max_log_size_mb: int = 10,
    max_log_files: int = 5,
) -> None:
    """
    Convenience function to set up application logging.

    Args:
        level: Logging level
        console_output: Whether to output to console
        debug_mode: Enable debug mode
        max_log_size_mb: Maximum log file size
        max_log_files: Maximum number of log files
    """
    _logging_config.setup_logging(level, console_output, debug_mode, max_log_size_mb, max_log_files)


def log_error_with_context(
    error: Exception,
    context: Optional[str] = None,
    user_action: Optional[str] = None,
    additional_data: Optional[dict[str, Any]] = None,
) -> None:
    """
    Convenience function to log errors with context.

    Args:
        error: Exception that occurred
        context: Context information
        user_action: User action that triggered the error
        additional_data: Additional diagnostic data
    """
    _logging_config.log_error_details(error, context, user_action, additional_data)


def log_performance(
    operation: str, duration_seconds: float, additional_metrics: Optional[dict[str, Any]] = None
) -> None:
    """
    Convenience function to log performance metrics.

    Args:
        operation: Operation name
        duration_seconds: Duration in seconds
        additional_metrics: Additional metrics
    """
    _logging_config.log_performance_metrics(operation, duration_seconds, additional_metrics)


def log_user_action(action: str, success: bool, additional_data: Optional[dict[str, Any]] = None) -> None:
    """
    Convenience function to log user actions.

    Args:
        action: Action name
        success: Whether action succeeded
        additional_data: Additional data
    """
    _logging_config.log_user_action(action, success, additional_data)


def get_logging_config() -> LoggingConfig:
    """Get the global logging configuration instance."""
    return _logging_config

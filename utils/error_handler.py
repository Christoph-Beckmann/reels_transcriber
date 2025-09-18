"""
Enhanced error handling system with user-friendly error classification and recovery suggestions.

This module provides comprehensive error handling that transforms technical errors into
actionable user guidance while maintaining detailed logging for troubleshooting.
"""

import logging
import re
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for prioritization and user feedback."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors for specific user guidance."""

    NETWORK = "network"
    PERMISSION = "permission"
    DISK_SPACE = "disk_space"
    INVALID_INPUT = "invalid_input"
    CONTENT_ACCESS = "content_access"
    AUDIO_PROCESSING = "audio_processing"
    SYSTEM_RESOURCE = "system_resource"
    DEPENDENCY = "dependency"
    UNKNOWN = "unknown"


@dataclass
class ErrorDetails:
    """Comprehensive error information for user feedback and logging."""

    category: ErrorCategory
    severity: ErrorSeverity
    user_message: str
    technical_details: str
    recovery_suggestions: list[str]
    retry_recommended: bool = False
    contact_support: bool = False
    error_code: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class ErrorClassifier:
    """
    Classifies technical errors into user-friendly categories with recovery suggestions.
    """

    def __init__(self):
        """Initialize error classifier with pattern matching rules."""
        self.network_patterns = [
            r"connection.*timeout",
            r"network.*unreachable",
            r"failed to connect",
            r"connection.*refused",
            r"dns.*resolution",
            r"ssl.*error",
            r"certificate.*verify",
            r"http.*error.*5\d\d",
            r"rate.*limit",
            r"too many requests",
        ]

        self.permission_patterns = [
            r"permission.*denied",
            r"access.*denied",
            r"forbidden",
            r"unauthorized",
            r"authentication.*failed",
            r"login.*required",
            r"private.*account",
            r"not.*allowed",
        ]

        self.disk_space_patterns = [
            r"no space left",
            r"disk.*full",
            r"insufficient.*space",
            r"cannot create.*file",
            r"write.*failed.*space",
        ]

        self.content_patterns = [
            r"not found",
            r"404",
            r"post.*not.*exist",
            r"deleted",
            r"unavailable",
            r"no.*video.*found",
            r"no.*audio.*detected",
            r"empty.*file",
            r"corrupted.*file",
        ]

        self.audio_patterns = [
            r"audio.*extraction.*failed",
            r"ffmpeg.*error",
            r"codec.*not.*supported",
            r"invalid.*audio",
            r"no.*audio.*stream",
            r"audio.*too.*short",
            r"whisper.*error",
            r"transcription.*failed",
        ]

        self.dependency_patterns = [
            r"module.*not.*found",
            r"import.*error",
            r"command.*not.*found",
            r"ffmpeg.*not.*found",
            r"missing.*dependency",
        ]

    def classify_error(self, error: Exception, context: Optional[str] = None) -> ErrorDetails:
        """
        Classify an error and generate user-friendly guidance.

        Args:
            error: The exception to classify
            context: Optional context about what was being done when error occurred

        Returns:
            ErrorDetails: Comprehensive error information
        """
        error_text = str(error).lower()

        # Check for specific error patterns
        if self._matches_patterns(error_text, self.network_patterns):
            return self._create_network_error(error, error_text, context)
        elif self._matches_patterns(error_text, self.permission_patterns):
            return self._create_permission_error(error, error_text, context)
        elif self._matches_patterns(error_text, self.disk_space_patterns):
            return self._create_disk_space_error(error, error_text, context)
        elif self._matches_patterns(error_text, self.content_patterns):
            return self._create_content_error(error, error_text, context)
        elif self._matches_patterns(error_text, self.audio_patterns):
            return self._create_audio_error(error, error_text, context)
        elif self._matches_patterns(error_text, self.dependency_patterns):
            return self._create_dependency_error(error, error_text, context)
        else:
            return self._create_unknown_error(error, error_text, context)

    def _matches_patterns(self, text: str, patterns: list[str]) -> bool:
        """Check if text matches any of the given patterns."""
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

    def _create_network_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for network-related issues."""
        if "rate limit" in error_text or "too many requests" in error_text:
            return ErrorDetails(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.WARNING,
                user_message="Instagram is temporarily limiting requests. Please wait before trying again.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Wait 15-30 minutes before trying again",
                    "Try processing a different Reel",
                    "Check if Instagram is experiencing service issues",
                ],
                retry_recommended=True,
                error_code="NET_RATE_LIMIT",
            )
        elif "timeout" in error_text:
            return ErrorDetails(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.ERROR,
                user_message="The connection to Instagram timed out. This may be due to slow internet or Instagram server issues.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Check your internet connection",
                    "Try again in a few minutes",
                    "Ensure no firewall is blocking the connection",
                    "Try connecting from a different network",
                ],
                retry_recommended=True,
                error_code="NET_TIMEOUT",
            )
        else:
            return ErrorDetails(
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.ERROR,
                user_message="Cannot connect to Instagram. Please check your internet connection and try again.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Verify your internet connection is working",
                    "Try restarting your network connection",
                    "Check if Instagram is accessible in your browser",
                    "Try again in a few minutes",
                ],
                retry_recommended=True,
                error_code="NET_CONNECTION",
            )

    def _create_permission_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for permission/access issues."""
        if "private" in error_text or "login" in error_text:
            return ErrorDetails(
                category=ErrorCategory.CONTENT_ACCESS,
                severity=ErrorSeverity.WARNING,
                user_message="This Reel is from a private account and cannot be accessed.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Make sure the Reel is from a public account",
                    "Ask the account owner to make the Reel public",
                    "Try a different Reel from a public account",
                ],
                retry_recommended=False,
                error_code="ACCESS_PRIVATE",
            )
        else:
            return ErrorDetails(
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.ERROR,
                user_message="Access denied. You may not have permission to access this content or file location.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Ensure the Reel URL is correct and publicly accessible",
                    "Try running the app as administrator if on Windows",
                    "Check file system permissions for the app directory",
                ],
                retry_recommended=False,
                error_code="PERM_DENIED",
            )

    def _create_disk_space_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for disk space issues."""
        return ErrorDetails(
            category=ErrorCategory.DISK_SPACE,
            severity=ErrorSeverity.ERROR,
            user_message="Not enough disk space to download and process the video.",
            technical_details=str(error),
            recovery_suggestions=[
                "Free up disk space by deleting unnecessary files",
                "Close other applications that might be using disk space",
                "Choose a different temporary directory with more space",
                "Delete old transcription files if any",
            ],
            retry_recommended=True,
            error_code="DISK_FULL",
            metadata={"available_space_mb": self._get_available_disk_space()},
        )

    def _create_content_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for content-related issues."""
        if "not found" in error_text or "404" in error_text:
            return ErrorDetails(
                category=ErrorCategory.CONTENT_ACCESS,
                severity=ErrorSeverity.WARNING,
                user_message="This Reel was not found. It may have been deleted or the URL is incorrect.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Double-check the URL is complete and correct",
                    "Ensure the Reel hasn't been deleted",
                    "Try copying the URL again from Instagram",
                    "Make sure you're using a Reel URL, not a regular post",
                ],
                retry_recommended=False,
                error_code="CONTENT_NOT_FOUND",
            )
        elif "no.*audio" in error_text:
            return ErrorDetails(
                category=ErrorCategory.AUDIO_PROCESSING,
                severity=ErrorSeverity.WARNING,
                user_message="No audio was found in this video. The Reel might be silent or have audio issues.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try a different Reel that has clear speech",
                    "Check if the Reel has audio when played on Instagram",
                    "Some Reels may only have background music without speech",
                ],
                retry_recommended=False,
                error_code="NO_AUDIO",
            )
        else:
            return ErrorDetails(
                category=ErrorCategory.CONTENT_ACCESS,
                severity=ErrorSeverity.ERROR,
                user_message="The video content could not be processed. It may be corrupted or in an unsupported format.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try a different Reel",
                    "Check if the Reel plays normally on Instagram",
                    "The Reel format might not be supported",
                ],
                retry_recommended=False,
                error_code="CONTENT_INVALID",
            )

    def _create_audio_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for audio processing issues."""
        if "ffmpeg" in error_text:
            return ErrorDetails(
                category=ErrorCategory.DEPENDENCY,
                severity=ErrorSeverity.CRITICAL,
                user_message="Audio processing failed. FFmpeg may not be installed or accessible.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Install FFmpeg if not already installed",
                    "Ensure FFmpeg is in your system PATH",
                    "Restart the application after installing FFmpeg",
                    "Check the application requirements documentation",
                ],
                retry_recommended=True,
                contact_support=True,
                error_code="AUDIO_FFMPEG",
            )
        elif "whisper" in error_text:
            return ErrorDetails(
                category=ErrorCategory.AUDIO_PROCESSING,
                severity=ErrorSeverity.ERROR,
                user_message="Speech recognition failed. The audio quality might be too poor or contain no speech.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try a Reel with clearer speech",
                    "Ensure the Reel contains actual speech, not just music",
                    "Try a Reel in a supported language (English, German)",
                    "Check if the Reel has background noise interfering with speech",
                ],
                retry_recommended=False,
                error_code="TRANSCRIBE_FAILED",
            )
        else:
            return ErrorDetails(
                category=ErrorCategory.AUDIO_PROCESSING,
                severity=ErrorSeverity.ERROR,
                user_message="Audio processing failed. The audio format may not be supported or the file is corrupted.",
                technical_details=str(error),
                recovery_suggestions=[
                    "Try a different Reel",
                    "Ensure the Reel has good audio quality",
                    "Check system audio processing capabilities",
                ],
                retry_recommended=True,
                error_code="AUDIO_PROCESS",
            )

    def _create_dependency_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for missing dependencies."""
        return ErrorDetails(
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.CRITICAL,
            user_message="A required system component is missing. The application cannot continue without it.",
            technical_details=str(error),
            recovery_suggestions=[
                "Install all required dependencies from requirements.txt",
                "Ensure Python environment is properly set up",
                "Check the installation documentation",
                "Reinstall the application if necessary",
            ],
            retry_recommended=False,
            contact_support=True,
            error_code="DEPENDENCY_MISSING",
        )

    def _create_unknown_error(self, error: Exception, error_text: str, context: Optional[str]) -> ErrorDetails:
        """Create error details for unknown/unclassified errors."""
        return ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            user_message="An unexpected error occurred during processing. This may be a temporary issue.",
            technical_details=str(error),
            recovery_suggestions=[
                "Try the operation again",
                "Restart the application",
                "Try with a different Reel",
                "Check the application logs for more details",
            ],
            retry_recommended=True,
            contact_support=True,
            error_code="UNKNOWN",
            metadata={"error_type": type(error).__name__, "context": context},
        )

    def _get_available_disk_space(self) -> Optional[float]:
        """Get available disk space in MB."""
        try:
            import tempfile

            temp_dir = tempfile.gettempdir()
            free_bytes = shutil.disk_usage(temp_dir).free
            return free_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return None


class UserFeedbackManager:
    """
    Manages user feedback for different error scenarios with contextual help.
    """

    def __init__(self):
        """Initialize feedback manager."""
        self.classifier = ErrorClassifier()

    def process_error(
        self, error: Exception, context: Optional[str] = None, user_action: Optional[str] = None
    ) -> ErrorDetails:
        """
        Process an error and generate comprehensive user feedback.

        Args:
            error: The exception that occurred
            context: Context about what was happening when error occurred
            user_action: What the user was trying to do

        Returns:
            ErrorDetails: Complete error information with user guidance
        """
        # Log the full technical error first
        logger.error(f"Error in context '{context}': {str(error)}", exc_info=True)

        # Classify and generate user-friendly information
        error_details = self.classifier.classify_error(error, context)

        # Add user action context if provided
        if user_action:
            if not error_details.metadata:
                error_details.metadata = {}
            error_details.metadata["user_action"] = user_action

        # Log the user-friendly classification
        logger.info(
            f"Error classified as {error_details.category.value} "
            f"(severity: {error_details.severity.value}, "
            f"code: {error_details.error_code})"
        )

        return error_details

    def get_contextual_help(self, error_category: ErrorCategory) -> dict[str, Any]:
        """
        Get contextual help information for specific error categories.

        Args:
            error_category: The category of error

        Returns:
            Dict containing help information
        """
        help_content = {
            ErrorCategory.NETWORK: {
                "title": "Network Connection Issues",
                "description": "Problems connecting to Instagram or downloading content",
                "common_solutions": [
                    "Check your internet connection",
                    "Try again in a few minutes",
                    "Disable VPN if using one",
                    "Check firewall settings",
                ],
                "prevention_tips": [
                    "Ensure stable internet connection before starting",
                    "Avoid peak usage times",
                    "Use wired connection if possible",
                ],
            },
            ErrorCategory.CONTENT_ACCESS: {
                "title": "Content Access Problems",
                "description": "Issues accessing or finding the requested Reel",
                "common_solutions": [
                    "Verify the Reel URL is correct and complete",
                    "Ensure the Reel is public and not deleted",
                    "Try copying the URL again from Instagram",
                ],
                "prevention_tips": [
                    "Only use Reels from public accounts",
                    "Copy URLs directly from the Instagram app/website",
                    "Verify Reels are still available before processing",
                ],
            },
            ErrorCategory.AUDIO_PROCESSING: {
                "title": "Audio Processing Issues",
                "description": "Problems extracting or transcribing audio from videos",
                "common_solutions": [
                    "Choose Reels with clear speech",
                    "Avoid Reels with only music",
                    "Try Reels in supported languages",
                ],
                "prevention_tips": [
                    "Select Reels with clear, audible speech",
                    "Avoid heavily music-focused content",
                    "Check that Reels contain actual speech before processing",
                ],
            },
            ErrorCategory.DISK_SPACE: {
                "title": "Storage Space Issues",
                "description": "Insufficient disk space for processing",
                "common_solutions": [
                    "Free up disk space by deleting old files",
                    "Change temporary directory to drive with more space",
                    "Close other applications using storage",
                ],
                "prevention_tips": [
                    "Maintain at least 1GB free space",
                    "Regularly clean temporary files",
                    "Monitor disk usage before processing large files",
                ],
            },
            ErrorCategory.DEPENDENCY: {
                "title": "System Requirements",
                "description": "Missing or incompatible system components",
                "common_solutions": [
                    "Install all required dependencies",
                    "Check Python environment setup",
                    "Reinstall the application if necessary",
                ],
                "prevention_tips": [
                    "Follow installation instructions carefully",
                    "Keep dependencies up to date",
                    "Use recommended Python version",
                ],
            },
        }

        return help_content.get(
            error_category,
            {
                "title": "General Help",
                "description": "General troubleshooting information",
                "common_solutions": [
                    "Try the operation again",
                    "Restart the application",
                    "Check application logs for details",
                ],
                "prevention_tips": [
                    "Keep the application updated",
                    "Report persistent issues",
                    "Follow recommended usage guidelines",
                ],
            },
        )

    def format_error_for_display(self, error_details: ErrorDetails) -> str:
        """
        Format error details for user display.

        Args:
            error_details: Error information to format

        Returns:
            Formatted error message for display
        """
        message_parts = [error_details.user_message]

        if error_details.recovery_suggestions:
            message_parts.append("\nWhat you can try:")
            for i, suggestion in enumerate(error_details.recovery_suggestions, 1):
                message_parts.append(f"  {i}. {suggestion}")

        if error_details.retry_recommended:
            message_parts.append("\n✓ You can try this operation again")

        if error_details.contact_support:
            message_parts.append("\n⚠ If this problem persists, please report it")

        return "\n".join(message_parts)

    def should_show_technical_details(self, error_details: ErrorDetails) -> bool:
        """
        Determine if technical details should be shown to user.

        Args:
            error_details: Error information

        Returns:
            True if technical details should be shown
        """
        # Show technical details for critical errors or when user might need them
        return (
            error_details.severity == ErrorSeverity.CRITICAL
            or error_details.contact_support
            or error_details.category == ErrorCategory.DEPENDENCY
        )


class DiagnosticCollector:
    """
    Collects diagnostic information for troubleshooting while respecting user privacy.
    """

    def __init__(self):
        """Initialize diagnostic collector."""
        pass

    def collect_system_info(self) -> dict[str, Any]:
        """
        Collect privacy-safe system information for diagnostics.

        Returns:
            Dictionary containing system diagnostic information
        """
        try:
            diagnostics = {
                "platform": self._get_platform_info(),
                "python_version": self._get_python_version(),
                "memory": self._get_memory_info(),
                "disk_space": self._get_disk_space_info(),
                "dependencies": self._check_dependencies(),
                "app_version": "1.0.0",  # Would be loaded from app metadata
                "timestamp": self._get_timestamp(),
            }
            return diagnostics
        except Exception as e:
            logger.warning(f"Failed to collect system diagnostics: {e}")
            return {"error": "diagnostic_collection_failed"}

    def _get_platform_info(self) -> dict[str, str]:
        """Get platform information."""
        import platform

        return {"system": platform.system(), "release": platform.release(), "architecture": platform.architecture()[0]}

    def _get_python_version(self) -> str:
        """Get Python version."""
        import sys

        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    def _get_memory_info(self) -> dict[str, float]:
        """Get memory usage information."""
        try:
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent,
            }
        except Exception:
            return {"error": "memory_info_unavailable"}

    def _get_disk_space_info(self) -> dict[str, float]:
        """Get disk space information."""
        try:
            import tempfile

            disk_usage = shutil.disk_usage(tempfile.gettempdir())
            return {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "percent_free": round((disk_usage.free / disk_usage.total) * 100, 1),
            }
        except Exception:
            return {"error": "disk_info_unavailable"}

    def _check_dependencies(self) -> dict[str, str]:
        """Check status of critical dependencies."""
        dependencies = {}

        # Check FFmpeg
        try:
            import subprocess

            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split("\n")[0]
                dependencies["ffmpeg"] = version_line.split()[2] if len(version_line.split()) > 2 else "installed"
            else:
                dependencies["ffmpeg"] = "not_found"
        except Exception:
            dependencies["ffmpeg"] = "not_available"

        # Check Python packages
        packages_to_check = [
            ("instaloader", "instaloader"),
            ("faster-whisper", "faster_whisper"),  # Package name vs import name
            ("FreeSimpleGUI", "FreeSimpleGUI"),
            ("psutil", "psutil"),
            ("yt-dlp", "yt_dlp"),
            ("moviepy", "moviepy"),
        ]
        for display_name, import_name in packages_to_check:
            try:
                import importlib

                module = importlib.import_module(import_name)
                if hasattr(module, "__version__"):
                    dependencies[display_name] = module.__version__
                elif hasattr(module, "VERSION"):
                    dependencies[display_name] = module.VERSION
                else:
                    dependencies[display_name] = "installed"
            except ImportError:
                dependencies[display_name] = "not_installed"

        return dependencies

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def sanitize_url_for_logging(self, url: str) -> str:
        """
        Sanitize Instagram URL for privacy-safe logging.

        Args:
            url: Original Instagram URL

        Returns:
            Sanitized URL safe for logging
        """
        try:
            import re

            # Replace shortcode with placeholder but keep structure
            if "instagram.com" in url and "reel" in url:
                # Extract and replace shortcode
                pattern = r"(/reel/)([^/?]+)"
                sanitized = re.sub(pattern, r"\1[REEL_ID]", url)
                return sanitized
            else:
                return "[NON_INSTAGRAM_URL]"
        except Exception:
            return "[URL_SANITIZATION_FAILED]"


# Global instances for convenience
_error_classifier = ErrorClassifier()
_feedback_manager = UserFeedbackManager()
_diagnostic_collector = DiagnosticCollector()


def classify_error(error: Exception, context: Optional[str] = None) -> ErrorDetails:
    """
    Convenience function to classify an error.

    Args:
        error: Exception to classify
        context: Optional context information

    Returns:
        ErrorDetails with user-friendly information
    """
    return _error_classifier.classify_error(error, context)


def process_error_for_user(
    error: Exception, context: Optional[str] = None, user_action: Optional[str] = None
) -> ErrorDetails:
    """
    Convenience function to process error for user feedback.

    Args:
        error: Exception that occurred
        context: Context of the error
        user_action: What user was trying to do

    Returns:
        ErrorDetails with comprehensive user guidance
    """
    return _feedback_manager.process_error(error, context, user_action)


def get_system_diagnostics() -> dict[str, Any]:
    """
    Convenience function to collect system diagnostics.

    Returns:
        Dictionary with system diagnostic information
    """
    return _diagnostic_collector.collect_system_info()


def sanitize_url(url: str) -> str:
    """
    Convenience function to sanitize URL for logging.

    Args:
        url: URL to sanitize

    Returns:
        Privacy-safe URL for logging
    """
    return _diagnostic_collector.sanitize_url_for_logging(url)

"""
Comprehensive test suite for error handling system targeting 70%+ coverage.

Tests cover error classification, user feedback generation, diagnostic collection,
and all production-ready error scenarios.
"""

from unittest.mock import MagicMock, patch

from utils.error_handler import (
    DiagnosticCollector,
    ErrorCategory,
    ErrorClassifier,
    ErrorDetails,
    ErrorSeverity,
    UserFeedbackManager,
    classify_error,
    get_system_diagnostics,
    process_error_for_user,
    sanitize_url,
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert ErrorSeverity.INFO.value == "info"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.CRITICAL.value == "critical"


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_category_values(self):
        """Test all error category values."""
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.PERMISSION.value == "permission"
        assert ErrorCategory.DISK_SPACE.value == "disk_space"
        assert ErrorCategory.INVALID_INPUT.value == "invalid_input"
        assert ErrorCategory.CONTENT_ACCESS.value == "content_access"
        assert ErrorCategory.AUDIO_PROCESSING.value == "audio_processing"
        assert ErrorCategory.SYSTEM_RESOURCE.value == "system_resource"
        assert ErrorCategory.DEPENDENCY.value == "dependency"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestErrorDetails:
    """Test ErrorDetails dataclass."""

    def test_error_details_creation(self):
        """Test creating ErrorDetails with all fields."""
        details = ErrorDetails(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            user_message="Network error occurred",
            technical_details="Connection timeout",
            recovery_suggestions=["Check internet", "Retry"],
            retry_recommended=True,
            contact_support=False,
            error_code="NET_001",
            metadata={"url": "example.com"},
        )

        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.ERROR
        assert details.user_message == "Network error occurred"
        assert details.technical_details == "Connection timeout"
        assert details.recovery_suggestions == ["Check internet", "Retry"]
        assert details.retry_recommended is True
        assert details.contact_support is False
        assert details.error_code == "NET_001"
        assert details.metadata == {"url": "example.com"}

    def test_error_details_defaults(self):
        """Test ErrorDetails with default values."""
        details = ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.INFO,
            user_message="Test message",
            technical_details="Test details",
            recovery_suggestions=[],
        )

        assert details.retry_recommended is False
        assert details.contact_support is False
        assert details.error_code is None
        assert details.metadata is None


class TestErrorClassifier:
    """Test ErrorClassifier functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.classifier = ErrorClassifier()

    def test_init_patterns_loaded(self):
        """Test that classifier initializes with all pattern sets."""
        assert len(self.classifier.network_patterns) > 0
        assert len(self.classifier.permission_patterns) > 0
        assert len(self.classifier.disk_space_patterns) > 0
        assert len(self.classifier.content_patterns) > 0
        assert len(self.classifier.audio_patterns) > 0
        assert len(self.classifier.dependency_patterns) > 0

    def test_matches_patterns_true(self):
        """Test pattern matching with matching text."""
        patterns = [r"connection.*timeout", r"network.*error"]
        text = "connection timeout occurred"
        assert self.classifier._matches_patterns(text, patterns) is True

    def test_matches_patterns_false(self):
        """Test pattern matching with non-matching text."""
        patterns = [r"connection.*timeout", r"network.*error"]
        text = "disk space full"
        assert self.classifier._matches_patterns(text, patterns) is False

    def test_matches_patterns_case_insensitive(self):
        """Test pattern matching is case insensitive."""
        patterns = [r"CONNECTION.*TIMEOUT"]
        text = "connection timeout occurred"
        assert self.classifier._matches_patterns(text, patterns) is True

    def test_classify_network_rate_limit_error(self):
        """Test classification of rate limit errors."""
        error = Exception("Rate limit exceeded, too many requests")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.WARNING
        assert "rate limit" in details.user_message.lower()
        assert details.error_code == "NET_RATE_LIMIT"
        assert details.retry_recommended is True

    def test_classify_network_timeout_error(self):
        """Test classification of timeout errors."""
        error = Exception("Connection timeout after 30 seconds")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.ERROR
        assert "timeout" in details.user_message.lower()
        assert details.error_code == "NET_TIMEOUT"
        assert details.retry_recommended is True

    def test_classify_network_generic_error(self):
        """Test classification of generic network errors."""
        error = Exception("Failed to connect to server")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "NET_CONNECTION"
        assert details.retry_recommended is True

    def test_classify_permission_private_account_error(self):
        """Test classification of private account errors."""
        error = Exception("Private account access not allowed")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.CONTENT_ACCESS
        assert details.severity == ErrorSeverity.WARNING
        assert "private account" in details.user_message.lower()
        assert details.error_code == "ACCESS_PRIVATE"
        assert details.retry_recommended is False

    def test_classify_permission_generic_error(self):
        """Test classification of generic permission errors."""
        error = Exception("Permission denied to access resource")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.PERMISSION
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "PERM_DENIED"
        assert details.retry_recommended is False

    def test_classify_disk_space_error(self):
        """Test classification of disk space errors."""
        error = Exception("No space left on device")

        with patch.object(self.classifier, "_get_available_disk_space", return_value=50.5):
            details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.DISK_SPACE
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "DISK_FULL"
        assert details.retry_recommended is True
        assert details.metadata["available_space_mb"] == 50.5

    def test_classify_content_not_found_error(self):
        """Test classification of content not found errors."""
        error = Exception("Post not found - 404 error")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.CONTENT_ACCESS
        assert details.severity == ErrorSeverity.WARNING
        assert details.error_code == "CONTENT_NOT_FOUND"
        assert details.retry_recommended is False

    def test_classify_content_no_audio_error(self):
        """Test classification of no audio errors."""
        error = Exception("No audio stream detected in video")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.AUDIO_PROCESSING
        assert details.severity == ErrorSeverity.WARNING
        assert details.error_code == "NO_AUDIO"
        assert details.retry_recommended is False

    def test_classify_content_generic_error(self):
        """Test classification of generic content errors."""
        error = Exception("Corrupted file cannot be processed")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.CONTENT_ACCESS
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "CONTENT_INVALID"
        assert details.retry_recommended is False

    def test_classify_audio_ffmpeg_error(self):
        """Test classification of FFmpeg errors."""
        error = Exception("FFmpeg processing failed")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.DEPENDENCY
        assert details.severity == ErrorSeverity.CRITICAL
        assert details.error_code == "AUDIO_FFMPEG"
        assert details.retry_recommended is True
        assert details.contact_support is True

    def test_classify_audio_whisper_error(self):
        """Test classification of Whisper errors."""
        error = Exception("Whisper transcription error occurred")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.AUDIO_PROCESSING
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "TRANSCRIBE_FAILED"
        assert details.retry_recommended is False

    def test_classify_audio_generic_error(self):
        """Test classification of generic audio errors."""
        error = Exception("Audio extraction failed")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.AUDIO_PROCESSING
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "AUDIO_PROCESS"
        assert details.retry_recommended is True

    def test_classify_dependency_error(self):
        """Test classification of dependency errors."""
        error = Exception("Module not found: required_package")
        details = self.classifier.classify_error(error)

        assert details.category == ErrorCategory.DEPENDENCY
        assert details.severity == ErrorSeverity.CRITICAL
        assert details.error_code == "DEPENDENCY_MISSING"
        assert details.retry_recommended is False
        assert details.contact_support is True

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Some unexpected error occurred")
        details = self.classifier.classify_error(error, context="test_context")

        assert details.category == ErrorCategory.UNKNOWN
        assert details.severity == ErrorSeverity.ERROR
        assert details.error_code == "UNKNOWN"
        assert details.retry_recommended is True
        assert details.contact_support is True
        assert details.metadata["error_type"] == "Exception"
        assert details.metadata["context"] == "test_context"

    def test_get_available_disk_space_success(self):
        """Test successful disk space retrieval."""
        mock_usage = MagicMock()
        mock_usage.free = 1024 * 1024 * 1024  # 1GB

        with patch("tempfile.gettempdir", return_value="/tmp"):
            with patch("shutil.disk_usage", return_value=mock_usage):
                space_mb = self.classifier._get_available_disk_space()
                assert space_mb == 1024.0  # 1GB in MB

    def test_get_available_disk_space_error(self):
        """Test disk space retrieval error handling."""
        with patch("shutil.disk_usage", side_effect=OSError("Disk error")):
            space_mb = self.classifier._get_available_disk_space()
            assert space_mb is None


class TestUserFeedbackManager:
    """Test UserFeedbackManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.feedback_manager = UserFeedbackManager()

    def test_init(self):
        """Test feedback manager initialization."""
        assert isinstance(self.feedback_manager.classifier, ErrorClassifier)

    @patch("utils.error_handler.logger")
    def test_process_error_basic(self, mock_logger):
        """Test basic error processing."""
        error = Exception("Test error")
        details = self.feedback_manager.process_error(error, context="test_context")

        assert isinstance(details, ErrorDetails)
        mock_logger.error.assert_called_once()
        mock_logger.info.assert_called_once()

    @patch("utils.error_handler.logger")
    def test_process_error_with_user_action(self, mock_logger):
        """Test error processing with user action context."""
        error = Exception("Test error")
        details = self.feedback_manager.process_error(error, context="test_context", user_action="downloading video")

        assert details.metadata is not None
        assert details.metadata["user_action"] == "downloading video"

    def test_get_contextual_help_network(self):
        """Test contextual help for network errors."""
        help_info = self.feedback_manager.get_contextual_help(ErrorCategory.NETWORK)

        assert help_info["title"] == "Network Connection Issues"
        assert "common_solutions" in help_info
        assert "prevention_tips" in help_info
        assert len(help_info["common_solutions"]) > 0

    def test_get_contextual_help_content_access(self):
        """Test contextual help for content access errors."""
        help_info = self.feedback_manager.get_contextual_help(ErrorCategory.CONTENT_ACCESS)

        assert help_info["title"] == "Content Access Problems"
        assert "description" in help_info
        assert "common_solutions" in help_info

    def test_get_contextual_help_unknown_category(self):
        """Test contextual help for unknown category."""
        help_info = self.feedback_manager.get_contextual_help(ErrorCategory.UNKNOWN)

        assert help_info["title"] == "General Help"
        assert "common_solutions" in help_info

    def test_format_error_for_display_basic(self):
        """Test basic error formatting for display."""
        details = ErrorDetails(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            user_message="Connection failed",
            technical_details="Timeout",
            recovery_suggestions=["Check internet", "Retry"],
        )

        formatted = self.feedback_manager.format_error_for_display(details)

        assert "Connection failed" in formatted
        assert "What you can try:" in formatted
        assert "1. Check internet" in formatted
        assert "2. Retry" in formatted

    def test_format_error_for_display_with_retry(self):
        """Test error formatting with retry recommendation."""
        details = ErrorDetails(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.ERROR,
            user_message="Connection failed",
            technical_details="Timeout",
            recovery_suggestions=["Check internet"],
            retry_recommended=True,
        )

        formatted = self.feedback_manager.format_error_for_display(details)

        assert "✓ You can try this operation again" in formatted

    def test_format_error_for_display_with_support(self):
        """Test error formatting with support contact."""
        details = ErrorDetails(
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.CRITICAL,
            user_message="Missing dependency",
            technical_details="Module not found",
            recovery_suggestions=["Install package"],
            contact_support=True,
        )

        formatted = self.feedback_manager.format_error_for_display(details)

        assert "⚠ If this problem persists, please report it" in formatted

    def test_should_show_technical_details_critical(self):
        """Test showing technical details for critical errors."""
        details = ErrorDetails(
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.CRITICAL,
            user_message="Critical error",
            technical_details="System failure",
            recovery_suggestions=[],
        )

        should_show = self.feedback_manager.should_show_technical_details(details)
        assert should_show is True

    def test_should_show_technical_details_support(self):
        """Test showing technical details when support contact recommended."""
        details = ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            user_message="Unknown error",
            technical_details="Stack trace",
            recovery_suggestions=[],
            contact_support=True,
        )

        should_show = self.feedback_manager.should_show_technical_details(details)
        assert should_show is True

    def test_should_show_technical_details_dependency(self):
        """Test showing technical details for dependency errors."""
        details = ErrorDetails(
            category=ErrorCategory.DEPENDENCY,
            severity=ErrorSeverity.ERROR,
            user_message="Dependency error",
            technical_details="Import failed",
            recovery_suggestions=[],
        )

        should_show = self.feedback_manager.should_show_technical_details(details)
        assert should_show is True

    def test_should_show_technical_details_normal(self):
        """Test not showing technical details for normal errors."""
        details = ErrorDetails(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            user_message="Network warning",
            technical_details="Minor issue",
            recovery_suggestions=[],
        )

        should_show = self.feedback_manager.should_show_technical_details(details)
        assert should_show is False


class TestDiagnosticCollector:
    """Test DiagnosticCollector functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.collector = DiagnosticCollector()

    def test_init(self):
        """Test diagnostic collector initialization."""
        assert self.collector is not None

    @patch("utils.error_handler.DiagnosticCollector._get_platform_info")
    @patch("utils.error_handler.DiagnosticCollector._get_python_version")
    @patch("utils.error_handler.DiagnosticCollector._get_memory_info")
    @patch("utils.error_handler.DiagnosticCollector._get_disk_space_info")
    @patch("utils.error_handler.DiagnosticCollector._check_dependencies")
    @patch("utils.error_handler.DiagnosticCollector._get_timestamp")
    def test_collect_system_info_success(
        self, mock_timestamp, mock_deps, mock_disk, mock_memory, mock_python, mock_platform
    ):
        """Test successful system info collection."""
        # Setup mocks
        mock_platform.return_value = {"system": "Linux"}
        mock_python.return_value = "3.9.0"
        mock_memory.return_value = {"total_gb": 8.0}
        mock_disk.return_value = {"total_gb": 100.0}
        mock_deps.return_value = {"ffmpeg": "4.4.0"}
        mock_timestamp.return_value = "2023-01-01T00:00:00"

        diagnostics = self.collector.collect_system_info()

        assert diagnostics["platform"] == {"system": "Linux"}
        assert diagnostics["python_version"] == "3.9.0"
        assert diagnostics["memory"] == {"total_gb": 8.0}
        assert diagnostics["disk_space"] == {"total_gb": 100.0}
        assert diagnostics["dependencies"] == {"ffmpeg": "4.4.0"}
        assert diagnostics["app_version"] == "1.0.0"
        assert diagnostics["timestamp"] == "2023-01-01T00:00:00"

    @patch("utils.error_handler.DiagnosticCollector._get_platform_info")
    @patch("utils.error_handler.logger")
    def test_collect_system_info_error(self, mock_logger, mock_platform):
        """Test system info collection with error."""
        mock_platform.side_effect = Exception("Platform error")

        diagnostics = self.collector.collect_system_info()

        assert diagnostics == {"error": "diagnostic_collection_failed"}
        mock_logger.warning.assert_called_once()

    @patch("platform.system")
    @patch("platform.release")
    @patch("platform.architecture")
    def test_get_platform_info(self, mock_arch, mock_release, mock_system):
        """Test platform info collection."""
        mock_system.return_value = "Linux"
        mock_release.return_value = "5.4.0"
        mock_arch.return_value = ("64bit", "ELF")

        platform_info = self.collector._get_platform_info()

        assert platform_info["system"] == "Linux"
        assert platform_info["release"] == "5.4.0"
        assert platform_info["architecture"] == "64bit"

    def test_get_python_version(self):
        """Test Python version retrieval."""
        version = self.collector._get_python_version()
        assert isinstance(version, str)
        assert "." in version  # Should be in format like "3.9.0"

    @patch("psutil.virtual_memory")
    def test_get_memory_info_success(self, mock_memory):
        """Test successful memory info retrieval."""
        mock_memory.return_value = MagicMock(
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB
            percent=50.0,
        )

        memory_info = self.collector._get_memory_info()

        assert memory_info["total_gb"] == 8.0
        assert memory_info["available_gb"] == 4.0
        assert memory_info["percent_used"] == 50.0

    @patch("psutil.virtual_memory")
    def test_get_memory_info_error(self, mock_memory):
        """Test memory info retrieval error."""
        mock_memory.side_effect = Exception("Memory error")

        memory_info = self.collector._get_memory_info()

        assert memory_info == {"error": "memory_info_unavailable"}

    @patch("tempfile.gettempdir")
    @patch("shutil.disk_usage")
    def test_get_disk_space_info_success(self, mock_disk_usage, mock_tempdir):
        """Test successful disk space info retrieval."""
        mock_tempdir.return_value = "/tmp"
        mock_disk_usage.return_value = MagicMock(
            total=100 * 1024**3,  # 100GB
            free=50 * 1024**3,  # 50GB
        )

        disk_info = self.collector._get_disk_space_info()

        assert disk_info["total_gb"] == 100.0
        assert disk_info["free_gb"] == 50.0
        assert disk_info["percent_free"] == 50.0

    @patch("shutil.disk_usage")
    def test_get_disk_space_info_error(self, mock_disk_usage):
        """Test disk space info retrieval error."""
        mock_disk_usage.side_effect = Exception("Disk error")

        disk_info = self.collector._get_disk_space_info()

        assert disk_info == {"error": "disk_info_unavailable"}

    @patch("subprocess.run")
    def test_check_dependencies_ffmpeg_success(self, mock_run):
        """Test FFmpeg dependency check success."""
        mock_run.return_value = MagicMock(returncode=0, stdout="ffmpeg version 4.4.0 Copyright...")

        with patch("importlib.import_module", side_effect=ImportError()):
            dependencies = self.collector._check_dependencies()

        assert "ffmpeg" in dependencies
        assert dependencies["ffmpeg"] == "4.4.0"

    @patch("subprocess.run")
    def test_check_dependencies_ffmpeg_not_found(self, mock_run):
        """Test FFmpeg dependency check when not found."""
        mock_run.return_value = MagicMock(returncode=1)

        with patch("importlib.import_module", side_effect=ImportError()):
            dependencies = self.collector._check_dependencies()

        assert dependencies["ffmpeg"] == "not_found"

    @patch("subprocess.run")
    def test_check_dependencies_ffmpeg_error(self, mock_run):
        """Test FFmpeg dependency check with error."""
        mock_run.side_effect = Exception("Command error")

        with patch("importlib.import_module", side_effect=ImportError()):
            dependencies = self.collector._check_dependencies()

        assert dependencies["ffmpeg"] == "not_available"

    def test_check_dependencies_python_packages(self):
        """Test Python package dependency checking."""
        mock_module = MagicMock()
        mock_module.__version__ = "1.2.3"

        with patch("importlib.import_module", return_value=mock_module):
            with patch("subprocess.run", return_value=MagicMock(returncode=1)):
                dependencies = self.collector._check_dependencies()

        # Should check multiple packages
        assert len(dependencies) > 1
        # Mock module should show version
        for package in ["instaloader", "whisper", "FreeSimpleGUI", "psutil"]:
            if package in dependencies:
                assert dependencies[package] in ["1.2.3", "not_installed"]

    def test_get_timestamp(self):
        """Test timestamp generation."""
        timestamp = self.collector._get_timestamp()
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format

    def test_sanitize_url_instagram_reel(self):
        """Test URL sanitization for Instagram reels."""
        url = "https://www.instagram.com/reel/ABC123DEF/?utm_source=test"
        sanitized = self.collector.sanitize_url_for_logging(url)

        assert sanitized == "https://www.instagram.com/reel/[REEL_ID]/?utm_source=test"

    def test_sanitize_url_non_instagram(self):
        """Test URL sanitization for non-Instagram URLs."""
        url = "https://example.com/video/123"
        sanitized = self.collector.sanitize_url_for_logging(url)

        assert sanitized == "[NON_INSTAGRAM_URL]"

    def test_sanitize_url_error(self):
        """Test URL sanitization error handling."""
        with patch("re.sub", side_effect=Exception("Regex error")):
            sanitized = self.collector.sanitize_url_for_logging("any_url")
            assert sanitized == "[URL_SANITIZATION_FAILED]"


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("utils.error_handler._error_classifier")
    def test_classify_error_function(self, mock_classifier):
        """Test classify_error convenience function."""
        error = Exception("Test error")
        mock_details = ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            user_message="Test",
            technical_details="Details",
            recovery_suggestions=[],
        )
        mock_classifier.classify_error.return_value = mock_details

        result = classify_error(error, "test_context")

        assert result == mock_details
        mock_classifier.classify_error.assert_called_once_with(error, "test_context")

    @patch("utils.error_handler._feedback_manager")
    def test_process_error_for_user_function(self, mock_manager):
        """Test process_error_for_user convenience function."""
        error = Exception("Test error")
        mock_details = ErrorDetails(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.ERROR,
            user_message="Test",
            technical_details="Details",
            recovery_suggestions=[],
        )
        mock_manager.process_error.return_value = mock_details

        result = process_error_for_user(error, "context", "action")

        assert result == mock_details
        mock_manager.process_error.assert_called_once_with(error, "context", "action")

    @patch("utils.error_handler._diagnostic_collector")
    def test_get_system_diagnostics_function(self, mock_collector):
        """Test get_system_diagnostics convenience function."""
        mock_diagnostics = {"platform": "Linux", "python": "3.9"}
        mock_collector.collect_system_info.return_value = mock_diagnostics

        result = get_system_diagnostics()

        assert result == mock_diagnostics
        mock_collector.collect_system_info.assert_called_once()

    @patch("utils.error_handler._diagnostic_collector")
    def test_sanitize_url_function(self, mock_collector):
        """Test sanitize_url convenience function."""
        mock_collector.sanitize_url_for_logging.return_value = "[SANITIZED]"

        result = sanitize_url("test_url")

        assert result == "[SANITIZED]"
        mock_collector.sanitize_url_for_logging.assert_called_once_with("test_url")


class TestErrorHandlingIntegration:
    """Integration tests for error handling workflows."""

    def test_complete_error_processing_workflow(self):
        """Test complete error processing from classification to user feedback."""
        # Create a network timeout error
        error = ConnectionError("Connection timeout after 30 seconds")

        # Process through the complete workflow
        feedback_manager = UserFeedbackManager()
        details = feedback_manager.process_error(
            error, context="downloading Instagram Reel", user_action="clicking download button"
        )

        # Verify complete processing
        assert details.category == ErrorCategory.NETWORK
        assert details.severity == ErrorSeverity.ERROR
        assert "timeout" in details.user_message.lower()
        assert details.error_code == "NET_TIMEOUT"
        assert details.retry_recommended is True
        assert details.metadata["user_action"] == "clicking download button"

        # Test user-facing formatting
        formatted = feedback_manager.format_error_for_display(details)
        assert len(formatted) > 0
        assert "What you can try:" in formatted

        # Test contextual help
        help_info = feedback_manager.get_contextual_help(details.category)
        assert help_info["title"] == "Network Connection Issues"

    def test_error_chain_classification(self):
        """Test classification of various error types."""
        test_cases = [
            (ConnectionError("Network unreachable"), ErrorCategory.NETWORK),
            (PermissionError("Access denied"), ErrorCategory.PERMISSION),
            (OSError("No space left on device"), ErrorCategory.DISK_SPACE),
            (FileNotFoundError("Reel not found"), ErrorCategory.CONTENT_ACCESS),
            (ImportError("Module not found"), ErrorCategory.DEPENDENCY),
            (ValueError("Invalid input format"), ErrorCategory.UNKNOWN),
        ]

        classifier = ErrorClassifier()
        for error, expected_category in test_cases:
            details = classifier.classify_error(error)
            assert details.category == expected_category

    def test_diagnostic_collection_workflow(self):
        """Test diagnostic information collection for support."""
        collector = DiagnosticCollector()

        # Collect diagnostics
        diagnostics = collector.collect_system_info()

        # Verify essential diagnostic information is present
        if "error" not in diagnostics:
            assert "platform" in diagnostics
            assert "python_version" in diagnostics
            assert "dependencies" in diagnostics
            assert "timestamp" in diagnostics

    def test_production_error_handling_patterns(self):
        """Test production-ready error handling patterns."""
        feedback_manager = UserFeedbackManager()

        # Test critical error with support contact
        critical_error = ImportError("FFmpeg not found")
        details = feedback_manager.process_error(critical_error)

        assert details.severity == ErrorSeverity.CRITICAL
        assert details.contact_support is True
        assert feedback_manager.should_show_technical_details(details) is True

        # Test user-friendly warning
        warning_error = Exception("Rate limit exceeded")
        details = feedback_manager.process_error(warning_error)

        assert details.severity == ErrorSeverity.WARNING
        assert details.retry_recommended is True
        assert len(details.recovery_suggestions) > 0

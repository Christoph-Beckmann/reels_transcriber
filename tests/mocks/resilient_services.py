"""
Mock services for testing resilience patterns without external dependencies.

This module provides comprehensive mocks for Instagram API, yt-dlp, and other external
services to enable isolated integration testing.
"""

import logging
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MockFailureMode(Enum):
    """Different failure modes for testing resilience patterns."""

    NONE = "none"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    RATE_LIMIT = "rate_limit"
    ACCESS_DENIED = "access_denied"
    NOT_FOUND = "not_found"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERMITTENT = "intermittent"


@dataclass
class MockServiceConfig:
    """Configuration for mock service behavior."""

    failure_mode: MockFailureMode = MockFailureMode.NONE
    failure_rate: float = 0.0  # 0.0 to 1.0
    latency_ms: int = 100
    max_latency_ms: int = 500
    circuit_breaker_trigger_count: int = 3
    recovery_after_seconds: int = 30


class MockInstagramAPI:
    """Mock Instagram API for testing without real network calls."""

    def __init__(self, config: Optional[MockServiceConfig] = None):
        """Initialize mock Instagram API."""
        self.config = config or MockServiceConfig()
        self.call_count = 0
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_service_degraded = False

        logger.info(f"Mock Instagram API initialized with config: {self.config}")

    def download_reel(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Mock Instagram Reel download."""
        self.call_count += 1

        # Simulate progress updates
        if progress_callback:
            progress_callback(10, "Connecting to Instagram...")
            time.sleep(0.1)
            progress_callback(30, "Authenticating...")
            time.sleep(0.1)
            progress_callback(60, "Downloading video...")

        # Check if we should simulate failure
        should_fail = self._should_simulate_failure()

        if should_fail:
            self.failure_count += 1
            self.last_failure_time = time.time()

            # Simulate different failure types
            error_msg = self._get_failure_message()

            if progress_callback:
                progress_callback(0, f"Error: {error_msg}")

            logger.warning(f"Mock Instagram API simulated failure: {error_msg}")
            return False, None, error_msg

        # Simulate successful download
        if progress_callback:
            progress_callback(90, "Processing video...")
            time.sleep(0.1)
            progress_callback(100, "Download complete!")

        # Create mock file path
        mock_file_path = f"/tmp/mock_instagram_{hash(url)}.mp4"

        logger.info(f"Mock Instagram API successful download: {mock_file_path}")
        return True, mock_file_path, None

    def validate_reel_url(self, url: str) -> tuple[bool, Optional[str], Optional[str]]:
        """Mock URL validation."""
        if not url or "instagram.com" not in url:
            return False, "Invalid Instagram URL", None

        if "reel" not in url and "reels" not in url:
            return False, "URL is not a Reel", None

        # Extract mock shortcode
        shortcode = f"MOCK{hash(url) % 10000}"
        return True, None, shortcode

    def is_healthy(self) -> bool:
        """Mock health check."""
        # Service is unhealthy during simulated outages
        if self.config.failure_mode == MockFailureMode.SERVICE_UNAVAILABLE:
            return False

        # Service recovers after configured time
        if self.is_service_degraded:
            if time.time() - self.last_failure_time > self.config.recovery_after_seconds:
                self.is_service_degraded = False
                self.failure_count = 0

        return not self.is_service_degraded

    def _should_simulate_failure(self) -> bool:
        """Determine if we should simulate a failure."""
        # Never fail in NONE mode
        if self.config.failure_mode == MockFailureMode.NONE:
            return False

        # Always fail in SERVICE_UNAVAILABLE mode
        if self.config.failure_mode == MockFailureMode.SERVICE_UNAVAILABLE:
            return True

        # Trigger circuit breaker after threshold
        if self.failure_count >= self.config.circuit_breaker_trigger_count:
            self.is_service_degraded = True
            return True

        # Random failures based on failure rate
        if self.config.failure_mode == MockFailureMode.INTERMITTENT:
            return random.random() < self.config.failure_rate

        # Other failure modes trigger once then recover
        if self.failure_count == 0:
            return True

        return False

    def _get_failure_message(self) -> str:
        """Get appropriate failure message for current failure mode."""
        failure_messages = {
            MockFailureMode.TIMEOUT: "Connection timed out",
            MockFailureMode.NETWORK_ERROR: "Network connection failed",
            MockFailureMode.RATE_LIMIT: "Rate limit exceeded. Please try again later.",
            MockFailureMode.ACCESS_DENIED: "Access denied. Content may be private.",
            MockFailureMode.NOT_FOUND: "Reel not found. It may have been deleted.",
            MockFailureMode.SERVICE_UNAVAILABLE: "Instagram service is currently unavailable",
            MockFailureMode.INTERMITTENT: "Temporary service error",
        }

        return failure_messages.get(self.config.failure_mode, "Unknown error")

    def get_stats(self) -> dict[str, Any]:
        """Get mock service statistics."""
        return {
            "total_calls": self.call_count,
            "total_failures": self.failure_count,
            "success_rate": (self.call_count - self.failure_count) / max(self.call_count, 1),
            "is_healthy": self.is_healthy(),
            "failure_mode": self.config.failure_mode.value,
            "is_degraded": self.is_service_degraded,
        }

    def reset_stats(self):
        """Reset statistics for new test."""
        self.call_count = 0
        self.failure_count = 0
        self.last_failure_time = 0
        self.is_service_degraded = False


class MockYtDlpService:
    """Mock yt-dlp service for testing fallback mechanisms."""

    def __init__(self, config: Optional[MockServiceConfig] = None, available: bool = True):
        """Initialize mock yt-dlp service."""
        self.config = config or MockServiceConfig()
        self.available = available
        self.call_count = 0
        self.failure_count = 0

        logger.info(f"Mock yt-dlp service initialized: available={available}")

    def is_available(self) -> bool:
        """Check if yt-dlp is available."""
        return self.available

    def download(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Mock yt-dlp download."""
        if not self.available:
            return False, None, "yt-dlp is not available"

        self.call_count += 1

        # Simulate progress
        if progress_callback:
            progress_callback(15, "Initializing yt-dlp...")
            time.sleep(0.1)
            progress_callback(40, "Extracting video information...")
            time.sleep(0.1)
            progress_callback(70, "Downloading with yt-dlp...")

        # Check for simulated failures
        if self._should_fail():
            self.failure_count += 1
            error_msg = self._get_ytdlp_error_message()

            if progress_callback:
                progress_callback(0, f"yt-dlp error: {error_msg}")

            logger.warning(f"Mock yt-dlp simulated failure: {error_msg}")
            return False, None, error_msg

        # Simulate successful download
        if progress_callback:
            progress_callback(100, "yt-dlp download complete!")

        mock_file_path = f"/tmp/mock_ytdlp_{hash(url)}.mp4"
        logger.info(f"Mock yt-dlp successful download: {mock_file_path}")
        return True, mock_file_path, None

    def _should_fail(self) -> bool:
        """Determine if yt-dlp should fail."""
        if self.config.failure_mode == MockFailureMode.NONE:
            return False

        # yt-dlp typically has different failure patterns than Instagram API
        return random.random() < (self.config.failure_rate * 0.5)  # Less likely to fail

    def _get_ytdlp_error_message(self) -> str:
        """Get yt-dlp specific error message."""
        ytdlp_errors = {
            MockFailureMode.TIMEOUT: "yt-dlp: download timeout",
            MockFailureMode.NETWORK_ERROR: "yt-dlp: network connection failed",
            MockFailureMode.ACCESS_DENIED: "yt-dlp: HTTP 403 Forbidden",
            MockFailureMode.NOT_FOUND: "yt-dlp: HTTP 404 Not Found",
            MockFailureMode.SERVICE_UNAVAILABLE: "yt-dlp: service unavailable",
        }

        return ytdlp_errors.get(self.config.failure_mode, "yt-dlp: unknown error")

    def get_capabilities(self) -> dict[str, Any]:
        """Get mock yt-dlp capabilities."""
        return {
            "available": self.available,
            "version_info": (
                {"version": "2023.01.06", "major": 2023, "minor": 1, "patch": 6} if self.available else None
            ),
            "call_count": self.call_count,
            "failure_count": self.failure_count,
        }


class MockCircuitBreakerTestScenarios:
    """Test scenarios for circuit breaker behavior validation."""

    @staticmethod
    def create_failure_cascade_scenario() -> list[MockServiceConfig]:
        """Create scenario where primary service fails and triggers circuit breaker."""
        return [
            MockServiceConfig(
                failure_mode=MockFailureMode.NETWORK_ERROR,
                failure_rate=1.0,  # Always fail
                circuit_breaker_trigger_count=3,
            )
        ]

    @staticmethod
    def create_recovery_scenario() -> list[MockServiceConfig]:
        """Create scenario where service recovers after circuit breaker opens."""
        return [
            MockServiceConfig(
                failure_mode=MockFailureMode.INTERMITTENT,
                failure_rate=0.8,  # High failure rate initially
                recovery_after_seconds=10,
            ),
            MockServiceConfig(
                failure_mode=MockFailureMode.NONE,  # Recovery phase
                failure_rate=0.0,
            ),
        ]

    @staticmethod
    def create_fallback_success_scenario() -> tuple[MockServiceConfig, MockServiceConfig]:
        """Create scenario where primary fails but fallback succeeds."""
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.RATE_LIMIT, failure_rate=1.0)

        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE, failure_rate=0.0)

        return primary_config, fallback_config

    @staticmethod
    def create_both_services_fail_scenario() -> tuple[MockServiceConfig, MockServiceConfig]:
        """Create scenario where both primary and fallback fail."""
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.ACCESS_DENIED, failure_rate=1.0)

        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NOT_FOUND, failure_rate=1.0)

        return primary_config, fallback_config


class MockResilientDownloaderFactory:
    """Factory for creating mock resilient downloaders for testing."""

    @staticmethod
    def create_healthy_downloader(temp_dir: str):
        """Create a mock downloader where all services are healthy."""
        from core.downloader import EnhancedInstagramDownloader
        from core.resilience import CircuitBreakerConfig

        # Create mock services with no failures
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        # Create downloader with mocked services
        downloader = EnhancedInstagramDownloader(
            download_dir=temp_dir,
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout=10, success_threshold=2),
        )

        # Replace actual services with mocks
        downloader._mock_primary = MockInstagramAPI(primary_config)
        downloader._mock_fallback = MockYtDlpService(fallback_config)

        return downloader

    @staticmethod
    def create_circuit_breaker_test_downloader(temp_dir: str, scenario: str):
        """Create downloader configured for specific circuit breaker test scenario."""
        from core.downloader import EnhancedInstagramDownloader
        from core.resilience import CircuitBreakerConfig

        if scenario == "failure_cascade":
            configs = MockCircuitBreakerTestScenarios.create_failure_cascade_scenario()
            primary_config = configs[0]
            fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        elif scenario == "recovery":
            configs = MockCircuitBreakerTestScenarios.create_recovery_scenario()
            primary_config = configs[0]  # Start with high failure rate
            fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        elif scenario == "fallback_success":
            primary_config, fallback_config = MockCircuitBreakerTestScenarios.create_fallback_success_scenario()

        elif scenario == "both_fail":
            primary_config, fallback_config = MockCircuitBreakerTestScenarios.create_both_services_fail_scenario()

        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        # Create downloader with aggressive circuit breaker for testing
        downloader = EnhancedInstagramDownloader(
            download_dir=temp_dir,
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2,  # Trigger quickly for testing
                recovery_timeout=5,  # Recover quickly for testing
                success_threshold=1,  # Close quickly for testing
            ),
        )

        # Replace with configured mock services
        downloader._mock_primary = MockInstagramAPI(primary_config)
        downloader._mock_fallback = MockYtDlpService(fallback_config)

        return downloader


class MockNetworkConditions:
    """Simulate different network conditions for testing resilience."""

    @staticmethod
    def simulate_slow_network(delay_seconds: float = 2.0):
        """Simulate slow network conditions."""
        return MockServiceConfig(latency_ms=int(delay_seconds * 1000), max_latency_ms=int(delay_seconds * 1500))

    @staticmethod
    def simulate_unstable_network(packet_loss_rate: float = 0.3):
        """Simulate unstable network with packet loss."""
        return MockServiceConfig(failure_mode=MockFailureMode.INTERMITTENT, failure_rate=packet_loss_rate)

    @staticmethod
    def simulate_rate_limited_network():
        """Simulate rate-limited network conditions."""
        return MockServiceConfig(
            failure_mode=MockFailureMode.RATE_LIMIT,
            failure_rate=0.5,  # 50% chance of rate limiting
            recovery_after_seconds=15,
        )


def create_mock_download_test_suite() -> dict[str, Any]:
    """Create comprehensive test suite with various mock scenarios."""
    return {
        "healthy_services": {
            "primary": MockServiceConfig(failure_mode=MockFailureMode.NONE),
            "fallback": MockServiceConfig(failure_mode=MockFailureMode.NONE),
            "expected_success": True,
            "expected_service": "primary",
        },
        "primary_fails_fallback_succeeds": {
            "primary": MockServiceConfig(failure_mode=MockFailureMode.ACCESS_DENIED, failure_rate=1.0),
            "fallback": MockServiceConfig(failure_mode=MockFailureMode.NONE),
            "expected_success": True,
            "expected_service": "fallback",
        },
        "both_services_fail": {
            "primary": MockServiceConfig(failure_mode=MockFailureMode.NOT_FOUND, failure_rate=1.0),
            "fallback": MockServiceConfig(failure_mode=MockFailureMode.NOT_FOUND, failure_rate=1.0),
            "expected_success": False,
            "expected_service": None,
        },
        "circuit_breaker_opens": {
            "primary": MockServiceConfig(
                failure_mode=MockFailureMode.TIMEOUT, failure_rate=1.0, circuit_breaker_trigger_count=2
            ),
            "fallback": MockServiceConfig(failure_mode=MockFailureMode.NONE),
            "expected_success": True,  # After circuit opens, fallback should work
            "expected_service": "fallback",
            "test_sequence": ["fail", "fail", "circuit_open", "fallback_success"],
        },
        "service_recovery": {
            "primary": MockServiceConfig(
                failure_mode=MockFailureMode.INTERMITTENT, failure_rate=0.8, recovery_after_seconds=5
            ),
            "fallback": MockServiceConfig(failure_mode=MockFailureMode.NONE),
            "expected_success": True,
            "expected_service": "varies",  # Should switch between services
            "test_sequence": ["fail", "fallback", "wait", "primary_recovers"],
        },
    }


# Utility functions for test integration


def patch_downloader_with_mocks(downloader, primary_config: MockServiceConfig, fallback_config: MockServiceConfig):
    """Patch a real downloader with mock services for testing."""
    mock_primary = MockInstagramAPI(primary_config)
    mock_fallback = MockYtDlpService(fallback_config)

    # Store original methods
    downloader._original_download_reel = downloader.download_reel
    downloader._original_ytdlp_download = downloader.ytdlp.download if downloader.ytdlp else None

    # Replace with mocks
    downloader.download_reel = mock_primary.download_reel
    if downloader.ytdlp:
        downloader.ytdlp.download = mock_fallback.download

    # Store mock references for inspection
    downloader._mock_primary = mock_primary
    downloader._mock_fallback = mock_fallback

    return downloader


def restore_downloader_from_mocks(downloader):
    """Restore a downloader from mocked state."""
    if hasattr(downloader, "_original_download_reel"):
        downloader.download_reel = downloader._original_download_reel
        delattr(downloader, "_original_download_reel")

    if hasattr(downloader, "_original_ytdlp_download") and downloader.ytdlp:
        downloader.ytdlp.download = downloader._original_ytdlp_download
        delattr(downloader, "_original_ytdlp_download")

    # Clean up mock references
    if hasattr(downloader, "_mock_primary"):
        delattr(downloader, "_mock_primary")
    if hasattr(downloader, "_mock_fallback"):
        delattr(downloader, "_mock_fallback")


def get_mock_stats(downloader) -> dict[str, Any]:
    """Get statistics from mock services attached to downloader."""
    stats = {}

    if hasattr(downloader, "_mock_primary"):
        stats["primary"] = downloader._mock_primary.get_stats()

    if hasattr(downloader, "_mock_fallback"):
        stats["fallback"] = downloader._mock_fallback.get_capabilities()

    return stats

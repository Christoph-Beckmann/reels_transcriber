"""
Circuit breaker and resilience patterns for external API dependencies.

This module provides robust patterns for handling external service failures,
including circuit breakers, retry strategies, and fallback mechanisms.
"""

import logging
import random
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure mode, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Failures to trigger open state
    recovery_timeout: int = 60  # Seconds before trying to recover
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds

    # Exponential backoff settings
    base_delay: float = 1.0  # Base delay for retries
    max_delay: float = 60.0  # Maximum delay between retries
    backoff_multiplier: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    circuit_trips: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    current_failure_count: int = 0
    current_success_count: int = 0


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class ExternalServiceInterface(ABC):
    """Abstract interface for external services with circuit breaker support."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the external service call."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        pass


class CircuitBreaker:
    """
    Circuit breaker implementation for external service resilience.

    Implements the circuit breaker pattern to prevent cascading failures
    when external services are unavailable or degraded.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.

        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.stats = CircuitBreakerStats()
        self._lock = threading.RLock()
        self._last_failure_time = 0.0

        logger.info(f"Circuit breaker initialized with config: {self.config}")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerException: When circuit is open
        """
        with self._lock:
            self.stats.total_requests += 1

            # Check if circuit should remain open
            if self.state == CircuitState.OPEN:
                if time.time() - self._last_failure_time < self.config.recovery_timeout:
                    raise CircuitBreakerException(
                        f"Circuit breaker is OPEN. Retry after "
                        f"{self.config.recovery_timeout - (time.time() - self._last_failure_time):.1f}s"
                    )
                else:
                    # Transition to half-open for testing
                    self.state = CircuitState.HALF_OPEN
                    self.stats.current_success_count = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN state for testing")

            try:
                # Execute the function with timeout
                result = self._execute_with_timeout(func, *args, **kwargs)

                # Handle success
                self._handle_success()
                return result

            except Exception as e:
                # Handle failure
                self._handle_failure(e)
                raise

    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with timeout protection."""
        import signal

        class TimeoutError(Exception):
            pass

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function timed out after {self.config.timeout}s")

        # Set up timeout for Unix systems
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.config.timeout))

            try:
                result = func(*args, **kwargs)
                signal.alarm(0)  # Cancel alarm
                return result
            finally:
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # Fallback for Windows - no timeout protection
            return func(*args, **kwargs)

    def _handle_success(self):
        """Handle successful function execution."""
        self.stats.successful_requests += 1
        self.stats.last_success_time = time.time()
        self.stats.current_failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.stats.current_success_count += 1
            if self.stats.current_success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                logger.info("Circuit breaker CLOSED - service recovered")

        elif self.state == CircuitState.OPEN:
            # Shouldn't happen, but handle gracefully
            self.state = CircuitState.CLOSED
            logger.warning("Circuit breaker forced CLOSED due to unexpected success")

    def _handle_failure(self, exception: Exception):
        """Handle failed function execution."""
        self.stats.failed_requests += 1
        self.stats.current_failure_count += 1
        self.stats.last_failure_time = time.time()
        self._last_failure_time = self.stats.last_failure_time

        # Check if we should open the circuit
        if (
            self.state in (CircuitState.CLOSED, CircuitState.HALF_OPEN)
            and self.stats.current_failure_count >= self.config.failure_threshold
        ):
            self.state = CircuitState.OPEN
            self.stats.circuit_trips += 1
            logger.error(
                f"Circuit breaker OPENED after {self.stats.current_failure_count} failures. Last error: {exception}"
            )

    def get_state(self) -> CircuitState:
        """Get current circuit breaker state."""
        return self.state

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self.stats

    def force_open(self):
        """Force circuit breaker to open state (for testing)."""
        with self._lock:
            self.state = CircuitState.OPEN
            self._last_failure_time = time.time()
            logger.warning("Circuit breaker forced to OPEN state")

    def force_close(self):
        """Force circuit breaker to close state (for recovery)."""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.stats.current_failure_count = 0
            self.stats.current_success_count = 0
            logger.info("Circuit breaker forced to CLOSED state")


class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize retry strategy."""
        self.config = config or CircuitBreakerConfig()

    def execute_with_retry(
        self, func: Callable, max_retries: int = 3, retryable_exceptions: tuple = (Exception,), *args, **kwargs
    ) -> Any:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            max_retries: Maximum number of retry attempts
            retryable_exceptions: tuple of exceptions that should trigger retries
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Last exception if all retries failed
        """
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"Function succeeded on attempt {attempt + 1}")
                return result

            except retryable_exceptions as e:
                last_exception = e

                if attempt < max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")

        raise last_exception

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(self.config.base_delay * (self.config.backoff_multiplier**attempt), self.config.max_delay)

        if self.config.jitter:
            # Add random jitter (±25% of delay)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)


class ResilientService:
    """
    Wrapper for external services with circuit breaker and retry patterns.
    """

    def __init__(
        self,
        service: ExternalServiceInterface,
        circuit_config: Optional[CircuitBreakerConfig] = None,
        fallback_service: Optional[ExternalServiceInterface] = None,
    ):
        """
        Initialize resilient service wrapper.

        Args:
            service: Primary external service
            circuit_config: Circuit breaker configuration
            fallback_service: Optional fallback service
        """
        self.primary_service = service
        self.fallback_service = fallback_service
        self.circuit_breaker = CircuitBreaker(circuit_config)
        self.retry_strategy = RetryStrategy(circuit_config)

        logger.info(f"Resilient service initialized with primary: {type(service).__name__}")
        if fallback_service:
            logger.info(f"Fallback service available: {type(fallback_service).__name__}")

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute service call with resilience patterns.

        Args:
            *args: Service arguments
            **kwargs: Service keyword arguments

        Returns:
            Service result
        """
        try:
            # Try primary service with circuit breaker and retry
            return self.circuit_breaker.call(
                self.retry_strategy.execute_with_retry,
                self.primary_service.execute,
                3,  # max_retries
                (ConnectionError, TimeoutError, Exception),
                *args,
                **kwargs,
            )

        except (CircuitBreakerException, Exception) as e:
            logger.warning(f"Primary service failed: {e}")

            # Try fallback service if available
            if self.fallback_service:
                logger.info("Attempting fallback service...")
                try:
                    result = self.fallback_service.execute(*args, **kwargs)
                    logger.info("Fallback service succeeded")
                    return result
                except Exception as fallback_error:
                    logger.error(f"Fallback service also failed: {fallback_error}")
                    raise Exception(
                        f"Both primary and fallback services failed. Primary: {e}, Fallback: {fallback_error}"
                    ) from fallback_error
            else:
                raise

    def is_healthy(self) -> bool:
        """Check if the resilient service is healthy."""
        primary_healthy = False
        fallback_healthy = False

        try:
            primary_healthy = self.primary_service.is_healthy()
        except Exception as e:
            logger.debug(f"Primary service health check failed: {e}")

        if self.fallback_service:
            try:
                fallback_healthy = self.fallback_service.is_healthy()
            except Exception as e:
                logger.debug(f"Fallback service health check failed: {e}")

        return primary_healthy or fallback_healthy

    def get_circuit_state(self) -> CircuitState:
        """Get circuit breaker state."""
        return self.circuit_breaker.get_state()

    def get_stats(self) -> CircuitBreakerStats:
        """Get circuit breaker statistics."""
        return self.circuit_breaker.get_stats()

    def force_circuit_open(self):
        """Force circuit breaker open (for testing)."""
        self.circuit_breaker.force_open()

    def force_circuit_close(self):
        """Force circuit breaker closed (for recovery)."""
        self.circuit_breaker.force_close()


class HealthChecker:
    """
    Health monitoring for external services.
    """

    def __init__(self, check_interval: int = 60):
        """
        Initialize health checker.

        Args:
            check_interval: Seconds between health checks
        """
        self.check_interval = check_interval
        self.services: dict[str, ExternalServiceInterface] = {}
        self.health_status: dict[str, bool] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def register_service(self, name: str, service: ExternalServiceInterface):
        """Register a service for health monitoring."""
        self.services[name] = service
        self.health_status[name] = True  # Assume healthy initially
        logger.info(f"Registered service for health monitoring: {name}")

    def start_monitoring(self):
        """Start background health monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring."""
        self._running = False
        if self._thread:
            self._thread.join()
        logger.info("Health monitoring stopped")

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._running:
            for name, service in self.services.items():
                try:
                    healthy = service.is_healthy()
                    previous_status = self.health_status.get(name, True)

                    if healthy != previous_status:
                        status_change = "recovered" if healthy else "degraded"
                        logger.info(f"Service {name} status changed: {status_change}")

                    self.health_status[name] = healthy

                except Exception as e:
                    logger.warning(f"Health check failed for {name}: {e}")
                    self.health_status[name] = False

            time.sleep(self.check_interval)

    def get_health_status(self) -> dict[str, bool]:
        """Get current health status of all services."""
        return self.health_status.copy()

    def is_service_healthy(self, name: str) -> bool:
        """Check if specific service is healthy."""
        return self.health_status.get(name, False)


class ResilientInstagramService(ExternalServiceInterface):
    """Instagram service wrapper with resilience patterns."""

    def __init__(self, downloader):
        """Initialize with Instagram downloader."""
        self.downloader = downloader

    def execute(
        self, url: str, progress_callback: Optional[Callable] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Execute Instagram download with resilience."""
        return self.downloader.download_reel(url, progress_callback)

    def is_healthy(self) -> bool:
        """Check Instagram service health."""
        try:
            # Simple health check - validate a known public URL format
            # This doesn't make actual requests, just checks downloader readiness
            return hasattr(self.downloader, "download_reel") and callable(self.downloader.download_reel)
        except Exception:
            return False


class ResilientYtDlpService(ExternalServiceInterface):
    """yt-dlp service wrapper with resilience patterns."""

    def __init__(self, downloader):
        """Initialize with yt-dlp downloader."""
        self.downloader = downloader

    def execute(
        self, url: str, progress_callback: Optional[Callable] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Execute yt-dlp download with resilience."""
        return self.downloader.download(url, progress_callback)

    def is_healthy(self) -> bool:
        """Check yt-dlp service health."""
        return self.downloader.is_available()


def create_resilient_downloader(primary_downloader, fallback_downloader=None, config=None):
    """
    Factory function to create a resilient downloader with circuit breaker protection.

    Args:
        primary_downloader: Primary downloader instance
        fallback_downloader: Optional fallback downloader
        config: Circuit breaker configuration

    Returns:
        ResilientService configured for downloading
    """
    # Wrap downloaders in service interfaces
    primary_service = ResilientInstagramService(primary_downloader)
    fallback_service = ResilientYtDlpService(fallback_downloader) if fallback_downloader else None

    # Create resilient service with circuit breaker
    resilient_service = ResilientService(
        service=primary_service, circuit_config=config, fallback_service=fallback_service
    )

    return resilient_service


# Global health checker instance
_health_checker = HealthChecker()


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    return _health_checker


def start_health_monitoring():
    """Start global health monitoring."""
    _health_checker.start_monitoring()


def stop_health_monitoring():
    """Stop global health monitoring."""
    _health_checker.stop_monitoring()

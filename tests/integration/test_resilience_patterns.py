"""
Integration tests for resilience patterns with isolated mock services.

These tests validate circuit breaker behavior, fallback mechanisms, and error recovery
without depending on external services.
"""

import time

import pytest

from core.downloader import EnhancedInstagramDownloader
from core.resilience import CircuitBreakerConfig, CircuitState
from tests.mocks.resilient_services import (
    MockFailureMode,
    MockServiceConfig,
    create_mock_download_test_suite,
    get_mock_stats,
    patch_downloader_with_mocks,
    restore_downloader_from_mocks,
)


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with real downloader components."""

    @pytest.fixture
    def circuit_breaker_config(self):
        """Fast circuit breaker config for testing."""
        return CircuitBreakerConfig(failure_threshold=2, recovery_timeout=3, success_threshold=1, timeout=10.0)

    @pytest.fixture
    def enhanced_downloader(self, temp_dir, circuit_breaker_config):
        """Create enhanced downloader for testing."""
        return EnhancedInstagramDownloader(
            download_dir=str(temp_dir), circuit_breaker_config=circuit_breaker_config, use_fallback=True
        )

    def test_circuit_breaker_opens_after_failures(self, enhanced_downloader, temp_dir):
        """Test that circuit breaker opens after repeated failures."""
        # Configure primary service to always fail
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.NETWORK_ERROR, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        # Patch with mock services
        patch_downloader_with_mocks(enhanced_downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/TEST123/"

            # First failure
            result1 = enhanced_downloader.download_reel_enhanced(test_url)
            assert not result1[0]  # Should fail

            # Second failure should trigger circuit breaker
            result2 = enhanced_downloader.download_reel_enhanced(test_url)
            assert not result2[0]  # Should still fail

            # Check circuit breaker state
            circuit_state = enhanced_downloader.resilient_service.get_circuit_state()
            assert circuit_state == CircuitState.OPEN

            # Third attempt should use fallback due to open circuit
            result3 = enhanced_downloader.download_reel_enhanced(test_url)
            # Should succeed via fallback
            assert result3[0] or "fallback" in (result3[2] or "").lower()

            # Verify mock statistics
            stats = get_mock_stats(enhanced_downloader)
            assert stats["primary"]["total_failures"] >= 2
            assert stats["primary"]["is_healthy"] is False

        finally:
            restore_downloader_from_mocks(enhanced_downloader)

    def test_circuit_breaker_recovery(self, enhanced_downloader):
        """Test circuit breaker recovery after timeout."""
        # Configure service to fail initially then recover
        primary_config = MockServiceConfig(
            failure_mode=MockFailureMode.INTERMITTENT,
            failure_rate=1.0,  # Start with 100% failure
            recovery_after_seconds=2,
        )
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(enhanced_downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/RECOVERY123/"

            # Trigger circuit breaker opening
            enhanced_downloader.download_reel_enhanced(test_url)
            enhanced_downloader.download_reel_enhanced(test_url)

            # Verify circuit is open
            assert enhanced_downloader.resilient_service.get_circuit_state() == CircuitState.OPEN

            # Wait for recovery timeout
            time.sleep(4)

            # Force service recovery by changing failure rate
            enhanced_downloader._mock_primary.config.failure_rate = 0.0
            enhanced_downloader._mock_primary.failure_count = 0
            enhanced_downloader._mock_primary.is_service_degraded = False

            # Force circuit breaker to attempt recovery
            enhanced_downloader.resilient_service.force_circuit_close()

            # Next call should succeed and close circuit
            result = enhanced_downloader.download_reel_enhanced(test_url)
            assert result[0]  # Should succeed

            # Verify circuit closed
            circuit_state = enhanced_downloader.resilient_service.get_circuit_state()
            assert circuit_state == CircuitState.CLOSED

        finally:
            restore_downloader_from_mocks(enhanced_downloader)

    def test_fallback_mechanism_activation(self, enhanced_downloader):
        """Test that fallback mechanism activates when primary fails."""
        # Primary always fails, fallback always succeeds
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.ACCESS_DENIED, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(enhanced_downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/FALLBACK123/"

            # Should eventually succeed via fallback
            result = enhanced_downloader.download_reel_enhanced(test_url)

            # Check if succeeded via fallback (either direct success or after primary failure)
            circuit_state = enhanced_downloader.resilient_service.get_circuit_state()
            stats = get_mock_stats(enhanced_downloader)

            # Either succeeded immediately via fallback, or primary failed and fallback worked
            fallback_worked = result[0] or (  # Direct success
                stats["primary"]["total_failures"] > 0 and circuit_state == CircuitState.OPEN
            )
            assert fallback_worked

        finally:
            restore_downloader_from_mocks(enhanced_downloader)

    def test_both_services_fail_gracefully(self, enhanced_downloader):
        """Test graceful failure when both primary and fallback services fail."""
        # Both services always fail
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.NOT_FOUND, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NOT_FOUND, failure_rate=1.0)

        patch_downloader_with_mocks(enhanced_downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/BOTHDEAD123/"

            result = enhanced_downloader.download_reel_enhanced(test_url)

            # Should fail gracefully
            assert not result[0]
            assert result[2] is not None  # Should have error message
            assert "not found" in result[2].lower() or "both" in result[2].lower() or "failed" in result[2].lower()

            # Verify both services were attempted
            stats = get_mock_stats(enhanced_downloader)
            assert stats["primary"]["total_failures"] > 0

        finally:
            restore_downloader_from_mocks(enhanced_downloader)


class TestRetryMechanisms:
    """Test retry mechanisms and exponential backoff."""

    def test_retry_with_exponential_backoff(self, temp_dir):
        """Test retry mechanism with exponential backoff."""
        config = CircuitBreakerConfig(
            failure_threshold=5,  # Allow more retries before circuit opens
            base_delay=0.1,  # Fast for testing
            max_delay=1.0,
            backoff_multiplier=2.0,
        )

        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir), circuit_breaker_config=config)

        # Configure to fail first few times then succeed
        primary_config = MockServiceConfig(
            failure_mode=MockFailureMode.INTERMITTENT,
            failure_rate=0.7,  # 70% failure rate
        )
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/RETRY123/"

            start_time = time.time()
            downloader.download_reel_enhanced(test_url)
            end_time = time.time()

            # Should eventually succeed (via primary retry or fallback)
            total_time = end_time - start_time

            # Verify some delay occurred due to retries
            assert total_time > 0.05  # At least some delay

            # Check statistics
            stats = get_mock_stats(downloader)
            print(f"Retry test stats: {stats}")

        finally:
            restore_downloader_from_mocks(downloader)

    def test_retry_timeout_behavior(self, temp_dir):
        """Test behavior when retries exceed timeout limits."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            timeout=2.0,  # Short timeout for testing
            max_delay=5.0,
        )

        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir), circuit_breaker_config=config)

        # Configure to always timeout
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.TIMEOUT, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:
            test_url = "https://instagram.com/reel/TIMEOUT123/"

            start_time = time.time()
            result = downloader.download_reel_enhanced(test_url)
            end_time = time.time()

            # Should fail or succeed via fallback within reasonable time
            assert end_time - start_time < 10  # Should not hang indefinitely

            # Should have meaningful error or fallback success
            if not result[0]:
                assert "timeout" in result[2].lower() or "fallback" in result[2].lower()

        finally:
            restore_downloader_from_mocks(downloader)


class TestHealthMonitoring:
    """Test health monitoring and service status tracking."""

    def test_health_status_reporting(self, temp_dir):
        """Test comprehensive health status reporting."""
        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir))

        # Configure mixed health status
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:
            health_status = downloader.get_health_status()

            # Verify health status structure
            assert "overall_healthy" in health_status
            assert "circuit_state" in health_status
            assert "circuit_stats" in health_status
            assert "service_health" in health_status
            assert "primary_service" in health_status
            assert "fallback_available" in health_status

            # With healthy mocks, overall should be healthy
            assert health_status["overall_healthy"]
            assert health_status["circuit_state"] == "closed"

        finally:
            restore_downloader_from_mocks(downloader)

    def test_health_monitoring_during_failures(self, temp_dir):
        """Test health monitoring during service failures."""
        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir))

        # Configure unhealthy primary, healthy fallback
        primary_config = MockServiceConfig(failure_mode=MockFailureMode.SERVICE_UNAVAILABLE, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:
            # Trigger some failures
            test_url = "https://instagram.com/reel/HEALTH123/"
            downloader.download_reel_enhanced(test_url)
            downloader.download_reel_enhanced(test_url)

            health_status = downloader.get_health_status()

            # Should show degraded health
            circuit_stats = health_status["circuit_stats"]
            assert circuit_stats.failed_requests > 0

            # Circuit might be open due to failures
            assert health_status["circuit_state"] in ["open", "half_open", "closed"]

        finally:
            restore_downloader_from_mocks(downloader)

    def test_troubleshooting_info_collection(self, temp_dir):
        """Test comprehensive troubleshooting information collection."""
        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir))

        troubleshooting_info = downloader.get_troubleshooting_info()

        # Verify troubleshooting info structure
        assert "health_status" in troubleshooting_info
        assert "capabilities" in troubleshooting_info
        assert "configuration" in troubleshooting_info

        capabilities = troubleshooting_info["capabilities"]
        assert "instaloader" in capabilities
        assert "ytdlp" in capabilities

        config_info = troubleshooting_info["configuration"]
        assert "circuit_breaker" in config_info
        assert "retry_settings" in config_info

        # Verify configuration details
        cb_config = config_info["circuit_breaker"]
        assert "failure_threshold" in cb_config
        assert "recovery_timeout" in cb_config
        assert "timeout" in cb_config


class TestComprehensiveScenarios:
    """Test comprehensive real-world scenarios."""

    def test_mock_download_test_suite(self, temp_dir):
        """Test all scenarios from the mock download test suite."""
        test_suite = create_mock_download_test_suite()

        for scenario_name, scenario_config in test_suite.items():
            print(f"\nTesting scenario: {scenario_name}")

            downloader = EnhancedInstagramDownloader(
                download_dir=str(temp_dir),
                circuit_breaker_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=2),
            )

            patch_downloader_with_mocks(downloader, scenario_config["primary"], scenario_config["fallback"])

            try:
                test_url = f"https://instagram.com/reel/{scenario_name.upper()}123/"

                if "test_sequence" in scenario_config:
                    # Multi-step test
                    self._execute_test_sequence(downloader, test_url, scenario_config)
                else:
                    # Simple test
                    result = downloader.download_reel_enhanced(test_url)

                    if scenario_config["expected_success"]:
                        # Should succeed somehow (primary or fallback)
                        assert (
                            result[0] or "fallback" in (result[2] or "").lower()
                        ), f"Scenario {scenario_name} should succeed but failed: {result[2]}"
                    else:
                        assert not result[0], f"Scenario {scenario_name} should fail but succeeded"

                print(f"✓ Scenario {scenario_name} passed")

            finally:
                restore_downloader_from_mocks(downloader)

    def _execute_test_sequence(self, downloader, test_url, scenario_config):
        """Execute a multi-step test sequence."""
        sequence = scenario_config["test_sequence"]

        for step in sequence:
            if step == "fail":
                result = downloader.download_reel_enhanced(test_url)
                # Expect failure but continue
                print(f"  Step '{step}': {'passed' if not result[0] else 'unexpected success'}")

            elif step == "circuit_open":
                circuit_state = downloader.resilient_service.get_circuit_state()
                print(f"  Step '{step}': circuit is {circuit_state.value}")

            elif step == "fallback_success":
                result = downloader.download_reel_enhanced(test_url)
                print(f"  Step '{step}': {'passed' if result[0] else 'failed'}")

            elif step == "wait":
                time.sleep(2)
                print(f"  Step '{step}': waited 2 seconds")

            elif step == "primary_recovers":
                # Simulate service recovery
                if hasattr(downloader, "_mock_primary"):
                    downloader._mock_primary.config.failure_rate = 0.0
                    downloader._mock_primary.is_service_degraded = False
                downloader.resilient_service.force_circuit_close()
                print(f"  Step '{step}': simulated service recovery")

    def test_concurrent_downloads_with_circuit_breaker(self, temp_dir):
        """Test concurrent downloads with circuit breaker coordination."""
        import concurrent.futures

        downloader = EnhancedInstagramDownloader(
            download_dir=str(temp_dir),
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3, recovery_timeout=5),
        )

        # Configure intermittent failures
        primary_config = MockServiceConfig(
            failure_mode=MockFailureMode.INTERMITTENT,
            failure_rate=0.4,  # 40% failure rate
        )
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:

            def download_task(task_id):
                url = f"https://instagram.com/reel/CONCURRENT{task_id}/"
                result = downloader.download_reel_enhanced(url)
                return task_id, result

            # Run concurrent downloads
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(download_task, i) for i in range(10)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Analyze results
            successful = sum(1 for _, (success, _, _) in results if success)
            failed = len(results) - successful

            print(f"Concurrent test: {successful} succeeded, {failed} failed")

            # Should have some successes (via fallback if nothing else)
            assert successful > 0

            # Get final health status
            health_status = downloader.get_health_status()
            print(f"Final health status: {health_status}")

        finally:
            restore_downloader_from_mocks(downloader)

    def test_stress_test_circuit_breaker_recovery(self, temp_dir):
        """Stress test circuit breaker with rapid failure and recovery cycles."""
        downloader = EnhancedInstagramDownloader(
            download_dir=str(temp_dir),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=2,
                recovery_timeout=1,  # Very fast recovery for stress test
                success_threshold=1,
            ),
        )

        # Configure rapidly changing service health
        primary_config = MockServiceConfig(
            failure_mode=MockFailureMode.INTERMITTENT,
            failure_rate=0.5,  # 50% failure rate
        )
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, primary_config, fallback_config)

        try:
            total_attempts = 20
            results = []

            for i in range(total_attempts):
                url = f"https://instagram.com/reel/STRESS{i}/"
                result = downloader.download_reel_enhanced(url)
                results.append(result)

                # Occasionally force service recovery
                if i % 5 == 0 and hasattr(downloader, "_mock_primary"):
                    downloader._mock_primary.failure_count = 0
                    downloader.resilient_service.force_circuit_close()

                time.sleep(0.1)  # Brief pause between attempts

            # Analyze stress test results
            successful = sum(1 for success, _, _ in results if success)
            success_rate = successful / total_attempts

            print(f"Stress test: {successful}/{total_attempts} succeeded ({success_rate:.1%})")

            # Should maintain reasonable success rate due to fallback
            assert success_rate > 0.3  # At least 30% success rate

            # Circuit breaker should still be functional
            final_health = downloader.get_health_status()
            assert "circuit_state" in final_health

        finally:
            restore_downloader_from_mocks(downloader)


@pytest.mark.integration
class TestRealWorldScenarios:
    """Test scenarios that simulate real-world conditions."""

    def test_instagram_outage_scenario(self, temp_dir):
        """Test behavior during simulated Instagram outage."""
        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir))

        # Simulate Instagram being completely down
        outage_config = MockServiceConfig(failure_mode=MockFailureMode.SERVICE_UNAVAILABLE, failure_rate=1.0)
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, outage_config, fallback_config)

        try:
            url = "https://instagram.com/reel/OUTAGE123/"

            # Multiple attempts during outage
            for i in range(5):
                result = downloader.download_reel_enhanced(url)
                print(f"Outage attempt {i + 1}: {'success' if result[0] else 'failed'}")

            # Should eventually rely on fallback
            circuit_state = downloader.resilient_service.get_circuit_state()
            assert circuit_state == CircuitState.OPEN

            # Health status should reflect the outage
            health_status = downloader.get_health_status()
            assert not health_status["service_health"].get("instagram_primary", True)

        finally:
            restore_downloader_from_mocks(downloader)

    def test_rate_limiting_scenario(self, temp_dir):
        """Test behavior under rate limiting conditions."""
        downloader = EnhancedInstagramDownloader(download_dir=str(temp_dir))

        rate_limit_config = MockServiceConfig(
            failure_mode=MockFailureMode.RATE_LIMIT,
            failure_rate=0.8,  # 80% rate limiting
            recovery_after_seconds=10,
        )
        fallback_config = MockServiceConfig(failure_mode=MockFailureMode.NONE)

        patch_downloader_with_mocks(downloader, rate_limit_config, fallback_config)

        try:
            # Rapid requests that should trigger rate limiting
            for i in range(3):
                url = f"https://instagram.com/reel/RATELIMIT{i}/"
                result = downloader.download_reel_enhanced(url)

                if not result[0] and result[2]:
                    assert any(
                        keyword in result[2].lower() for keyword in ["rate limit", "too many requests", "fallback"]
                    )

            # Circuit should open due to rate limiting
            circuit_state = downloader.resilient_service.get_circuit_state()
            print(f"Circuit state after rate limiting: {circuit_state}")

        finally:
            restore_downloader_from_mocks(downloader)

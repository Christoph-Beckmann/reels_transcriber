"""
Integration tests using real Instagram URLs from mock_data.py.
Tests end-to-end functionality with actual Instagram content.
"""

import os
import sys
import time
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.downloader import InstagramDownloader
from core.file_manager import TempFileManager
from core.pipeline import TranscriptionPipeline
from core.transcriber import TranscriptionResult
from tests.fixtures.mock_data import MockInstagramData
from utils.logging_config import LoggingConfig
from utils.progress import ProgressTracker


class TestRealInstagramUrls:
    """Test suite for real Instagram URL processing."""

    @pytest.fixture
    def setup_pipeline(self, tmp_path):
        """Set up pipeline with test configuration."""
        # Configure logging
        LoggingConfig.setup_logging(log_level="INFO", log_dir=str(tmp_path / "logs"))

        # Create file manager
        file_manager = TempFileManager(base_temp_dir=str(tmp_path / "temp"), cleanup_on_exit=True)

        # Create progress tracker
        progress_tracker = ProgressTracker()

        # Create pipeline with test configuration
        pipeline = TranscriptionPipeline(
            file_manager=file_manager,
            progress_callback=progress_tracker.update_progress,
            model_config={"model_size": "base", "device": "cpu", "compute_type": "int8"},
            audio_config={"sample_rate": 16000, "mono": True},
            enable_parallel=False,  # Disable for testing
        )

        yield pipeline, file_manager, progress_tracker

        # Cleanup
        try:
            pipeline.cleanup()
            file_manager.cleanup_all()
        except Exception as e:
            print(f"Cleanup error: {e}")

    def test_url_validation_with_real_urls(self, tmp_path):
        """Test URL validation with real Instagram URLs."""
        # Create a downloader instance for validation
        downloader = InstagramDownloader(download_dir=str(tmp_path))

        # Test valid reel URLs
        for url in MockInstagramData.SAMPLE_URLS["real_test_reels"]:
            is_valid, error, reel_id = downloader.validate_reel_url(url)
            assert is_valid is True, f"URL should be valid: {url}"
            assert error is None
            assert reel_id is not None
            print(f"✓ Valid reel URL: {url} -> ID: {reel_id}")

        # Test invalid URLs
        for url in MockInstagramData.SAMPLE_URLS["invalid_urls"]:
            is_valid, error, reel_id = downloader.validate_reel_url(url)
            assert is_valid is False, f"URL should be invalid: {url}"
            assert error is not None
            print(f"✓ Invalid URL rejected: {url}")

    @pytest.mark.network
    @pytest.mark.skipif(os.getenv("SKIP_NETWORK_TESTS", "false").lower() == "true", reason="Network tests disabled")
    def test_download_real_reel(self, setup_pipeline, tmp_path):
        """Test downloading a real Instagram reel."""
        pipeline, file_manager, progress_tracker = setup_pipeline

        # Use the first test URL
        test_url = MockInstagramData.SAMPLE_URLS["real_test_reels"][0]
        print(f"\nTesting download of: {test_url}")

        downloader = InstagramDownloader(output_dir=str(tmp_path / "downloads"), file_manager=file_manager)

        # Validate URL first
        is_valid, error, reel_id = downloader.validate_reel_url(test_url)
        assert is_valid is True, f"Failed to validate URL: {error}"

        # Attempt download (may fail due to network/auth requirements)
        try:
            success, video_path, error = downloader.download_reel(
                test_url, progress_callback=progress_tracker.update_progress
            )

            if success:
                assert video_path is not None
                assert Path(video_path).exists()
                print(f"✓ Successfully downloaded to: {video_path}")
            else:
                # Download may fail due to Instagram restrictions
                print(f"⚠ Download failed (expected): {error}")
                pytest.skip(f"Instagram download not available: {error}")

        except Exception as e:
            print(f"⚠ Download exception (expected): {e}")
            pytest.skip(f"Instagram download not available: {str(e)}")

    @pytest.mark.slow
    @pytest.mark.network
    @pytest.mark.skipif(os.getenv("SKIP_NETWORK_TESTS", "false").lower() == "true", reason="Network tests disabled")
    def test_end_to_end_transcription(self, setup_pipeline):
        """Test complete end-to-end transcription with real URL."""
        pipeline, file_manager, progress_tracker = setup_pipeline

        # Use test URL
        test_url = MockInstagramData.SAMPLE_URLS["real_test_reels"][0]
        metadata = MockInstagramData.REAL_URL_METADATA.get(test_url, {})

        print(f"\nEnd-to-end test for: {test_url}")
        print(f"Expected characteristics: {metadata}")

        try:
            # Run transcription
            result = pipeline.transcribe_reel(test_url)

            if result["success"]:
                assert "transcription" in result
                assert result["transcription"] is not None

                transcription = result["transcription"]
                print("\n✓ Transcription successful!")
                print(f"  - Text length: {len(transcription.text)} chars")
                print(f"  - Language: {transcription.language}")
                print(f"  - Segments: {len(transcription.segments)}")
                print(f"  - Duration: {transcription.metadata.get('audio_duration', 0):.1f}s")

                # Validate against expected characteristics
                if "expected_duration_range" in metadata:
                    duration = transcription.metadata.get("audio_duration", 0)
                    min_dur, max_dur = metadata["expected_duration_range"]
                    assert (
                        min_dur <= duration <= max_dur
                    ), f"Duration {duration} outside expected range {metadata['expected_duration_range']}"

            else:
                # Transcription may fail due to network/auth
                error = result.get("error", "Unknown error")
                print(f"⚠ Transcription failed (may be expected): {error}")

                # Check if it's a known/expected failure
                if "download" in error.lower() or "network" in error.lower():
                    pytest.skip(f"Expected network-related failure: {error}")
                else:
                    # Unexpected failure
                    pytest.fail(f"Unexpected transcription failure: {error}")

        except Exception as e:
            print(f"⚠ Exception during transcription: {e}")
            if "network" in str(e).lower() or "connection" in str(e).lower():
                pytest.skip(f"Network-related exception: {e}")
            else:
                raise

    def test_mock_url_processing(self, setup_pipeline, tmp_path):
        """Test processing with mock URLs (no network required)."""
        pipeline, file_manager, progress_tracker = setup_pipeline
        downloader = InstagramDownloader(download_dir=str(tmp_path))

        # Test with mock URLs that don't require network
        for url in MockInstagramData.SAMPLE_URLS["valid_reels_mock"]:
            is_valid, error, reel_id = downloader.validate_reel_url(url)
            assert is_valid is True
            assert reel_id is not None
            print(f"✓ Mock URL validated: {url} -> ID: {reel_id}")

    def test_performance_characteristics(self):
        """Test performance characteristics match expectations."""
        from tests.fixtures.mock_data import MockInstagramData

        # Verify test data structures
        assert len(MockInstagramData.SAMPLE_URLS["real_test_reels"]) == 4
        assert len(MockInstagramData.REAL_URL_METADATA) == 4

        # Check metadata completeness
        for url in MockInstagramData.SAMPLE_URLS["real_test_reels"]:
            metadata = MockInstagramData.REAL_URL_METADATA.get(url)
            assert metadata is not None, f"Missing metadata for {url}"
            assert "expected_category" in metadata
            assert "test_scenarios" in metadata
            assert "privacy_notes" in metadata
            print(f"✓ Metadata complete for: {metadata['expected_category']}")


class TestMockDataIntegration:
    """Test integration with mock data generators."""

    def test_mock_audio_generation(self, tmp_path):
        """Test mock audio file generation."""
        from tests.fixtures.mock_data import MockDataGenerator

        # Generate audio with speech
        audio_file = MockDataGenerator.create_realistic_audio_file(
            str(tmp_path / "test_speech.wav"), duration=3.0, include_speech=True
        )

        assert Path(audio_file).exists()
        assert Path(audio_file).stat().st_size > 0
        print(f"✓ Generated speech audio: {audio_file}")

        # Generate silence
        silent_file = MockDataGenerator.create_realistic_audio_file(
            str(tmp_path / "test_silent.wav"), duration=2.0, include_speech=False
        )

        assert Path(silent_file).exists()
        print(f"✓ Generated silent audio: {silent_file}")

    def test_mock_transcription_result(self):
        """Test mock transcription result generation."""
        from tests.fixtures.mock_data import MockDataGenerator

        text = "This is a test transcription with multiple segments."
        result = MockDataGenerator.create_mock_transcription_result(text=text, language="en", model_size="base")

        assert isinstance(result, TranscriptionResult)
        assert result.text == text
        assert result.language == "en"
        assert len(result.segments) > 0
        assert result.metadata["model_size"] == "base"
        print(f"✓ Generated mock transcription with {len(result.segments)} segments")

    def test_test_category_definitions(self):
        """Test that test categories are properly defined."""
        categories = MockInstagramData.TEST_CATEGORIES

        # Verify all categories have required fields
        for category_name, category_data in categories.items():
            assert "urls" in category_data
            assert "purpose" in category_data
            assert "expected_success" in category_data
            assert "timeout_seconds" in category_data
            assert len(category_data["urls"]) > 0
            print(f"✓ Category '{category_name}': {category_data['purpose']}")

        # Verify URL references are valid
        all_defined_urls = []
        for url_list in MockInstagramData.SAMPLE_URLS.values():
            all_defined_urls.extend(url_list)

        for category_data in categories.values():
            for url in category_data["urls"]:
                assert url in all_defined_urls, f"URL {url} not defined in SAMPLE_URLS"


@pytest.mark.benchmark
class TestPerformanceWithRealData:
    """Performance benchmarks using real test data."""

    def test_validation_performance_real_urls(self, benchmark, tmp_path):
        """Benchmark URL validation with real URLs."""
        urls = MockInstagramData.SAMPLE_URLS["real_test_reels"]
        downloader = InstagramDownloader(download_dir=str(tmp_path))

        def validate_all():
            results = []
            for url in urls:
                is_valid, error, reel_id = downloader.validate_reel_url(url)
                results.append((is_valid, reel_id))
            return results

        results = benchmark(validate_all)
        assert len(results) == len(urls)
        assert all(r[0] for r in results)  # All should be valid

    def test_mock_data_generation_performance(self, benchmark, tmp_path):
        """Benchmark mock data generation."""
        from tests.fixtures.mock_data import MockDataGenerator

        def generate_test_data():
            # Generate audio file
            audio = MockDataGenerator.create_realistic_audio_file(
                str(tmp_path / f"perf_test_{time.time()}.wav"), duration=1.0, include_speech=True
            )
            # Generate transcription
            result = MockDataGenerator.create_mock_transcription_result("Performance test text", language="en")
            return audio, result

        audio, result = benchmark(generate_test_data)
        assert Path(audio).exists()
        assert result is not None


if __name__ == "__main__":
    # Run tests with specific markers
    import subprocess

    # Run only non-network tests by default
    print("Running Instagram URL integration tests...")
    subprocess.run(["pytest", __file__, "-v", "-m", "not network", "--tb=short"])

    # To run network tests, set SKIP_NETWORK_TESTS=false
    if os.getenv("RUN_NETWORK_TESTS", "false").lower() == "true":
        print("\nRunning network tests...")
        subprocess.run(["pytest", __file__, "-v", "-m", "network", "--tb=short"])

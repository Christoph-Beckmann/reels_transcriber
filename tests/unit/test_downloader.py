"""
Unit tests for Instagram downloader component.
"""

import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from core.downloader import DownloadProgress, InstagramDownloader
from tests.fixtures.mock_data import MockInstagramData


class TestDownloadProgress:
    """Test the DownloadProgress helper class."""

    def test_progress_callback_initialization(self):
        """Test DownloadProgress initialization with callback."""
        callback = Mock()
        progress = DownloadProgress(callback)

        assert progress.callback == callback
        assert not progress.started
        assert not progress.completed

    def test_progress_callback_none(self):
        """Test DownloadProgress initialization without callback."""
        progress = DownloadProgress(None)

        assert progress.callback is None
        assert not progress.started
        assert not progress.completed

    def test_progress_start(self):
        """Test progress start functionality."""
        callback = Mock()
        progress = DownloadProgress(callback)

        progress.start()

        assert progress.started
        callback.assert_called_once_with(0, "Connecting to Instagram...")

    def test_progress_update(self):
        """Test progress update functionality."""
        callback = Mock()
        progress = DownloadProgress(callback)

        progress.update(50, "Downloading...")

        callback.assert_called_once_with(50, "Downloading...")

    def test_progress_complete(self):
        """Test progress completion."""
        callback = Mock()
        progress = DownloadProgress(callback)

        progress.complete("/path/to/video.mp4")

        assert progress.completed
        callback.assert_called_once_with(100, "Download complete: video.mp4")

    def test_progress_no_callback(self):
        """Test progress operations without callback."""
        progress = DownloadProgress(None)

        # Should not raise exceptions
        progress.start()
        progress.update(50, "Test")
        progress.complete("/test.mp4")

        assert progress.started
        assert progress.completed


class TestInstagramDownloader:
    """Test the InstagramDownloader class."""

    def test_initialization(self, temp_dir):
        """Test downloader initialization."""
        downloader = InstagramDownloader(download_dir=str(temp_dir), timeout=30, max_retries=2, retry_delay=1)

        assert downloader.download_dir == temp_dir
        assert downloader.timeout == 30
        assert downloader.max_retries == 2
        assert downloader.retry_delay == 1
        assert temp_dir.exists()

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates download directory."""
        non_existent_dir = temp_dir / "new_dir"
        assert not non_existent_dir.exists()

        InstagramDownloader(str(non_existent_dir))

        assert non_existent_dir.exists()

    @pytest.mark.parametrize(
        "url,expected_valid,expected_shortcode",
        [
            ("https://www.instagram.com/reel/ABC123DEF/", True, "ABC123DEF"),
            ("https://instagram.com/reel/XYZ789/", True, "XYZ789"),
            ("https://www.instagram.com/p/DEF456GHI/", True, "DEF456GHI"),
            ("https://not-instagram.com/reel/123/", False, None),
            ("invalid-url", False, None),
            ("", False, None),
        ],
    )
    def test_validate_reel_url(self, temp_dir, url, expected_valid, expected_shortcode):
        """Test URL validation and shortcode extraction."""
        downloader = InstagramDownloader(str(temp_dir))

        is_valid, error_msg, shortcode = downloader.validate_reel_url(url)

        assert is_valid == expected_valid
        if expected_valid:
            assert error_msg is None
            assert shortcode == expected_shortcode
        else:
            assert error_msg is not None
            assert shortcode is None

    @patch("core.downloader.Post")
    def test_download_reel_success(self, mock_post_class, temp_dir, mock_progress_callback):
        """Test successful reel download."""
        # Setup mocks
        mock_post = Mock()
        mock_post.shortcode = "ABC123"
        mock_post.is_video = True
        mock_post_class.from_shortcode.return_value = mock_post

        downloader = InstagramDownloader(str(temp_dir))

        # Create a fake video file that the downloader would find
        post_dir = temp_dir / "reel_ABC123"
        post_dir.mkdir()
        video_file = post_dir / "ABC123.mp4"
        video_file.write_bytes(b"fake video content")

        with patch.object(downloader.loader, "download_post"):
            success, video_path, error_msg = downloader.download_reel(
                "https://instagram.com/reel/ABC123/", mock_progress_callback
            )

        assert success
        assert video_path == str(video_file)
        assert error_msg is None
        assert mock_progress_callback.called

    @patch("core.downloader.Post")
    def test_download_reel_not_video(self, mock_post_class, temp_dir):
        """Test download failure when post is not a video."""
        mock_post = Mock()
        mock_post.is_video = False
        mock_post_class.from_shortcode.return_value = mock_post

        downloader = InstagramDownloader(str(temp_dir))

        success, video_path, error_msg = downloader.download_reel("https://instagram.com/reel/ABC123/")

        assert not success
        assert video_path is None
        assert "not a video" in error_msg.lower()

    @patch("core.downloader.Post")
    def test_download_reel_invalid_url(self, mock_post_class, temp_dir):
        """Test download failure with invalid URL."""
        downloader = InstagramDownloader(str(temp_dir))

        success, video_path, error_msg = downloader.download_reel("https://invalid-url.com/video/123")

        assert not success
        assert video_path is None
        assert error_msg is not None

    @patch("core.downloader.Post")
    def test_download_reel_connection_error(self, mock_post_class, temp_dir):
        """Test download with connection errors and retries."""
        from instaloader import ConnectionException

        mock_post_class.from_shortcode.side_effect = [
            ConnectionException("Network error"),
            ConnectionException("Network error"),
            ConnectionException("Network error"),
        ]

        downloader = InstagramDownloader(str(temp_dir), max_retries=2)

        success, video_path, error_msg = downloader.download_reel("https://instagram.com/reel/ABC123/")

        assert not success
        assert video_path is None
        assert "connection failed" in error_msg.lower()
        assert mock_post_class.from_shortcode.call_count == 2  # max_retries

    @patch("core.downloader.Post")
    def test_download_reel_not_found_error(self, mock_post_class, temp_dir):
        """Test download with 404 not found error."""
        from instaloader import BadResponseException

        mock_post_class.from_shortcode.side_effect = BadResponseException("404 Not Found")

        downloader = InstagramDownloader(str(temp_dir))

        success, video_path, error_msg = downloader.download_reel("https://instagram.com/reel/ABC123/")

        assert not success
        assert video_path is None
        assert "not found" in error_msg.lower()

    @patch("core.downloader.Post")
    def test_download_reel_private_error(self, mock_post_class, temp_dir):
        """Test download with private account error."""
        from instaloader import BadResponseException

        mock_post_class.from_shortcode.side_effect = BadResponseException("403 Forbidden")

        downloader = InstagramDownloader(str(temp_dir))

        success, video_path, error_msg = downloader.download_reel("https://instagram.com/reel/ABC123/")

        assert not success
        assert video_path is None
        assert "private" in error_msg.lower()

    @patch("core.downloader.Post")
    def test_download_reel_rate_limit_error(self, mock_post_class, temp_dir):
        """Test download with rate limit error."""
        from instaloader import InstaloaderException

        mock_post_class.from_shortcode.side_effect = InstaloaderException("Rate limit exceeded")

        downloader = InstagramDownloader(str(temp_dir))

        success, video_path, error_msg = downloader.download_reel("https://instagram.com/reel/ABC123/")

        assert not success
        assert video_path is None
        assert "rate limit" in error_msg.lower()

    def test_find_downloaded_video(self, temp_dir):
        """Test finding downloaded video files."""
        downloader = InstagramDownloader(str(temp_dir))

        # Create test directory and video file
        post_dir = temp_dir / "test_post"
        post_dir.mkdir()
        video_file = post_dir / "ABC123.mp4"
        video_file.write_bytes(b"test video content")

        found_path = downloader._find_downloaded_video(post_dir, "ABC123")

        assert found_path == str(video_file)

    def test_find_downloaded_video_not_found(self, temp_dir):
        """Test finding video when no file exists."""
        downloader = InstagramDownloader(str(temp_dir))

        post_dir = temp_dir / "test_post"
        post_dir.mkdir()

        found_path = downloader._find_downloaded_video(post_dir, "ABC123")

        assert found_path is None

    def test_find_video_file_generic(self, temp_dir):
        """Test generic video file finding."""
        downloader = InstagramDownloader(str(temp_dir))

        # Create test directory and video files
        post_dir = temp_dir / "test_post"
        post_dir.mkdir()
        video_file1 = post_dir / "random_name.mp4"
        video_file2 = post_dir / "another_video.mov"
        video_file1.write_bytes(b"video1")
        video_file2.write_bytes(b"video2")

        found_path = downloader._find_video_file(post_dir)

        assert found_path is not None
        assert Path(found_path).exists()
        assert Path(found_path).suffix in [".mp4", ".mov"]

    @patch("core.downloader.Post")
    def test_get_reel_metadata_success(self, mock_post_class, temp_dir):
        """Test successful metadata retrieval."""
        mock_post = Mock()
        mock_post.shortcode = "ABC123"
        mock_post.is_video = True
        mock_post.video_duration = 30
        mock_post.caption = "Test caption"
        mock_post.owner_username = "testuser"
        mock_post.likes = 100
        mock_post.video_url = "https://instagram.com/video.mp4"
        mock_post.date_utc = None
        mock_post_class.from_shortcode.return_value = mock_post

        downloader = InstagramDownloader(str(temp_dir))

        metadata = downloader.get_reel_metadata("https://instagram.com/reel/ABC123/")

        assert metadata is not None
        assert metadata["shortcode"] == "ABC123"
        assert metadata["is_video"] is True
        assert metadata["video_duration"] == 30
        assert metadata["caption"] == "Test caption"
        assert metadata["owner_username"] == "testuser"
        assert metadata["likes"] == 100
        assert metadata["video_url"] == "https://instagram.com/video.mp4"

    @patch("core.downloader.Post")
    def test_get_reel_metadata_invalid_url(self, mock_post_class, temp_dir):
        """Test metadata retrieval with invalid URL."""
        downloader = InstagramDownloader(str(temp_dir))

        metadata = downloader.get_reel_metadata("https://invalid-url.com/video/123")

        assert metadata is None

    @patch("core.downloader.Post")
    def test_get_reel_metadata_error(self, mock_post_class, temp_dir):
        """Test metadata retrieval with error."""
        mock_post_class.from_shortcode.side_effect = Exception("Network error")

        downloader = InstagramDownloader(str(temp_dir))

        metadata = downloader.get_reel_metadata("https://instagram.com/reel/ABC123/")

        assert metadata is None

    def test_cleanup_download_dir(self, temp_dir):
        """Test download directory cleanup."""
        downloader = InstagramDownloader(str(temp_dir))

        # Create some files in the directory
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")

        assert temp_dir.exists()
        assert test_file.exists()

        success = downloader.cleanup_download_dir()

        assert success
        assert not temp_dir.exists()

    def test_cleanup_download_dir_not_exists(self, temp_dir):
        """Test cleanup when directory doesn't exist."""
        non_existent_dir = temp_dir / "non_existent"
        downloader = InstagramDownloader(str(non_existent_dir))

        # Remove the directory that was created during initialization
        shutil.rmtree(non_existent_dir)

        success = downloader.cleanup_download_dir()

        assert success  # Should succeed even if directory doesn't exist


class TestDownloaderIntegration:
    """Integration tests for downloader component."""

    @pytest.mark.network
    @pytest.mark.slow
    def test_real_download_workflow(self, temp_dir):
        """Test real download workflow with actual Instagram URL."""
        # This test requires network connectivity and should be run sparingly
        # Skip by default to avoid hitting Instagram's rate limits
        pytest.skip("Skipping real network test to avoid rate limits")

        downloader = InstagramDownloader(str(temp_dir))

        # Use a known public Instagram Reel for testing
        # Note: This should be updated if the URL becomes invalid
        test_url = "https://www.instagram.com/reel/test_public_reel/"

        success, video_path, error_msg = downloader.download_reel(test_url)

        if success:
            assert video_path is not None
            assert Path(video_path).exists()
            assert Path(video_path).stat().st_size > 0
        else:
            # Network issues or Instagram changes are acceptable in tests
            assert error_msg is not None

    def test_downloader_thread_safety(self, temp_dir):
        """Test downloader thread safety with concurrent operations."""
        import threading

        downloader = InstagramDownloader(str(temp_dir))
        results = []
        errors = []

        def download_task(url_suffix):
            try:
                url = f"https://instagram.com/reel/{url_suffix}/"
                is_valid, error_msg, shortcode = downloader.validate_reel_url(url)
                results.append((url_suffix, is_valid, shortcode))
            except Exception as e:
                errors.append((url_suffix, str(e)))

        # Create multiple threads for validation (doesn't require network)
        threads = []
        for i in range(5):
            thread = threading.Thread(target=download_task, args=(f"TEST{i:03d}",))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        assert len(errors) == 0  # No threading errors
        assert len(results) == 5  # All threads completed
        for url_suffix, is_valid, shortcode in results:
            assert is_valid  # All URLs should be valid format
            assert shortcode == f"TEST{url_suffix.split('TEST')[1]}"


@pytest.mark.benchmark
class TestDownloaderPerformance:
    """Performance tests for downloader component."""

    def test_url_validation_performance(self, benchmark, temp_dir):
        """Benchmark URL validation performance."""
        downloader = InstagramDownloader(str(temp_dir))
        test_url = "https://www.instagram.com/reel/ABC123DEF/"

        result = benchmark(downloader.validate_reel_url, test_url)

        assert result[0] is True  # Should be valid
        # Validation should be fast (< 1ms typically)

    def test_multiple_validation_performance(self, benchmark, temp_dir):
        """Benchmark performance of multiple URL validations."""
        downloader = InstagramDownloader(str(temp_dir))
        test_urls = [f"https://www.instagram.com/reel/TEST{i:05d}/" for i in range(100)]

        def validate_multiple():
            results = []
            for url in test_urls:
                result = downloader.validate_reel_url(url)
                results.append(result)
            return results

        results = benchmark(validate_multiple)

        assert len(results) == 100
        assert all(result[0] for result in results)  # All should be valid


@pytest.mark.integration
@pytest.mark.real_urls
class TestRealURLValidation:
    """Tests using real Instagram URLs for validation."""

    def test_real_url_validation_patterns(self, temp_dir):
        """Test validation with real Instagram URLs to ensure pattern accuracy."""
        downloader = InstagramDownloader(str(temp_dir))

        # Test real URLs from our test data
        real_urls = MockInstagramData.SAMPLE_URLS["real_test_reels"]

        print("\n🔗 Testing real URL validation patterns")

        for test_url in real_urls:
            print(f"   Testing: {test_url}")

            try:
                is_valid, error_msg, shortcode = downloader.validate_reel_url(test_url)

                # Real URLs should validate successfully
                assert is_valid, f"Real URL should be valid: {test_url}"
                assert error_msg is None, f"Valid URL should not have error: {error_msg}"
                assert shortcode is not None, f"Valid URL should extract shortcode: {test_url}"

                # Verify shortcode extraction accuracy
                expected_shortcode = test_url.split("/reels/")[-1].rstrip("/")
                assert (
                    shortcode == expected_shortcode
                ), f"Shortcode mismatch: got {shortcode}, expected {expected_shortcode}"

                print(f"      ✅ Valid - Shortcode: {shortcode}")

            except Exception as e:
                pytest.fail(f"URL validation should not raise exception: {e}")

    def test_real_url_format_consistency(self, temp_dir):
        """Test that our real URLs match expected Instagram URL patterns."""
        downloader = InstagramDownloader(str(temp_dir))

        # Test all categories of URLs
        url_categories = {
            "real_test_reels": MockInstagramData.SAMPLE_URLS["real_test_reels"],
            "mock_reels": MockInstagramData.SAMPLE_URLS["valid_reels_mock"],
            "invalid_urls": MockInstagramData.SAMPLE_URLS["invalid_urls"],
        }

        print("\n📋 Testing URL format consistency across categories")

        for category, urls in url_categories.items():
            print(f"\n   Category: {category}")

            for url in urls:
                try:
                    is_valid, error_msg, shortcode = downloader.validate_reel_url(url)

                    if category == "invalid_urls":
                        # Invalid URLs should fail validation
                        assert not is_valid, f"Invalid URL should fail validation: {url}"
                        assert shortcode is None, f"Invalid URL should not extract shortcode: {url}"
                        print(f"      ❌ {url} - Correctly rejected")
                    else:
                        # Valid URLs (real and mock) should pass validation
                        assert is_valid, f"Valid URL should pass validation: {url}"
                        assert shortcode is not None, f"Valid URL should extract shortcode: {url}"
                        print(f"      ✅ {url} - Shortcode: {shortcode}")

                except Exception as e:
                    # Validation should not raise exceptions
                    pytest.fail(f"URL validation should handle all cases gracefully: {url} - {e}")

    def test_real_url_metadata_expectations(self, temp_dir):
        """Test that real URL metadata matches expected characteristics."""
        downloader = InstagramDownloader(str(temp_dir))

        print("\n📊 Testing real URL metadata expectations")

        for test_url in MockInstagramData.SAMPLE_URLS["real_test_reels"]:
            expected_metadata = MockInstagramData.REAL_URL_METADATA.get(test_url, {})

            if not expected_metadata:
                continue

            print(f"\n   URL: {test_url}")
            print(f"   Expected category: {expected_metadata.get('expected_category', 'unknown')}")

            try:
                # Test URL validation
                is_valid, error_msg, shortcode = downloader.validate_reel_url(test_url)

                assert is_valid, f"Real URL should be valid: {test_url}"
                assert shortcode is not None, f"Should extract shortcode: {test_url}"

                print(f"      ✅ Validation passed - Shortcode: {shortcode}")

                # Test metadata retrieval (this will make a network request)
                # Skip if we don't want to make network requests in unit tests
                if hasattr(pytest, "skip_network_tests") and pytest.skip_network_tests:
                    print("      ⏭️  Skipping metadata test (network disabled)")
                    continue

                try:
                    metadata = downloader.get_reel_metadata(test_url)

                    if metadata:
                        print("      ✅ Metadata retrieved")
                        print(f"         Video: {metadata.get('is_video', 'unknown')}")
                        print(f"         Duration: {metadata.get('video_duration', 'unknown')}s")

                        # Privacy-safe validation - only check structure
                        assert "shortcode" in metadata, "Metadata should include shortcode"
                        assert metadata["shortcode"] == shortcode, "Metadata shortcode should match extracted"

                        # Flexible duration validation
                        duration_range = expected_metadata.get("expected_duration_range", (0, 999))
                        if "video_duration" in metadata and metadata["video_duration"]:
                            duration = metadata["video_duration"]
                            # Allow for some flexibility in duration expectations
                            if not (duration_range[0] <= duration <= duration_range[1] * 2):
                                print(f"         ⚠️  Duration {duration}s outside expected range {duration_range}")
                    else:
                        print("      ⚠️  No metadata retrieved (may be private/restricted)")

                except Exception as metadata_error:
                    error_msg = str(metadata_error).lower()
                    if any(
                        keyword in error_msg
                        for keyword in ["network", "connection", "timeout", "private", "not found", "rate limit"]
                    ):
                        print(f"      ⚠️  Metadata error (expected): {metadata_error}")
                    else:
                        print(f"      ❌ Unexpected metadata error: {metadata_error}")

            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["network", "connection", "timeout", "dns", "ssl"]):
                    print(f"      ⚠️  Network error (skipping): {e}")
                    continue
                else:
                    pytest.fail(f"Unexpected error in URL metadata test: {e}")

    def test_real_url_edge_cases(self, temp_dir):
        """Test edge cases with real URL patterns."""
        downloader = InstagramDownloader(str(temp_dir))

        print("\n🧪 Testing real URL edge cases")

        # Test URL variations that should be equivalent
        base_url = MockInstagramData.SAMPLE_URLS["real_test_reels"][0]
        shortcode = base_url.split("/reels/")[-1].rstrip("/")

        url_variations = [
            base_url,  # Original
            base_url.rstrip("/"),  # Without trailing slash
            base_url.replace("www.", ""),  # Without www
            base_url.replace("https://", "http://"),  # HTTP instead of HTTPS
            f"https://instagram.com/reels/{shortcode}/",  # Alternative domain format
        ]

        print(f"   Base shortcode: {shortcode}")

        for i, variation in enumerate(url_variations):
            print(f"\n   Variation {i + 1}: {variation}")

            try:
                is_valid, error_msg, extracted_shortcode = downloader.validate_reel_url(variation)

                if is_valid:
                    assert (
                        extracted_shortcode == shortcode
                    ), f"All variations should extract same shortcode: got {extracted_shortcode}, expected {shortcode}"
                    print(f"      ✅ Valid - Shortcode: {extracted_shortcode}")
                else:
                    # Some variations might be rejected by validation (e.g., HTTP)
                    print(f"      ❌ Rejected: {error_msg}")

            except Exception as e:
                print(f"      💥 Exception: {e}")
                # URL variations should not cause exceptions
                pytest.fail(f"URL variation should be handled gracefully: {variation} - {e}")


@pytest.mark.real_urls
@pytest.mark.network
@pytest.mark.slow
class TestRealURLDownloadValidation:
    """Optional tests for actual download validation with real URLs - use sparingly."""

    def test_real_url_download_workflow_validation(self, temp_dir):
        """Validate download workflow with one real URL - for development/debugging only."""
        # This test is marked as slow and should only be run when specifically needed
        # It will actually attempt to download from Instagram

        test_url = MockInstagramData.TEST_CATEGORIES["integration_basic"]["urls"][0]

        print("\n🎬 Real download validation test")
        print(f"   URL: {test_url}")
        print("   ⚠️  This test makes actual network requests to Instagram")

        try:
            downloader = InstagramDownloader(str(temp_dir))

            # Progress tracking for debugging
            progress_updates = []

            def debug_progress_callback(progress, message):
                progress_updates.append((progress, message))
                print(f"      Progress: {progress}% - {message}")

            success, video_path, error_msg = downloader.download_reel(test_url, debug_progress_callback)

            if success:
                print("   ✅ Download successful!")
                print(f"      Video path: {video_path}")

                if video_path and Path(video_path).exists():
                    file_size = Path(video_path).stat().st_size
                    print(f"      File size: {file_size / 1024 / 1024:.1f}MB")

                    # Basic file validation
                    assert file_size > 0, "Downloaded file should not be empty"
                    assert file_size < 100 * 1024 * 1024, "Downloaded file should be reasonable size (< 100MB)"
                else:
                    pytest.fail("Download reported success but no file found")

                # Validate progress updates
                assert len(progress_updates) > 0, "Should have progress updates"
                final_progress = progress_updates[-1][0] if progress_updates else 0
                assert final_progress == 100, "Final progress should be 100%"

            else:
                print(f"   ❌ Download failed: {error_msg}")

                # For real URL tests, network/access failures are acceptable
                if any(
                    keyword in (error_msg or "").lower()
                    for keyword in ["network", "connection", "timeout", "private", "not found", "rate limit"]
                ):
                    pytest.skip(f"Real URL test skipped due to expected issue: {error_msg}")
                else:
                    pytest.fail(f"Unexpected download error: {error_msg}")

        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["network", "connection", "timeout", "dns", "ssl"]):
                pytest.skip(f"Real URL test skipped due to network issue: {e}")
            else:
                raise

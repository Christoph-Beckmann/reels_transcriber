"""
Comprehensive test suite for URL validation utilities.

Tests cover Instagram URL validation, normalization, and ID extraction
with edge cases and error scenarios.
"""

from unittest.mock import patch

from utils.validators import (
    INSTAGRAM_REEL_PATTERN,
    INSTAGRAM_REEL_REGEX,
    extract_reel_id,
    is_instagram_reel_url,
    normalize_instagram_url,
    validate_instagram_url,
)


class TestInstagramReelUrlValidation:
    """Test Instagram Reel URL validation functionality."""

    def test_valid_reel_urls(self):
        """Test validation of valid Instagram Reel URLs."""
        valid_urls = [
            "https://www.instagram.com/reel/ABC123DEF/",
            "https://instagram.com/reel/XYZ789/",
            "http://www.instagram.com/reel/test123/",
            "https://www.instagram.com/reels/ABC123DEF/",  # plural form
            "https://instagram.com/reels/XYZ789/",
            "https://www.instagram.com/reel/A1B2C3D4E5F6/",
            "https://www.instagram.com/reel/test_reel-123/",
        ]

        for url in valid_urls:
            assert is_instagram_reel_url(url), f"Expected {url} to be valid"

    def test_invalid_reel_urls(self):
        """Test validation of invalid Instagram URLs."""
        invalid_urls = [
            "",  # empty
            None,  # None
            "not_a_url",
            "https://youtube.com/watch?v=123",
            "https://www.instagram.com/tv/ABC123DEF/",  # IGTV
            "https://www.instagram.com/stories/user/123/",  # story
            "https://www.instagram.com/user/",  # profile
            "https://www.instagram.com/",  # base domain
            "instagram.com/reel/ABC123/",  # missing protocol
            "https://facebook.com/reel/ABC123/",  # wrong domain
            "https://www.instagram.com/reel/",  # missing ID
        ]

        for url in invalid_urls:
            assert not is_instagram_reel_url(url), f"Expected {url} to be invalid"

    def test_url_input_types(self):
        """Test various input types for URL validation."""
        # Non-string inputs
        assert not is_instagram_reel_url(123)
        assert not is_instagram_reel_url([])
        assert not is_instagram_reel_url({})
        assert not is_instagram_reel_url(True)

    def test_url_whitespace_handling(self):
        """Test handling of URLs with whitespace."""
        url_with_spaces = "  https://www.instagram.com/reel/ABC123/  "
        assert is_instagram_reel_url(url_with_spaces)

        empty_after_strip = "   "
        assert not is_instagram_reel_url(empty_after_strip)


class TestValidateInstagramUrl:
    """Test comprehensive Instagram URL validation with error messages."""

    def test_validate_valid_reel_url(self):
        """Test validation of valid Reel URL."""
        url = "https://www.instagram.com/reel/ABC123DEF/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is True
        assert error is None

    def test_validate_empty_input(self):
        """Test validation of empty inputs."""
        test_cases = [
            ("", "Please enter a URL"),
            (None, "Please enter a URL"),
            ("   ", "Please enter a URL"),
        ]

        for url, expected_error in test_cases:
            is_valid, error = validate_instagram_url(url)
            assert is_valid is False
            assert error == expected_error

    def test_validate_non_string_input(self):
        """Test validation of non-string input."""
        is_valid, error = validate_instagram_url(123)
        assert is_valid is False
        assert error == "URL must be a text string"

    def test_validate_invalid_url_structure(self):
        """Test validation of invalid URL structures."""
        test_cases = [
            ("not_a_url", "URL must start with http:// or https://"),
            ("://missing_scheme", "Invalid URL format"),
            ("https://", "Invalid URL format"),
        ]

        for url, expected_error in test_cases:
            is_valid, error = validate_instagram_url(url)
            assert is_valid is False
            assert expected_error in error

    def test_validate_non_instagram_domain(self):
        """Test validation of non-Instagram domains."""
        url = "https://www.youtube.com/watch?v=123"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is False
        assert error == "URL must be from instagram.com"

    def test_validate_instagram_post_url(self):
        """Test validation of Instagram post URL (should be valid for video transcription)."""
        url = "https://www.instagram.com/p/ABC123DEF/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is True
        assert error is None

    def test_validate_instagram_tv_url(self):
        """Test validation of Instagram TV URL."""
        url = "https://www.instagram.com/tv/ABC123DEF/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is False
        assert "IGTV video, not a Reel" in error

    def test_validate_instagram_story_url(self):
        """Test validation of Instagram story URL."""
        url = "https://www.instagram.com/stories/user/123/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is False
        assert "Instagram story, not a Reel" in error

    def test_validate_instagram_profile_url(self):
        """Test validation of Instagram profile URL."""
        url = "https://www.instagram.com/username/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is False
        assert "valid Instagram Reel URL" in error

    @patch("utils.validators.logger")
    def test_validate_logs_successful_validation(self, mock_logger):
        """Test that successful validation is logged."""
        url = "https://www.instagram.com/reel/ABC123DEF/"
        is_valid, error = validate_instagram_url(url)

        assert is_valid is True
        mock_logger.info.assert_called_once()

    def test_validate_url_parsing_error(self):
        """Test handling of URL parsing errors."""
        # Create a malformed URL that might cause urlparse to fail
        with patch("utils.validators.urlparse", side_effect=Exception("Parse error")):
            is_valid, error = validate_instagram_url("https://example.com")
            assert is_valid is False
            assert error == "Invalid URL format"


class TestNormalizeInstagramUrl:
    """Test Instagram URL normalization functionality."""

    def test_normalize_http_to_https(self):
        """Test conversion of HTTP to HTTPS."""
        url = "http://www.instagram.com/reel/ABC123/"
        normalized = normalize_instagram_url(url)
        assert normalized.startswith("https://")

    def test_normalize_add_https_prefix(self):
        """Test adding HTTPS prefix when missing."""
        url = "www.instagram.com/reel/ABC123/"
        normalized = normalize_instagram_url(url)
        assert normalized.startswith("https://")

    def test_normalize_add_www_subdomain(self):
        """Test adding www subdomain when missing."""
        url = "https://instagram.com/reel/ABC123/"
        normalized = normalize_instagram_url(url)
        assert "https://www.instagram.com" in normalized

    def test_normalize_remove_query_parameters(self):
        """Test removal of query parameters."""
        url = "https://www.instagram.com/reel/ABC123/?utm_source=test&ref=share"
        normalized = normalize_instagram_url(url)
        assert "?" not in normalized
        assert "utm_source" not in normalized

    def test_normalize_remove_fragment(self):
        """Test removal of URL fragments."""
        url = "https://www.instagram.com/reel/ABC123/#fragment"
        normalized = normalize_instagram_url(url)
        assert "#" not in normalized

    def test_normalize_add_trailing_slash(self):
        """Test adding trailing slash when needed."""
        url = "https://www.instagram.com/reel"
        normalized = normalize_instagram_url(url)
        assert normalized.endswith("/")

    def test_normalize_complex_url(self):
        """Test normalization of complex URL with multiple issues."""
        url = "http://instagram.com/reel/ABC123?utm_source=test#fragment"
        normalized = normalize_instagram_url(url)

        assert normalized == "https://www.instagram.com/reel/ABC123"

    def test_normalize_whitespace_handling(self):
        """Test normalization handles whitespace."""
        url = "  https://www.instagram.com/reel/ABC123/  "
        normalized = normalize_instagram_url(url)
        assert not normalized.startswith(" ")
        assert not normalized.endswith(" ")

    def test_normalize_already_normalized_url(self):
        """Test normalization of already normalized URL."""
        url = "https://www.instagram.com/reel/ABC123/"
        normalized = normalize_instagram_url(url)
        assert normalized == url


class TestExtractReelId:
    """Test Reel ID extraction functionality."""

    def test_extract_valid_reel_id(self):
        """Test extraction of valid Reel IDs."""
        test_cases = [
            ("https://www.instagram.com/reel/ABC123DEF/", "ABC123DEF"),
            ("https://instagram.com/reel/XYZ789/", "XYZ789"),
            ("https://www.instagram.com/reels/TEST123/", "TEST123"),
            ("https://www.instagram.com/reel/A1B2C3D4E5F6", "A1B2C3D4E5F6"),
            ("https://www.instagram.com/reel/test_reel-123/", "test_reel-123"),
        ]

        for url, expected_id in test_cases:
            reel_id = extract_reel_id(url)
            assert reel_id == expected_id, f"Expected {expected_id} from {url}, got {reel_id}"

    def test_extract_reel_id_with_trailing_slash(self):
        """Test extraction handles URLs with and without trailing slash."""
        url_with_slash = "https://www.instagram.com/reel/ABC123/"
        url_without_slash = "https://www.instagram.com/reel/ABC123"

        id_with_slash = extract_reel_id(url_with_slash)
        id_without_slash = extract_reel_id(url_without_slash)

        assert id_with_slash == "ABC123"
        assert id_without_slash == "ABC123"

    def test_extract_reel_id_plural_form(self):
        """Test extraction from plural 'reels' URLs."""
        url = "https://www.instagram.com/reels/ABC123/"
        reel_id = extract_reel_id(url)
        assert reel_id == "ABC123"

    def test_extract_reel_id_invalid_urls(self):
        """Test extraction from invalid URLs returns None."""
        invalid_urls = [
            "https://www.instagram.com/reel/",  # missing ID
            "https://www.instagram.com/user/",  # not a reel
            "https://youtube.com/watch?v=123",  # wrong domain
            "",  # empty
        ]

        for url in invalid_urls:
            reel_id = extract_reel_id(url)
            assert reel_id is None, f"Expected None from {url}, got {reel_id}"

    def test_extract_reel_id_invalid_characters(self):
        """Test extraction rejects IDs with invalid characters."""
        urls_with_invalid_ids = [
            "https://www.instagram.com/reel/ABC@123/",  # contains @
            "https://www.instagram.com/reel/ABC 123/",  # contains space
            "https://www.instagram.com/reel/ABC#123/",  # contains #
            "https://www.instagram.com/reel/ABC%123/",  # contains %
        ]

        for url in urls_with_invalid_ids:
            reel_id = extract_reel_id(url)
            assert reel_id is None

    def test_extract_reel_id_with_query_params(self):
        """Test extraction from URLs with query parameters."""
        url = "https://www.instagram.com/reel/ABC123/?utm_source=test"
        reel_id = extract_reel_id(url)
        assert reel_id == "ABC123"

    @patch("utils.validators.logger")
    def test_extract_reel_id_exception_handling(self, mock_logger):
        """Test extraction handles exceptions gracefully."""
        # Simulate an exception during processing
        with patch("utils.validators.re.match", side_effect=Exception("Regex error")):
            reel_id = extract_reel_id("https://www.instagram.com/reel/ABC123/")
            assert reel_id is None
            mock_logger.warning.assert_called_once()

    def test_extract_reel_id_edge_cases(self):
        """Test extraction with edge cases."""
        edge_cases = [
            ("https://www.instagram.com/reel/1/", "1"),  # single character
            ("https://www.instagram.com/reel/123456789012345/", "123456789012345"),  # long ID
            ("https://www.instagram.com/reel/A-B_C/", "A-B_C"),  # with dashes and underscores
        ]

        for url, expected_id in edge_cases:
            reel_id = extract_reel_id(url)
            assert reel_id == expected_id


class TestRegexPatterns:
    """Test regex patterns and compiled expressions."""

    def test_instagram_reel_pattern_constant(self):
        """Test the INSTAGRAM_REEL_PATTERN constant."""
        assert isinstance(INSTAGRAM_REEL_PATTERN, str)
        assert "instagram" in INSTAGRAM_REEL_PATTERN.lower()
        assert "reel" in INSTAGRAM_REEL_PATTERN.lower()

    def test_instagram_reel_regex_compiled(self):
        """Test the compiled regex pattern."""
        import re

        assert isinstance(INSTAGRAM_REEL_REGEX, re.Pattern)

        # Test pattern matches valid URLs
        valid_url = "https://www.instagram.com/reel/ABC123/"
        assert INSTAGRAM_REEL_REGEX.match(valid_url) is not None

        # Test pattern also matches post URLs (for video transcription)
        post_url = "https://www.instagram.com/p/ABC123/"
        assert INSTAGRAM_REEL_REGEX.match(post_url) is not None

        # Test pattern rejects truly invalid URLs
        invalid_url = "https://www.instagram.com/tv/ABC123/"
        assert INSTAGRAM_REEL_REGEX.match(invalid_url) is None


class TestValidationIntegration:
    """Integration tests for validation workflows."""

    def test_complete_validation_workflow(self):
        """Test complete URL validation and processing workflow."""
        # Start with a messy URL
        messy_url = "  http://instagram.com/reel/ABC123DEF?utm_source=test#fragment  "

        # Validate the URL
        is_valid, error = validate_instagram_url(messy_url)
        assert is_valid is True
        assert error is None

        # Normalize the URL
        normalized = normalize_instagram_url(messy_url)
        assert normalized == "https://www.instagram.com/reel/ABC123DEF"

        # Extract the Reel ID
        reel_id = extract_reel_id(normalized)
        assert reel_id == "ABC123DEF"

        # Verify the normalized URL is still valid
        is_valid_normalized, _ = validate_instagram_url(normalized)
        assert is_valid_normalized is True

    def test_error_handling_workflow(self):
        """Test error handling throughout validation workflow."""
        invalid_url = "https://www.youtube.com/watch?v=123"

        # Validation should fail
        is_valid, error = validate_instagram_url(invalid_url)
        assert is_valid is False
        assert error is not None

        # Normalization should still work (but result will be invalid)
        normalized = normalize_instagram_url(invalid_url)
        assert normalized.startswith("https://")

        # ID extraction should return None
        reel_id = extract_reel_id(invalid_url)
        assert reel_id is None

    def test_edge_case_workflow(self):
        """Test workflow with edge cases."""
        edge_case_url = "https://www.instagram.com/reels/A/"

        # Should be valid (single character ID)
        is_valid, error = validate_instagram_url(edge_case_url)
        assert is_valid is True

        # Should normalize correctly
        normalized = normalize_instagram_url(edge_case_url)
        assert "reels/A" in normalized

        # Should extract ID correctly
        reel_id = extract_reel_id(normalized)
        assert reel_id == "A"

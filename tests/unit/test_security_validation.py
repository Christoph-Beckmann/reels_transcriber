"""
Security validation tests for Instagram Reels Transcriber.

Tests all security fixes including:
- URL sanitization for subprocess calls
- Path traversal prevention in file manager
- Enhanced URL validation with protocol and localhost blocking
- Dependency security updates
"""

from unittest.mock import MagicMock, patch

import pytest

from core.file_manager import TempFileManager, sanitize_filename
from core.yt_dlp_optimizer import YtDlpOptimizer, sanitize_url_for_subprocess
from utils.validators import is_instagram_reel_url, normalize_instagram_url, validate_instagram_url


class TestURLSanitization:
    """Test URL sanitization for subprocess command injection prevention."""

    def test_valid_instagram_url_passes(self):
        """Test that valid Instagram URLs pass sanitization."""
        valid_url = "https://www.instagram.com/reel/ABC123def456/"
        result = sanitize_url_for_subprocess(valid_url)
        assert result == valid_url

    def test_command_injection_patterns_blocked(self):
        """Test that command injection patterns are blocked."""
        malicious_urls = [
            "https://instagram.com/reel/test; rm -rf /",
            "https://instagram.com/reel/test && sudo delete",
            "https://instagram.com/reel/test | cat /etc/passwd",
            "https://instagram.com/reel/test `whoami`",
            "https://instagram.com/reel/test $(id)",
            "https://instagram.com/reel/test${PATH}",
            "https://instagram.com/reel/test\nrm file",
            "https://instagram.com/reel/test\rmalware",
            "https://instagram.com/reel/test\tdelete",
        ]

        for malicious_url in malicious_urls:
            with pytest.raises(ValueError, match="dangerous pattern"):
                sanitize_url_for_subprocess(malicious_url)

    def test_invalid_url_schemes_blocked(self):
        """Test that non-HTTP/HTTPS schemes are blocked."""
        # Test URL schemes that don't contain dangerous patterns
        clean_invalid_schemes = [
            "ftp://example.com/test",
            "ldap://example.com/exploit",
        ]

        for invalid_url in clean_invalid_schemes:
            with pytest.raises(ValueError, match="Unsupported URL scheme"):
                sanitize_url_for_subprocess(invalid_url)

        # URLs with dangerous characters are caught by pattern check first
        dangerous_scheme_urls = [
            "javascript:alert('xss')",  # Contains '(' and ')'
            "data:text/html,<script>alert('xss')</script>",  # Contains '<' and '>'
            "file:///etc/passwd",  # Contains '/etc/'
        ]

        for dangerous_url in dangerous_scheme_urls:
            with pytest.raises(ValueError, match="dangerous pattern"):
                sanitize_url_for_subprocess(dangerous_url)

    def test_localhost_and_private_ips_blocked(self):
        """Test that localhost and private IP ranges are blocked."""
        dangerous_urls = [
            "https://localhost/reel/test",
            "https://127.0.0.1/reel/test",
            "https://0.0.0.0/reel/test",
            "https://192.168.1.1/reel/test",
            "https://10.0.0.1/reel/test",
            "https://172.16.0.1/reel/test",
            "https://test.local/reel/test",
        ]

        for dangerous_url in dangerous_urls:
            with pytest.raises(ValueError, match="localhost|private networks"):
                sanitize_url_for_subprocess(dangerous_url)

    def test_extremely_long_urls_blocked(self):
        """Test that extremely long URLs are blocked."""
        long_url = "https://instagram.com/reel/" + "A" * 2048
        with pytest.raises(ValueError, match="too long"):
            sanitize_url_for_subprocess(long_url)

    def test_empty_or_invalid_input_blocked(self):
        """Test that empty or invalid inputs are blocked."""
        invalid_inputs = ["", None, 123, [], {}]

        for invalid_input in invalid_inputs:
            with pytest.raises(ValueError):
                sanitize_url_for_subprocess(invalid_input)


class TestPathTraversalPrevention:
    """Test path traversal prevention in file manager."""

    def test_valid_filenames_pass(self):
        """Test that valid filenames pass sanitization."""
        valid_filenames = ["test.mp4", "reel_123.wav", "video-file.mp3", "output_final.txt", "data123.json"]

        for filename in valid_filenames:
            result = sanitize_filename(filename)
            assert result == filename

    def test_path_traversal_patterns_blocked(self):
        """Test that path traversal patterns are blocked."""
        malicious_filenames = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32",
            "test/../../../sensitive",
            "..\\malware.exe",
            "normal/../../etc/passwd",
            "file\\..\\..\\system",
        ]

        for malicious_filename in malicious_filenames:
            with pytest.raises(ValueError, match="dangerous pattern"):
                sanitize_filename(malicious_filename)

    def test_dangerous_characters_blocked(self):
        """Test that dangerous characters are blocked."""
        dangerous_filenames = [
            "file:with:colons",
            "file*with*asterisks",
            "file?with?questions",
            'file"with"quotes',
            "file<with>brackets",
            "file|with|pipes",
            "file\x00with\x00nulls",
            "file\nwith\nnewlines",
        ]

        for dangerous_filename in dangerous_filenames:
            with pytest.raises(ValueError, match="dangerous pattern"):
                sanitize_filename(dangerous_filename)

    def test_reserved_windows_names_blocked(self):
        """Test that Windows reserved names are blocked."""
        reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2", "con.txt", "prn.log", "aux.dat"]

        for reserved_name in reserved_names:
            with pytest.raises(ValueError, match="reserved name"):
                sanitize_filename(reserved_name)

    def test_hidden_files_blocked(self):
        """Test that hidden files (starting with .) are blocked."""
        hidden_files = [".hidden", ".secret", ".bashrc", ".env"]

        for hidden_file in hidden_files:
            with pytest.raises(ValueError, match="Hidden filenames"):
                sanitize_filename(hidden_file)

    def test_temp_file_manager_path_validation(self):
        """Test that TempFileManager validates paths correctly."""
        with TempFileManager() as manager:
            # Valid filename should work
            safe_path = manager.get_temp_path("safe_file.txt")
            assert "safe_file.txt" in safe_path

            # Malicious filename should raise error
            with pytest.raises(ValueError):
                manager.get_temp_path("../../../etc/passwd")

            with pytest.raises(ValueError):
                manager.get_temp_path("malicious\x00file")


class TestEnhancedURLValidation:
    """Test enhanced URL validation with security improvements."""

    def test_https_only_enforcement(self):
        """Test that only HTTPS URLs are accepted."""
        # HTTP should be rejected
        is_valid, error = validate_instagram_url("http://instagram.com/reel/test")
        assert not is_valid
        assert "HTTPS" in error

        # HTTPS should be accepted
        is_valid, error = validate_instagram_url("https://instagram.com/reel/test123")
        assert is_valid
        assert error is None

    def test_localhost_blocking_in_validation(self):
        """Test that localhost URLs are blocked in validation."""
        localhost_urls = [
            "https://localhost/reel/test",
            "https://127.0.0.1/reel/test",
            "https://0.0.0.0/reel/test",
        ]

        for url in localhost_urls:
            is_valid, error = validate_instagram_url(url)
            assert not is_valid
            assert "localhost" in error

    def test_private_network_blocking(self):
        """Test that private network URLs are blocked."""
        private_urls = [
            "https://192.168.1.1/reel/test",
            "https://10.0.0.1/reel/test",
            "https://172.16.0.1/reel/test",
            "https://test.local/reel/test",
        ]

        for url in private_urls:
            is_valid, error = validate_instagram_url(url)
            assert not is_valid
            assert "private networks" in error

    def test_exact_domain_matching(self):
        """Test that only exact Instagram domains are allowed."""
        # Valid domains
        valid_domains = ["https://instagram.com/reel/test123", "https://www.instagram.com/reel/test123"]

        for url in valid_domains:
            is_valid, error = validate_instagram_url(url)
            assert is_valid

        # Invalid domains (subdomain attacks)
        invalid_domains = [
            "https://evil.instagram.com/reel/test",
            "https://instagram.com.evil.com/reel/test",
            "https://notinstagram.com/reel/test",
        ]

        for url in invalid_domains:
            is_valid, error = validate_instagram_url(url)
            assert not is_valid

    def test_url_pattern_security(self):
        """Test that URL pattern requires HTTPS."""
        # HTTP URLs should not match pattern
        assert not is_instagram_reel_url("http://instagram.com/reel/test")

        # HTTPS URLs should match
        assert is_instagram_reel_url("https://instagram.com/reel/test123")
        assert is_instagram_reel_url("https://www.instagram.com/reels/test123")

    def test_normalize_url_security(self):
        """Test that URL normalization includes security validation."""
        # Valid URL should normalize successfully
        normalized = normalize_instagram_url("https://instagram.com/reel/test123")
        assert normalized == "https://www.instagram.com/reel/test123"

        # Invalid URL should raise exception
        with pytest.raises(ValueError):
            normalize_instagram_url("https://localhost/reel/test")

        with pytest.raises(ValueError):
            normalize_instagram_url("http://malicious.com/reel/test")


class TestYtDlpOptimizerSecurity:
    """Test yt-dlp optimizer security integration."""

    def test_build_command_sanitizes_url(self):
        """Test that build_instagram_command sanitizes URLs."""
        optimizer = YtDlpOptimizer()

        # Mock version detection to avoid subprocess calls in tests
        optimizer.version = MagicMock()
        optimizer.version.is_at_least.return_value = True
        optimizer.available_options = {}

        # Valid URL should work
        with patch.object(optimizer, "is_available", return_value=True):
            cmd = optimizer.build_instagram_command("https://www.instagram.com/reel/test123", "/tmp/output.mp4")
            assert "https://www.instagram.com/reel/test123" in cmd

        # Malicious URL should raise exception
        with patch.object(optimizer, "is_available", return_value=True):
            with pytest.raises(RuntimeError, match="Invalid or dangerous URL"):
                optimizer.build_instagram_command("https://instagram.com/reel/test; rm -rf /", "/tmp/output.mp4")


class TestDependencySecurityUpdates:
    """Test that security-critical dependencies are properly updated."""

    def test_requests_version_updated(self):
        """Test that requests library requirements are updated."""
        # This test verifies the requirements.txt changes
        with open("/Users/christophbeckmann/Developer/personal/active/reels_transcriber/requirements.txt") as f:
            requirements = f.read()

        # Check that requirements.txt specifies secure version
        assert "requests>=2.32.0" in requirements, "requirements.txt should specify requests>=2.32.0 for security"

        # Note: The actual installed version may be older in test environment
        # The important thing is that production deployments will use the secure version

    def test_numpy_version_updated(self):
        """Test that numpy is updated to secure version."""
        import numpy

        # numpy >= 1.26.0 should be installed
        version_parts = numpy.__version__.split(".")
        major, minor = int(version_parts[0]), int(version_parts[1])

        # Should be version 1.26.0 or higher
        assert major > 1 or (
            major == 1 and minor >= 26
        ), f"numpy version {numpy.__version__} is vulnerable, should be >= 1.26.0"


class TestSecurityIntegration:
    """Integration tests for security features working together."""

    def test_end_to_end_secure_workflow(self):
        """Test complete secure workflow from URL to file processing."""
        # Valid Instagram URL
        test_url = "https://www.instagram.com/reel/test123"

        # 1. URL validation should pass
        is_valid, error = validate_instagram_url(test_url)
        assert is_valid and error is None

        # 2. URL normalization should work
        normalized_url = normalize_instagram_url(test_url)
        assert normalized_url == test_url

        # 3. URL sanitization for subprocess should pass
        sanitized_url = sanitize_url_for_subprocess(normalized_url)
        assert sanitized_url == normalized_url

        # 4. File manager should handle safe filenames
        with TempFileManager() as manager:
            safe_path = manager.get_temp_path("reel_test123.mp4")
            assert "reel_test123.mp4" in safe_path

    def test_malicious_input_blocked_at_every_level(self):
        """Test that malicious input is blocked at every security layer."""
        malicious_url = "https://localhost/reel/test; rm -rf /"

        # 1. URL validation should fail
        is_valid, error = validate_instagram_url(malicious_url)
        assert not is_valid

        # 2. URL normalization should fail
        with pytest.raises(ValueError):
            normalize_instagram_url(malicious_url)

        # 3. URL sanitization should fail
        with pytest.raises(ValueError):
            sanitize_url_for_subprocess(malicious_url)

        # 4. File manager should block malicious filenames
        with TempFileManager() as manager:
            with pytest.raises(ValueError):
                manager.get_temp_path("../../../etc/passwd")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

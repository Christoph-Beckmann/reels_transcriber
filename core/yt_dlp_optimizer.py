"""
Optimized yt-dlp configuration and command builder with deprecated argument handling.

This module provides intelligent yt-dlp command construction that adapts to different
versions and handles deprecated arguments gracefully.
"""

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def sanitize_url_for_subprocess(url: str) -> str:
    """
    Sanitize URL to prevent command injection attacks in subprocess calls.

    Args:
        url: URL to sanitize

    Returns:
        str: Sanitized URL safe for subprocess execution

    Raises:
        ValueError: If URL is malicious or contains dangerous patterns
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty or whitespace only")

    # Check for obvious command injection patterns
    dangerous_patterns = [
        ";",
        "&",
        "|",
        "`",
        "$",
        "(",
        ")",
        "{",
        "}",
        "\n",
        "\r",
        "\t",
        "&&",
        "||",
        "$(",
        "${",
        "sudo",
        "rm ",
        "del ",
        "format",
        "chmod",
        "/etc/",
        "/usr/",
        "/var/",
        "/sys/",
        "/proc/",
        "\\",
        '"',
        "'",
        "<",
        ">",
    ]

    for pattern in dangerous_patterns:
        if pattern in url:
            raise ValueError(f"URL contains potentially dangerous pattern: {pattern}")

    # Validate URL structure
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL structure")

        # Only allow HTTP/HTTPS schemes
        if parsed.scheme.lower() not in ["http", "https"]:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")

        # Block localhost and private IPs to prevent SSRF
        hostname = parsed.hostname
        if hostname:
            hostname_lower = hostname.lower()
            if (
                hostname_lower in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]
                or hostname_lower.startswith("192.168.")
                or hostname_lower.startswith("10.")
                or hostname_lower.startswith("172.16.")
                or hostname_lower.endswith(".local")
            ):
                raise ValueError("URLs pointing to localhost or private networks are not allowed")

    except Exception as e:
        raise ValueError(f"URL validation failed: {e}") from e

    # Additional length check to prevent extremely long URLs
    if len(url) > 2048:
        raise ValueError("URL is too long (max 2048 characters)")

    # Return the original URL if it passes all checks
    return url


@dataclass
class YtDlpVersion:
    """yt-dlp version information."""

    major: int
    minor: int
    patch: int
    raw_version: str

    def __str__(self) -> str:
        return self.raw_version

    def is_at_least(self, major: int, minor: int, patch: int = 0) -> bool:
        """Check if version is at least the specified version."""
        return (self.major, self.minor, self.patch) >= (major, minor, patch)


@dataclass
class CommandOption:
    """yt-dlp command option with version constraints."""

    option: str
    value: Optional[str] = None
    min_version: Optional[tuple[int, int, int]] = None
    max_version: Optional[tuple[int, int, int]] = None
    deprecated_since: Optional[tuple[int, int, int]] = None
    replacement: Optional[str] = None
    description: str = ""


class YtDlpOptimizer:
    """
    Optimizes yt-dlp command construction based on version and feature detection.

    Handles deprecated arguments, version compatibility, and provides intelligent
    fallbacks for different yt-dlp versions.
    """

    def __init__(self):
        """Initialize yt-dlp optimizer."""
        self.version: Optional[YtDlpVersion] = None
        self.available_options: dict[str, bool] = {}
        self._option_registry = self._build_option_registry()

        # Detect yt-dlp version and capabilities
        self._detect_version_and_capabilities()

    def _build_option_registry(self) -> dict[str, CommandOption]:
        """Build registry of yt-dlp options with version constraints."""
        return {
            # Output options
            "output_template": CommandOption(option="-o", description="Output filename template"),
            "format_selector": CommandOption(option="--format", description="Video format selection"),
            # Network options
            "user_agent": CommandOption(option="--user-agent", description="User agent string"),
            "no_check_certificate": CommandOption(
                option="--no-check-certificate", description="Skip SSL certificate verification"
            ),
            "socket_timeout": CommandOption(
                option="--socket-timeout", min_version=(2021, 12, 1), description="Socket timeout in seconds"
            ),
            # Authentication and access
            "username": CommandOption(option="--username", description="Login username"),
            "password": CommandOption(option="--password", description="Login password"),
            "cookies": CommandOption(option="--cookies", description="Cookie file path"),
            # Download options
            "no_playlist": CommandOption(option="--no-playlist", description="Download only single video"),
            # Post-processing
            "extract_audio": CommandOption(
                option="--extract-audio",
                deprecated_since=(2021, 6, 1),
                replacement='--format "bestaudio/best"',
                description="Extract audio (deprecated)",
            ),
            "audio_format": CommandOption(
                option="--audio-format",
                deprecated_since=(2021, 6, 1),
                replacement="--format",
                description="Audio format (deprecated)",
            ),
            # Output control
            "quiet": CommandOption(option="--quiet", description="Suppress output"),
            "no_warnings": CommandOption(option="--no-warnings", description="Suppress warnings"),
            "verbose": CommandOption(option="--verbose", description="Verbose output"),
            # Instagram-specific options
            "instagram_include_stories": CommandOption(
                option="--instagram-include-stories", min_version=(2022, 1, 1), description="Include Instagram stories"
            ),
            # Retry and error handling
            "retries": CommandOption(option="--retries", description="Number of retries"),
            "abort_on_error": CommandOption(option="--abort-on-error", description="Abort on first error"),
            "continue_on_error": CommandOption(option="--ignore-errors", description="Continue on errors"),
            # Rate limiting
            "limit_rate": CommandOption(option="--limit-rate", description="Download rate limit"),
            "sleep_interval": CommandOption(
                option="--sleep-interval", min_version=(2021, 9, 1), description="Sleep between downloads"
            ),
            # Metadata and info
            "write_info_json": CommandOption(option="--write-info-json", description="Write info JSON file"),
            "write_description": CommandOption(option="--write-description", description="Write description file"),
            # Geo and proxy
            "geo_bypass": CommandOption(option="--geo-bypass", description="Bypass geographic restrictions"),
            "proxy": CommandOption(option="--proxy", description="HTTP/HTTPS/SOCKS proxy URL"),
        }

    def _detect_version_and_capabilities(self):
        """Detect yt-dlp version and available capabilities."""
        try:
            # Get version information
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                version_text = result.stdout.strip()
                self.version = self._parse_version(version_text)
                logger.info(f"Detected yt-dlp version: {self.version}")

                # Test available options
                self._test_available_options()
            else:
                logger.warning("yt-dlp version detection failed")

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"yt-dlp not available or failed to detect: {e}")
            self.version = None

    def _parse_version(self, version_text: str) -> Optional[YtDlpVersion]:
        """Parse yt-dlp version string."""
        try:
            # Common version patterns for yt-dlp
            patterns = [
                r"(\d+)\.(\d+)\.(\d+)",  # X.Y.Z
                r"(\d{4})\.(\d{2})\.(\d{2})",  # YYYY.MM.DD
                r"(\d+)\.(\d+)",  # X.Y
            ]

            for pattern in patterns:
                match = re.search(pattern, version_text)
                if match:
                    groups = match.groups()
                    major = int(groups[0])
                    minor = int(groups[1]) if len(groups) > 1 else 0
                    patch = int(groups[2]) if len(groups) > 2 else 0

                    return YtDlpVersion(major=major, minor=minor, patch=patch, raw_version=version_text)

            logger.warning(f"Could not parse yt-dlp version: {version_text}")
            return None

        except Exception as e:
            logger.error(f"Error parsing yt-dlp version: {e}")
            return None

    def _test_available_options(self):
        """Test which options are available in current yt-dlp version."""
        if not self.version:
            return

        # Test key options that might not be available
        test_options = ["--socket-timeout", "--sleep-interval", "--instagram-include-stories", "--geo-bypass"]

        for option in test_options:
            try:
                result = subprocess.run(["yt-dlp", "--help"], capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    self.available_options[option] = option in result.stdout
                else:
                    self.available_options[option] = False

            except Exception:
                self.available_options[option] = False

        logger.debug(f"Available options: {self.available_options}")

    def build_instagram_command(
        self, url: str, output_path: str, options: Optional[dict[str, Any]] = None
    ) -> list[str]:
        """
        Build optimized yt-dlp command for Instagram downloads.

        Args:
            url: Instagram URL to download
            output_path: Output file path template
            options: Additional options dictionary

        Returns:
            List of command arguments
        """
        if not self.is_available():
            raise RuntimeError("yt-dlp is not available")

        cmd = ["yt-dlp"]
        options = options or {}

        # Core Instagram download configuration
        core_config = {
            "output_template": output_path,
            "format_selector": "best[ext=mp4]/best",
            "no_playlist": True,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "no_check_certificate": True,
            "quiet": not options.get("verbose", False),
            "no_warnings": not options.get("verbose", False),
            "retries": options.get("max_retries", 3),
            "continue_on_error": True,
        }

        # Add rate limiting if available
        if self.available_options.get("--sleep-interval", True):
            core_config["sleep_interval"] = "1-3"

        # Add socket timeout if available
        if self.available_options.get("--socket-timeout", True):
            core_config["socket_timeout"] = str(options.get("timeout", 30))

        # Merge with user options
        core_config.update(options)

        # Build command with version-appropriate options
        for config_key, value in core_config.items():
            if config_key in self._option_registry:
                option_def = self._option_registry[config_key]

                # Check if option is supported in current version
                if self._is_option_supported(option_def):
                    if value is True:
                        cmd.append(option_def.option)
                    elif value is not False and value is not None:
                        cmd.extend([option_def.option, str(value)])
                else:
                    # Handle deprecated options
                    replacement = self._get_option_replacement(option_def, value)
                    if replacement:
                        cmd.extend(replacement)
                        logger.debug(f"Replaced deprecated option {option_def.option} with {replacement}")

        # Sanitize URL before adding to command to prevent injection attacks
        try:
            sanitized_url = sanitize_url_for_subprocess(url)
            cmd.append(sanitized_url)
        except ValueError as e:
            logger.error(f"URL sanitization failed: {e}")
            raise RuntimeError(f"Invalid or dangerous URL provided: {e}") from e

        logger.debug(f"Built yt-dlp command: {' '.join(cmd)}")
        return cmd

    def _is_option_supported(self, option_def: CommandOption) -> bool:
        """Check if an option is supported in current version."""
        if not self.version:
            return True  # Assume supported if version unknown

        # Check minimum version requirement
        if option_def.min_version:
            if not self.version.is_at_least(*option_def.min_version):
                return False

        # Check maximum version requirement
        if option_def.max_version:
            if self.version.is_at_least(*option_def.max_version):
                return False

        # Check if deprecated
        if option_def.deprecated_since:
            if self.version.is_at_least(*option_def.deprecated_since):
                return False

        # Check availability from testing
        if option_def.option in self.available_options:
            return self.available_options[option_def.option]

        return True

    def _get_option_replacement(self, option_def: CommandOption, value: Any) -> Optional[list[str]]:
        """Get replacement for deprecated option."""
        if not option_def.replacement:
            return None

        # Handle simple string replacements
        if isinstance(option_def.replacement, str):
            if value is True:
                return [option_def.replacement]
            elif value is not False and value is not None:
                return [option_def.replacement, str(value)]

        return None

    def is_available(self) -> bool:
        """Check if yt-dlp is available."""
        return self.version is not None

    def get_version_info(self) -> Optional[dict[str, Any]]:
        """Get detailed version information."""
        if not self.version:
            return None

        return {
            "version": str(self.version),
            "major": self.version.major,
            "minor": self.version.minor,
            "patch": self.version.patch,
            "available_options": self.available_options,
            "supported_features": self._get_supported_features(),
        }

    def _get_supported_features(self) -> dict[str, bool]:
        """Get supported features for current version."""
        if not self.version:
            return {}

        features = {
            "instagram_stories": self.version.is_at_least(2022, 1, 1),
            "socket_timeout": self.version.is_at_least(2021, 12, 1),
            "sleep_interval": self.version.is_at_least(2021, 9, 1),
            "modern_format_selection": self.version.is_at_least(2021, 6, 1),
            "geo_bypass": True,  # Generally available
            "cookies_support": True,  # Generally available
            "proxy_support": True,  # Generally available
        }

        return features

    def validate_instagram_url(self, url: str) -> tuple[bool, Optional[str]]:
        """
        Validate Instagram URL for yt-dlp compatibility.

        Args:
            url: Instagram URL to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not url:
            return False, "URL is required"

        # Basic Instagram URL patterns that yt-dlp supports
        instagram_patterns = [
            r"https?://(?:www\.)?instagram\.com/p/[^/]+",  # Posts
            r"https?://(?:www\.)?instagram\.com/reel/[^/]+",  # Reels (preferred)
            r"https?://(?:www\.)?instagram\.com/reels/[^/]+",  # Reels variant
            r"https?://(?:www\.)?instagram\.com/tv/[^/]+",  # IGTV
            r"https?://(?:www\.)?instagram\.com/stories/[^/]+",  # Stories
        ]

        for pattern in instagram_patterns:
            if re.match(pattern, url):
                # Additional validation for Reels (preferred format)
                if "/reel/" in url or "/reels/" in url:
                    return True, None
                elif "/p/" in url:
                    return True, "Note: Post URL detected. Reels work better for video content."
                elif "/tv/" in url:
                    return True, "Note: IGTV URL detected. Some features may be limited."
                elif "/stories/" in url:
                    if self._get_supported_features().get("instagram_stories", False):
                        return True, "Note: Stories may have limited availability."
                    else:
                        return False, "Stories are not supported in this yt-dlp version."
                else:
                    return True, None

        return False, "URL does not appear to be a valid Instagram URL."

    def get_optimal_format_selector(self, prefer_quality: str = "best") -> str:
        """
        Get optimal format selector for Instagram content.

        Args:
            prefer_quality: Preferred quality ('best', 'worst', 'medium')

        Returns:
            Format selector string
        """
        if not self.version:
            return "best"

        # Modern format selection (post-2021.6.1)
        if self.version.is_at_least(2021, 6, 1):
            if prefer_quality == "best":
                return "best[ext=mp4]/best[ext=webm]/best"
            elif prefer_quality == "medium":
                return "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best"
            elif prefer_quality == "worst":
                return "worst[ext=mp4]/worst"
            else:
                return "best[ext=mp4]/best"
        else:
            # Legacy format selection
            return prefer_quality if prefer_quality in ["best", "worst"] else "best"

    def create_safe_output_template(self, directory: str, filename_prefix: str = "instagram") -> str:
        """
        Create safe output template for yt-dlp.

        Args:
            directory: Output directory
            filename_prefix: Prefix for filename

        Returns:
            Safe output template string
        """
        # Ensure directory exists and is writable
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        # Create safe template with fallbacks
        template = str(directory_path / f"{filename_prefix}_%(id)s.%(ext)s")

        # Normalize path separators
        template = template.replace("\\", "/")

        return template

    def get_troubleshooting_info(self) -> dict[str, Any]:
        """Get troubleshooting information for yt-dlp issues."""
        info = {"yt_dlp_available": self.is_available(), "version_info": self.get_version_info(), "common_issues": []}

        if not self.is_available():
            info["common_issues"].extend(
                [
                    "yt-dlp is not installed or not in PATH",
                    "Try: pip install yt-dlp",
                    "Ensure yt-dlp is accessible from command line",
                ]
            )
        elif self.version and self.version.is_at_least(2020, 1, 1):
            info["common_issues"].extend(
                [
                    "Very old yt-dlp version detected",
                    "Consider updating: pip install --upgrade yt-dlp",
                    "Some features may not work with old versions",
                ]
            )

        return info


# Global optimizer instance
_yt_dlp_optimizer = None


def get_yt_dlp_optimizer() -> YtDlpOptimizer:
    """Get global yt-dlp optimizer instance."""
    global _yt_dlp_optimizer
    if _yt_dlp_optimizer is None:
        _yt_dlp_optimizer = YtDlpOptimizer()
    return _yt_dlp_optimizer


def build_optimized_command(url: str, output_path: str, **options) -> list[str]:
    """
    Convenience function to build optimized yt-dlp command.

    Args:
        url: URL to download
        output_path: Output path template
        **options: Additional options

    Returns:
        Optimized command list
    """
    optimizer = get_yt_dlp_optimizer()
    return optimizer.build_instagram_command(url, output_path, options)


def validate_instagram_url_for_ytdlp(url: str) -> tuple[bool, Optional[str]]:
    """
    Convenience function to validate Instagram URL for yt-dlp.

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    optimizer = get_yt_dlp_optimizer()
    return optimizer.validate_instagram_url(url)

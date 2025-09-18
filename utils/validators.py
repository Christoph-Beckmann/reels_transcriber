"""URL validation utilities for Instagram Reels."""

import logging
import re
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Instagram Reel URL pattern (supports both http and https, /reel/, /reels/, and /p/)
INSTAGRAM_REEL_PATTERN = r"https?://(www\.)?instagram\.com/(reels?|p)/[A-Za-z0-9_-]+/?"

# Compiled regex for better performance
INSTAGRAM_REEL_REGEX = re.compile(INSTAGRAM_REEL_PATTERN)


def is_instagram_reel_url(url: str) -> bool:
    """
    Check if URL is a valid Instagram Reel URL format.

    Args:
        url: URL string to validate

    Returns:
        bool: True if valid Instagram Reel URL, False otherwise
    """
    if not url or not isinstance(url, str):
        return False

    url = url.strip()
    if not url:
        return False

    return bool(INSTAGRAM_REEL_REGEX.match(url))


def validate_instagram_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate Instagram Reel URL and return detailed error if invalid.

    Args:
        url: URL string to validate

    Returns:
        tuple[bool, Optional[str]]: (is_valid, error_message)
            - is_valid: True if URL is valid
            - error_message: None if valid, descriptive error if invalid
    """
    # Basic input validation
    if not url:
        return False, "Please enter a URL"

    if not isinstance(url, str):
        return False, "URL must be a text string"

    url = url.strip()
    if not url:
        return False, "Please enter a URL"

    # Check if it's a valid URL structure
    try:
        parsed = urlparse(url)
        # Check for malformed URLs that start with :// (invalid syntax)
        if url.startswith("://"):
            return False, "Invalid URL format"
        if not parsed.scheme:
            return False, "URL must start with http:// or https://"
        if not parsed.netloc:
            return False, "Invalid URL format"
    except Exception:
        return False, "Invalid URL format"

    # Allow both HTTP and HTTPS protocols
    if parsed.scheme.lower() not in ["http", "https"]:
        return False, "URL must start with http:// or https://"

    # Security: Block localhost and private IP addresses to prevent SSRF attacks
    hostname = parsed.hostname
    if hostname:
        hostname_lower = hostname.lower()
        # Block localhost variations
        if hostname_lower in ["localhost", "127.0.0.1", "0.0.0.0", "::1"]:
            return False, "URLs pointing to localhost are not allowed"

        # Block private IP ranges
        if (
            hostname_lower.startswith("192.168.")
            or hostname_lower.startswith("10.")
            or hostname_lower.startswith("172.16.")
            or hostname_lower.startswith("172.17.")
            or hostname_lower.startswith("172.18.")
            or hostname_lower.startswith("172.19.")
            or hostname_lower.startswith("172.2")
            or hostname_lower.startswith("172.30.")
            or hostname_lower.startswith("172.31.")
            or hostname_lower.endswith(".local")
        ):
            return False, "URLs pointing to private networks are not allowed"

    # Check if it's an Instagram domain (exact match for security)
    if parsed.netloc.lower() not in ["instagram.com", "www.instagram.com"]:
        return False, "URL must be from instagram.com"

    # Check if it's specifically a Reel URL
    if not is_instagram_reel_url(url):
        if "/tv/" in url:
            return False, "This appears to be an IGTV video, not a Reel. Please use a Reel URL"
        elif "/stories/" in url:
            return False, "This appears to be an Instagram story, not a Reel. Please use a Reel URL"
        else:
            return False, "Please enter a valid Instagram Reel URL (format: instagram.com/reel/...)"

    logger.info(f"URL validation successful: {url}")
    return True, None


def normalize_instagram_url(url: str) -> str:
    """
    Normalize Instagram URL to consistent format with security validation.

    Args:
        url: Instagram URL to normalize

    Returns:
        str: Normalized URL with https and www

    Raises:
        ValueError: If URL is malicious or invalid
    """
    if not url or not isinstance(url, str):
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")

    # Security: Force HTTPS only (no HTTP allowed)
    if url.startswith("http://"):
        url = url.replace("http://", "https://", 1)
    elif not url.startswith("https://"):
        url = "https://" + url

    # Basic URL structure validation (but very permissive)
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL structure")
    except Exception as e:
        raise ValueError("Invalid URL format") from e

    # Ensure www for consistency
    if "https://instagram.com" in url:
        url = url.replace("https://instagram.com", "https://www.instagram.com", 1)

    # Remove trailing parameters and fragments for security
    if "?" in url:
        url = url.split("?")[0]
    if "#" in url:
        url = url.split("#")[0]

    # Ensure trailing slash for consistency
    if url.endswith("/reel") or url.endswith("/reels"):
        url += "/"

    return url


def extract_reel_id(url: str) -> Optional[str]:
    """
    Extract the Reel ID from an Instagram Reel URL.

    Args:
        url: Instagram Reel URL

    Returns:
        Optional[str]: Reel ID if extractable, None otherwise
    """
    try:
        # Remove trailing slash and split by /
        url = url.rstrip("/")
        parts = url.split("/")

        # Find 'reel', 'reels', or 'p' in URL parts
        if "reel" in parts or "reels" in parts or "p" in parts:
            if "reels" in parts:
                reel_index = parts.index("reels")
            elif "reel" in parts:
                reel_index = parts.index("reel")
            else:
                reel_index = parts.index("p")

            if reel_index + 1 < len(parts):
                reel_id = parts[reel_index + 1]
                # Validate reel ID format (alphanumeric, underscore, dash)
                if re.match(r"^[A-Za-z0-9_-]+$", reel_id):
                    return reel_id

        return None

    except Exception as e:
        logger.warning(f"Failed to extract reel ID from {url}: {e}")
        return None

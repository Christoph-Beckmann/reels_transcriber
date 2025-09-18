"""
Refactored Instagram Reels downloader with eliminated duplication and improved patterns.

This version consolidates common download patterns, eliminates code duplication,
and provides cleaner separation of concerns.
"""

import logging
import os
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

from instaloader import BadResponseException, ConnectionException, Instaloader, InstaloaderException, Post

from utils.common_patterns import (
    OperationResult,
    PathManager,
    handle_operation_error,
    retry_with_backoff,
)
from utils.validators import extract_reel_id, validate_instagram_url

logger = logging.getLogger(__name__)


class DownloadProgress:
    """Backward compatibility for DownloadProgress."""

    def __init__(self, callback=None):
        self.callback = callback
        self.progress = 0
        self.message = ""

    def update(self, progress: int, message: str):
        """Update download progress."""
        self.progress = progress
        self.message = message
        if self.callback:
            self.callback(progress, message)


class DownloadMethod(Enum):
    """Available download methods."""

    INSTALOADER = "instaloader"
    YT_DLP = "yt-dlp"


@dataclass
class DownloadConfig:
    """Configuration for download operations."""

    timeout: int = 60
    max_retries: int = 3
    retry_delay: int = 2
    use_fallback: bool = True
    download_dir: str = "downloads"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "use_fallback": self.use_fallback,
            "download_dir": self.download_dir,
        }


class DownloadProgressTracker:
    """Improved progress tracking for download operations."""

    def __init__(self, callback: Optional[Callable[[int, str], None]] = None):
        """
        Initialize progress tracker.

        Args:
            callback: Function to call with (progress_percent, status_message)
        """
        self.callback = callback
        self.started = False
        self.completed = False
        self.last_progress = 0

    def start(self, message: str = "Connecting to Instagram..."):
        """Mark download as started."""
        self.started = True
        self.update(0, message)

    def update(self, progress: int, message: str):
        """Update progress with validation."""
        # Ensure progress only moves forward
        progress = max(self.last_progress, min(100, max(0, progress)))
        self.last_progress = progress

        if self.callback:
            try:
                self.callback(progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def complete(self, file_path: str):
        """Mark download as completed."""
        self.completed = True
        filename = os.path.basename(file_path)
        self.update(100, f"Download complete: {filename}")


class BaseDownloader(ABC):
    """Abstract base class for download implementations."""

    def __init__(self, config: DownloadConfig):
        """
        Initialize base downloader.

        Args:
            config: Download configuration
        """
        self.config = config
        self.download_dir = PathManager.ensure_directory(config.download_dir)
        logger.debug(f"Initialized {self.__class__.__name__} with config: {config.to_dict()}")

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this downloader is available for use."""
        pass

    @abstractmethod
    def download_implementation(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Implementation-specific download logic.

        Args:
            url: URL to download
            progress_callback: Progress callback function

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        pass

    def download(self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None) -> OperationResult:
        """
        Download with standardized error handling and retry logic.

        Args:
            url: URL to download
            progress_callback: Progress callback function

        Returns:
            OperationResult with download results
        """
        progress = DownloadProgressTracker(progress_callback)

        try:
            progress.start(f"Starting {self.__class__.__name__} download...")

            @retry_with_backoff(
                max_retries=self.config.max_retries,
                base_delay=self.config.retry_delay,
                exceptions=(ConnectionException, BadResponseException, subprocess.TimeoutExpired),
            )
            def download_with_retry():
                return self.download_implementation(url, progress_callback)

            success, video_path, error_msg = download_with_retry()

            if success and video_path:
                # Validate downloaded file
                if not PathManager.validate_file_not_empty(video_path):
                    return OperationResult(success=False, error_message="Downloaded file is empty or corrupted")

                progress.complete(video_path)
                file_size = Path(video_path).stat().st_size

                return OperationResult(
                    success=True,
                    data={"video_path": video_path},
                    metadata={"file_size": file_size, "method": self.__class__.__name__},
                )
            else:
                return OperationResult(
                    success=False, error_message=error_msg or f"{self.__class__.__name__} download failed"
                )

        except Exception as e:
            return handle_operation_error(e, f"{self.__class__.__name__}_download")

    def find_video_file(self, search_dir: Path, shortcode: Optional[str] = None) -> Optional[str]:
        """
        Common logic for finding downloaded video files.

        Args:
            search_dir: Directory to search in
            shortcode: Optional shortcode to match

        Returns:
            Path to video file if found
        """
        video_extensions = [".mp4", ".mov", ".avi", ".webm"]

        # Try shortcode-specific patterns first
        if shortcode:
            for ext in video_extensions:
                patterns = [f"{shortcode}{ext}", f"*{shortcode}{ext}", f"{shortcode}_*{ext}", f"*_{shortcode}{ext}"]

                for pattern in patterns:
                    files = list(search_dir.glob(pattern))
                    if files:
                        return str(files[0])

        # Fall back to finding any video file
        for ext in video_extensions:
            video_files = list(search_dir.glob(f"*{ext}"))
            if video_files:
                # Sort by modification time and take the newest
                video_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return str(video_files[0])

        return None


class InstaloaderDownloader(BaseDownloader):
    """Instagram downloader using instaloader library."""

    def __init__(self, config: DownloadConfig):
        """Initialize instaloader downloader."""
        super().__init__(config)
        self._initialize_instaloader()

    def _initialize_instaloader(self):
        """Initialize instaloader with optimized settings."""
        self.loader = Instaloader(
            dirname_pattern=str(self.download_dir),
            download_pictures=False,
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
            post_metadata_txt_pattern="",
            storyitem_metadata_txt_pattern="",
        )

    def is_available(self) -> bool:
        """Instaloader is always available if imported."""
        return True

    def download_implementation(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Download using instaloader.

        Args:
            url: Instagram URL to download
            progress_callback: Progress callback

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        try:
            # Extract shortcode from URL
            shortcode = extract_reel_id(url)
            if not shortcode:
                return False, None, "Could not extract Reel ID from URL"

            if progress_callback:
                progress_callback(20, "Retrieving Reel information...")

            # Get post object
            post = Post.from_shortcode(self.loader.context, shortcode)
            if not post:
                return False, None, f"Could not retrieve Reel data for shortcode: {shortcode}"

            # Verify it's a video
            if not post.is_video:
                return False, None, "This Instagram post is not a video/Reel"

            if progress_callback:
                progress_callback(50, "Downloading video file...")

            # Create specific directory for this download
            post_dir = self.download_dir / f"reel_{shortcode}"
            post_dir.mkdir(exist_ok=True)

            # Configure loader for this download
            self.loader.dirname_pattern = str(post_dir)

            # Download the post
            self.loader.download_post(post, target=str(post_dir))

            if progress_callback:
                progress_callback(90, "Locating downloaded video...")

            # Find the downloaded video file
            video_path = self.find_video_file(post_dir, shortcode)
            if not video_path:
                return False, None, "Downloaded video file not found"

            logger.info(f"Successfully downloaded with instaloader: {video_path}")
            return True, video_path, None

        except ConnectionException as e:
            return False, None, f"Network connection failed: {e}"
        except BadResponseException as e:
            error_msg = str(e).lower()
            if "not found" in error_msg or "404" in error_msg:
                return False, None, "This Reel was not found. It may be deleted or private."
            elif "private" in error_msg or "403" in error_msg:
                return False, None, "This Reel is private and cannot be accessed."
            else:
                return False, None, f"Instagram returned an error: {e}"
        except InstaloaderException as e:
            error_msg = str(e).lower()
            if "login" in error_msg:
                return False, None, "This Reel requires login to access"
            elif "rate" in error_msg or "limit" in error_msg:
                return False, None, "Instagram rate limit reached. Please try again later."
            else:
                return False, None, f"Download failed: {e}"
        except KeyError as e:
            return False, None, f"Instagram API returned unexpected data format: {e}"
        except Exception as e:
            return False, None, f"Unexpected error: {e}"


class YtDlpDownloader(BaseDownloader):
    """Fallback downloader using yt-dlp."""

    def is_available(self) -> bool:
        """Check if yt-dlp is available."""
        try:
            result = subprocess.run(["yt-dlp", "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def download_implementation(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Download using yt-dlp.

        Args:
            url: Instagram URL to download
            progress_callback: Progress callback

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        try:
            if progress_callback:
                progress_callback(10, "Initializing yt-dlp...")

            # Prepare output template
            output_template = str(self.download_dir / "%(id)s.%(ext)s")

            # Build yt-dlp command with enhanced anti-detection measures
            cmd = [
                "yt-dlp",
                "--quiet",
                "--no-warnings",
                "--no-playlist",
                "-o",
                output_template,
                "--format",
                "best[ext=mp4]/best",
                "--no-check-certificate",
                "--user-agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
                "--add-header",
                "Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "--add-header",
                "Accept-Language: en-US,en;q=0.5",
                "--add-header",
                "DNT: 1",
                "--add-header",
                "Upgrade-Insecure-Requests: 1",
                "--add-header",
                "Sec-Fetch-Dest: document",
                "--add-header",
                "Sec-Fetch-Mode: navigate",
                "--add-header",
                "Sec-Fetch-Site: none",
                url,
            ]

            if progress_callback:
                progress_callback(30, "Executing yt-dlp download...")

            # Execute download
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config.timeout, cwd=str(self.download_dir)
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                return False, None, self._parse_ytdlp_error(error_msg)

            if progress_callback:
                progress_callback(80, "Locating downloaded file...")

            # Find the downloaded file
            video_path = self.find_video_file(self.download_dir)
            if not video_path:
                return False, None, "Downloaded file not found"

            logger.info(f"Successfully downloaded with yt-dlp: {video_path}")
            return True, video_path, None

        except subprocess.TimeoutExpired:
            return False, None, "Download timed out"
        except Exception as e:
            return False, None, f"yt-dlp error: {e}"

    def _parse_ytdlp_error(self, error_msg: str) -> str:
        """Parse yt-dlp error messages into user-friendly format."""
        error_lower = error_msg.lower()

        # Check for specific error patterns
        if "private" in error_lower and "login" not in error_lower:
            return "This Reel is private and cannot be accessed"
        elif "404" in error_lower or "not found" in error_lower:
            return "This Reel was not found or has been deleted"
        elif "rate-limit" in error_lower or "rate limit" in error_lower:
            return "Instagram is blocking access. This may be due to anti-bot protection. Try again in a few minutes or use browser cookies."
        elif "login required" in error_lower and ("rate-limit" in error_lower or "rate limit" in error_lower):
            # This is Instagram's generic blocking message, not necessarily a login requirement
            return "Instagram is blocking automated access. The content may be accessible through a browser."
        elif "login" in error_lower:
            # Only report login required if it's not combined with rate-limit
            return "Instagram reports this content may require login, though it could be anti-bot protection."
        elif "csrf token" in error_lower or "unable to extract" in error_lower:
            return "Instagram's structure has changed. The downloader may need an update."
        else:
            # Return truncated error for unknown errors
            return f"Download failed: {error_msg[:200]}"


class DownloadMethodManager:
    """Manages multiple download methods with intelligent fallback."""

    def __init__(self, config: DownloadConfig):
        """
        Initialize download manager.

        Args:
            config: Download configuration
        """
        self.config = config
        self.downloaders = self._initialize_downloaders()
        self.available_methods = self._check_available_methods()

        logger.info(f"Download manager initialized. Available methods: {list(self.available_methods.keys())}")

    def _initialize_downloaders(self) -> dict[DownloadMethod, BaseDownloader]:
        """Initialize all available downloaders."""
        return {
            DownloadMethod.INSTALOADER: InstaloaderDownloader(self.config),
            DownloadMethod.YT_DLP: YtDlpDownloader(self.config),
        }

    def _check_available_methods(self) -> dict[DownloadMethod, bool]:
        """Check which download methods are available."""
        return {method: downloader.is_available() for method, downloader in self.downloaders.items()}

    def download_with_fallback(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> OperationResult:
        """
        Download with automatic fallback to alternative methods.

        Args:
            url: URL to download
            progress_callback: Progress callback

        Returns:
            OperationResult with download results
        """
        # Determine download order
        method_order = [DownloadMethod.INSTALOADER]
        if self.config.use_fallback:
            method_order.append(DownloadMethod.YT_DLP)

        errors_encountered = []

        for i, method in enumerate(method_order):
            if not self.available_methods.get(method, False):
                continue

            try:
                if progress_callback:
                    method_name = method.value
                    progress_callback(5, f"Trying {method_name} (method {i + 1}/{len(method_order)})...")

                logger.info(f"Attempting download with {method.value}")
                downloader = self.downloaders[method]
                result = downloader.download(url, progress_callback)

                if result.success:
                    logger.info(f"Download successful with {method.value}")
                    return result

                errors_encountered.append(f"{method.value}: {result.error_message}")
                logger.warning(f"Download failed with {method.value}: {result.error_message}")

                # Wait before trying next method
                if i < len(method_order) - 1:
                    time.sleep(2)

            except Exception as e:
                error_msg = f"Critical error with {method.value}: {e}"
                errors_encountered.append(error_msg)
                logger.error(error_msg)

        # All methods failed
        combined_error = f"All download methods failed. Errors: {'; '.join(errors_encountered)}"

        return OperationResult(
            success=False,
            error_message=combined_error,
            metadata={
                "attempted_methods": [m.value for m in method_order],
                "all_errors": errors_encountered,
                "available_methods": self.available_methods,
            },
        )


class RefactoredInstagramDownloader:
    """
    Refactored Instagram downloader with consolidated patterns and improved structure.
    """

    def __init__(
        self,
        download_dir: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 2,
        use_fallback: bool = True,
    ):
        """
        Initialize refactored downloader.

        Args:
            download_dir: Directory to download files to
            timeout: Network timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            use_fallback: Whether to use fallback methods
        """
        self.config = DownloadConfig(
            download_dir=download_dir,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_fallback=use_fallback,
        )

        # Backward compatibility attributes expected by tests
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Initialize instaloader for compatibility
        self.loader = Instaloader(
            dirname_pattern=str(self.download_dir),
            filename_pattern="{shortcode}",
            download_videos=True,
            download_video_thumbnails=False,
            download_geotags=False,
            download_comments=False,
            save_metadata=False,
            compress_json=False,
        )

        self.download_manager = DownloadMethodManager(self.config)

        logger.info(f"RefactoredInstagramDownloader initialized: {self.config.to_dict()}")

    def validate_reel_url(self, url: str) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Validate Instagram Reel URL and extract shortcode.

        Args:
            url: Instagram Reel URL to validate

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (is_valid, error_message, shortcode)
        """
        try:
            # Basic URL validation
            is_valid, error_msg = validate_instagram_url(url)
            if not is_valid:
                return False, error_msg, None

            # Extract shortcode/reel ID
            reel_id = extract_reel_id(url)
            if not reel_id:
                return False, "Could not extract Reel ID from URL", None

            logger.debug(f"URL validation successful: {reel_id}")
            return True, None, reel_id

        except Exception as e:
            return False, f"URL validation error: {e}", None

    def download_reel(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Download Instagram Reel using direct instaloader method (for test compatibility).

        Args:
            url: Instagram Reel URL
            progress_callback: Callback for progress updates

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        # Validate URL first
        is_valid, error_msg, shortcode = self.validate_reel_url(url)
        if not is_valid:
            return False, None, error_msg

        last_error = None

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                if progress_callback:
                    progress_callback(10, f"Attempt {attempt + 1}: Initializing Instagram connection...")

                # Get post using instaloader
                post = Post.from_shortcode(self.loader.context, shortcode)

                if progress_callback:
                    progress_callback(30, "Fetching post information...")

                # Check if it's a video
                if not getattr(post, "is_video", True):
                    return False, None, "This post is not a video"

                if progress_callback:
                    progress_callback(50, "Downloading video file...")

                # Create specific directory for this download
                post_dir = self.download_dir / f"reel_{shortcode}"
                post_dir.mkdir(exist_ok=True)

                # Configure loader for this download
                self.loader.dirname_pattern = str(post_dir)

                # Download the post
                self.loader.download_post(post, target=str(post_dir))

                if progress_callback:
                    progress_callback(90, "Locating downloaded video...")

                # Find the downloaded video file
                video_path = self.find_video_file(post_dir, shortcode)
                if not video_path:
                    return False, None, "Downloaded video file not found"

                if progress_callback:
                    progress_callback(100, "Download complete")

                logger.info(f"Successfully downloaded reel: {video_path}")
                return True, video_path, None

            except ConnectionException as e:
                last_error = e
                logger.warning(f"Connection failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break
            except BadResponseException as e:
                error_str = str(e).lower()
                if "403" in error_str or "forbidden" in error_str:
                    return False, None, "This Reel is private and cannot be accessed"
                elif "404" in error_str or "not found" in error_str:
                    return False, None, "This Reel was not found or has been deleted"
                else:
                    return False, None, f"Instagram API error: {str(e)}"
            except Exception as e:
                error_msg = f"Download failed: {str(e)}"
                logger.error(error_msg)
                return False, None, error_msg

        # If we get here, all retries failed
        error_msg = f"Network connection failed: {str(last_error)}"
        logger.error(error_msg)
        return False, None, error_msg

    def download_reel_enhanced(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Enhanced download with automatic fallback methods.

        Args:
            url: Instagram Reel URL
            progress_callback: Callback for progress updates

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        # Validate URL first
        is_valid, error_msg, shortcode = self.validate_reel_url(url)
        if not is_valid:
            return False, None, error_msg

        # Use download manager with fallback
        result = self.download_manager.download_with_fallback(url, progress_callback)

        return result.success, result.data.get("video_path") if result.data else None, result.error_message

    def get_reel_metadata(self, url: str) -> Optional[dict[str, Any]]:
        """
        Get metadata for an Instagram Reel without downloading.

        Args:
            url: Instagram Reel URL

        Returns:
            Optional[dict[str, Any]]: Reel metadata if successful
        """
        try:
            is_valid, error_msg, shortcode = self.validate_reel_url(url)
            if not is_valid:
                logger.warning(f"Invalid URL for metadata: {error_msg}")
                return None

            # Use instaloader for metadata
            downloader = self.download_manager.downloaders[DownloadMethod.INSTALOADER]
            post = Post.from_shortcode(downloader.loader.context, shortcode)
            if not post:
                return None

            metadata = {
                "shortcode": post.shortcode,
                "is_video": post.is_video,
                "video_duration": getattr(post, "video_duration", None),
                "caption": post.caption,
                "date_utc": post.date_utc.isoformat() if post.date_utc else None,
                "owner_username": post.owner_username,
                "likes": post.likes,
                "video_url": post.video_url if post.is_video else None,
            }

            logger.debug(f"Retrieved metadata for Reel {shortcode}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to get metadata for {url}: {e}")
            return None

    def get_health_status(self) -> dict[str, Any]:
        """Get health status of all download methods."""
        return {
            "available_methods": self.download_manager.available_methods,
            "config": self.config.to_dict(),
            "primary_healthy": self.download_manager.available_methods.get(DownloadMethod.INSTALOADER, False),
            "fallback_available": self.download_manager.available_methods.get(DownloadMethod.YT_DLP, False),
        }

    def _find_downloaded_video(self, download_dir: Path, shortcode: str) -> Optional[str]:
        """Find downloaded video file (backward compatibility method)."""
        return self.find_video_file(download_dir, shortcode)

    def _find_video_file(self, directory: Path, pattern: str = "*") -> Optional[str]:
        """Find video file in directory (backward compatibility method)."""
        return self.find_video_file(directory, pattern)

    def find_video_file(self, directory: Path, pattern: str = "*") -> Optional[str]:
        """
        Find video file in the specified directory.

        Args:
            directory: Directory to search in
            pattern: Pattern to match (usually shortcode)

        Returns:
            Path to video file if found, None otherwise
        """
        try:
            video_extensions = [".mp4", ".mkv", ".avi", ".mov", ".webm"]

            # Search for video files
            for ext in video_extensions:
                if pattern == "*":
                    search_pattern = f"*{ext}"
                else:
                    search_pattern = f"*{pattern}*{ext}"

                video_files = list(directory.glob(search_pattern))
                if video_files:
                    # Return the first video file found
                    video_path = str(video_files[0])
                    logger.info(f"Found video file: {video_path}")
                    return video_path

            # If no specific pattern match, try any video file
            if pattern != "*":
                for ext in video_extensions:
                    video_files = list(directory.glob(f"*{ext}"))
                    if video_files:
                        video_path = str(video_files[0])
                        logger.info(f"Found fallback video file: {video_path}")
                        return video_path

            logger.warning(f"No video files found in {directory}")
            return None

        except Exception as e:
            logger.error(f"Error finding video file: {e}")
            return None

    def cleanup_download_dir(self) -> bool:
        """Clean up the download directory."""
        try:
            download_path = Path(self.config.download_dir)
            if download_path.exists():
                shutil.rmtree(download_path)
                logger.info(f"Cleaned up download directory: {download_path}")
                return True
            return True
        except Exception as e:
            logger.error(f"Failed to cleanup download directory: {e}")
            return False


class EnhancedInstagramDownloader:
    """
    Enhanced Instagram downloader with circuit breaker support.
    """

    def __init__(
        self,
        download_dir: str,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 2,
        use_fallback: bool = True,
        circuit_breaker_config=None,
    ):
        """
        Initialize enhanced downloader with circuit breaker support.

        Args:
            download_dir: Directory to download files to
            timeout: Network timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            use_fallback: Whether to use fallback methods
            circuit_breaker_config: Optional circuit breaker configuration
        """
        # Create base downloader
        self.base_downloader = RefactoredInstagramDownloader(
            download_dir=download_dir,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            use_fallback=use_fallback,
        )

        # If circuit breaker config provided, wrap with resilience
        if circuit_breaker_config:
            from core.resilience import create_resilient_downloader

            self.resilient_downloader = create_resilient_downloader(
                primary_downloader=self.base_downloader, fallback_downloader=None, config=circuit_breaker_config
            )
        else:
            self.resilient_downloader = None

        # Expose base downloader methods and attributes for testing compatibility
        self.validate_reel_url = self.base_downloader.validate_reel_url
        self.get_reel_metadata = self.base_downloader.get_reel_metadata
        self.get_health_status = self.base_downloader.get_health_status
        self.cleanup_download_dir = self.base_downloader.cleanup_download_dir

        # For test compatibility - expose ytdlp downloader
        self.ytdlp = None  # Will be set if YT-DLP downloader exists
        if hasattr(self.base_downloader, "download_manager"):
            if DownloadMethod.YT_DLP in self.base_downloader.download_manager.downloaders:
                self.ytdlp = self.base_downloader.download_manager.downloaders[DownloadMethod.YT_DLP]

    def download_reel(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Download Instagram Reel with circuit breaker if configured.

        Args:
            url: Instagram Reel URL
            progress_callback: Callback for progress updates

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        if self.resilient_downloader:
            # Use resilient downloader with circuit breaker
            try:
                result = self.resilient_downloader.execute(url, progress_callback)
                if hasattr(result, "success"):
                    # Result is an OperationResult
                    return result.success, result.data.get("video_path") if result.data else None, result.error_message
                else:
                    # Result is a tuple
                    return result
            except Exception as e:
                return False, None, str(e)
        else:
            # Use base downloader directly
            return self.base_downloader.download_reel(url, progress_callback)

    def download_reel_enhanced(
        self, url: str, progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """
        Enhanced download with automatic fallback methods.

        Args:
            url: Instagram Reel URL
            progress_callback: Callback for progress updates

        Returns:
            tuple[bool, Optional[str], Optional[str]]: (success, video_path, error_message)
        """
        return self.base_downloader.download_reel_enhanced(url, progress_callback)


# Backward compatibility aliases
InstagramDownloader = RefactoredInstagramDownloader
# DownloadProgress already defined above as a class


# Factory function for backward compatibility
def create_enhanced_downloader(download_dir: str, **kwargs) -> RefactoredInstagramDownloader:
    """
    Factory function to create a refactored enhanced downloader.

    Args:
        download_dir: Directory for downloads
        **kwargs: Additional parameters for the downloader

    Returns:
        RefactoredInstagramDownloader: Configured downloader instance
    """
    return RefactoredInstagramDownloader(download_dir, **kwargs)

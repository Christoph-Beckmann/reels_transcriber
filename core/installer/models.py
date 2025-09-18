"""
Whisper model downloader with progress tracking and resume capability.
"""

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional


class WhisperModelDownloader:
    """
    Downloads Whisper AI models with progress tracking and resume support.
    """

    # Model information
    MODEL_SIZES = {
        "tiny": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
            "size_mb": 39,
            "sha256": "65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9",
        },
        "base": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
            "size_mb": 74,
            "sha256": "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e",
        },
        "small": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
            "size_mb": 244,
            "sha256": "9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794",
        },
        "medium": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
            "size_mb": 769,
            "sha256": "345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1",
        },
        "large": {
            "url": "https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6884409c/large-v2.pt",
            "size_mb": 1550,
            "sha256": "81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6884409c",
        },
    }

    def __init__(
        self,
        progress_callback: Optional[Callable] = None,
        download_callback: Optional[Callable] = None,
        verbose: bool = False,
    ):
        """
        Initialize model downloader.

        Args:
            progress_callback: Callback for overall progress (percent, message)
            download_callback: Callback for download progress (bytes_downloaded, total_bytes)
            verbose: Enable verbose output
        """
        self.progress_callback = progress_callback
        self.download_callback = download_callback
        self.verbose = verbose
        self.cache_dir = self._get_cache_dir()

    def _get_cache_dir(self) -> Path:
        """Get or create Whisper model cache directory."""
        if os.name == "nt":  # Windows
            cache_dir = Path.home() / "AppData" / "Local" / "whisper"
        else:  # macOS/Linux
            cache_dir = Path.home() / ".cache" / "whisper"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def download_model(
        self, model_size: str = "base", force_redownload: bool = False
    ) -> tuple[bool, str, Optional[Path]]:
        """
        Download a Whisper model with resume support.

        Args:
            model_size: Size of model to download (tiny, base, small, medium, large)
            force_redownload: Force redownload even if model exists

        Returns:
            Tuple of (success, message, model_path)
        """
        if model_size not in self.MODEL_SIZES:
            return False, f"Invalid model size: {model_size}", None

        model_info = self.MODEL_SIZES[model_size]
        model_filename = f"{model_size}.pt"
        model_path = self.cache_dir / model_filename
        temp_path = self.cache_dir / f"{model_filename}.download"

        # Check if model already exists
        if model_path.exists() and not force_redownload:
            if self._verify_model(model_path, model_info["sha256"]):
                if self.progress_callback:
                    self.progress_callback(100, f"Model '{model_size}' already downloaded")
                return True, f"Model '{model_size}' already exists and verified", model_path
            else:
                if self.verbose:
                    print(f"Model '{model_size}' exists but verification failed, re-downloading")

        # Download the model
        try:
            success = self._download_with_resume(
                url=model_info["url"],
                dest_path=temp_path,
                expected_size=model_info["size_mb"] * 1024 * 1024,
                model_name=model_size,
            )

            if not success:
                return False, f"Failed to download model '{model_size}'", None

            # Verify downloaded file
            if self.progress_callback:
                self.progress_callback(90, "Verifying model integrity...")

            if self._verify_model(temp_path, model_info["sha256"]):
                # Move to final location
                temp_path.rename(model_path)
                if self.progress_callback:
                    self.progress_callback(100, f"Model '{model_size}' downloaded successfully")
                return True, f"Model '{model_size}' downloaded and verified", model_path
            else:
                temp_path.unlink(missing_ok=True)
                return False, f"Model '{model_size}' verification failed", None

        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            return False, f"Error downloading model: {str(e)}", None

    def _download_with_resume(self, url: str, dest_path: Path, expected_size: int, model_name: str) -> bool:
        """
        Download file with resume capability.

        Args:
            url: URL to download from
            dest_path: Destination file path
            expected_size: Expected file size in bytes
            model_name: Model name for progress display

        Returns:
            True if download successful
        """
        headers = {}
        mode = "wb"
        resume_pos = 0

        # Check for partial download
        if dest_path.exists():
            resume_pos = dest_path.stat().st_size
            if resume_pos < expected_size:
                headers["Range"] = f"bytes={resume_pos}-"
                mode = "ab"
                if self.verbose:
                    print(f"Resuming download from {resume_pos / 1024 / 1024:.1f}MB")

        # Create request
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                # Get total size
                if "Content-Range" in response.headers:
                    total_size = int(response.headers["Content-Range"].split("/")[-1])
                elif "Content-Length" in response.headers:
                    total_size = int(response.headers["Content-Length"]) + resume_pos
                else:
                    total_size = expected_size

                # Download with progress
                with open(dest_path, mode) as f:
                    downloaded = resume_pos
                    chunk_size = 8192
                    last_progress = 0

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress
                        progress = int((downloaded / total_size) * 100)

                        if self.download_callback:
                            self.download_callback(downloaded, total_size)

                        if self.progress_callback and progress > last_progress:
                            self.progress_callback(
                                progress,
                                f"Downloading {model_name} model: {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f}MB",
                            )
                            last_progress = progress

                return downloaded >= total_size

        except urllib.error.HTTPError as e:
            if e.code == 416:  # Range not satisfiable - file already complete
                return dest_path.stat().st_size >= expected_size
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            if self.verbose:
                print(f"Download error: {e}")
            return False

    def _verify_model(self, model_path: Path, expected_sha256: str) -> bool:
        """
        Verify model file integrity using SHA256.

        Args:
            model_path: Path to model file
            expected_sha256: Expected SHA256 hash

        Returns:
            True if verification passes
        """
        if not model_path.exists():
            return False

        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        calculated = sha256_hash.hexdigest()
        return calculated == expected_sha256

    def list_downloaded_models(self) -> dict[str, dict]:
        """
        List all downloaded models.

        Returns:
            Dictionary of model_name: {path, size_mb, verified}
        """
        models = {}

        for model_name, model_info in self.MODEL_SIZES.items():
            model_path = self.cache_dir / f"{model_name}.pt"
            if model_path.exists():
                models[model_name] = {
                    "path": str(model_path),
                    "size_mb": model_path.stat().st_size / 1024 / 1024,
                    "verified": self._verify_model(model_path, model_info["sha256"]),
                }

        return models

    def get_model_path(self, model_size: str) -> Optional[Path]:
        """
        Get path to a downloaded model.

        Args:
            model_size: Model size name

        Returns:
            Path to model if it exists and is verified, None otherwise
        """
        if model_size not in self.MODEL_SIZES:
            return None

        model_path = self.cache_dir / f"{model_size}.pt"
        if model_path.exists():
            if self._verify_model(model_path, self.MODEL_SIZES[model_size]["sha256"]):
                return model_path

        return None

    def estimate_download_time(self, model_size: str, bandwidth_mbps: float = 10.0) -> float:
        """
        Estimate download time for a model.

        Args:
            model_size: Model size name
            bandwidth_mbps: Estimated bandwidth in Mbps

        Returns:
            Estimated time in seconds
        """
        if model_size not in self.MODEL_SIZES:
            return 0

        size_mb = self.MODEL_SIZES[model_size]["size_mb"]
        time_seconds = (size_mb * 8) / bandwidth_mbps
        return time_seconds

    def cleanup_partial_downloads(self):
        """Remove any partial download files."""
        for temp_file in self.cache_dir.glob("*.download"):
            try:
                temp_file.unlink()
            except Exception:
                pass

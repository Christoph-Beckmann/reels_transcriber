"""Configuration settings for the transcription app."""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    """Application configuration settings."""

    # Whisper settings
    whisper_model: str = "large"  # Using large model for best transcription quality
    auto_detect_language: bool = True
    supported_languages: list = field(default_factory=lambda: ["de", "en"])  # Prioritize German

    # Download settings
    download_timeout: int = 60  # seconds
    max_retries: int = 3
    retry_delay: int = 2  # seconds

    # Audio processing settings
    target_sample_rate: int = 16000  # Hz (Whisper requirement)
    audio_chunk_duration: int = 30  # seconds for memory efficiency

    # File management
    temp_dir: str = field(default_factory=lambda: os.path.join(tempfile.gettempdir(), "reels_transcriber"))
    cleanup_on_exit: bool = True
    max_disk_space_mb: int = 1000  # Maximum disk space for temp files

    # GUI settings
    window_title: str = "Instagram Reels Transcriber"
    window_size: tuple = (800, 600)
    theme: str = "Default"

    # Logging settings
    log_level: str = "INFO"
    log_file: str = "transcriber.log"
    max_log_files: int = 5

    # Performance settings
    max_memory_mb: int = 500
    enable_gpu: bool = False  # For future GPU acceleration

    def __post_init__(self):
        """Validate and setup configuration after initialization."""
        # Ensure temp directory exists
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        # Validate whisper model
        valid_models = ["tiny", "base", "small", "medium", "large"]
        if self.whisper_model not in valid_models:
            logger.warning(f"Invalid Whisper model '{self.whisper_model}', using 'base'")
            self.whisper_model = "base"

        # Validate languages
        all_supported_languages = ["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"]
        self.supported_languages = [lang for lang in self.supported_languages if lang in all_supported_languages]
        if not self.supported_languages:
            logger.warning("No valid languages configured, using English and German")
            self.supported_languages = ["en", "de"]

        # Validate timeouts and limits
        self.download_timeout = max(10, min(300, self.download_timeout))  # 10s to 5min
        self.max_retries = max(1, min(10, self.max_retries))  # 1 to 10 retries
        self.max_memory_mb = max(100, min(4000, self.max_memory_mb))  # 100MB to 4GB

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "AppConfig":
        """Create AppConfig from dictionary."""
        return cls(**config_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            "whisper_model": self.whisper_model,
            "auto_detect_language": self.auto_detect_language,
            "supported_languages": self.supported_languages,
            "download_timeout": self.download_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "target_sample_rate": self.target_sample_rate,
            "audio_chunk_duration": self.audio_chunk_duration,
            "temp_dir": self.temp_dir,
            "cleanup_on_exit": self.cleanup_on_exit,
            "max_disk_space_mb": self.max_disk_space_mb,
            "window_title": self.window_title,
            "window_size": self.window_size,
            "theme": self.theme,
            "log_level": self.log_level,
            "log_file": self.log_file,
            "max_log_files": self.max_log_files,
            "max_memory_mb": self.max_memory_mb,
            "enable_gpu": self.enable_gpu,
        }

    def get_temp_path(self, filename: str = None) -> str:
        """Get path for temporary file."""
        if filename:
            return os.path.join(self.temp_dir, filename)
        return self.temp_dir

    def check_disk_space(self) -> bool:
        """Check if sufficient disk space is available."""
        try:
            import shutil

            free_space = shutil.disk_usage(self.temp_dir).free
            free_space_mb = free_space / (1024 * 1024)
            return free_space_mb > self.max_disk_space_mb
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")
            return True  # Assume OK if we can't check

    def get_log_level(self) -> int:
        """Get logging level as integer."""
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return levels.get(self.log_level.upper(), logging.INFO)


# Default configuration instance
DEFAULT_CONFIG = AppConfig()


def load_config(config_file: str = None) -> AppConfig:
    """
    Load configuration from file or return default.

    Args:
        config_file: Path to configuration file (JSON format)

    Returns:
        AppConfig: Loaded or default configuration
    """
    if config_file and os.path.exists(config_file):
        try:
            import json

            with open(config_file) as f:
                config_dict = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return AppConfig.from_dict(config_dict)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_file}: {e}")

    logger.info("Using default configuration")
    return DEFAULT_CONFIG


def save_config(config: AppConfig, config_file: str) -> bool:
    """
    Save configuration to file.

    Args:
        config: Configuration to save
        config_file: Path to save configuration

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import json

        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {config_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_file}: {e}")
        return False

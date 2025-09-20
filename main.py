#!/usr/bin/env python3
"""
Instagram Reels Transcription App
Entry point for the desktop application.
"""

import logging
import sys

from config.settings import DEFAULT_CONFIG, load_config
from gui import MainWindow
from utils.error_handler import get_system_diagnostics
from utils.logging_config import setup_application_logging

logger = logging.getLogger(__name__)


def main():
    """Main application entry point."""
    # Load configuration (from config.json if created by installer, otherwise use defaults)
    config = load_config("config.json")

    # Set up enhanced logging first
    setup_application_logging(
        level=config.log_level,
        console_output=True,
        debug_mode=False,
        max_log_size_mb=config.max_log_files,
        max_log_files=config.max_log_files,
    )

    logger.info("Starting Instagram Reels Transcriber")

    # Log system diagnostics at startup
    try:
        diagnostics = get_system_diagnostics()
        logger.info(
            f"System diagnostics: Platform={diagnostics.get('platform', {}).get('system', 'Unknown')}, "
            f"Python={diagnostics.get('python_version', 'Unknown')}, "
            f"Memory={diagnostics.get('memory', {}).get('available_gb', 'Unknown')}GB"
        )
    except Exception as e:
        logger.warning(f"Failed to collect system diagnostics: {e}")

    try:
        # Create and run the main window
        app = MainWindow(config=config)
        app.run()

    except ImportError as e:
        error_msg = f"Missing required dependency: {e}"
        logger.error(error_msg)
        print(f"\nERROR: {error_msg}")
        print("Please install the required dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Application failed to start: {e}", exc_info=True)
        print("\nCRITICAL ERROR: Application failed to start")
        print(f"Error: {e}")
        print("Check the log file for more details.")
        sys.exit(1)

    logger.info("Application shutting down")


if __name__ == "__main__":
    main()

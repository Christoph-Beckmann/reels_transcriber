#!/usr/bin/env python3
"""
Command-line interface for testing the Instagram Reels transcription pipeline.

This CLI tool demonstrates the TranscriptionPipeline orchestrator functionality
and can be used for testing and debugging outside of the GUI environment.
"""

import argparse
import sys
import time
from typing import Any

from config.settings import DEFAULT_CONFIG
from core.pipeline import PipelineResult, process_reel
from utils.error_handler import get_system_diagnostics, process_error_for_user
from utils.logging_config import log_user_action, setup_application_logging
from utils.progress import ProcessingStage, ProgressUpdate


def setup_logging(verbose: bool = False) -> None:
    """Configure enhanced logging for CLI output."""
    level = "DEBUG" if verbose else "INFO"
    setup_application_logging(
        level=level,
        console_output=True,
        debug_mode=verbose,
        max_log_size_mb=5,  # Smaller log files for CLI
        max_log_files=3,
    )


def progress_callback(progress_update: ProgressUpdate) -> None:
    """
    Progress callback that prints updates to console.

    Args:
        progress_update: Progress update from pipeline
    """
    stage_icons = {
        ProcessingStage.VALIDATING: "🔍",
        ProcessingStage.DOWNLOADING: "⬇️",
        ProcessingStage.EXTRACTING_AUDIO: "🎵",
        ProcessingStage.TRANSCRIBING: "✍️",
        ProcessingStage.CLEANING_UP: "🧹",
        ProcessingStage.COMPLETED: "✅",
        ProcessingStage.ERROR: "❌",
    }

    icon = stage_icons.get(progress_update.stage, "⚙️")
    progress_bar = "█" * (progress_update.progress // 5) + "░" * (20 - progress_update.progress // 5)

    print(f"\r{icon} [{progress_bar}] {progress_update.progress:3d}% | {progress_update.message}", end="", flush=True)

    # Add newline for completed states
    if progress_update.stage in (ProcessingStage.COMPLETED, ProcessingStage.ERROR):
        print()  # New line for completion


def print_result(result: PipelineResult) -> None:
    """
    Print pipeline execution result with enhanced error information.

    Args:
        result: Pipeline result to display
    """
    print("\n" + "=" * 80)

    if result.success:
        print("🎉 TRANSCRIPTION SUCCESSFUL!")
        print(f"Language detected: {result.detected_language or 'Unknown'}")
        print(f"Execution time: {result.execution_time:.2f} seconds")
        print(f"Stages completed: {result.stages_completed}")

        if result.metadata:
            print(f"Model used: {result.metadata.get('model_size', 'Unknown')}")
            print(f"Audio duration: {result.metadata.get('audio_duration', 0):.1f}s")

        print("\n📝 TRANSCRIPT:")
        print("-" * 40)
        print(result.transcript)
        print("-" * 40)

        # Log successful completion
        log_user_action(
            "cli_transcription_success",
            True,
            {
                "execution_time": result.execution_time,
                "transcript_length": len(result.transcript) if result.transcript else 0,
                "language": result.detected_language,
            },
        )

    else:
        print("❌ TRANSCRIPTION FAILED!")
        print(f"Error: {result.error_message}")
        if result.execution_time:
            print(f"Failed after: {result.execution_time:.2f} seconds")
        print(f"Stages completed: {result.stages_completed}")

        # Try to provide enhanced error guidance for CLI users
        try:
            error_exception = Exception(result.error_message or "Unknown error")
            error_details = process_error_for_user(error_exception, "cli_processing", "transcribe_reel")

            if error_details.recovery_suggestions:
                print("\n💡 SUGGESTIONS:")
                for i, suggestion in enumerate(error_details.recovery_suggestions[:3], 1):
                    print(f"  {i}. {suggestion}")

            if error_details.retry_recommended:
                print("\n🔄 You can try running the command again.")

        except Exception:
            print("Note: Run with --verbose for more diagnostic information.")

        # Log failed completion
        log_user_action(
            "cli_transcription_failed",
            False,
            {
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "stages_completed": result.stages_completed,
            },
        )

    print("=" * 80)


def create_config(args) -> dict[str, Any]:
    """
    Create configuration dictionary from CLI arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        dict[str, Any]: Configuration dictionary
    """
    config = DEFAULT_CONFIG.to_dict()

    # Override with CLI arguments
    if args.model:
        config["whisper_model"] = args.model

    if args.language:
        config["supported_languages"] = [args.language]
        config["auto_detect_language"] = False

    if args.temp_dir:
        config["temp_dir"] = args.temp_dir

    if args.timeout:
        config["download_timeout"] = args.timeout

    return config


def interactive_mode() -> None:
    """Run interactive mode where user can input URLs."""
    from tests.fixtures.mock_data import MockInstagramData

    print("\n🎬 Instagram Reels Transcriber - Interactive Mode")
    print("=" * 50)
    print("Enter Instagram Reel URLs to transcribe, or 'quit' to exit.")
    print("Examples (Real URLs for testing):")

    # Show real test URLs as examples
    real_urls = MockInstagramData.SAMPLE_URLS["real_test_reels"][:2]  # Show first 2
    for url in real_urls:
        print(f"  {url}")

    print("Format examples:")
    print("  https://instagram.com/reel/ABC123/")
    print("  https://www.instagram.com/reels/ABC123/")
    print()

    while True:
        try:
            url = input("📱 Enter Instagram Reel URL: ").strip()

            if not url:
                continue

            if url.lower() in ("quit", "exit", "q"):
                print("👋 Goodbye!")
                break

            print(f"\n🚀 Processing: {url}")

            # Use default config for interactive mode
            result = process_reel(url, progress_callback=progress_callback)
            print_result(result)

            print("\n" + "-" * 50)

        except KeyboardInterrupt:
            print("\n\n⚠️ Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def demo_mode(args) -> None:
    """Run demo mode with real test URLs."""
    from tests.fixtures.mock_data import MockInstagramData

    print("\n🎬 Instagram Reels Transcriber - Demo Mode")
    print("=" * 50)
    print("Testing with real Instagram URLs from our test suite")
    print("This demonstrates the full pipeline functionality.\n")

    # Get demo URLs
    demo_urls = MockInstagramData.TEST_CATEGORIES["integration_basic"]["urls"]
    demo_urls.extend(MockInstagramData.TEST_CATEGORIES["performance_benchmark"]["urls"][:1])  # Add one more

    # Create configuration
    config = create_config(args)

    print("📊 Configuration:")
    print(f"   Model: {config.get('whisper_model', 'base')}")
    auto_detect = config.get("auto_detect_language", True)
    language = "Auto-detect" if auto_detect else config.get("supported_languages", ["en"])[0]
    print(f"   Language: {language}")
    print()

    successful = 0
    failed = 0

    for i, test_url in enumerate(demo_urls, 1):
        print(f"\n{'=' * 60}")
        print(f"🎬 DEMO {i}/{len(demo_urls)}: {test_url}")
        print(f"{'=' * 60}")

        # Get expected metadata for context
        metadata = MockInstagramData.REAL_URL_METADATA.get(test_url, {})
        if metadata:
            category = metadata.get("expected_category", "unknown")
            duration_range = metadata.get("expected_duration_range", (0, 0))
            print(f"📝 Expected: {category}, ~{duration_range[0]}-{duration_range[1]}s duration")

        print("🚀 Processing URL...")

        try:
            # Process the URL
            start_time = time.time()
            result = process_reel(test_url, config, progress_callback)
            end_time = time.time()

            # Display results
            print_result(result)

            if result.success:
                successful += 1
                print(f"\n✅ Demo {i} completed successfully in {end_time - start_time:.2f}s")
            else:
                failed += 1
                print(f"\n❌ Demo {i} failed after {end_time - start_time:.2f}s")

        except KeyboardInterrupt:
            print("\n\n⚠️ Demo interrupted by user")
            break
        except Exception as e:
            failed += 1
            print(f"\n💥 Demo {i} crashed: {e}")

        # Brief pause between demos unless it's the last one
        if i < len(demo_urls):
            print("\n⏳ Brief pause before next demo...")
            time.sleep(2)

    # Final summary
    print(f"\n{'=' * 60}")
    print("🏁 DEMO SUMMARY")
    print(f"{'=' * 60}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(
        f"📊 Success rate: {successful / (successful + failed) * 100:.1f}%"
        if (successful + failed) > 0
        else "No tests completed"
    )
    print()

    if successful > 0:
        print("🎉 Demo completed! The transcription pipeline is working.")
        if failed > 0:
            print("ℹ️  Some failures are expected due to network/access limitations.")
    else:
        print("⚠️  Demo failed. Check your network connection and dependencies.")

    print("\n💡 Try interactive mode with: python3 cli.py --interactive")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Instagram Reels Transcription Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Transcribe a single reel (using real test URL)
  python3 cli.py https://www.instagram.com/reels/DOqtiuYiLV5/

  # Use specific Whisper model
  python3 cli.py --model small https://www.instagram.com/reels/DOq10tEAnZ1/

  # Force specific language
  python3 cli.py --language en https://www.instagram.com/reels/DOGoywOiMqA/

  # Interactive mode
  python3 cli.py --interactive

  # Demo mode with real test URLs
  python3 cli.py --demo

  # Verbose logging
  python3 cli.py --verbose https://www.instagram.com/reels/DMXM6vuo1t0/
        """,
    )

    # Positional argument
    parser.add_argument("url", nargs="?", help="Instagram Reel URL to transcribe")

    # Model options
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)",
    )

    parser.add_argument(
        "--language",
        choices=["en", "de", "fr", "es", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh"],
        help="Force specific language (default: auto-detect)",
    )

    # Processing options
    parser.add_argument("--timeout", type=int, default=60, help="Download timeout in seconds (default: 60)")

    parser.add_argument("--temp-dir", help="Custom temporary directory")

    # Mode options
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")

    parser.add_argument("--demo", action="store_true", help="Run demo mode with real test URLs")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # Convenience options
    parser.add_argument(
        "--test-pipeline", action="store_true", help="Test pipeline with hardcoded URL (for development)"
    )

    parser.add_argument("--diagnostics", action="store_true", help="Show system diagnostics and exit")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    print("🎬 Instagram Reels Transcription Pipeline")
    print("=" * 45)

    # Handle different modes
    if args.diagnostics:
        print("\n🔧 SYSTEM DIAGNOSTICS")
        print("=" * 30)
        try:
            diagnostics = get_system_diagnostics()

            # Platform info
            platform_info = diagnostics.get("platform", {})
            print(f"Platform: {platform_info.get('system', 'Unknown')} {platform_info.get('release', '')}")
            print(f"Architecture: {platform_info.get('architecture', 'Unknown')}")
            print(f"Python Version: {diagnostics.get('python_version', 'Unknown')}")

            # Memory info
            memory_info = diagnostics.get("memory", {})
            if "error" not in memory_info:
                avail_gb = memory_info.get("available_gb", 0)
                total_gb = memory_info.get("total_gb", 0)
                print(f"Memory: {avail_gb:.1f}GB available / {total_gb:.1f}GB total")
                print(f"Memory Usage: {memory_info.get('percent_used', 0):.1f}%")
            else:
                print("Memory: Unable to determine")

            # Disk info
            disk_info = diagnostics.get("disk_space", {})
            if "error" not in disk_info:
                print(
                    f"Disk Space: {disk_info.get('free_gb', 0):.1f}GB free / {disk_info.get('total_gb', 0):.1f}GB total"
                )
                print(f"Disk Usage: {100 - disk_info.get('percent_free', 0):.1f}%")
            else:
                print("Disk Space: Unable to determine")

            # Dependencies
            deps_info = diagnostics.get("dependencies", {})
            print("\nDependencies:")
            for dep, status in deps_info.items():
                status_icon = "✅" if status not in ["not_found", "not_available", "not_installed"] else "❌"
                print(f"  {status_icon} {dep}: {status}")

            print(f"\nTimestamp: {diagnostics.get('timestamp', 'Unknown')}")

        except Exception as e:
            print(f"❌ Failed to collect diagnostics: {e}")

        return 0

    if args.interactive:
        interactive_mode()
        return

    if args.demo:
        demo_mode(args)
        return

    if args.test_pipeline:
        # Test with a real URL from our test suite
        from tests.fixtures.mock_data import MockInstagramData

        test_url = MockInstagramData.TEST_CATEGORIES["integration_basic"]["urls"][0]
        print(f"🧪 Testing pipeline with real URL: {test_url}")
        config = create_config(args)
        result = process_reel(test_url, config, progress_callback)
        print_result(result)
        return

    # Single URL processing
    if not args.url:
        print("❌ Error: No URL provided. Use --interactive or provide a URL.")
        parser.print_help()
        return 1

    print(f"🚀 Processing URL: {args.url}")
    print(f"📊 Model: {args.model}")
    if args.language:
        print(f"🌍 Language: {args.language}")
    else:
        print("🌍 Language: Auto-detect")

    print()

    # Create configuration
    config = create_config(args)

    # Process the URL
    result = process_reel(args.url, config, progress_callback)

    # Display results
    print_result(result)

    # Return appropriate exit code
    return 0 if result.success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)

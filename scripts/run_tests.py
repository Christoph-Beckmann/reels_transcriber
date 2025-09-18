#!/usr/bin/env python3
"""
Test runner script for Instagram Reels Transcriber.

Provides convenient commands for running different test suites with appropriate
configurations and report generation.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Manages test execution with different configurations."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.reports_dir = self.project_root / "reports"
        self.ensure_reports_dir()

    def ensure_reports_dir(self):
        """Ensure reports directory exists."""
        self.reports_dir.mkdir(exist_ok=True)

    def run_command(self, command, description=""):
        """Run a shell command and handle errors."""
        print(f"\n{'=' * 60}")
        print(f"🔧 {description}")
        print(f"{'=' * 60}")
        print(f"Running: {' '.join(command)}")
        print()

        try:
            result = subprocess.run(command, check=True, cwd=self.project_root)
            print(f"✅ {description} completed successfully")
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"❌ Command not found: {command[0]}")
            print("Please ensure all dependencies are installed.")
            return False

    def run_unit_tests(self, coverage=True, verbose=True):
        """Run unit tests with coverage reporting."""
        command = ["pytest", "tests/unit/"]

        if coverage:
            command.extend(
                [
                    "--cov=core",
                    "--cov=gui",
                    "--cov=utils",
                    "--cov-report=html:reports/coverage",
                    "--cov-report=xml:reports/coverage.xml",
                    "--cov-report=term-missing",
                ]
            )

        if verbose:
            command.append("-v")

        command.extend(["--junitxml=reports/junit-unit.xml", "--html=reports/unit-tests.html", "--self-contained-html"])

        return self.run_command(command, "Unit Tests")

    def run_integration_tests(self, verbose=True, include_real_urls=False):
        """Run integration tests."""
        command = ["pytest", "tests/integration/"]

        if verbose:
            command.append("-v")

        if include_real_urls:
            command.extend(["-m", "not (real_urls and network)", "-m", "real_urls"])
            print("🔗 Including real URL tests (this may take longer and requires network access)")
        else:
            command.extend(["-m", "not (real_urls and network)"])
            print("⚠️  Excluding real URL tests (use --real-urls to include them)")

        command.extend(
            [
                "--junitxml=reports/junit-integration.xml",
                "--html=reports/integration-tests.html",
                "--self-contained-html",
            ]
        )

        return self.run_command(command, "Integration Tests")

    def run_gui_tests(self, headed=False, verbose=True):
        """GUI tests are not available - this project uses desktop GUI, not web GUI."""
        print("⚠️ GUI tests are not available for this desktop application")
        print("   This project uses FreeSimpleGUI/PySimpleGUI (tkinter-based), not web GUI")
        print("   Consider writing unit tests for GUI components instead")
        return False

    def run_performance_tests(self, verbose=True, include_real_urls=False):
        """Run performance and benchmark tests."""
        command = [
            "pytest",
            "tests/performance/",
            "--benchmark-only",
            "--benchmark-json=reports/benchmark.json",
            "--benchmark-html=reports/benchmark.html",
        ]

        if verbose:
            command.append("-v")

        if include_real_urls:
            command.extend(["-m", "benchmark or real_urls"])
            print("🔗 Including real URL performance tests")
        else:
            command.extend(["-m", "benchmark and not (real_urls and network)"])
            print("⚠️  Excluding real URL performance tests (use --real-urls to include them)")

        command.extend(["--junitxml=reports/junit-performance.xml"])

        return self.run_command(command, "Performance Tests")

    def run_real_url_tests(self, verbose=True):
        """Run only real URL tests (requires network access)."""
        command = ["pytest", "-m", "real_urls", "--tb=short"]

        if verbose:
            command.append("-v")

        command.extend(
            ["--junitxml=reports/junit-real-urls.xml", "--html=reports/real-url-tests.html", "--self-contained-html"]
        )

        print("🔗 Running real URL tests (requires network access to Instagram)")
        print("⚠️  These tests may take longer and could fail due to network issues")

        return self.run_command(command, "Real URL Tests")

    def run_all_tests(self, include_gui=False, include_slow=False, include_real_urls=False):
        """Run the complete test suite."""
        results = []

        # Unit tests
        results.append(self.run_unit_tests())

        # Integration tests
        results.append(self.run_integration_tests(include_real_urls=include_real_urls))

        # GUI tests (optional)
        if include_gui:
            results.append(self.run_gui_tests())

        # Performance tests
        if include_slow:
            results.append(self.run_performance_tests(include_real_urls=include_real_urls))

        # Real URL tests (if specifically requested)
        if include_real_urls:
            results.append(self.run_real_url_tests())

        # Summary
        passed = sum(results)
        total = len(results)

        print(f"\n{'=' * 60}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Passed: {passed}/{total}")

        if include_real_urls:
            print("ℹ️  Real URL tests were included - some failures may be due to network issues")

        if passed == total:
            print("🎉 All test suites passed!")
            return True
        else:
            print("❌ Some test suites failed!")
            return False

    def run_quick_tests(self):
        """Run a quick subset of tests for development."""
        command = [
            "pytest",
            "tests/unit/",
            "tests/integration/",
            "-x",  # Stop on first failure
            "--tb=short",
            "-q",  # Quiet output
        ]

        return self.run_command(command, "Quick Tests")

    def run_specific_test(self, test_path, verbose=True):
        """Run a specific test file or test function."""
        command = ["pytest", test_path]

        if verbose:
            command.extend(["-v", "-s"])

        return self.run_command(command, f"Specific Test: {test_path}")

    def check_coverage(self, threshold=90):
        """Check if coverage meets threshold."""
        command = [
            "pytest",
            "tests/unit/",
            "--cov=core",
            "--cov=gui",
            "--cov=utils",
            "--cov-fail-under=" + str(threshold),
            "--cov-report=term-missing",
            "-q",
        ]

        return self.run_command(command, f"Coverage Check (>={threshold}%)")

    def lint_code(self):
        """Run code linting and formatting checks."""
        commands = [
            (["black", "--check", "."], "Black formatting check"),
            (["isort", "--check-only", "."], "Import sorting check"),
            (["flake8", "."], "Flake8 linting"),
        ]

        results = []
        for command, description in commands:
            try:
                subprocess.run(command, check=True, cwd=self.project_root)
                print(f"✅ {description} passed")
                results.append(True)
            except subprocess.CalledProcessError:
                print(f"❌ {description} failed")
                results.append(False)
            except FileNotFoundError:
                print(f"⚠️ {description} skipped (tool not installed)")
                results.append(True)  # Don't fail if tool is missing

        return all(results)

    def setup_test_environment(self):
        """Set up the test environment."""
        print("🔧 Setting up test environment...")

        # Note: Playwright browser installation removed as this is a desktop app

        # Create necessary directories
        self.ensure_reports_dir()
        print("✅ Reports directory created")

        print("✅ Test environment setup complete")

    def clean_test_artifacts(self):
        """Clean up test artifacts and cache files."""
        patterns_to_clean = [".pytest_cache", "reports/*", "htmlcov", "**/__pycache__", "**/*.pyc", ".coverage"]

        print("🧹 Cleaning test artifacts...")

        for pattern in patterns_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    path.unlink()
                    print(f"  Removed file: {path}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"  Removed directory: {path}")

        print("✅ Test artifacts cleaned")

    def generate_report_summary(self):
        """Generate a summary of available test reports."""
        print(f"\n{'=' * 60}")
        print("📋 AVAILABLE REPORTS")
        print(f"{'=' * 60}")

        report_files = [
            ("reports/coverage/index.html", "Code Coverage Report"),
            ("reports/unit-tests.html", "Unit Test Results"),
            ("reports/integration-tests.html", "Integration Test Results"),
            ("reports/gui-tests.html", "GUI Test Results"),
            ("reports/benchmark.html", "Performance Benchmarks"),
        ]

        for file_path, description in report_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                print(f"✅ {description}: {file_path}")
            else:
                print(f"❌ {description}: Not available")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for Instagram Reels Transcriber",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 run_tests.py unit              # Run unit tests only
  python3 run_tests.py all               # Run all tests
  python3 run_tests.py all --gui         # Run all tests including GUI
  python3 run_tests.py all --real-urls   # Include real URL integration tests
  python3 run_tests.py integration --real-urls  # Run integration tests with real URLs
  python3 run_tests.py quick             # Run quick development tests
  python3 run_tests.py coverage --threshold 95  # Check 95% coverage
  python3 run_tests.py lint              # Run code linting
  python3 run_tests.py setup             # Setup test environment
  python3 run_tests.py clean             # Clean test artifacts
        """,
    )

    parser.add_argument(
        "command",
        choices=[
            "unit",
            "integration",
            "gui",
            "performance",
            "all",
            "quick",
            "coverage",
            "lint",
            "setup",
            "clean",
            "test",
            "reports",
            "real-urls",
        ],
        help="Test command to run",
    )

    parser.add_argument("--gui", action="store_true", help="Include GUI tests (when running 'all')")

    parser.add_argument("--slow", action="store_true", help="Include slow tests (when running 'all')")

    parser.add_argument("--headed", action="store_true", help="Run GUI tests in headed mode (visible browser)")

    parser.add_argument("--threshold", type=int, default=90, help="Coverage threshold percentage (default: 90)")

    parser.add_argument("--test-path", help="Specific test file or path to run (use with 'test' command)")

    parser.add_argument("--real-urls", action="store_true", help="Include real URL tests (requires network access)")

    parser.add_argument("--no-verbose", action="store_true", help="Reduce test output verbosity")

    args = parser.parse_args()

    runner = TestRunner()
    verbose = not args.no_verbose

    # Set environment for testing
    os.environ["PYTEST_RUNNING"] = "1"

    success = True

    if args.command == "unit":
        success = runner.run_unit_tests(verbose=verbose)

    elif args.command == "integration":
        success = runner.run_integration_tests(verbose=verbose, include_real_urls=args.real_urls)

    elif args.command == "gui":
        success = runner.run_gui_tests(headed=args.headed, verbose=verbose)

    elif args.command == "performance":
        success = runner.run_performance_tests(verbose=verbose, include_real_urls=args.real_urls)

    elif args.command == "all":
        success = runner.run_all_tests(include_gui=args.gui, include_slow=args.slow, include_real_urls=args.real_urls)

    elif args.command == "real-urls":
        success = runner.run_real_url_tests(verbose=verbose)

    elif args.command == "quick":
        success = runner.run_quick_tests()

    elif args.command == "coverage":
        success = runner.check_coverage(threshold=args.threshold)

    elif args.command == "lint":
        success = runner.lint_code()

    elif args.command == "setup":
        runner.setup_test_environment()

    elif args.command == "clean":
        runner.clean_test_artifacts()

    elif args.command == "test":
        if args.test_path:
            success = runner.run_specific_test(args.test_path, verbose=verbose)
        else:
            print("Error: --test-path is required with 'test' command")
            sys.exit(1)

    elif args.command == "reports":
        runner.generate_report_summary()

    # Generate report summary for test commands
    if args.command in ["unit", "integration", "gui", "performance", "all"]:
        runner.generate_report_summary()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

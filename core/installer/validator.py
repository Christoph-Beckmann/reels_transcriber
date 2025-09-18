"""
System validation for pre-installation checks.
"""

import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import psutil


@dataclass
class ValidationResult:
    """Result of system validation check."""

    passed: bool
    message: str
    details: Optional[dict] = None
    can_continue: bool = True


class SystemValidator:
    """
    Validates system requirements before installation.
    """

    # Minimum requirements
    MIN_PYTHON_VERSION = (3, 9)
    MIN_DISK_SPACE_GB = 2.0  # For models and dependencies
    MIN_MEMORY_GB = 4.0  # Recommended for Whisper
    REQUIRED_COMMANDS = ["git"]  # Required system commands

    def __init__(self, verbose: bool = False):
        """
        Initialize system validator.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.validation_results = []

    def validate_all(self) -> tuple[bool, list[ValidationResult]]:
        """
        Run all validation checks.

        Returns:
            Tuple of (all_passed, list of validation results)
        """
        self.validation_results = []

        # Run all checks
        checks = [
            ("Python Version", self.check_python_version),
            ("Operating System", self.check_operating_system),
            ("Disk Space", self.check_disk_space),
            ("Memory", self.check_memory),
            ("Internet Connection", self.check_internet_connection),
            ("System Commands", self.check_system_commands),
            ("Write Permissions", self.check_write_permissions),
            ("Package Managers", self.check_package_managers),
        ]

        for check_name, check_func in checks:
            try:
                result = check_func()
                self.validation_results.append(result)
            except Exception as e:
                result = ValidationResult(
                    passed=False,
                    message=f"{check_name} check failed with error",
                    details={"error": str(e)},
                    can_continue=False,
                )
                self.validation_results.append(result)

        # Check if all critical checks passed
        all(r.passed or not r.can_continue for r in self.validation_results)
        can_continue = all(r.can_continue for r in self.validation_results)

        return can_continue, self.validation_results

    def check_python_version(self) -> ValidationResult:
        """Check if Python version meets requirements."""
        current_version = sys.version_info[:2]
        required = self.MIN_PYTHON_VERSION

        if current_version >= required:
            return ValidationResult(
                passed=True,
                message=f"Python {current_version[0]}.{current_version[1]} meets requirements",
                details={"version": f"{current_version[0]}.{current_version[1]}"},
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"Python {current_version[0]}.{current_version[1]} is below minimum {required[0]}.{required[1]}",
                details={
                    "current": f"{current_version[0]}.{current_version[1]}",
                    "required": f"{required[0]}.{required[1]}",
                },
                can_continue=False,
            )

    def check_operating_system(self) -> ValidationResult:
        """Check operating system compatibility."""
        os_name = platform.system()
        os_version = platform.version()
        arch = platform.machine()

        supported_os = ["Windows", "Darwin", "Linux"]

        if os_name in supported_os:
            return ValidationResult(
                passed=True,
                message=f"{os_name} {arch} is supported",
                details={"os": os_name, "version": os_version, "architecture": arch},
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"{os_name} may not be fully supported",
                details={"os": os_name},
                can_continue=True,  # Can try anyway
            )

    def check_disk_space(self) -> ValidationResult:
        """Check available disk space."""
        try:
            path = Path.cwd()
            stat = shutil.disk_usage(path)
            available_gb = stat.free / (1024**3)

            if available_gb >= self.MIN_DISK_SPACE_GB:
                return ValidationResult(
                    passed=True,
                    message=f"{available_gb:.1f}GB available disk space",
                    details={"available_gb": available_gb},
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Only {available_gb:.1f}GB available, need {self.MIN_DISK_SPACE_GB}GB",
                    details={"available_gb": available_gb, "required_gb": self.MIN_DISK_SPACE_GB},
                    can_continue=False,
                )
        except Exception as e:
            return ValidationResult(
                passed=False, message="Could not check disk space", details={"error": str(e)}, can_continue=True
            )

    def check_memory(self) -> ValidationResult:
        """Check system memory."""
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            if total_gb >= self.MIN_MEMORY_GB:
                return ValidationResult(
                    passed=True,
                    message=f"{total_gb:.1f}GB total memory ({available_gb:.1f}GB available)",
                    details={"total_gb": total_gb, "available_gb": available_gb},
                )
            else:
                return ValidationResult(
                    passed=False,
                    message=f"Only {total_gb:.1f}GB memory, {self.MIN_MEMORY_GB}GB recommended",
                    details={"total_gb": total_gb, "recommended_gb": self.MIN_MEMORY_GB},
                    can_continue=True,  # Can run with less memory, just slower
                )
        except Exception as e:
            return ValidationResult(
                passed=True,
                message="Could not check memory, assuming sufficient",
                details={"error": str(e)},
                can_continue=True,
            )

    def check_internet_connection(self) -> ValidationResult:
        """Check internet connectivity."""
        import socket

        test_hosts = [
            ("pypi.org", 443),
            ("github.com", 443),
            ("cdn.openai.com", 443),  # For Whisper models
        ]

        failed_hosts = []
        for host, port in test_hosts:
            try:
                socket.create_connection((host, port), timeout=5).close()
            except (OSError, socket.timeout):
                failed_hosts.append(host)

        if not failed_hosts:
            return ValidationResult(
                passed=True,
                message="Internet connection verified",
                details={"tested_hosts": [h for h, _ in test_hosts]},
            )
        elif len(failed_hosts) < len(test_hosts):
            return ValidationResult(
                passed=False,
                message=f"Limited connectivity, cannot reach: {', '.join(failed_hosts)}",
                details={"failed_hosts": failed_hosts},
                can_continue=True,
            )
        else:
            return ValidationResult(
                passed=False,
                message="No internet connection detected",
                details={"failed_hosts": failed_hosts},
                can_continue=False,
            )

    def check_system_commands(self) -> ValidationResult:
        """Check for required system commands."""
        missing_commands = []

        for cmd in self.REQUIRED_COMMANDS:
            if not shutil.which(cmd):
                missing_commands.append(cmd)

        if not missing_commands:
            return ValidationResult(
                passed=True, message="All required system commands found", details={"commands": self.REQUIRED_COMMANDS}
            )
        else:
            return ValidationResult(
                passed=False,
                message=f"Missing commands: {', '.join(missing_commands)}",
                details={"missing": missing_commands},
                can_continue=False,
            )

    def check_write_permissions(self) -> ValidationResult:
        """Check write permissions in current directory."""
        test_file = Path.cwd() / ".install_test_write"

        try:
            test_file.touch()
            test_file.unlink()
            return ValidationResult(
                passed=True, message="Write permissions verified", details={"path": str(Path.cwd())}
            )
        except Exception as e:
            return ValidationResult(
                passed=False,
                message="No write permission in current directory",
                details={"error": str(e), "path": str(Path.cwd())},
                can_continue=False,
            )

    def check_package_managers(self) -> ValidationResult:
        """Check for Python package managers."""
        managers = {
            "pip": shutil.which("pip") or shutil.which("pip3"),
            "uv": shutil.which("uv"),
        }

        available = [name for name, path in managers.items() if path]

        if "uv" in available:
            return ValidationResult(
                passed=True, message="UV package manager found (recommended)", details={"managers": available}
            )
        elif "pip" in available:
            return ValidationResult(
                passed=True, message="pip found (will use for installation)", details={"managers": available}
            )
        else:
            return ValidationResult(
                passed=False,
                message="No Python package manager found",
                details={"checked": list(managers.keys())},
                can_continue=False,
            )

    def get_system_info(self) -> dict:
        """
        Get comprehensive system information.

        Returns:
            Dictionary with system details
        """
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "compiler": platform.python_compiler(),
            },
            "resources": {
                "cpu_count": os.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3) if psutil else None,
                "disk_gb": shutil.disk_usage(Path.cwd()).free / (1024**3),
            },
        }

    def print_summary(self, results: list[ValidationResult]):
        """
        Print validation summary.

        Args:
            results: List of validation results
        """
        print("\nSystem Validation Summary:")
        print("-" * 40)

        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"{status} {result.message}")

            if self.verbose and result.details:
                for key, value in result.details.items():
                    print(f"    {key}: {value}")

        can_continue = all(r.can_continue for r in results)
        print("-" * 40)
        if can_continue:
            print("System validation: PASSED ✓")
        else:
            print("System validation: FAILED ✗")
            print("\nPlease fix the issues above before continuing.")

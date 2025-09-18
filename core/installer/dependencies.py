"""
Dependency installation with progress tracking and retry logic.
"""

import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PackageInfo:
    """Information about a package to install."""

    name: str
    version: Optional[str] = None
    extras: Optional[list[str]] = None

    def to_spec(self) -> str:
        """Convert to pip specification string."""
        spec = self.name
        if self.extras:
            spec += f"[{','.join(self.extras)}]"
        if self.version:
            spec += self.version
        return spec


class DependencyInstaller:
    """
    Manages Python package installation with progress tracking and retry logic.
    """

    def __init__(self, progress_callback=None, verbose: bool = False):
        """
        Initialize dependency installer.

        Args:
            progress_callback: Callback function for progress updates
            verbose: Enable verbose output
        """
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.use_uv = shutil.which("uv") is not None
        self.package_manager = "uv" if self.use_uv else "pip"
        self.venv_path = Path.cwd() / ".venv"

    def create_virtual_environment(self) -> tuple[bool, str]:
        """
        Create a virtual environment.

        Returns:
            Tuple of (success, message)
        """
        try:
            if self.use_uv:
                # Use uv to create venv (much faster)
                cmd = ["uv", "venv"]
                if self.verbose:
                    cmd.append("--verbose")
            else:
                # Fallback to standard venv
                cmd = [sys.executable, "-m", "venv", str(self.venv_path)]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode == 0:
                return True, f"Virtual environment created with {self.package_manager}"
            else:
                return False, f"Failed to create venv: {result.stderr}"

        except Exception as e:
            return False, f"Error creating virtual environment: {str(e)}"

    def get_venv_python(self) -> str:
        """Get path to Python executable in virtual environment."""
        if sys.platform == "win32":
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            python_path = self.venv_path / "bin" / "python"

        return str(python_path)

    def install_packages(
        self, requirements_file: Optional[str] = None, packages: Optional[list[str]] = None, max_retries: int = 3
    ) -> tuple[bool, str]:
        """
        Install Python packages with retry logic.

        Args:
            requirements_file: Path to requirements file
            packages: List of package specifications
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple of (success, message)
        """
        if not requirements_file and not packages:
            return False, "No packages specified"

        # Determine installation command
        if self.use_uv:
            base_cmd = ["uv", "pip", "sync" if requirements_file else "install"]
        else:
            venv_pip = str(self.venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
            base_cmd = [venv_pip, "install"]

        # Build full command
        cmd = base_cmd.copy()
        if requirements_file:
            if self.use_uv and not requirements_file.startswith("requirements"):
                cmd.extend([requirements_file])
            else:
                cmd.extend(["-r" if not self.use_uv or "install" in cmd else "", requirements_file])
                cmd = [c for c in cmd if c]  # Remove empty strings
        elif packages:
            cmd.extend(packages)

        if not self.use_uv:
            cmd.extend(["--no-cache-dir", "--progress-bar", "on"])

        # Retry logic
        for attempt in range(max_retries):
            try:
                if self.progress_callback:
                    self.progress_callback(
                        int((attempt / max_retries) * 30), f"Installing packages (attempt {attempt + 1}/{max_retries})"
                    )

                # Run installation
                result = self._run_install_command(cmd)

                if result.returncode == 0:
                    if self.progress_callback:
                        self.progress_callback(100, "Package installation complete")
                    return True, "All packages installed successfully"

                # Handle specific error cases
                error_msg = result.stderr.lower()
                if "no space left" in error_msg:
                    return False, "Insufficient disk space for installation"
                elif "permission denied" in error_msg:
                    return False, "Permission denied - try running with administrator privileges"
                elif attempt < max_retries - 1:
                    # Retry with exponential backoff
                    wait_time = 2**attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return False, f"Installation failed after {max_retries} attempts: {result.stderr}"

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)
                    continue
                return False, f"Installation error: {str(e)}"

        return False, "Installation failed after maximum retries"

    def _run_install_command(self, cmd: list[str]) -> subprocess.CompletedProcess:
        """
        Run installation command with progress tracking.

        Args:
            cmd: Command to execute

        Returns:
            CompletedProcess result
        """
        if self.verbose:
            print(f"Running: {' '.join(cmd)}")

        # For UV, we can capture output normally
        if self.use_uv:
            return subprocess.run(cmd, capture_output=True, text=True, check=False)

        # For pip, parse progress output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
        )

        output_lines = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                output_lines.append(line)

                # Parse pip progress
                if self.progress_callback and "Collecting" in line:
                    self.progress_callback(30, "Collecting packages...")
                elif self.progress_callback and "Downloading" in line:
                    self.progress_callback(50, "Downloading packages...")
                elif self.progress_callback and "Installing" in line:
                    self.progress_callback(80, "Installing packages...")

                if self.verbose:
                    print(line.strip())

        returncode = process.poll()
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout="".join(output_lines), stderr="")

    def install_rich_fallback(self) -> tuple[bool, str]:
        """
        Install rich library for better progress display.

        Returns:
            Tuple of (success, message)
        """
        try:
            # Try to install rich for better UI
            success, msg = self.install_packages(packages=["rich>=13.0.0"])
            if success:
                return True, "Rich library installed for enhanced UI"
            return True, "Continuing without rich library"  # Not critical
        except Exception:
            return True, "Continuing with basic progress display"

    def verify_installation(self, required_packages: list[str]) -> tuple[bool, list[str]]:
        """
        Verify that required packages are installed.

        Args:
            required_packages: List of package names to verify

        Returns:
            Tuple of (all_installed, list of missing packages)
        """
        venv_python = self.get_venv_python()
        missing = []

        for package in required_packages:
            # Extract package name (remove version specifiers)
            pkg_name = package.split("[")[0].split("=")[0].split(">")[0].split("<")[0]

            try:
                result = subprocess.run(
                    [venv_python, "-c", f"import {pkg_name.replace('-', '_')}"], capture_output=True, check=False
                )

                if result.returncode != 0:
                    missing.append(pkg_name)

            except Exception:
                missing.append(pkg_name)

        return len(missing) == 0, missing

    def get_installed_packages(self) -> dict[str, str]:
        """
        Get list of installed packages with versions.

        Returns:
            Dictionary of package: version
        """
        try:
            if self.use_uv:
                cmd = ["uv", "pip", "list", "--format", "json"]
            else:
                venv_pip = str(self.venv_path / ("Scripts" if sys.platform == "win32" else "bin") / "pip")
                cmd = [venv_pip, "list", "--format", "json"]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            packages = json.loads(result.stdout)
            return {pkg["name"]: pkg["version"] for pkg in packages}

        except Exception as e:
            if self.verbose:
                print(f"Could not get package list: {e}")
            return {}

    def cleanup_cache(self):
        """Clean up pip cache to free disk space."""
        try:
            if self.use_uv:
                # UV manages its own cache
                pass
            else:
                subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], capture_output=True, check=False)
        except Exception:
            pass  # Cache cleanup is optional

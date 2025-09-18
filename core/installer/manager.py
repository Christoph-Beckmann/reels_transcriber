"""
Main installation manager that orchestrates the complete installation process.
"""

import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .dependencies import DependencyInstaller
from .models import WhisperModelDownloader
from .progress import InstallationStep, ProgressTracker, StepResult
from .validator import SystemValidator


@dataclass
class InstallationOptions:
    """Configuration options for installation."""

    model_size: str = "base"
    install_mode: str = "full"  # minimal, full, dev
    verbose: bool = False
    force_reinstall: bool = False
    skip_validation: bool = False
    create_shortcuts: bool = True
    requirements_file: str = "requirements.txt"


class InstallationManager:
    """
    Orchestrates the complete installation process with progress tracking.
    """

    def __init__(self, options: Optional[InstallationOptions] = None):
        """
        Initialize installation manager.

        Args:
            options: Installation configuration options
        """
        self.options = options or InstallationOptions()
        self.progress = ProgressTracker(total_steps=6, verbose=self.options.verbose)
        self.validator = SystemValidator(verbose=self.options.verbose)
        self.dep_installer = DependencyInstaller(
            progress_callback=self._update_step_progress, verbose=self.options.verbose
        )
        self.model_downloader = WhisperModelDownloader(
            progress_callback=self._update_step_progress,
            download_callback=self._update_download_progress,
            verbose=self.options.verbose,
        )

        self.installation_log = []
        self.installation_state = {}

    def run(self) -> tuple[bool, str]:
        """
        Run the complete installation process.

        Returns:
            Tuple of (success, message)
        """
        self.progress.start()

        try:
            # Step 1: System validation
            if not self.options.skip_validation:
                if not self._run_validation():
                    return False, "System validation failed"

            # Step 2: Environment setup
            if not self._setup_environment():
                return False, "Environment setup failed"

            # Step 3: Install dependencies
            if not self._install_dependencies():
                return False, "Dependency installation failed"

            # Step 4: Download Whisper model
            if not self._download_model():
                return False, "Model download failed"

            # Step 5: Verification
            if not self._verify_installation():
                return False, "Installation verification failed"

            # Step 6: Finalization
            if not self._finalize_installation():
                return False, "Installation finalization failed"

            # Success!
            self._save_installation_state()
            self.progress.finish()
            return True, "Installation completed successfully"

        except KeyboardInterrupt:
            self.progress.error("Installation cancelled by user")
            self._cleanup_failed_installation()
            return False, "Installation cancelled"

        except Exception as e:
            self.progress.error(f"Unexpected error: {str(e)}", e)
            self._cleanup_failed_installation()
            return False, f"Installation failed: {str(e)}"

        finally:
            self.progress.finish()

    def _run_validation(self) -> bool:
        """Run system validation checks."""
        self.progress.start_step(InstallationStep.SYSTEM_CHECK)

        can_continue, results = self.validator.validate_all()

        # Update progress based on results
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        progress = int((passed / total) * 100)
        self.progress.update_step(progress, f"Checked {passed}/{total} requirements")

        # Log results
        for result in results:
            if not result.passed and not result.can_continue:
                self.progress.error(result.message)

        if can_continue:
            self.progress.complete_step(
                StepResult(success=True, message=f"System validation passed ({passed}/{total} checks)")
            )
            return True
        else:
            self.progress.complete_step(StepResult(success=False, message="System requirements not met"))
            return False

    def _setup_environment(self) -> bool:
        """Set up virtual environment."""
        self.progress.start_step(InstallationStep.ENV_SETUP)

        # Check if venv already exists
        venv_path = Path.cwd() / ".venv"
        if venv_path.exists() and not self.options.force_reinstall:
            self.progress.update_step(100, "Virtual environment already exists")
            self.progress.complete_step(StepResult(success=True, message="Using existing virtual environment"))
            return True

        # Create virtual environment
        self.progress.update_step(30, "Creating virtual environment...")
        success, message = self.dep_installer.create_virtual_environment()

        if success:
            self.progress.update_step(100, "Virtual environment created")
            self.progress.complete_step(StepResult(success=True, message=message))
            return True
        else:
            self.progress.complete_step(StepResult(success=False, message=message))
            return False

    def _install_dependencies(self) -> bool:
        """Install Python dependencies."""
        self.progress.start_step(InstallationStep.DEPENDENCY_INSTALL)

        # First, try to install rich for better UI
        self.dep_installer.install_rich_fallback()

        # Determine requirements file based on mode
        requirements_files = {
            "minimal": "requirements.txt",
            "full": "requirements-complete.txt",
            "dev": "requirements-dev.txt",
        }

        req_file = requirements_files.get(self.options.install_mode, self.options.requirements_file)

        # Check if requirements file exists
        if not Path(req_file).exists():
            self.progress.error(f"Requirements file not found: {req_file}")
            self.progress.complete_step(StepResult(success=False, message=f"Requirements file not found: {req_file}"))
            return False

        # Install packages
        self.progress.update_step(10, f"Installing from {req_file}...")
        success, message = self.dep_installer.install_packages(requirements_file=req_file)

        if success:
            self.progress.complete_step(StepResult(success=True, message=f"Dependencies installed from {req_file}"))
            return True
        else:
            self.progress.complete_step(StepResult(success=False, message=message))
            return False

    def _download_model(self) -> bool:
        """Download Whisper AI model."""
        self.progress.start_step(InstallationStep.MODEL_DOWNLOAD)

        # Check if model already exists
        existing_path = self.model_downloader.get_model_path(self.options.model_size)
        if existing_path and not self.options.force_reinstall:
            self.progress.update_step(100, f"Model '{self.options.model_size}' already downloaded")
            self.progress.complete_step(
                StepResult(success=True, message=f"Using existing '{self.options.model_size}' model")
            )
            return True

        # Estimate download time
        est_time = self.model_downloader.estimate_download_time(self.options.model_size)
        self.progress.info(f"Estimated download time: {est_time / 60:.1f} minutes")

        # Download model
        success, message, model_path = self.model_downloader.download_model(
            model_size=self.options.model_size, force_redownload=self.options.force_reinstall
        )

        if success:
            self.progress.complete_step(
                StepResult(success=True, message=message, details={"model_path": str(model_path)})
            )
            return True
        else:
            self.progress.complete_step(StepResult(success=False, message=message))
            return False

    def _verify_installation(self) -> bool:
        """Verify the installation is complete and working."""
        self.progress.start_step(InstallationStep.VERIFICATION)

        checks_passed = 0
        total_checks = 3

        # Check 1: Verify critical packages
        self.progress.update_step(30, "Verifying package installation...")
        critical_packages = ["whisper", "FreeSimpleGUI", "yt_dlp", "moviepy"]
        packages_ok, missing = self.dep_installer.verify_installation(critical_packages)

        if packages_ok:
            checks_passed += 1
        else:
            self.progress.warning(f"Missing packages: {', '.join(missing)}")

        # Check 2: Verify model file
        self.progress.update_step(60, "Verifying AI model...")
        model_path = self.model_downloader.get_model_path(self.options.model_size)
        if model_path and model_path.exists():
            checks_passed += 1
        else:
            self.progress.warning("Model file not found")

        # Check 3: Test import
        self.progress.update_step(90, "Testing imports...")
        try:
            venv_python = self.dep_installer.get_venv_python()
            import subprocess

            result = subprocess.run(
                [venv_python, "-c", "import whisper; import FreeSimpleGUI"], capture_output=True, check=False
            )
            if result.returncode == 0:
                checks_passed += 1
        except Exception:
            pass

        # Complete verification
        if checks_passed == total_checks:
            self.progress.complete_step(
                StepResult(success=True, message=f"All {total_checks} verification checks passed")
            )
            return True
        else:
            self.progress.complete_step(
                StepResult(
                    success=checks_passed > 1, message=f"{checks_passed}/{total_checks} verification checks passed"
                )
            )
            return checks_passed > 1  # Allow partial success

    def _finalize_installation(self) -> bool:
        """Create launcher scripts and finalize setup."""
        self.progress.start_step(InstallationStep.FINALIZATION)

        try:
            # Create launcher scripts if they don't exist
            self.progress.update_step(30, "Creating launcher scripts...")
            self._create_launcher_scripts()

            # Save installation info
            self.progress.update_step(60, "Saving installation info...")
            self._save_installation_info()

            # Clean up temporary files
            self.progress.update_step(90, "Cleaning up...")
            self.model_downloader.cleanup_partial_downloads()
            self.dep_installer.cleanup_cache()

            self.progress.complete_step(StepResult(success=True, message="Installation finalized successfully"))
            return True

        except Exception as e:
            self.progress.complete_step(StepResult(success=False, message=f"Finalization error: {str(e)}"))
            return False

    def _create_launcher_scripts(self):
        """Create platform-specific launcher scripts."""
        # Scripts should already exist, but ensure they're executable
        if platform.system() != "Windows":
            script_path = Path("run_app.sh")
            if script_path.exists():
                os.chmod(script_path, 0o755)

    def _save_installation_info(self):
        """Save installation information for troubleshooting."""
        info = {
            "installation_date": datetime.now().isoformat(),
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "model_size": self.options.model_size,
            "install_mode": self.options.install_mode,
            "installed_packages": self.dep_installer.get_installed_packages(),
        }

        info_path = Path(".installation_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    def _save_installation_state(self):
        """Save installation state for recovery."""
        state_file = Path(".installation_state.json")
        state = {"completed": True, "timestamp": datetime.now().isoformat(), "options": asdict(self.options)}
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _cleanup_failed_installation(self):
        """Clean up after failed installation."""
        # Remove partial downloads
        self.model_downloader.cleanup_partial_downloads()

        # Remove installation state
        state_file = Path(".installation_state.json")
        if state_file.exists():
            state_file.unlink()

    def _update_step_progress(self, progress: int, message: str):
        """Update progress for current step."""
        self.progress.update_step(progress, message)

    def _update_download_progress(self, downloaded: int, total: int):
        """Update download progress."""
        if total > 0:
            progress = int((downloaded / total) * 100)
            size_mb = total / 1024 / 1024
            downloaded_mb = downloaded / 1024 / 1024
            self.progress.update_step(progress, f"Downloading: {downloaded_mb:.1f}/{size_mb:.1f}MB ({progress}%)")

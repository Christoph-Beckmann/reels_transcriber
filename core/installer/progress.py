"""
Progress tracking system for installation with rich terminal UI.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback to simple progress


class InstallationStep(Enum):
    """Installation steps with descriptions."""

    SYSTEM_CHECK = ("System Requirements", "Checking system compatibility...")
    ENV_SETUP = ("Environment Setup", "Creating virtual environment...")
    DEPENDENCY_INSTALL = ("Dependencies", "Installing Python packages...")
    MODEL_DOWNLOAD = ("AI Model", "Downloading Whisper model...")
    VERIFICATION = ("Verification", "Verifying installation...")
    FINALIZATION = ("Finalization", "Creating launcher scripts...")


@dataclass
class StepResult:
    """Result of an installation step."""

    success: bool
    message: str
    details: Optional[dict[str, Any]] = None


class ProgressTracker:
    """
    Manages progress display for installation with rich UI.
    Falls back to simple progress if rich is not available.
    """

    def __init__(self, total_steps: int = 6, verbose: bool = False):
        """
        Initialize progress tracker.

        Args:
            total_steps: Total number of installation steps
            verbose: Enable verbose output
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE else None
        self.progress = None
        self.overall_task = None
        self.current_task = None
        self.download_task = None
        self.step_results = []

    def start(self):
        """Start the progress tracking."""
        if RICH_AVAILABLE:
            # Create rich progress bars
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=self.console,
                expand=True,
            )

            # Add overall progress task
            self.overall_task = self.progress.add_task("🚀 Overall Installation Progress", total=self.total_steps)

            self.progress.start()
            self._print_header()
        else:
            print("\n" + "=" * 60)
            print("Instagram Reels Transcriber - Installation")
            print("=" * 60 + "\n")

    def _print_header(self):
        """Print installation header with rich formatting."""
        if self.console:
            header = Panel.fit(
                "[bold cyan]Instagram Reels Transcriber[/bold cyan]\n[dim]Unified Installation System v1.0[/dim]",
                border_style="bright_blue",
            )
            self.console.print(header)
            self.console.print()

    def start_step(self, step: InstallationStep):
        """
        Start a new installation step.

        Args:
            step: The installation step to start
        """
        self.current_step += 1
        # Safely get step value, handling cases where step might not be a proper enum
        step_value = getattr(step, 'value', (str(step), 'Unknown step')) if step is not None else ('unknown_step', 'Unknown step')
        step_name, step_desc = step_value if isinstance(step_value, tuple) and len(step_value) >= 2 else (str(step_value), 'Unknown step')

        if RICH_AVAILABLE and self.progress:
            # Update overall progress
            self.progress.update(
                self.overall_task, advance=1, description=f"🚀 Step {self.current_step}/{self.total_steps}: {step_name}"
            )

            # Add task for current step
            self.current_task = self.progress.add_task(f"   ⚙️  {step_desc}", total=100)
        else:
            print(f"\n[{self.current_step}/{self.total_steps}] {step_name}")
            print(f"    {step_desc}")

    def update_step(self, progress: int, message: Optional[str] = None):
        """
        Update current step progress.

        Args:
            progress: Progress percentage (0-100)
            message: Optional status message
        """
        if RICH_AVAILABLE and self.progress and self.current_task is not None:
            update_kwargs = {"completed": progress}
            if message:
                update_kwargs["description"] = f"   ⚙️  {message}"
            self.progress.update(self.current_task, **update_kwargs)
        else:
            if message:
                print(f"    {message} ({progress}%)")

    def complete_step(self, result: StepResult):
        """
        Mark current step as complete.

        Args:
            result: Result of the step execution
        """
        self.step_results.append(result)

        if RICH_AVAILABLE and self.progress and self.current_task is not None:
            # Complete the current task
            self.progress.update(self.current_task, completed=100)

            # Update with result
            status = "✅" if result.success else "❌"
            desc = f"   {status} {result.message}"
            self.progress.update(self.current_task, description=desc)

            # Remove task after a short delay for visibility
            time.sleep(0.5)
            self.progress.remove_task(self.current_task)
            self.current_task = None
        else:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"    [{status}] {result.message}")

    def start_download(self, file_name: str, total_size: int):
        """
        Start tracking a file download with size information.

        Args:
            file_name: Name of file being downloaded
            total_size: Total size in bytes
        """
        if RICH_AVAILABLE and self.progress:
            # Create specialized download progress
            download_progress = Progress(
                TextColumn("   📥 [bold blue]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=self.console,
            )

            self.download_task = download_progress.add_task(f"Downloading {file_name}", total=total_size)

            # Add to main progress display
            self.progress = download_progress
        else:
            print(f"    Downloading {file_name} ({total_size / 1024 / 1024:.1f} MB)")

    def update_download(self, bytes_downloaded: int):
        """
        Update download progress.

        Args:
            bytes_downloaded: Number of bytes downloaded so far
        """
        if RICH_AVAILABLE and self.progress and self.download_task is not None:
            self.progress.update(self.download_task, completed=bytes_downloaded)
        else:
            # Simple percentage for non-rich mode
            pass

    def finish(self):
        """Finish progress tracking and show summary."""
        if RICH_AVAILABLE and self.progress:
            self.progress.stop()
            self._print_summary()
        else:
            self._print_simple_summary()

    def _print_summary(self):
        """Print installation summary with rich formatting."""
        if not self.console:
            return

        # Create summary table
        table = Table(title="Installation Summary", show_header=True)
        table.add_column("Step", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for i, result in enumerate(self.step_results, 1):
            status = "✅ Success" if result.success else "❌ Failed"
            style = "green" if result.success else "red"
            table.add_row(f"Step {i}", f"[{style}]{status}[/{style}]", result.message)

        self.console.print()
        self.console.print(table)

        # Overall status
        all_success = all(r.success for r in self.step_results)
        if all_success:
            self.console.print(
                Panel.fit(
                    "[bold green]✨ Installation completed successfully![/bold green]\n"
                    "[dim]You can now run the application with: python main.py[/dim]",
                    border_style="green",
                )
            )
        else:
            self.console.print(
                Panel.fit(
                    "[bold red]⚠️  Installation completed with errors[/bold red]\n"
                    "[dim]Please check the errors above and try again[/dim]",
                    border_style="red",
                )
            )

    def _print_simple_summary(self):
        """Print simple summary for non-rich terminals."""
        print("\n" + "=" * 60)
        print("Installation Summary")
        print("=" * 60)

        for i, result in enumerate(self.step_results, 1):
            status = "SUCCESS" if result.success else "FAILED"
            print(f"Step {i}: [{status}] {result.message}")

        all_success = all(r.success for r in self.step_results)
        print("\n" + "=" * 60)
        if all_success:
            print("✨ Installation completed successfully!")
            print("You can now run the application with: python main.py")
        else:
            print("⚠️  Installation completed with errors")
            print("Please check the errors above and try again")
        print("=" * 60)

    def error(self, message: str, exception: Optional[Exception] = None):
        """
        Display an error message.

        Args:
            message: Error message to display
            exception: Optional exception for details
        """
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[bold red]❌ Error:[/bold red] {message}")
            if exception and self.verbose:
                self.console.print(f"[dim]Details: {str(exception)}[/dim]")
        else:
            print(f"ERROR: {message}")
            if exception and self.verbose:
                print(f"Details: {str(exception)}")

    def info(self, message: str):
        """
        Display an info message.

        Args:
            message: Info message to display
        """
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[blue]ℹ️  {message}[/blue]")
        else:
            print(f"INFO: {message}")

    def success(self, message: str):
        """
        Display a success message.

        Args:
            message: Success message to display
        """
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(f"SUCCESS: {message}")

    def warning(self, message: str):
        """
        Display a warning message.

        Args:
            message: Warning message to display
        """
        if RICH_AVAILABLE and self.console:
            self.console.print(f"[yellow]⚠️  {message}[/yellow]")
        else:
            print(f"WARNING: {message}")

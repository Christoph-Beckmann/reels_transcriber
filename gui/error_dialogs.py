"""
Enhanced error dialog components for user-friendly error handling and guidance.

This module provides comprehensive error dialogs with contextual help, recovery suggestions,
and diagnostic information collection while maintaining a professional user experience.
"""

import logging
from typing import Any, Optional

try:
    import FreeSimpleGUI as sg

    GUI_AVAILABLE = True
except ImportError:
    try:
        import PySimpleGUI as sg

        GUI_AVAILABLE = True
    except ImportError:
        sg = None
        GUI_AVAILABLE = False

from utils.error_handler import ErrorCategory, ErrorDetails, ErrorSeverity, get_system_diagnostics

logger = logging.getLogger(__name__)


class EnhancedErrorDialog:
    """
    Enhanced error dialog with contextual help and recovery suggestions.
    """

    @staticmethod
    def show_error_with_guidance(error_details: ErrorDetails, parent_title: str = "Error") -> str:
        """
        Show enhanced error dialog with guidance and recovery options.

        Args:
            error_details: Comprehensive error information
            parent_title: Title for the dialog

        Returns:
            User's choice: 'retry', 'help', 'report', or 'close'
        """
        if not GUI_AVAILABLE:
            logger.error("GUI not available for error dialog")
            return "close"

        try:
            # Create error message with severity indicator
            severity_icon = {
                ErrorSeverity.INFO: "ℹ️",
                ErrorSeverity.WARNING: "⚠️",
                ErrorSeverity.ERROR: "❌",
                ErrorSeverity.CRITICAL: "🚨",
            }

            icon = severity_icon.get(error_details.severity, "❌")
            # Safely get severity value, handling cases where severity might not be a proper enum
            severity_value = getattr(error_details.severity, 'value', str(error_details.severity)) if error_details.severity is not None else 'unknown_severity'
            title = f"{icon} {severity_value.title()}: {parent_title}"

            # Build dialog layout
            layout = [
                # Error message
                [sg.Text(error_details.user_message, font=("Helvetica", 11), size=(60, None), text_color="#333333")],
                [sg.HorizontalSeparator(pad=((0, 0), (10, 10)))],
                # Recovery suggestions
                [sg.Text("What you can try:", font=("Helvetica", 10, "bold"))],
            ]

            # Add recovery suggestions
            for i, suggestion in enumerate(error_details.recovery_suggestions[:4], 1):  # Limit to 4
                layout.append(
                    [sg.Text(f"  {i}. {suggestion}", font=("Helvetica", 10), size=(55, None), text_color="#555555")]
                )

            layout.append([sg.HorizontalSeparator(pad=((0, 0), (10, 10)))])

            # Buttons based on error characteristics
            button_row = []

            if error_details.retry_recommended:
                button_row.append(
                    sg.Button(
                        "Try Again", key="-RETRY-", button_color=("white", "#4CAF50"), font=("Helvetica", 10, "bold")
                    )
                )

            button_row.append(
                sg.Button("More Help", key="-HELP-", button_color=("white", "#2196F3"), font=("Helvetica", 10))
            )

            if error_details.contact_support:
                button_row.append(
                    sg.Button("Report Issue", key="-REPORT-", button_color=("white", "#FF9800"), font=("Helvetica", 10))
                )

            button_row.append(
                sg.Button("Close", key="-CLOSE-", button_color=("white", "#757575"), font=("Helvetica", 10))
            )

            layout.append([sg.Push()] + button_row + [sg.Push()])

            # Create and show dialog
            window = sg.Window(
                title,
                layout,
                modal=True,
                finalize=True,
                font=("Helvetica", 10),
                margins=(20, 20),
                element_justification="left",
            )

            while True:
                event, values = window.read()

                if event in (sg.WIN_CLOSED, "-CLOSE-"):
                    result = "close"
                    break
                elif event == "-RETRY-":
                    result = "retry"
                    break
                elif event == "-HELP-":
                    EnhancedErrorDialog.show_contextual_help(error_details.category)
                    continue  # Stay in dialog
                elif event == "-REPORT-":
                    EnhancedErrorDialog.show_report_dialog(error_details)
                    continue  # Stay in dialog

            window.close()
            return result

        except Exception as e:
            logger.error(f"Failed to show enhanced error dialog: {e}")
            # Fallback to simple error popup
            if GUI_AVAILABLE:
                sg.popup_error(error_details.user_message, title=parent_title)
            return "close"

    @staticmethod
    def show_contextual_help(error_category: ErrorCategory) -> None:
        """
        Show contextual help for specific error categories.

        Args:
            error_category: Category of error to show help for
        """
        if not GUI_AVAILABLE:
            return

        from utils.error_handler import _feedback_manager

        help_info = _feedback_manager.get_contextual_help(error_category)

        layout = [
            [sg.Text(help_info["title"], font=("Helvetica", 14, "bold"), text_color="#2196F3")],
            [sg.Text(help_info["description"], font=("Helvetica", 10), size=(60, None))],
            [sg.HorizontalSeparator(pad=((0, 0), (10, 10)))],
            [sg.Text("Common Solutions:", font=("Helvetica", 11, "bold"))],
        ]

        for solution in help_info["common_solutions"]:
            layout.append([sg.Text(f"• {solution}", font=("Helvetica", 10), size=(55, None))])

        layout.extend(
            [
                [sg.HorizontalSeparator(pad=((0, 0), (10, 10)))],
                [sg.Text("Prevention Tips:", font=("Helvetica", 11, "bold"))],
            ]
        )

        for tip in help_info["prevention_tips"]:
            layout.append([sg.Text(f"• {tip}", font=("Helvetica", 10), size=(55, None), text_color="#555555")])

        layout.append([sg.Push(), sg.Button("Close", key="-CLOSE-"), sg.Push()])

        window = sg.Window("Help & Troubleshooting", layout, modal=True, finalize=True, margins=(20, 20))

        window.read()
        window.close()

    @staticmethod
    def show_report_dialog(error_details: ErrorDetails) -> None:
        """
        Show dialog for reporting issues with diagnostic information.

        Args:
            error_details: Error details to include in report
        """
        if not GUI_AVAILABLE:
            return

        # Collect system diagnostics
        diagnostics = get_system_diagnostics()

        report_info = [
            f"Error Code: {error_details.error_code or 'N/A'}",
            # Safely get enum values, handling cases where they might not be proper enums
            f"Category: {getattr(error_details.category, 'value', str(error_details.category)) if error_details.category is not None else 'unknown_category'}",
            f"Severity: {getattr(error_details.severity, 'value', str(error_details.severity)) if error_details.severity is not None else 'unknown_severity'}",
            f"Technical Details: {error_details.technical_details}",
            "",
            "System Information:",
            f"Platform: {diagnostics.get('platform', {}).get('system', 'Unknown')}",
            f"Python Version: {diagnostics.get('python_version', 'Unknown')}",
            f"Memory: {diagnostics.get('memory', {}).get('available_gb', 'Unknown')}GB available",
            f"Disk Space: {diagnostics.get('disk_space', {}).get('free_gb', 'Unknown')}GB free",
        ]

        layout = [
            [sg.Text("Issue Report Information", font=("Helvetica", 14, "bold"))],
            [sg.Text("The following information can help diagnose the issue:", font=("Helvetica", 10))],
            [
                sg.Multiline(
                    "\n".join(report_info), size=(60, 15), font=("Courier", 9), disabled=True, key="-REPORT-TEXT-"
                )
            ],
            [
                sg.Text(
                    "You can copy this information and include it when reporting the issue.",
                    font=("Helvetica", 9),
                    text_color="#666666",
                )
            ],
            [sg.Push(), sg.Button("Copy to Clipboard", key="-COPY-"), sg.Button("Close", key="-CLOSE-"), sg.Push()],
        ]

        window = sg.Window("Report Issue", layout, modal=True, finalize=True, margins=(15, 15))

        while True:
            event, values = window.read()

            if event in (sg.WIN_CLOSED, "-CLOSE-"):
                break
            elif event == "-COPY-":
                try:
                    sg.clipboard_set("\n".join(report_info))
                    sg.popup_quick_message("Report information copied to clipboard!", auto_close_duration=2)
                except Exception as e:
                    sg.popup_error(f"Failed to copy: {e}")

        window.close()


class ProcessingTimer:
    """
    Tracks processing time and provides estimates.
    """

    def __init__(self):
        """Initialize processing timer."""
        self.start_time: Optional[float] = None
        self.stage_times: dict[str, float] = {}
        self.estimated_total_time: Optional[float] = None

    def start(self) -> None:
        """Start timing."""
        import time

        self.start_time = time.time()
        self.stage_times.clear()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        import time

        return time.time() - self.start_time

    def estimate_remaining_time(self, current_progress: int) -> Optional[str]:
        """
        Estimate remaining processing time.

        Args:
            current_progress: Current progress percentage (0-100)

        Returns:
            Human-readable time estimate or None
        """
        if self.start_time is None or current_progress <= 0:
            return None

        elapsed = self.get_elapsed_time()
        if current_progress >= 100:
            return "Completing..."

        # Simple linear estimation
        total_estimated = elapsed * 100 / current_progress
        remaining = total_estimated - elapsed

        if remaining < 60:
            return f"~{int(remaining)}s remaining"
        elif remaining < 3600:
            minutes = int(remaining / 60)
            return f"~{minutes}m remaining"
        else:
            hours = int(remaining / 3600)
            minutes = int((remaining % 3600) / 60)
            return f"~{hours}h {minutes}m remaining"

    def format_elapsed_time(self) -> str:
        """Format elapsed time for display."""
        elapsed = self.get_elapsed_time()
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        elif elapsed < 3600:
            minutes = int(elapsed / 60)
            seconds = int(elapsed % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(elapsed / 3600)
            minutes = int((elapsed % 3600) / 60)
            return f"{hours}h {minutes}m"


class UserFeedbackHelper:
    """
    Helper class for user feedback and guidance.
    """

    @staticmethod
    def show_confirmation(message: str, title: str = "Confirm") -> bool:
        """
        Show confirmation dialog for destructive actions.

        Args:
            message: Confirmation message
            title: Dialog title

        Returns:
            True if user confirmed, False otherwise
        """
        if not GUI_AVAILABLE:
            return True  # Default to allowing action if GUI not available

        try:
            result = sg.popup_yes_no(message, title=title, font=("Helvetica", 10), button_color=("white", "#757575"))
            return result == "Yes"
        except Exception as e:
            logger.error(f"Failed to show confirmation dialog: {e}")
            return True  # Default to allowing action

    @staticmethod
    def show_processing_stats(completion_data: dict[str, Any]) -> None:
        """
        Show processing completion statistics.

        Args:
            completion_data: Processing completion data
        """
        if not GUI_AVAILABLE:
            return

        try:
            stats = [
                f"✓ Transcript Length: {len(completion_data.get('transcript', ''))} characters",
                f"✓ Language Detected: {completion_data.get('language', 'Unknown')}",
                f"✓ Processing Time: {completion_data.get('execution_time', 0):.1f} seconds",
                f"✓ Stages Completed: {completion_data.get('stages_completed', 0)}",
            ]

            sg.popup(
                "Transcription Completed Successfully!\n\n" + "\n".join(stats),
                title="Success",
                font=("Helvetica", 10),
                button_color=("white", "#4CAF50"),
            )

        except Exception as e:
            logger.error(f"Failed to show processing stats: {e}")

    @staticmethod
    def show_help() -> None:
        """Show general help information."""
        if not GUI_AVAILABLE:
            return

        help_text = """
Instagram Reels Transcriber Help

How to use:
1. Copy an Instagram Reel URL from the Instagram app or website
2. Paste the URL in the input field above
3. Click "Start Transcription" to begin processing
4. Wait for the transcription to complete
5. Copy the result to your clipboard when done

Supported URLs:
• https://instagram.com/reel/[ID]
• https://www.instagram.com/reel/[ID]
• https://instagram.com/p/[ID] (if it's a video post)

Tips for best results:
• Choose Reels with clear speech
• Avoid Reels with only background music
• Make sure the Reel is public (not from a private account)
• Ensure you have a stable internet connection

Supported languages:
• English
• German
• More languages may be added in future versions

If you encounter issues:
• Check your internet connection
• Verify the URL is correct and complete
• Try a different Reel if the current one fails
• Use the "More Help" button in error dialogs for specific guidance

For persistent problems, use the "Report Issue" feature in error dialogs.
        """

        layout = [
            [sg.Text("Help & Usage Guide", font=("Helvetica", 14, "bold"))],
            [
                sg.Multiline(
                    help_text.strip(), size=(70, 25), font=("Helvetica", 10), disabled=True, background_color="white"
                )
            ],
            [sg.Push(), sg.Button("Close", key="-CLOSE-"), sg.Push()],
        ]

        window = sg.Window("Help & Usage Guide", layout, modal=True, finalize=True, margins=(15, 15))

        window.read()
        window.close()

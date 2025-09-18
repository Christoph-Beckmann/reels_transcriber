"""Main GUI window for Instagram Reels transcription app."""

import logging
import queue
from typing import Any, Callable, Optional

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
        logging.error("Neither FreeSimpleGUI nor PySimpleGUI is available")

from config.settings import DEFAULT_CONFIG
from utils.error_handler import (
    ErrorDetails,
    process_error_for_user,
)
from utils.logging_config import log_user_action
from utils.progress import ProcessingStage, ProgressUpdate

from .error_dialogs import EnhancedErrorDialog, ProcessingTimer, UserFeedbackHelper
from .worker import ProcessingWorker

logger = logging.getLogger(__name__)


class MainWindow:
    """
    Main GUI window for the Instagram Reels transcription application.
    Enhanced with comprehensive error handling and user feedback.
    """

    def __init__(self, title: str = "Instagram Reels Transcriber"):
        """
        Initialize the main window.

        Args:
            title: Window title
        """
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI library not available. Please install FreeSimpleGUI: pip install FreeSimpleGUI")

        self.title = title
        self.window: Optional[sg.Window] = None
        self.message_queue: queue.Queue = queue.Queue()
        self.worker: Optional[ProcessingWorker] = None
        self.is_processing = False
        self.current_transcript = ""
        self.processing_timer = ProcessingTimer()
        self.last_error_details: Optional[ErrorDetails] = None

        # Initialize worker
        self.worker = ProcessingWorker(self.message_queue, DEFAULT_CONFIG.to_dict())

        # Configure GUI theme and settings
        sg.theme("LightBlue3")  # Professional, clean theme
        sg.set_options(
            element_padding=(5, 5),
            margins=(10, 10),
            button_element_size=(10, 1),
            auto_size_buttons=False,
            auto_size_text=False,
        )

        logger.info("Main window initialized with enhanced error handling")

    def create_layout(self) -> list:
        """
        Create the window layout.

        Returns:
            list: GUI layout definition
        """
        # Main content layout
        layout = [
            # Header
            [
                sg.Text(
                    "Instagram Reels Transcriber",
                    font=("Helvetica", 16, "bold"),
                    justification="center",
                    pad=((0, 0), (10, 20)),
                )
            ],
            # URL input section
            [sg.Text("Enter Instagram Reel URL:", font=("Helvetica", 10), pad=((0, 0), (0, 5)))],
            [
                sg.Input(
                    size=(70, 1),
                    key="-URL-",
                    font=("Helvetica", 10),
                    enable_events=True,
                    tooltip="Paste the Instagram Reel URL here (e.g., https://instagram.com/reel/...)",
                )
            ],
            # Control buttons
            [
                sg.Push(),
                sg.Button(
                    "Start Transcription",
                    key="-START-",
                    button_color=("white", "#4CAF50"),
                    font=("Helvetica", 10, "bold"),
                    size=(15, 1),
                    tooltip="Begin transcribing the Instagram Reel",
                ),
                sg.Button(
                    "Stop",
                    key="-STOP-",
                    button_color=("white", "#F44336"),
                    font=("Helvetica", 10),
                    size=(8, 1),
                    disabled=True,
                    tooltip="Stop current transcription",
                ),
                sg.Button(
                    "Clear",
                    key="-CLEAR-",
                    button_color=("white", "#FF9800"),
                    font=("Helvetica", 10),
                    size=(8, 1),
                    tooltip="Clear the URL and transcript",
                ),
                sg.Button(
                    "Help",
                    key="-HELP-",
                    button_color=("white", "#9C27B0"),
                    font=("Helvetica", 10),
                    size=(8, 1),
                    tooltip="Show help and troubleshooting information",
                ),
                sg.Push(),
            ],
            # Status and progress section
            [sg.Text("Status:", font=("Helvetica", 10, "bold"), pad=((0, 5), (20, 5)))],
            [
                sg.Text(
                    "Ready to transcribe Instagram Reels",
                    key="-STATUS-",
                    size=(80, 1),
                    font=("Helvetica", 10),
                    text_color="#333333",
                    relief=sg.RELIEF_SUNKEN,
                    pad=((5, 5), (0, 5)),
                )
            ],
            # Progress bar with time estimate
            [sg.Text("Progress:", font=("Helvetica", 10, "bold"), pad=((0, 5), (10, 5)))],
            [
                sg.ProgressBar(
                    100,
                    orientation="h",
                    size=(50, 20),
                    key="-PROGRESS-",
                    relief=sg.RELIEF_SUNKEN,
                    border_width=2,
                    bar_color=("#4CAF50", "#E0E0E0"),
                ),
                sg.Text("", key="-TIME-ESTIMATE-", font=("Helvetica", 9), text_color="#666666", size=(25, 1)),
            ],
            # Transcript section
            [sg.Text("Transcript:", font=("Helvetica", 10, "bold"), pad=((0, 5), (20, 5)))],
            [
                sg.Multiline(
                    default_text="The transcript will appear here once processing is complete...",
                    size=(80, 18),
                    key="-TRANSCRIPT-",
                    font=("Courier", 10),
                    text_color="#333333",
                    background_color="white",
                    disabled=False,
                    autoscroll=True,
                    auto_refresh=True,
                    reroute_stdout=False,
                    reroute_stderr=False,
                    reroute_cprint=False,
                    enable_events=False,
                    tooltip="The transcribed text will appear here. You can select and copy it.",
                )
            ],
            # Bottom buttons
            [
                sg.Push(),
                sg.Button(
                    "Copy to Clipboard",
                    key="-COPY-",
                    button_color=("white", "#2196F3"),
                    font=("Helvetica", 10),
                    size=(15, 1),
                    disabled=True,
                    tooltip="Copy the transcript to clipboard",
                ),
                sg.Button(
                    "Retry Last",
                    key="-RETRY-",
                    button_color=("white", "#FF5722"),
                    font=("Helvetica", 10),
                    size=(12, 1),
                    disabled=True,
                    tooltip="Retry the last failed operation",
                ),
                sg.Button(
                    "Exit",
                    key="-EXIT-",
                    button_color=("white", "#757575"),
                    font=("Helvetica", 10),
                    size=(8, 1),
                    tooltip="Close the application",
                ),
                sg.Push(),
            ],
        ]

        return layout

    def create_window(self) -> sg.Window:
        """
        Create and return the main window.

        Returns:
            sg.Window: Created window object
        """
        layout = self.create_layout()

        window = sg.Window(
            self.title,
            layout,
            resizable=False,
            enable_close_attempted_event=True,
            finalize=True,
            icon=None,  # Could add an icon file later
            margins=(15, 15),
            element_justification="left",
            font=("Helvetica", 10),
            return_keyboard_events=False,
            use_default_focus=False,
            disable_close=False,
            disable_minimize=False,
            location=(None, None),  # Center on screen
            size=(None, None),  # Auto-size
            alpha_channel=None,
        )

        # Set initial focus to URL input
        window["-URL-"].set_focus()

        return window

    def update_progress(self, progress_update: ProgressUpdate) -> None:
        """
        Update the progress bar and status from progress update.

        Args:
            progress_update: Progress update object
        """
        if not self.window:
            return

        try:
            # Update progress bar
            self.window["-PROGRESS-"].update(progress_update.progress)

            # Update status message
            status_message = progress_update.message
            if progress_update.details:
                status_message += f" ({progress_update.details})"

            self.window["-STATUS-"].update(status_message)

            # Update time estimate
            if self.is_processing:
                elapsed_time = self.processing_timer.format_elapsed_time()
                remaining_time = self.processing_timer.estimate_remaining_time(progress_update.progress)

                if remaining_time:
                    time_text = f"Elapsed: {elapsed_time} | {remaining_time}"
                else:
                    time_text = f"Elapsed: {elapsed_time}"

                self.window["-TIME-ESTIMATE-"].update(time_text)

            # Update button states based on stage
            if progress_update.stage == ProcessingStage.COMPLETED:
                self.window["-START-"].update(disabled=False)
                self.window["-STOP-"].update(disabled=True)
                self.window["-CLEAR-"].update(disabled=False)
                self.window["-COPY-"].update(disabled=False if self.current_transcript else True)
                self.window["-RETRY-"].update(disabled=True)
                self.window["-TIME-ESTIMATE-"].update(f"Completed in {self.processing_timer.format_elapsed_time()}")
                self.is_processing = False
            elif progress_update.stage == ProcessingStage.ERROR:
                self.window["-START-"].update(disabled=False)
                self.window["-STOP-"].update(disabled=True)
                self.window["-CLEAR-"].update(disabled=False)
                self.window["-RETRY-"].update(
                    disabled=False if self.last_error_details and self.last_error_details.retry_recommended else True
                )
                self.is_processing = False
            elif progress_update.stage != ProcessingStage.IDLE:
                self.window["-START-"].update(disabled=True)
                self.window["-STOP-"].update(disabled=False)
                self.window["-CLEAR-"].update(disabled=True)
                self.window["-COPY-"].update(disabled=True)
                self.window["-RETRY-"].update(disabled=True)
                self.is_processing = True

            # Refresh window
            self.window.refresh()

        except Exception as e:
            logger.error(f"Failed to update progress: {e}")

    def update_transcript(self, transcript: str) -> None:
        """
        Update the transcript display.

        Args:
            transcript: Transcript text to display
        """
        if not self.window:
            return

        try:
            self.current_transcript = transcript
            self.window["-TRANSCRIPT-"].update(transcript)
            self.window["-COPY-"].update(disabled=False if transcript.strip() else True)
            logger.debug(f"Transcript updated: {len(transcript)} characters")

        except Exception as e:
            logger.error(f"Failed to update transcript: {e}")

    def show_error(self, error_message: str, title: str = "Error", error_details: Optional[ErrorDetails] = None) -> str:
        """
        Show error message to user with enhanced error handling.

        Args:
            error_message: Error message to display
            title: Error dialog title
            error_details: Optional detailed error information

        Returns:
            User's response: 'retry', 'help', 'report', or 'close'
        """
        try:
            log_user_action("error_shown", False, {"error_message": error_message, "title": title})

            if error_details:
                self.last_error_details = error_details
                return EnhancedErrorDialog.show_error_with_guidance(error_details, title)
            else:
                # Fallback for simple errors
                sg.popup_error(error_message, title=title, font=("Helvetica", 10))
                return "close"

        except Exception as e:
            logger.error(f"Failed to show error dialog: {e}")
            # Ultimate fallback
            sg.popup_error(error_message, title=title)
            return "close"

    def show_info(self, message: str, title: str = "Information") -> None:
        """
        Show information message to user.

        Args:
            message: Information message to display
            title: Dialog title
        """
        try:
            sg.popup(message, title=title, font=("Helvetica", 10))
        except Exception as e:
            logger.error(f"Failed to show info popup: {e}")

    def show_confirmation(self, message: str, title: str = "Confirm") -> bool:
        """
        Show confirmation dialog for destructive actions.

        Args:
            message: Confirmation message
            title: Dialog title

        Returns:
            True if user confirmed, False otherwise
        """
        return UserFeedbackHelper.show_confirmation(message, title)

    def show_processing_stats(self, completion_data: dict[str, Any]) -> None:
        """
        Show processing completion statistics.

        Args:
            completion_data: Processing completion data
        """
        UserFeedbackHelper.show_processing_stats(completion_data)

    def show_help(self) -> None:
        """Show general help information."""
        UserFeedbackHelper.show_help()

    def copy_to_clipboard(self) -> bool:
        """
        Copy current transcript to clipboard.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.current_transcript.strip():
                self.show_error("No transcript to copy!", "Copy Error")
                return False

            # Copy to clipboard using FreeSimpleGUI
            sg.clipboard_set(self.current_transcript)
            self.show_info("Transcript copied to clipboard!", "Success")
            logger.info("Transcript copied to clipboard")
            return True

        except Exception as e:
            error_msg = f"Failed to copy to clipboard: {e}"
            logger.error(error_msg)
            self.show_error(error_msg, "Copy Error")
            return False

    def clear_all(self) -> None:
        """Clear URL input and transcript."""
        try:
            # Show confirmation for destructive action if there's content
            if self.current_transcript.strip() and not self.show_confirmation(
                "This will clear the current transcript. Are you sure?", "Clear Transcript"
            ):
                return

            if self.window:
                self.window["-URL-"].update("")
                self.window["-TRANSCRIPT-"].update("The transcript will appear here once processing is complete...")
                self.window["-STATUS-"].update("Ready to transcribe Instagram Reels")
                self.window["-PROGRESS-"].update(0)
                self.window["-TIME-ESTIMATE-"].update("")
                self.window["-COPY-"].update(disabled=True)
                self.window["-RETRY-"].update(disabled=True)
                self.current_transcript = ""
                self.last_error_details = None
                self.window["-URL-"].set_focus()

            log_user_action("interface_cleared", True)
            logger.debug("Interface cleared")

        except Exception as e:
            logger.error(f"Failed to clear interface: {e}")

    def validate_url_input(self, url: str) -> bool:
        """
        Basic URL validation for UI feedback.

        Args:
            url: URL to validate

        Returns:
            bool: True if URL looks valid
        """
        url = url.strip()
        return bool(url and "instagram.com" in url and "reel" in url)

    def set_processing_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback function for processing start.

        Args:
            callback: Function to call when processing should start
        """
        self.processing_callback = callback

    def run(self) -> None:
        """
        Run the main window event loop.
        """
        if not GUI_AVAILABLE:
            raise RuntimeError("GUI library not available")

        # Create window
        self.window = self.create_window()
        logger.info("Main window created and displayed")

        try:
            # Main event loop
            while True:
                # Read window events with timeout for message processing
                event, values = self.window.read(timeout=100)

                # Check for messages from worker thread
                try:
                    while True:
                        message = self.message_queue.get_nowait()
                        self._handle_worker_message(message)
                except queue.Empty:
                    pass

                # Handle window events
                if event == sg.WIN_CLOSED or event == "-EXIT-":
                    logger.info("Window close requested")
                    break

                elif event == "-START-":
                    self._handle_start_button(values)

                elif event == "-STOP-":
                    self._handle_stop_button()

                elif event == "-CLEAR-":
                    self.clear_all()

                elif event == "-COPY-":
                    self.copy_to_clipboard()

                elif event == "-RETRY-":
                    self._handle_retry_button()

                elif event == "-HELP-":
                    self.show_help()

                elif event == "-URL-":
                    # URL input changed - could add real-time validation feedback
                    url = values["-URL-"].strip()
                    if url:
                        if self.validate_url_input(url):
                            self.window["-STATUS-"].update("Instagram Reel URL detected")
                        else:
                            self.window["-STATUS-"].update("Please enter a valid Instagram Reel URL")
                    else:
                        self.window["-STATUS-"].update("Ready to transcribe Instagram Reels")

                elif event == sg.TIMEOUT_KEY:
                    # Regular timeout - used for message processing
                    continue

                else:
                    # Unknown event
                    logger.debug(f"Unhandled event: {event}")

        except Exception as e:
            logger.error(f"Error in main event loop: {e}")
            self.show_error(f"Application error: {e}", "Critical Error")

        finally:
            # Clean up worker
            if self.worker and self.worker.is_processing():
                logger.info("Stopping worker...")
                self.worker.stop_processing()

            # Clean up window
            if self.window:
                self.window.close()
                logger.info("Window closed")

    def _handle_start_button(self, values: dict) -> None:
        """Handle start button click."""
        url = values["-URL-"].strip()

        if not url:
            self.show_error("Please enter an Instagram Reel URL.", "Missing URL")
            return

        if not self.validate_url_input(url):
            self.show_error(
                "Please enter a valid Instagram Reel URL.\n\n"
                "Supported formats:\n"
                "• https://instagram.com/reel/[ID]\n"
                "• https://www.instagram.com/reel/[ID]\n"
                "• https://instagram.com/p/[ID] (for video posts)",
                "Invalid URL",
            )
            return

        if self.is_processing:
            logger.warning("Already processing, ignoring start request")
            return

        # Clear any previous error state
        self.last_error_details = None

        # Start processing with worker
        logger.info(f"Start processing requested for URL: {url}")
        log_user_action("transcription_started", True, {"url_length": len(url)})

        try:
            self.processing_timer.start()

            if self.worker and self.worker.start_processing(url):
                self.is_processing = True
                self.window["-START-"].update(disabled=True)
                self.window["-CLEAR-"].update(disabled=True)
                self.window["-RETRY-"].update(disabled=True)
                self.window["-STATUS-"].update("Starting transcription...")
                self.window["-PROGRESS-"].update(0)
                self.window["-TIME-ESTIMATE-"].update("Initializing...")
            else:
                self.show_error("Failed to start processing. Please try again.", "Processing Error")
        except Exception as e:
            logger.error(f"Failed to start worker: {e}")
            error_details = process_error_for_user(e, "starting_transcription", "start_transcription")
            self.show_error(f"Failed to start processing: {e}", "Processing Error", error_details)

    def _handle_retry_button(self) -> None:
        """Handle retry button click."""
        if not self.last_error_details or not self.last_error_details.retry_recommended:
            return

        # Get the current URL and retry
        url = self.window["-URL-"].get().strip()
        if url:
            log_user_action("retry_attempted", True, {"error_code": self.last_error_details.error_code})
            self._handle_start_button({"-URL-": url})
        else:
            self.show_error("No URL available to retry.", "Retry Error")

    def _handle_stop_button(self) -> None:
        """Handle stop button click."""
        if self.worker and self.is_processing:
            logger.info("Stop processing requested")
            try:
                self.worker.stop_processing()
                self.window["-STATUS-"].update("Processing stopped by user")
                self.window["-START-"].update(disabled=False)
                self.window["-STOP-"].update(disabled=True)
                self.window["-CLEAR-"].update(disabled=False)
                self.is_processing = False
            except Exception as e:
                logger.error(f"Failed to stop processing: {e}")
                self.show_error(f"Failed to stop processing: {e}", "Stop Error")

    def _handle_worker_message(self, message: dict) -> None:
        """Handle message from worker thread."""
        try:
            message_type = message.get("type")

            if message_type == "progress":
                progress_update = message.get("data")
                if progress_update:
                    self.update_progress(progress_update)

            elif message_type == "transcript":
                transcript = message.get("data", "")
                self.update_transcript(transcript)

            elif message_type == "error":
                error_msg = message.get("data", "Unknown error")

                # Try to process the error for better user feedback
                try:
                    # Create a generic exception for error processing
                    error_exception = Exception(error_msg)
                    error_details = process_error_for_user(error_exception, "transcription_pipeline", "transcribe_reel")

                    result = self.show_error(error_msg, "Processing Error", error_details)

                    # Handle user's response to error
                    if result == "retry" and error_details.retry_recommended:
                        # Auto-retry if user chooses to retry
                        self._handle_retry_button()

                except Exception as e:
                    logger.error(f"Failed to process error details: {e}")
                    self.show_error(error_msg, "Processing Error")

            elif message_type == "error_detailed":
                error_data = message.get("data", {})
                error_msg = error_data.get("message", "Unknown error")
                error_details = error_data.get("details")

                if error_details:
                    result = self.show_error(error_msg, "Processing Error", error_details)

                    # Handle user's response to error
                    if result == "retry" and error_details.retry_recommended:
                        self._handle_retry_button()
                else:
                    self.show_error(error_msg, "Processing Error")

            elif message_type == "complete":
                completion_data = message.get("data", {})

                if isinstance(completion_data, dict):
                    transcript = completion_data.get("transcript", "")
                    if transcript:
                        self.update_transcript(transcript)

                    # Show completion statistics
                    self.show_processing_stats(completion_data)

                    log_user_action(
                        "transcription_completed",
                        True,
                        {
                            "execution_time": completion_data.get("execution_time", 0),
                            "transcript_length": len(transcript),
                        },
                    )
                else:
                    # Backwards compatibility - completion_data might be just the transcript
                    transcript = completion_data if isinstance(completion_data, str) else str(completion_data)
                    self.update_transcript(transcript)

                self.window["-STATUS-"].update("Transcription completed successfully!")

            else:
                logger.warning(f"Unknown worker message type: {message_type}")

        except Exception as e:
            logger.error(f"Failed to handle worker message: {e}")

    def send_message_to_gui(self, message: dict) -> None:
        """
        Send message from worker thread to GUI.

        Args:
            message: Message dictionary to send
        """
        try:
            self.message_queue.put(message)
        except Exception as e:
            logger.error(f"Failed to send message to GUI: {e}")


# Convenience function
def create_main_window(title: str = "Instagram Reels Transcriber") -> MainWindow:
    """
    Create and return main window instance.

    Args:
        title: Window title

    Returns:
        MainWindow: Created window instance
    """
    return MainWindow(title)

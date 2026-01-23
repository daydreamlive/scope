"""Base class for event-driven async processors.

Event processors handle discrete, triggered operations (as opposed to
continuous frame processing). They run asynchronously and don't block
the main pipeline.

Examples:
    - Prompt enhancement (text → text)
    - Image generation (text → image)
    - Captioning (image → text)
    - Style extraction (image → embeddings)
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

logger = logging.getLogger(__name__)

# Type variables for input and output
TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput")


class ProcessorState(Enum):
    """Current state of an event processor."""

    IDLE = "idle"
    PROCESSING = "processing"
    CANCELLED = "cancelled"


@dataclass
class ProcessorResult(Generic[TOutput]):
    """Result from an event processor."""

    success: bool
    output: TOutput | None = None
    error: str | None = None
    cancelled: bool = False


@dataclass
class ProcessorConfig:
    """Configuration for an event processor."""

    cancel_on_new: bool = True  # Cancel in-flight work when new event arrives
    timeout: float | None = None  # Optional timeout in seconds
    name: str = "EventProcessor"  # Name for logging


class EventProcessor(ABC, Generic[TInput, TOutput]):
    """Base class for discrete, async event processing.

    Subclasses implement `process()` to handle events. The processor
    runs work in a background thread and calls the callback when done.

    Features:
        - Non-blocking: doesn't block the main pipeline
        - Cancellable: can cancel in-flight work when new event arrives
        - Thread-safe: safe to call submit() from any thread
        - Independent: multiple processors can run concurrently

    Example:
        class PromptEnhancer(EventProcessor[str, str]):
            def process(self, prompt: str) -> str:
                return enhance_with_llm(prompt)

        enhancer = PromptEnhancer()
        enhancer.submit("a cat", callback=lambda r: print(r.output))
    """

    def __init__(self, config: ProcessorConfig | None = None):
        """Initialize the event processor.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or ProcessorConfig()
        self._thread: threading.Thread | None = None
        self._cancel_requested = threading.Event()
        self._lock = threading.Lock()
        self._state = ProcessorState.IDLE
        self._current_callback: Callable[[ProcessorResult[TOutput]], None] | None = None

        # Track the latest result for polling (alternative to callbacks)
        self._latest_result: ProcessorResult[TOutput] | None = None
        self._result_ready = threading.Event()

    @property
    def state(self) -> ProcessorState:
        """Current processor state."""
        with self._lock:
            return self._state

    @property
    def is_processing(self) -> bool:
        """Whether the processor is currently working."""
        return self.state == ProcessorState.PROCESSING

    @property
    def latest_result(self) -> ProcessorResult[TOutput] | None:
        """Get the latest result (for polling instead of callbacks)."""
        with self._lock:
            return self._latest_result

    def submit(
        self,
        event: TInput,
        callback: Callable[[ProcessorResult[TOutput]], None] | None = None,
    ) -> bool:
        """Submit an event for async processing.

        If cancel_on_new is True and there's work in flight, it will be
        cancelled before starting the new work.

        Args:
            event: The input event to process.
            callback: Optional callback when processing completes.
                     Called with ProcessorResult containing output or error.

        Returns:
            True if submitted, False if rejected.
        """
        with self._lock:
            # Cancel in-flight work if configured
            if self.config.cancel_on_new and self._thread and self._thread.is_alive():
                logger.debug(
                    f"{self.config.name}: Cancelling in-flight work for new event"
                )
                self._cancel_requested.set()
                # Don't wait for thread - let it finish in background

            # Reset state
            self._cancel_requested.clear()
            self._result_ready.clear()
            self._state = ProcessorState.PROCESSING
            self._current_callback = callback

            # Start worker thread
            self._thread = threading.Thread(
                target=self._worker,
                args=(event,),
                daemon=True,
                name=f"{self.config.name}-worker",
            )
            self._thread.start()

            logger.debug(f"{self.config.name}: Submitted event for processing")
            return True

    def cancel(self) -> bool:
        """Request cancellation of in-flight work.

        Returns:
            True if cancellation was requested, False if nothing to cancel.
        """
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._cancel_requested.set()
                self._state = ProcessorState.CANCELLED
                logger.debug(f"{self.config.name}: Cancellation requested")
                return True
            return False

    def wait(self, timeout: float | None = None) -> ProcessorResult[TOutput] | None:
        """Wait for the current processing to complete.

        Args:
            timeout: Max seconds to wait. None = wait forever.

        Returns:
            The result, or None if timeout.
        """
        if self._result_ready.wait(timeout=timeout):
            return self._latest_result
        return None

    def _worker(self, event: TInput) -> None:
        """Worker thread that runs the processing."""
        result: ProcessorResult[TOutput]

        try:
            # Check for early cancellation
            if self._cancel_requested.is_set():
                result = ProcessorResult(success=False, cancelled=True)
            else:
                # Run the actual processing
                output = self.process(event)

                # Check if cancelled during processing
                if self._cancel_requested.is_set():
                    result = ProcessorResult(success=False, cancelled=True)
                else:
                    result = ProcessorResult(success=True, output=output)

        except Exception as e:
            logger.warning(f"{self.config.name}: Processing failed: {e}")
            result = ProcessorResult(success=False, error=str(e))

        # Store result and update state
        with self._lock:
            if not self._cancel_requested.is_set():
                self._latest_result = result
                self._state = ProcessorState.IDLE
                self._result_ready.set()

                # Call callback if provided
                if self._current_callback:
                    try:
                        self._current_callback(result)
                    except Exception as e:
                        logger.error(f"{self.config.name}: Callback error: {e}")

    @abstractmethod
    def process(self, event: TInput) -> TOutput:
        """Process an event. Override in subclass.

        This method runs in a background thread. It should:
        - Be thread-safe
        - Check self._cancel_requested periodically for long operations
        - Raise exceptions on error (they'll be caught and reported)

        Args:
            event: The input to process.

        Returns:
            The processed output.
        """
        raise NotImplementedError

    def check_cancelled(self) -> bool:
        """Check if cancellation was requested. Call periodically in process().

        Returns:
            True if should stop processing.
        """
        return self._cancel_requested.is_set()

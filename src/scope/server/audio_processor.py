"""Audio processor for streaming audio from VibeVoice and similar audio pipelines."""

import logging
import queue
import threading
import time
from typing import Generator

import numpy as np
import torch

from .pipeline_manager import PipelineManager, PipelineNotAvailableException

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 24000  # 24kHz sample rate (matches VibeVoice)
CHUNK_SIZE = 1920  # ~80ms chunks at 24kHz
OUTPUT_QUEUE_MAX_SIZE = 50  # Max number of audio chunks to buffer


class AudioProcessor:
    """Processor for audio pipelines like VibeVoice.

    This processes text prompts through audio pipelines and manages
    the output queue of audio chunks.
    """

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        initial_parameters: dict = None,
        notification_callback: callable = None,
    ):
        self.pipeline_manager = pipeline_manager
        self.parameters = initial_parameters or {}
        self.notification_callback = notification_callback

        self.output_queue = queue.Queue(maxsize=OUTPUT_QUEUE_MAX_SIZE)
        self.parameters_queue = queue.Queue(maxsize=8)

        self.worker_thread: threading.Thread | None = None
        self.shutdown_event = threading.Event()
        self.running = False
        self._generation_complete = False

        # Track current text for generation
        self._current_text = ""
        self._text_submitted = False

        logger.info("AudioProcessor initialized")

    def start(self):
        """Start the audio processing worker thread."""
        if self.running:
            return

        self.running = True
        self.shutdown_event.clear()
        self._generation_complete = False

        # Extract initial text from parameters
        if "prompts" in self.parameters:
            prompts = self.parameters["prompts"]
            if prompts and len(prompts) > 0:
                if isinstance(prompts[0], dict):
                    self._current_text = prompts[0].get("text", "")
                elif isinstance(prompts[0], str):
                    self._current_text = prompts[0]
                else:
                    self._current_text = str(prompts[0])
                self._text_submitted = True

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        logger.info("AudioProcessor started")

    def stop(self, error_message: str = None):
        """Stop the audio processor and clean up."""
        if not self.running:
            return

        self.running = False
        self.shutdown_event.set()

        if self.worker_thread and self.worker_thread.is_alive():
            if threading.current_thread() != self.worker_thread:
                self.worker_thread.join(timeout=5.0)

        # Clear the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("AudioProcessor stopped")

        # Notify callback that processor has stopped
        if self.notification_callback:
            try:
                message = {"type": "stream_stopped"}
                if error_message:
                    message["error_message"] = error_message
                self.notification_callback(message)
            except Exception as e:
                logger.error(f"Error in audio processor stop callback: {e}")

    def get(self) -> np.ndarray | None:
        """Get the next audio chunk from the output queue.

        Returns:
            Audio chunk as numpy array, or None if no chunk available.
        """
        if not self.running:
            return None

        try:
            chunk = self.output_queue.get_nowait()
            return chunk
        except queue.Empty:
            return None

    def is_complete(self) -> bool:
        """Check if audio generation is complete.

        Returns:
            True if generation is complete and queue is empty.
        """
        return self._generation_complete and self.output_queue.empty()

    def update_parameters(self, parameters: dict):
        """Update parameters for the next generation.

        Args:
            parameters: New parameters to apply.
        """
        # Handle prompt updates
        if "prompts" in parameters:
            prompts = parameters["prompts"]
            if prompts and len(prompts) > 0:
                if isinstance(prompts[0], dict):
                    self._current_text = prompts[0].get("text", "")
                elif isinstance(prompts[0], str):
                    self._current_text = prompts[0]
                else:
                    self._current_text = str(prompts[0])
                self._text_submitted = True
                logger.info(f"Text updated: {self._current_text[:50]}...")

        try:
            self.parameters_queue.put_nowait(parameters)
        except queue.Full:
            logger.warning("Parameter queue full, dropping update")

    def worker_loop(self):
        """Main worker loop that processes text through the audio pipeline."""
        logger.info("Audio worker thread started")

        while self.running and not self.shutdown_event.is_set():
            try:
                # Check for new parameters
                try:
                    new_parameters = self.parameters_queue.get_nowait()
                    self.parameters = {**self.parameters, **new_parameters}
                except queue.Empty:
                    pass

                # Check if we should be paused
                paused = self.parameters.get("paused", False)
                if paused:
                    self.shutdown_event.wait(0.01)
                    continue

                # Only generate if text has been submitted
                if not self._text_submitted:
                    self.shutdown_event.wait(0.01)
                    continue

                # Get the pipeline
                pipeline = self.pipeline_manager.get_pipeline()

                # Check if this is an audio pipeline
                if not hasattr(pipeline, "is_audio_pipeline") or not pipeline.is_audio_pipeline:
                    logger.error("Pipeline is not an audio pipeline")
                    self.shutdown_event.wait(0.1)
                    continue

                # Generate audio
                logger.info(f"Starting audio generation for: {self._current_text[:50]}...")
                self._generation_complete = False

                try:
                    # Call pipeline to get audio generator
                    audio_generator = pipeline(text=self._current_text, **self.parameters)

                    if audio_generator is None:
                        logger.warning("Pipeline returned None")
                        self._text_submitted = False
                        continue

                    # Process audio chunks
                    chunk_count = 0
                    for chunk_tensor in audio_generator:
                        if not self.running or self.shutdown_event.is_set():
                            break

                        # Convert tensor to numpy array
                        if isinstance(chunk_tensor, torch.Tensor):
                            chunk_np = chunk_tensor.cpu().numpy().astype(np.float32)
                        else:
                            chunk_np = np.asarray(chunk_tensor, dtype=np.float32)

                        # Ensure 1D
                        if chunk_np.ndim > 1:
                            chunk_np = chunk_np.reshape(-1)

                        # Put chunk in output queue
                        try:
                            self.output_queue.put(chunk_np, timeout=1.0)
                            chunk_count += 1
                        except queue.Full:
                            logger.warning("Output queue full, dropping chunk")

                    logger.info(f"Audio generation complete, {chunk_count} chunks produced")
                    self._generation_complete = True
                    self._text_submitted = False

                except Exception as e:
                    logger.error(f"Error during audio generation: {e}", exc_info=True)
                    self._text_submitted = False

            except PipelineNotAvailableException as e:
                logger.debug(f"Pipeline temporarily unavailable: {e}")
                self.shutdown_event.wait(0.1)
            except Exception as e:
                logger.error(f"Error in audio worker loop: {e}", exc_info=True)
                self.shutdown_event.wait(0.1)

        logger.info("Audio worker thread stopped")


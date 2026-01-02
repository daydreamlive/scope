#!/usr/bin/env python3
"""Standalone preprocessor worker process.

This script runs as a completely separate process (not subprocess) for preprocessing.
It is spawned via subprocess.Popen to ensure complete CUDA context isolation.

Usage:
    python -m scope.core.preprocessors.preprocessor_worker_process \
        --preprocessor-type depthanything \
        --encoder vits \
        --input-port 5555 \
        --output-port 5556 \
        --ready-file /tmp/preprocessor_worker_ready

The ready-file is created when the model is loaded and ready to accept frames.
"""

import argparse
import logging
import os
import pickle
import signal
import sys
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PreprocessorWorkerProcess")


def load_preprocessor_pipeline(preprocessor_type: str, encoder: str | None = None):
    """Load the appropriate preprocessor pipeline based on type."""
    import torch

    # Built-in preprocessors with custom initialization
    if preprocessor_type == "depthanything":
        from scope.core.pipelines.depthanything import DepthAnythingPipeline

        encoder = encoder or "vits"  # Default to vits
        logger.info(f"Loading DepthAnythingPipeline with encoder: {encoder}")
        pipeline = DepthAnythingPipeline(
            encoder=encoder,
            device=torch.device("cuda"),
            dtype=torch.float16,
            input_size=392,  # Default input size
            streaming=True,  # Use streaming mode for real-time processing
            output_format="rgb",  # Output format for depth preprocessing
        )
        pipeline.prepare()  # Load the model
        logger.info("DepthAnythingPipeline loaded successfully!")
        return pipeline

    elif preprocessor_type == "passthrough":
        from scope.core.pipelines.passthrough import PassthroughPipeline

        logger.info("Loading PassthroughPipeline")
        pipeline = PassthroughPipeline(
            height=512,  # Will be resized to match input
            width=512,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        logger.info("PassthroughPipeline loaded successfully!")
        return pipeline

    # Try to load from registry (for plugin preprocessors)
    else:
        try:
            from scope.core.preprocessors.registry import PreprocessorRegistry
            from scope.core.pipelines.registry import PipelineRegistry

            # Check PreprocessorRegistry first
            preprocessor_class = PreprocessorRegistry.get(preprocessor_type)

            # If not found, check PipelineRegistry (plugin preprocessors are pipelines)
            if preprocessor_class is None:
                preprocessor_class = PipelineRegistry.get(preprocessor_type)

            if preprocessor_class is None:
                raise ValueError(f"Preprocessor {preprocessor_type} not found in registry")

            logger.info(f"Loading plugin preprocessor: {preprocessor_type}")
            # Plugin pipelines are instantiated with load_params (which may be empty)
            # Try instantiating with no parameters first (plugin pipelines should handle defaults)
            try:
                pipeline = preprocessor_class()
            except TypeError:
                # If that fails, try with device and dtype (common pattern)
                try:
                    pipeline = preprocessor_class(
                        device=torch.device("cuda"),
                        dtype=torch.bfloat16,
                    )
                except TypeError:
                    # If that also fails, try with just device
                    pipeline = preprocessor_class(
                        device=torch.device("cuda"),
                    )

            logger.info(f"Plugin preprocessor {preprocessor_type} loaded successfully!")
            return pipeline
        except Exception as e:
            logger.error(f"Failed to load preprocessor {preprocessor_type} from registry: {e}")
            raise ValueError(f"Unknown preprocessor type: {preprocessor_type}") from e


def process_frames_depthanything(pipeline, frames, target_height, target_width):
    """Process frames using depthanything pipeline."""
    import torch

    # Convert numpy to torch tensor and prepare as list of frames for pipeline
    frames_tensor = torch.from_numpy(frames).float()  # [F, H, W, C]

    # Convert to list of frames, each with shape [H, W, C]
    frame_list = [frames_tensor[i] for i in range(frames_tensor.shape[0])]

    # Use the pipeline's __call__ method
    depth_output = pipeline(video=frame_list)  # Returns [T, H, W, 3] in [0, 1]

    # Convert pipeline output [T, H, W, 3] in [0, 1] to [1, 3, T, H, W] in [-1, 1]
    T, H, W, C = depth_output.shape

    # Resize to target dimensions if needed
    if H != target_height or W != target_width:
        depth_output = depth_output.permute(0, 3, 1, 2)  # [T, 3, H, W]
        import torch.nn.functional as F
        depth_output = F.interpolate(
            depth_output,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        depth_output = depth_output.permute(0, 2, 3, 1)  # [T, H, W, 3]

    # Convert [T, H, W, 3] -> [T, 3, H, W]
    depth_tensor = depth_output.permute(0, 3, 1, 2)  # [T, 3, H, W]

    # Convert from [0, 1] to [-1, 1]
    depth_tensor = depth_tensor * 2.0 - 1.0

    # Add batch dimension and rearrange to [1, 3, T, H, W]
    result = depth_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]

    return result.float().cpu()


def process_frames_passthrough(pipeline, frames, target_height, target_width):
    """Process frames using passthrough pipeline (just returns input)."""
    import torch
    from einops import rearrange

    # Convert numpy to torch tensor
    frames_tensor = torch.from_numpy(frames).float()  # [F, H, W, C]

    # Convert to list of frames, each with shape [1, H, W, C] for pipeline
    frame_list = [frames_tensor[i].unsqueeze(0) for i in range(frames_tensor.shape[0])]

    # Use the pipeline's __call__ method
    output = pipeline(video=frame_list)  # Returns [T, H, W, C] in [0, 1]

    # Convert [T, H, W, C] -> [T, C, H, W]
    output_tensor = output.permute(0, 3, 1, 2)  # [T, C, H, W]

    # Resize to target dimensions if needed
    if output_tensor.shape[2] != target_height or output_tensor.shape[3] != target_width:
        import torch.nn.functional as F
        output_tensor = F.interpolate(
            output_tensor,
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )

    # Passthrough should keep values in [0, 1] range (not convert to [-1, 1])
    # VACE expects conditioning inputs in [0, 1] range

    # Add batch dimension and rearrange to [1, C, T, H, W]
    # output_tensor is [T, C, H, W], need [1, C, T, H, W]
    result = output_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    return result.float().cpu()


def process_frames_generic(pipeline, frames, target_height, target_width):
    """Generic processing function for plugin preprocessors.

    This function handles plugin pipelines that may have different interfaces.
    It tries multiple calling conventions to accommodate different pipeline types.
    """
    import torch
    import torch.nn.functional as F

    # Convert numpy to torch tensor
    frames_tensor = torch.from_numpy(frames).float()  # [F, H, W, C]
    num_frames = frames_tensor.shape[0]

    # Try different calling conventions for plugin pipelines
    output = None

    # Try 1: List of frames with video= keyword (standard interface)
    try:
        frame_list = [frames_tensor[i].unsqueeze(0) for i in range(num_frames)]
        output = pipeline(video=frame_list)
    except (TypeError, AttributeError, KeyError):
        # Try 2: List of frames as positional argument
        try:
            frame_list = [frames_tensor[i] for i in range(num_frames)]
            output = pipeline(frame_list)
        except (TypeError, AttributeError):
            # Try 3: Tensor input [B, C, T, H, W] format
            try:
                # Convert [F, H, W, C] -> [1, C, F, H, W]
                input_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
                output = pipeline(input=input_tensor)
            except (TypeError, AttributeError):
                # Try 4: Direct tensor input
                try:
                    input_tensor = frames_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
                    output = pipeline(input_tensor)
                except Exception as e:
                    logger.error(f"Failed to call plugin pipeline with any known interface: {e}")
                    raise ValueError(f"Plugin pipeline does not support any known calling convention") from e

    if output is None:
        raise ValueError("Plugin pipeline returned None")

    # Log the raw output for debugging
    logger.debug(f"Plugin pipeline raw output type: {type(output)}, shape: {output.shape if isinstance(output, torch.Tensor) else 'N/A'}")

    # Normalize output to [T, H, W, C] format
    if isinstance(output, torch.Tensor):
        # Handle different output shapes
        if output.dim() == 5:  # [B, C, T, H, W] or [B, T, C, H, W]
            if output.shape[1] == 3 or output.shape[2] == 3:  # Channel dimension
                # Assume [B, C, T, H, W] -> [T, H, W, C]
                output = output.squeeze(0).permute(1, 2, 3, 0)  # [T, H, W, C]
            else:
                # Assume [B, T, C, H, W] -> [T, H, W, C]
                output = output.squeeze(0).permute(0, 2, 3, 1)  # [T, H, W, C]
        elif output.dim() == 4:  # [T, C, H, W] or [T, H, W, C]
            if output.shape[1] == 3:  # [T, C, H, W]
                output = output.permute(0, 2, 3, 1)  # [T, H, W, C]
            # else already [T, H, W, C]
        elif output.dim() == 3:  # [H, W, C] - single frame, repeat for all frames
            output = output.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # [T, H, W, C]
        else:
            raise ValueError(f"Unexpected output shape from plugin pipeline: {output.shape}")
    else:
        raise ValueError(f"Plugin pipeline returned non-tensor type: {type(output)}")

    # Ensure output is in [0, 1] range (normalize if needed)
    if output.max() > 1.0:
        output = output / 255.0
    elif output.min() < 0.0:
        # If in [-1, 1] range, convert to [0, 1]
        output = (output + 1.0) / 2.0

    # Convert [T, H, W, C] -> [T, C, H, W]
    output_tensor = output.permute(0, 3, 1, 2)  # [T, C, H, W]

    # Resize to target dimensions if needed
    if output_tensor.shape[2] != target_height or output_tensor.shape[3] != target_width:
        output_tensor = F.interpolate(
            output_tensor.unsqueeze(0),  # Add batch dim for interpolation
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)  # Remove batch dim

    # Validate output has frames
    if output_tensor.shape[0] == 0:
        raise ValueError(f"Plugin pipeline returned empty output (0 frames)")

    # Ensure we have exactly 3 channels (RGB) for VACE compatibility
    # VACE expects 3-channel RGB conditioning maps
    if output_tensor.shape[1] == 1:
        # Convert grayscale to RGB by repeating the channel
        output_tensor = output_tensor.repeat(1, 3, 1, 1)
    elif output_tensor.shape[1] != 3:
        logger.warning(f"Plugin pipeline returned {output_tensor.shape[1]} channels, expected 1 or 3. Converting to 3 channels.")
        # If more than 3 channels, take first 3; if less, pad with zeros
        if output_tensor.shape[1] > 3:
            output_tensor = output_tensor[:, :3, :, :]
        else:
            padding = torch.zeros(
                (output_tensor.shape[0], 3 - output_tensor.shape[1], output_tensor.shape[2], output_tensor.shape[3]),
                dtype=output_tensor.dtype,
                device=output_tensor.device
            )
            output_tensor = torch.cat([output_tensor, padding], dim=1)

    # Ensure we have at least 4 frames (VAE stream_encode requires chunks of 4)
    # If we have fewer frames, pad by repeating the last frame
    num_frames_output = output_tensor.shape[0]
    if num_frames_output < 4:
        logger.warning(
            f"Plugin pipeline returned {num_frames_output} frames, but VAE requires at least 4. "
            f"Padding by repeating the last frame."
        )
        last_frame = output_tensor[-1:].repeat(4 - num_frames_output, 1, 1, 1)
        output_tensor = torch.cat([output_tensor, last_frame], dim=0)

    # Add batch dimension and rearrange to [1, C, T, H, W]
    # output_tensor is [T, C, H, W], need [1, C, T, H, W]
    result = output_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    # Final validation - ensure result has correct shape [1, C, T, H, W]
    if result.shape[0] != 1 or result.shape[1] != 3 or result.shape[2] == 0:
        raise ValueError(f"Plugin pipeline returned invalid output shape: {result.shape}, expected [1, 3, T, H, W]")

    # Ensure we have at least 4 frames in the final result
    if result.shape[2] < 4:
        raise ValueError(f"Plugin pipeline returned {result.shape[2]} frames, but VAE requires at least 4 frames")

    logger.debug(f"Plugin preprocessor output shape: {result.shape}")
    return result.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Preprocessor worker process")
    parser.add_argument("--preprocessor-type", type=str, required=True,
                       help="Type of preprocessor to use (e.g., depthanything, passthrough, or plugin preprocessor ID)")
    parser.add_argument("--encoder", type=str, default="vits",
                       choices=["vits", "vitb", "vitl"],
                       help="Encoder size for depthanything (ignored for other types)")
    parser.add_argument("--input-port", type=int, required=True)
    parser.add_argument("--output-port", type=int, required=True)
    parser.add_argument("--ready-file", type=str, required=True,
                       help="File to create when ready (signals parent process)")
    args = parser.parse_args()

    logger.info(f"=" * 60)
    logger.info(f"Preprocessor Worker Process Starting")
    logger.info(f"PID: {os.getpid()}")
    logger.info(f"Preprocessor type: {args.preprocessor_type}")
    if args.preprocessor_type == "depthanything":
        logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Input port: {args.input_port}")
    logger.info(f"Output port: {args.output_port}")
    logger.info(f"=" * 60)

    # Import torch and check CUDA
    import torch
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    else:
        logger.warning("CUDA not available, falling back to CPU")

    # Import zmq
    import zmq

    # ZeroMQ configuration
    ZMQ_TIMEOUT_MS = 1000  # 1 second timeout for recv
    ZMQ_HWM = 100  # High water mark for socket buffers

    # Load preprocessor pipeline
    try:
        pipeline = load_preprocessor_pipeline(args.preprocessor_type, args.encoder)
    except Exception as e:
        logger.error(f"Failed to load preprocessor pipeline: {e}", exc_info=True)
        sys.exit(1)

    # Setup ZeroMQ sockets
    context = zmq.Context()

    # Pull socket for receiving frames
    input_socket = context.socket(zmq.PULL)
    input_socket.setsockopt(zmq.RCVHWM, ZMQ_HWM)
    input_socket.bind(f"tcp://*:{args.input_port}")
    input_socket.setsockopt(zmq.RCVTIMEO, ZMQ_TIMEOUT_MS)

    # Push socket for sending results
    output_socket = context.socket(zmq.PUSH)
    output_socket.setsockopt(zmq.SNDHWM, ZMQ_HWM)
    output_socket.bind(f"tcp://*:{args.output_port}")

    logger.info(f"ZeroMQ sockets bound (input={args.input_port}, output={args.output_port})")

    # Signal readiness by creating the ready file
    ready_path = Path(args.ready_file)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(f"ready:{os.getpid()}")
    logger.info(f"Ready signal written to {args.ready_file}")

    # Handle graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.info(f"Received signal {signum}, shutting down...")
        shutdown_requested = True

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Select processing function based on preprocessor type
    # Built-in preprocessors have specific processing functions
    if args.preprocessor_type == "depthanything":
        process_frames_fn = process_frames_depthanything
    elif args.preprocessor_type == "passthrough":
        process_frames_fn = process_frames_passthrough
    else:
        # Plugin preprocessors use generic processing that handles different interfaces
        logger.info(f"Using generic processing for plugin preprocessor: {args.preprocessor_type}")
        process_frames_fn = process_frames_generic

    # Main processing loop
    logger.info("Starting main processing loop...")
    try:
        while not shutdown_requested:
            try:
                # Receive frame data (with timeout to check shutdown flag)
                message = input_socket.recv()
                data = pickle.loads(message)

                chunk_id = data["chunk_id"]
                frames = data["frames"]  # numpy array [F, H, W, C]
                target_height = data["target_height"]
                target_width = data["target_width"]

                logger.debug(f"Received chunk {chunk_id}, frames shape: {frames.shape}")

                # Process frames using the appropriate function
                start_time = time.time()
                result_tensor = process_frames_fn(pipeline, frames, target_height, target_width)
                inference_time = time.time() - start_time
                num_frames = frames.shape[0]

                logger.info(
                    f"Chunk {chunk_id}: {num_frames} frames in "
                    f"{inference_time:.3f}s ({num_frames / inference_time:.1f} FPS)"
                )

                # Send result
                result = {
                    "chunk_id": chunk_id,
                    "data": result_tensor.numpy(),
                    "timestamp": time.time(),
                }
                output_socket.send(pickle.dumps(result))

                logger.debug(f"Sent result for chunk {chunk_id}, shape: {result_tensor.shape}")

            except zmq.error.Again:
                # Timeout, continue to check shutdown flag
                continue
            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=True)
                continue

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Shutting down...")
        input_socket.close()
        output_socket.close()
        context.term()

        # Cleanup pipeline if it has an offload method
        if hasattr(pipeline, "offload"):
            pipeline.offload()

        # Remove ready file
        if ready_path.exists():
            ready_path.unlink()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

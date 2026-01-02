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

    else:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")


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

    # Convert from [0, 1] to [-1, 1]
    output_tensor = output_tensor * 2.0 - 1.0

    # Add batch dimension and rearrange to [1, C, T, H, W]
    # output_tensor is [T, C, H, W], need [1, C, T, H, W]
    result = output_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, C, T, H, W]

    return result.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Preprocessor worker process")
    parser.add_argument("--preprocessor-type", type=str, required=True,
                       choices=["depthanything", "passthrough"],
                       help="Type of preprocessor to use")
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
    if args.preprocessor_type == "depthanything":
        process_frames_fn = process_frames_depthanything
    elif args.preprocessor_type == "passthrough":
        process_frames_fn = process_frames_passthrough
    else:
        logger.error(f"Unknown preprocessor type: {args.preprocessor_type}")
        sys.exit(1)

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

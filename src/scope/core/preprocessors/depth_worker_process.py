#!/usr/bin/env python3
"""Standalone depth worker process.

This script runs as a completely separate process (not subprocess) for depth estimation.
It is spawned via subprocess.Popen to ensure complete CUDA context isolation.

Usage:
    python -m scope.core.preprocessors.depth_worker_process \
        --encoder vitl \
        --input-port 5555 \
        --output-port 5556 \
        --ready-file /tmp/depth_worker_ready

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
logger = logging.getLogger("DepthWorkerProcess")


def main():
    parser = argparse.ArgumentParser(description="Depth worker process")
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-port", type=int, required=True)
    parser.add_argument("--output-port", type=int, required=True)
    parser.add_argument("--ready-file", type=str, required=True,
                        help="File to create when ready (signals parent process)")
    args = parser.parse_args()

    logger.info(f"=" * 60)
    logger.info(f"Depth Worker Process Starting")
    logger.info(f"PID: {os.getpid()}")
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

    # Load depth pipeline
    logger.info("Loading DepthAnythingPipeline...")
    from scope.core.pipelines.depthanything import DepthAnythingPipeline

    try:
        depth_pipeline = DepthAnythingPipeline(
            encoder=args.encoder,
            device=torch.device("cuda"),
            dtype=torch.float16,
            input_size=392,  # Default input size
            streaming=True,  # Use streaming mode for real-time processing
            output_format="rgb",  # Output format for depth preprocessing
        )
        depth_pipeline.prepare()  # Load the model
        logger.info("DepthAnythingPipeline loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load DepthAnythingPipeline: {e}", exc_info=True)
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

                # Convert numpy to torch tensor and prepare as list of frames for pipeline
                frames_tensor = torch.from_numpy(frames).float()  # [F, H, W, C]

                # Convert to list of frames, each with shape [H, W, C]
                frame_list = [frames_tensor[i] for i in range(frames_tensor.shape[0])]

                # Use the pipeline's __call__ method
                start_time = time.time()
                depth_output = depth_pipeline(video=frame_list)  # Returns [T, H, W, 3] in [0, 1]
                inference_time = time.time() - start_time
                num_frames = frames.shape[0]

                logger.info(
                    f"Chunk {chunk_id}: {num_frames} frames in "
                    f"{inference_time:.3f}s ({num_frames / inference_time:.1f} FPS)"
                )

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
                depth = depth_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]

                # Keep as float32 for numpy serialization
                depth_cpu = depth.float().cpu()

                # Send result
                result = {
                    "chunk_id": chunk_id,
                    "depth": depth_cpu.numpy(),
                    "timestamp": time.time(),
                }
                output_socket.send(pickle.dumps(result))

                logger.debug(f"Sent result for chunk {chunk_id}, depth shape: {depth_cpu.shape}")

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
        depth_pipeline.offload()

        # Remove ready file
        if ready_path.exists():
            ready_path.unlink()

        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()

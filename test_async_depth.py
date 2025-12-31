"""Simple test script for async depth preprocessor."""

import logging
import time

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_async_depth():
    """Test the async depth preprocessor."""
    from scope.core.preprocessors import DepthPreprocessorClient

    logger.info("=" * 60)
    logger.info("Testing Async Depth Preprocessor")
    logger.info("=" * 60)

    # Create client
    logger.info("\n1. Creating DepthPreprocessorClient...")
    client = DepthPreprocessorClient(encoder="vits")  # Use vits for faster loading

    # Start client (this should wait for worker to be ready)
    logger.info("\n2. Starting client (this will load the model)...")
    start_time = time.time()

    if not client.start(timeout=120.0):  # 2 minute timeout
        logger.error("Failed to start client!")
        return False

    startup_time = time.time() - start_time
    logger.info(f"Client started successfully in {startup_time:.1f}s")

    # Check worker is alive
    if not client.is_running():
        logger.error("Worker process is not running!")
        client.stop()
        return False

    logger.info("Worker process is running")

    # Create some test frames
    logger.info("\n3. Creating test frames...")
    frames = np.random.randint(0, 255, (12, 512, 512, 3), dtype=np.uint8)
    logger.info(f"Test frames shape: {frames.shape}")

    # Submit frames
    logger.info("\n4. Submitting frames...")
    submit_start = time.time()
    chunk_id = client.submit_frames(
        frames,
        target_height=512,
        target_width=512,
    )
    submit_time = time.time() - submit_start
    logger.info(f"Frames submitted (chunk_id={chunk_id}) in {submit_time:.3f}s")

    # Wait for result
    logger.info("\n5. Waiting for result...")
    wait_start = time.time()
    result = client.get_depth_result(wait=True, timeout=30.0)
    wait_time = time.time() - wait_start

    if result is None:
        logger.error(f"No result received after {wait_time:.1f}s!")
        client.stop()
        return False

    logger.info(f"Result received in {wait_time:.1f}s")
    logger.info(f"Result chunk_id: {result.chunk_id}")
    logger.info(f"Depth shape: {result.depth.shape}")
    logger.info(f"Depth dtype: {result.depth.dtype}")
    logger.info(f"Depth range: [{result.depth.min():.3f}, {result.depth.max():.3f}]")

    # Stop client
    logger.info("\n6. Stopping client...")
    client.stop()
    logger.info("Client stopped")

    logger.info("\n" + "=" * 60)
    logger.info("TEST PASSED!")
    logger.info("=" * 60)
    logger.info(f"Total time: {time.time() - start_time:.1f}s")
    logger.info(f"Startup: {startup_time:.1f}s")
    logger.info(f"Submit: {submit_time:.3f}s")
    logger.info(f"Wait for result: {wait_time:.1f}s")

    return True

if __name__ == "__main__":
    success = test_async_depth()
    exit(0 if success else 1)

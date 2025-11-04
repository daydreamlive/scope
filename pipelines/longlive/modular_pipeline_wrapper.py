"""Example: Wiring LongLive components with modular blocks using adapters.

This demonstrates how to integrate the three-layer architecture:
1. Protocol (interfaces.py) - defines contracts
2. Adapter (adapters.py) - makes LongLive components conform to protocols
3. ModularPipelineBlock (modular_blocks.py) - generic blocks that use protocols

The same VideoEncodeBlock/VideoDecodeBlock can work with StreamDiffusion, Krea,
or any other pipeline by just swapping the adapter.
"""

from types import SimpleNamespace

import torch
from diffusers.modular_pipelines import PipelineState

from lib.models_config import get_models_dir

from ..base.wan2_1.wrapper import WanVAEWrapper
from .adapters import LongLiveVAEAdapter
from .modular_blocks import VAECacheClearBlock, VideoDecodeBlock, VideoEncodeBlock


def create_longlive_modular_vae_pipeline():
    """Create a modular VAE pipeline for LongLive.

    This shows how to wire up:
    1. Concrete implementation (WanVAEWrapper)
    2. Adapter (LongLiveVAEAdapter)
    3. Generic blocks (VideoEncodeBlock, VideoDecodeBlock)

    Returns:
        Tuple of (components, encode_block, decode_block, clear_cache_block)
    """
    # Step 1: Load the concrete LongLive VAE implementation
    # Use same model_dir pattern as longlive/test.py
    models_dir = get_models_dir()
    wan_vae = WanVAEWrapper(model_dir=str(models_dir))

    # Step 2: Wrap it with adapter to conform to VAEInterface protocol
    vae_adapter = LongLiveVAEAdapter(wan_vae)

    # Step 3: Create components namespace for modular blocks
    # The blocks expect components.vae to implement VAEInterface
    components = SimpleNamespace(vae=vae_adapter)

    # Step 4: Create generic blocks that work with VAEInterface
    encode_block = VideoEncodeBlock()
    decode_block = VideoDecodeBlock()
    clear_cache_block = VAECacheClearBlock()

    return components, encode_block, decode_block, clear_cache_block


def example_encode_decode_pipeline():
    """Example: Using modular blocks for encode-decode pipeline."""

    # Setup
    components, encode_block, decode_block, clear_cache_block = (
        create_longlive_modular_vae_pipeline()
    )

    # Move VAE to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    components.vae.vae = components.vae.vae.to(device=device, dtype=dtype)

    # Create fake pixel input [B, C, T, H, W]
    pixels = torch.randn(1, 3, 4, 512, 512, device=device, dtype=dtype)

    # Create initial state
    state = PipelineState()
    state.set("pixels", pixels)

    # Step 1: Encode pixels to latents using generic block
    print("Encoding pixels to latents...")
    components, state = encode_block(components, state)
    latents = state.values.get("latents")
    print(f"  Latents shape: {latents.shape}")
    print(f"  Latents dtype: {latents.dtype}")

    # Step 2: Decode latents back to pixels using generic block
    print("\nDecoding latents to pixels...")
    state = PipelineState()
    state.set("latents", latents)
    state.set("use_cache", False)
    components, state = decode_block(components, state)
    decoded_pixels = state.values.get("pixels")
    print(f"  Pixels shape: {decoded_pixels.shape}")
    print(f"  Pixels dtype: {decoded_pixels.dtype}")

    # Step 3: Clear VAE cache
    print("\nClearing VAE cache...")
    state = PipelineState()
    components, state = clear_cache_block(components, state)
    print("  Cache cleared")

    # Verify dtype stability
    assert pixels.dtype == decoded_pixels.dtype, "Dtype not preserved!"
    print(f"\nDtype stability verified: {pixels.dtype} -> {decoded_pixels.dtype}")


def example_streaming_decode():
    """Example: Using modular blocks for streaming decode with cache."""

    # Setup
    components, _, decode_block, clear_cache_block = (
        create_longlive_modular_vae_pipeline()
    )

    # Move VAE to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    components.vae.vae = components.vae.vae.to(device=device, dtype=dtype)

    # Clear cache before streaming
    state = PipelineState()
    components, state = clear_cache_block(components, state)

    print("Streaming decode with cache...")

    # Simulate streaming: decode multiple chunks with cache
    for i in range(3):
        # Create fake latent chunk [B, T, C, H, W]
        latents = torch.randn(1, 1, 16, 64, 64, device=device, dtype=dtype)

        # Decode with cache
        state = PipelineState()
        state.set("latents", latents)
        state.set("use_cache", True)
        components, state = decode_block(components, state)

        decoded_pixels = state.values.get("pixels")
        print(
            f"  Chunk {i+1}: decoded shape {decoded_pixels.shape}, dtype {decoded_pixels.dtype}"
        )

    print("Streaming decode complete")


if __name__ == "__main__":
    print("=== LongLive Modular VAE Pipeline Example ===\n")

    print("Example 1: Encode-Decode Pipeline")
    print("-" * 50)
    example_encode_decode_pipeline()

    print("\n\nExample 2: Streaming Decode with Cache")
    print("-" * 50)
    example_streaming_decode()

    print("\n\n=== Key Insight ===")
    print("The same VideoEncodeBlock and VideoDecodeBlock can work with:")
    print("  - LongLive VAE (via LongLiveVAEAdapter)")
    print("  - StreamDiffusion VAE (via StreamDiffusionVAEAdapter)")
    print("  - Krea VAE (via KreaVAEAdapter)")
    print("  - Any other VAE (via custom adapter)")
    print(
        "\nThe blocks depend ONLY on VAEInterface protocol, not concrete implementations!"
    )

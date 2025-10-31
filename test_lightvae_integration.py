"""
Test script for LightVAE integration in LongLive pipeline.

Usage:
    python test_lightvae_integration.py
"""

import os
from pathlib import Path

import torch

from pipelines.base.wan2_1.lightvae_wrapper import LightVAEWrapper


def test_lightvae_wrapper():
    """Test LightVAE wrapper instantiation and basic round-trip."""

    print("test_lightvae_wrapper: Starting LightVAE wrapper test...")

    vae_path = os.path.expanduser(
        "~/.daydream-scope/models/Wan2.1-T2V-1.3B/lightvaew2_1.pth"
    )

    if not Path(vae_path).exists():
        print(
            f"test_lightvae_wrapper: WARNING - LightVAE checkpoint not found at {vae_path}"
        )
        print(
            "test_lightvae_wrapper: Download it from lightx2v/Autoencoders or adjust vae_path in model.yaml"
        )
        return False

    try:
        print(f"test_lightvae_wrapper: Loading LightVAE from {vae_path}")
        vae = LightVAEWrapper(vae_path=vae_path)
        print("test_lightvae_wrapper: Successfully loaded LightVAE wrapper")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16

        vae = vae.to(device=device, dtype=dtype)
        print(f"test_lightvae_wrapper: Moved model to {device} with dtype {dtype}")

        B, C, T, H, W = 1, 3, 5, 480, 832
        print(
            f"test_lightvae_wrapper: Creating test input tensor [{B}, {C}, {T}, {H}, {W}]"
        )
        pixel_input = torch.randn(B, C, T, H, W, device=device, dtype=dtype)

        print("test_lightvae_wrapper: Encoding to latent...")
        latent = vae.encode_to_latent(pixel_input)
        print(f"test_lightvae_wrapper: Latent shape: {latent.shape}")

        if latent.shape[0] != B:
            print(
                f"test_lightvae_wrapper: ERROR - Expected batch size {B}, got {latent.shape[0]}"
            )
            return False
        if latent.shape[2] != 16:
            print(
                f"test_lightvae_wrapper: ERROR - Expected 16 latent channels, got {latent.shape[2]}"
            )
            return False
        if latent.shape[3] != H // 8 or latent.shape[4] != W // 8:
            print(
                f"test_lightvae_wrapper: ERROR - Expected spatial size ({H//8}, {W//8}), got ({latent.shape[3]}, {latent.shape[4]})"
            )
            return False

        T_latent = latent.shape[1]
        print(
            f"test_lightvae_wrapper: Temporal compression: {T} input frames -> {T_latent} latent frames (compression ratio: {T/T_latent:.2f}x)"
        )

        if T_latent > T:
            print(
                f"test_lightvae_wrapper: ERROR - Latent temporal dimension {T_latent} should not exceed input {T}"
            )
            return False

        print("test_lightvae_wrapper: Decoding to pixel...")
        pixel_output = vae.decode_to_pixel(latent, use_cache=False)
        print(f"test_lightvae_wrapper: Decoded pixel shape: {pixel_output.shape}")

        T_output = pixel_output.shape[1]
        print(
            f"test_lightvae_wrapper: Temporal reconstruction: {T_latent} latent frames -> {T_output} output frames"
        )

        if pixel_output.shape[0] != B:
            print(
                f"test_lightvae_wrapper: ERROR - Expected batch size {B}, got {pixel_output.shape[0]}"
            )
            return False
        if pixel_output.shape[2] != C:
            print(
                f"test_lightvae_wrapper: ERROR - Expected {C} channels, got {pixel_output.shape[2]}"
            )
            return False
        if pixel_output.shape[3] != H or pixel_output.shape[4] != W:
            print(
                f"test_lightvae_wrapper: ERROR - Expected spatial size ({H}, {W}), got ({pixel_output.shape[3]}, {pixel_output.shape[4]})"
            )
            return False

        print("test_lightvae_wrapper: Testing cached decode...")
        vae.clear_cache()
        pixel_output_cached = vae.decode_to_pixel(latent, use_cache=True)
        print(
            f"test_lightvae_wrapper: Cached decoded pixel shape: {pixel_output_cached.shape}"
        )

        if pixel_output_cached.shape != pixel_output.shape:
            print(
                f"test_lightvae_wrapper: ERROR - Cached decode shape {pixel_output_cached.shape} differs from normal decode {pixel_output.shape}"
            )
            return False

        print("test_lightvae_wrapper: Round-trip test PASSED")
        print(
            f"test_lightvae_wrapper: Input shape: {list(pixel_input.shape)}, range: [{pixel_input.min():.3f}, {pixel_input.max():.3f}]"
        )
        print(
            f"test_lightvae_wrapper: Latent shape: {list(latent.shape)}, range: [{latent.min():.3f}, {latent.max():.3f}]"
        )
        print(
            f"test_lightvae_wrapper: Output shape: {list(pixel_output.shape)}, range: [{pixel_output.min():.3f}, {pixel_output.max():.3f}]"
        )

        return True

    except Exception as e:
        print(f"test_lightvae_wrapper: ERROR - {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_config_validation():
    """Test that invalid config values are rejected."""
    print("\ntest_config_validation: Testing config validation...")

    try:
        from pipelines.base.wan2_1.lightvae_wrapper import LightVAEWrapper

        try:
            vae = LightVAEWrapper(vae_path=None)
            print(
                "test_config_validation: ERROR - Should have raised ValueError for vae_path=None"
            )
            return False
        except ValueError as e:
            print(f"test_config_validation: Correctly rejected vae_path=None: {e}")

        try:
            vae = LightVAEWrapper(vae_path="/nonexistent/path.pth")
            print(
                "test_config_validation: ERROR - Should have raised FileNotFoundError for nonexistent path"
            )
            return False
        except FileNotFoundError as e:
            print(f"test_config_validation: Correctly rejected nonexistent path: {e}")

        print("test_config_validation: Config validation PASSED")
        return True

    except Exception as e:
        print(f"test_config_validation: ERROR - {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Starting LightVAE integration tests...\n")

    results = []

    results.append(("Config Validation", test_config_validation()))
    results.append(("LightVAE Wrapper Round-trip", test_lightvae_wrapper()))

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)

    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\nAll tests PASSED")
    else:
        print("\nSome tests FAILED")
        exit(1)

"""
Cross-platform LoRA downloader CLI.
"""

import argparse
import asyncio
import logging
import sys

from .lora_downloader import LoRADownloadRequest, download_lora
from .models_config import get_lora_dir

logger = logging.getLogger(__name__)


async def download_loras(
    url: str, expected_sha256: str | None = None, filename: str | None = None
) -> None:
    """Download a LoRA from a direct URL into the configured LoRA directory."""
    lora_dir = get_lora_dir()
    logger.info(f"Downloading LoRA to: {lora_dir}")

    request = LoRADownloadRequest(
        source="url",
        url=url,
        filename=filename,
        expected_sha256=expected_sha256,
    )
    result = await download_lora(request, lora_dir)

    print("Downloaded LoRA:")
    print(f"  Filename: {result.filename}")
    print(f"  Path: {result.path}")
    print(f"  SHA256: {result.sha256}")
    print(f"  Size: {result.size_bytes} bytes")


def main() -> None:
    """Main entry point for the download_loras script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download LoRAs for Daydream Scope from direct URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_loras.py --url https://example.com/style.safetensors
  python download_loras.py -u https://example.com/style.safetensors --expected-sha256 abc123
  python download_loras.py -u https://example.com/download/123 --filename style.safetensors
        """,
    )
    parser.add_argument(
        "--url",
        "-u",
        type=str,
        required=True,
        help="Direct URL for the LoRA file to download.",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help="Expected SHA-256 hash for integrity verification.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Filename to save the LoRA as. Useful for URLs that do not include a stable filename.",
    )

    args = parser.parse_args()

    try:
        asyncio.run(
            download_loras(
                args.url,
                expected_sha256=args.expected_sha256,
                filename=args.filename,
            )
        )
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

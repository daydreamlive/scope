"""Patch xformers flash.py to set FLASH_VER_LAST to 2.8.3."""

import importlib.util
import sys
from pathlib import Path


def find_xformers_flash_file() -> Path | None:
    """Find the xformers flash.py file in the installed package."""
    try:
        # Get the package path
        spec = importlib.util.find_spec("xformers")
        if spec is None or spec.origin is None:
            return None

        # xformers package location
        package_path = Path(spec.origin).parent
        flash_file = package_path / "ops" / "fmha" / "flash.py"

        if flash_file.exists():
            return flash_file

        return None
    except ImportError:
        print("Error: xformers is not installed", file=sys.stderr)
        return None


def patch_xformers_flash() -> bool:
    """Patch xformers flash.py to set FLASH_VER_LAST to 2.8.3."""
    flash_file = find_xformers_flash_file()

    if flash_file is None:
        print("Error: Could not find xformers flash.py file", file=sys.stderr)
        return False

    try:
        # Read the file
        content = flash_file.read_text(encoding="utf-8")

        # Check if already patched
        if 'FLASH_VER_LAST = parse_version("2.8.3")' in content:
            print(f"✓ xformers flash.py already patched: {flash_file}")
            return True

        # Find and replace the FLASH_VER_LAST line
        lines = content.splitlines()
        patched = False

        for i, line in enumerate(lines):
            # Look for FLASH_VER_LAST assignment
            if "FLASH_VER_LAST" in line and "parse_version" in line:
                # Preserve original indentation
                indent = len(line) - len(line.lstrip())
                # Replace with our desired version, preserving indentation
                lines[i] = (
                    " " * indent
                    + 'FLASH_VER_LAST = parse_version("2.8.3")  # last supported, inclusive'
                )
                patched = True
                break

        if not patched:
            print(
                "Warning: Could not find FLASH_VER_LAST line to patch",
                file=sys.stderr,
            )
            return False

        # Write the patched content back
        flash_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"✓ Successfully patched xformers flash.py: {flash_file}")
        return True

    except Exception as e:
        print(f"Error patching xformers flash.py: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Main entry point for the patch command."""
    success = patch_xformers_flash()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Build script for daydream-scope
"""

import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a command and exit on failure."""
    # Find the full path to the executable on Windows
    if cmd and not Path(cmd[0]).is_absolute():
        executable_path = shutil.which(cmd[0])
        if executable_path:
            cmd[0] = executable_path
        else:
            print(f"❌ Error: Could not find executable '{cmd[0]}' in PATH")
            sys.exit(1)

    try:
        result = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        if result.stdout:
            print(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {' '.join(cmd)}")
        if e.stderr:
            print(f"Error output: {e.stderr.strip()}")
        sys.exit(1)


def main() -> None:
    """Main build function."""
    print("🚀 Building daydream-scope...")

    project_root = Path(__file__).parent.parent.parent.parent

    # Check if we're in the right directory
    if not (project_root / "pyproject.toml").exists():
        print(
            "❌ Error: pyproject.toml not found. Please run this script from the project root."
        )
        sys.exit(1)

    # Build frontend
    print("📦 Building frontend...")
    frontend_dir = project_root / "frontend"

    if not frontend_dir.exists():
        print("❌ Error: frontend directory not found")
        sys.exit(1)

    # Always run npm install to ensure dependencies are up to date
    print("📥 Installing frontend dependencies...")
    run_command(["npm", "install"], cwd=frontend_dir)

    # Build the frontend
    print("🔨 Building frontend assets...")
    run_command(["npm", "run", "build"], cwd=frontend_dir)

    # Check if build was successful
    dist_dir = frontend_dir / "dist"
    if not dist_dir.exists():
        print("❌ Error: Frontend build failed - dist directory not found")
        sys.exit(1)

    print("✅ Frontend build completed successfully")


if __name__ == "__main__":
    main()

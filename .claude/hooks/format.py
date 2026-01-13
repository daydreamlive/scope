#!/usr/bin/env python3
"""
Claude Code formatter hook.

This script is triggered after Write/Edit operations and automatically
runs linting/formatting on the modified file using the project's existing tools.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

# File extensions to format
PYTHON_EXTENSIONS = {".py"}
JS_TS_EXTENSIONS = {".ts", ".tsx", ".js", ".jsx"}
PRETTIER_ONLY_EXTENSIONS = {".json", ".css", ".md"}
FRONTEND_EXTENSIONS = JS_TS_EXTENSIONS | PRETTIER_ONLY_EXTENSIONS


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent


def run_command(cmd: list[str], cwd: Path) -> bool:
    """
    Run a command silently, only printing stderr on failure.
    Returns True if command succeeded.
    """
    try:
        # On Windows, use shell=True to find .cmd executables like npx.cmd
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            shell=IS_WINDOWS,
        )
        if result.returncode != 0 and result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return result.returncode == 0
    except FileNotFoundError:
        # Command not found, skip silently
        return False
    except Exception as e:
        print(f"Error running {cmd[0]}: {e}", file=sys.stderr)
        return False


def format_python_file(file_path: Path, repo_root: Path) -> None:
    """Run ruff check --fix and ruff format on a Python file."""
    rel_path = str(file_path.relative_to(repo_root))

    # Run ruff check with auto-fix
    run_command(["uv", "run", "ruff", "check", "--fix", rel_path], repo_root)

    # Run ruff format
    run_command(["uv", "run", "ruff", "format", rel_path], repo_root)


def format_frontend_file(file_path: Path, repo_root: Path) -> None:
    """Run prettier and optionally eslint on a frontend file."""
    frontend_dir = repo_root / "frontend"
    rel_path = str(file_path.relative_to(frontend_dir))

    # Run prettier
    run_command(["npx", "prettier", "--write", rel_path], frontend_dir)

    # Run eslint only for JS/TS files
    if file_path.suffix in JS_TS_EXTENSIONS:
        run_command(["npx", "eslint", "--fix", rel_path], frontend_dir)


def should_format_python(file_path: Path, repo_root: Path) -> bool:
    """Check if file is a Python file in src/ or tests/."""
    if file_path.suffix not in PYTHON_EXTENSIONS:
        return False

    try:
        rel_path = file_path.relative_to(repo_root)
        parts = rel_path.parts
        return len(parts) > 0 and parts[0] in ("src", "tests")
    except ValueError:
        return False


def should_format_frontend(file_path: Path, repo_root: Path) -> bool:
    """Check if file is a frontend file in frontend/."""
    if file_path.suffix not in FRONTEND_EXTENSIONS:
        return False

    try:
        rel_path = file_path.relative_to(repo_root)
        parts = rel_path.parts
        return len(parts) > 0 and parts[0] == "frontend"
    except ValueError:
        return False


def main() -> None:
    """Main entry point for the formatter hook."""
    # Read JSON input from stdin
    try:
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError:
        # Invalid JSON, exit silently
        return

    # Extract file path from tool input
    tool_input = input_data.get("tool_input", {})
    file_path_str = tool_input.get("file_path")

    if not file_path_str:
        # No file path, exit silently
        return

    file_path = Path(file_path_str).resolve()
    repo_root = get_repo_root()

    # Check if file exists
    if not file_path.exists():
        return

    # Determine file type and format accordingly
    if should_format_python(file_path, repo_root):
        format_python_file(file_path, repo_root)
    elif should_format_frontend(file_path, repo_root):
        format_frontend_file(file_path, repo_root)
    # Otherwise, skip silently


if __name__ == "__main__":
    main()

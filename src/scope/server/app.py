import argparse
import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from importlib.metadata import version
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from .logs_config import (
    cleanup_old_logs,
    ensure_logs_dir,
    get_current_log_file,
)


class STUNErrorFilter(logging.Filter):
    """Filter to suppress STUN/TURN connection errors that are not critical."""

    def filter(self, record):
        # Suppress STUN  exeception that occurrs always during the stream restart
        if "Task exception was never retrieved" in record.getMessage():
            return False
        return True


# Ensure logs directory exists and clean up old logs
logs_dir = ensure_logs_dir()
cleanup_old_logs(max_age_days=1)  # Delete logs older than 1 day
log_file = get_current_log_file()

# Configure logging - set root to WARNING to keep non-app libraries quiet by default
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Console handler handles INFO
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(
        handler, RotatingFileHandler
    ):
        handler.setLevel(logging.INFO)

# Add rotating file handler
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=5 * 1024 * 1024,  # 5 MB per file
    backupCount=5,  # Keep 5 backup files
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
root_logger.addHandler(file_handler)

# Add the filter to suppress STUN/TURN errors
stun_filter = STUNErrorFilter()
logging.getLogger("asyncio").addFilter(stun_filter)

# Set INFO level for your app modules
logging.getLogger("scope.server").setLevel(logging.INFO)
logging.getLogger("scope.core").setLevel(logging.INFO)

# Set INFO level for uvicorn
logging.getLogger("uvicorn.error").setLevel(logging.INFO)

# Enable verbose logging for other libraries when needed
if os.getenv("VERBOSE_LOGGING"):
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("aiortc").setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# Track if API server is running
_api_server_thread = None
_api_server_running = False


def get_git_commit_hash() -> str:
    """
    Get the current git commit hash.

    Returns:
        Git commit hash if available, otherwise a fallback message.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,  # 5 second timeout
            cwd=Path(__file__).parent,  # Run in the project directory
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return "unknown (not a git repository)"
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return "unknown (git error)"
    except FileNotFoundError:
        return "unknown (git not installed)"
    except Exception:
        return "unknown"


def print_version_info():
    """Print version information and exit."""
    try:
        pkg_version = version("daydream-scope")
    except Exception:
        pkg_version = "unknown"

    git_hash = get_git_commit_hash()

    print(f"daydream-scope: {pkg_version}")
    print(f"git commit: {git_hash}")


app = FastAPI(
    title="Scope",
    description="Main server for Scope - use /reserve to start the API server",
    version=version("daydream-scope"),
)


@app.get("/ping")
async def ping():
    """Health check endpoint required by RunPod Serverless load balancer."""
    return {"status": "healthy"}


@app.post("/reserve")
async def reserve():
    """Start the API server on port 8080."""
    global _api_server_thread, _api_server_running

    if _api_server_running:
        return {
            "message": "API server is already running on port 8080",
            "status": "running",
        }

    def start_api_server():
        global _api_server_running
        try:
            from .api_server import run_api_server

            _api_server_running = True
            run_api_server(port=8080, host="0.0.0.0")
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            _api_server_running = False

    _api_server_thread = threading.Thread(target=start_api_server, daemon=True)
    _api_server_thread.start()

    # Give the server a moment to start
    time.sleep(0.5)

    return {
        "message": "API server started on port 8080",
        "status": "started",
        "api_url": "http://0.0.0.0:8080",
    }


def open_browser_when_ready(host: str, port: int, server):
    """Open browser when server is ready, with fallback to URL logging."""
    # Wait for server to be ready
    while not getattr(server, "started", False):
        time.sleep(0.1)

    # Determine the URL to open
    url = (
        f"http://localhost:{port}"
        if host in ["0.0.0.0", "127.0.0.1"]
        else f"http://{host}:{port}"
    )

    try:
        success = webbrowser.open(url)
        if success:
            logger.info(f"🌐 Opened browser at {url}")
    except Exception:
        success = False

    if not success:
        logger.info(f"🌐 UI is available at: {url}")


def main():
    """Main entry point for the daydream-scope command."""
    parser = argparse.ArgumentParser(
        description="Daydream Scope - Real-time AI video generation and streaming"
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information and exit",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development (default: False)",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="Port to bind to (default: PORT env var or 8000)",
    )
    parser.add_argument(
        "-N",
        "--no-browser",
        action="store_true",
        help="Do not automatically open a browser window after the server starts",
    )

    args = parser.parse_args()

    # Handle version flag
    if args.version:
        print_version_info()
        sys.exit(0)

    # Check if we're in production mode (frontend dist exists)
    frontend_dist = Path(__file__).parent.parent.parent / "frontend" / "dist"
    is_production = frontend_dist.exists()

    if is_production:
        # Create server instance for production mode
        config = uvicorn.Config(
            "scope.server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_config=None,  # Use our logging config, don't override it
        )
        server = uvicorn.Server(config)

        # Start browser opening thread (unless disabled)
        if not args.no_browser:
            browser_thread = threading.Thread(
                target=open_browser_when_ready,
                args=(args.host, args.port, server),
                daemon=True,
            )
            browser_thread.start()
        else:
            logger.info("main: Skipping browser auto-launch due to --no-browser")

        # Run the server
        try:
            server.run()
        except KeyboardInterrupt:
            pass  # Clean shutdown on Ctrl+C
    else:
        # Development mode - just run normally
        uvicorn.run(
            "scope.server.app:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_config=None,  # Use our logging config, don't override it
        )


if __name__ == "__main__":
    # For RunPod serverless mode, run uvicorn directly when PORT env var is set
    # Otherwise use main() with full argument parsing
    port_env = os.getenv("PORT")
    if port_env:
        # RunPod serverless mode - run uvicorn directly
        import uvicorn

        port = int(port_env)
        logger.info(f"Starting server on port {port} (RunPod serverless mode)")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_config=None,  # Use our logging config
        )
    else:
        # Normal mode - use main() with argument parsing
        main()

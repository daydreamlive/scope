import argparse
import asyncio
import json
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
from fastapi.responses import StreamingResponse

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


# Middleware to force keep-alive for streaming responses
@app.middleware("http")
async def force_keep_alive(request, call_next):
    response = await call_next(request)
    # Force Connection: keep-alive for streaming responses
    content_type = response.headers.get("content-type", "")
    if "text/event-stream" in content_type or "/reserve" in str(request.url):
        # Remove any existing connection header (case-insensitive)
        headers_to_remove = [
            k for k in response.headers.keys() if k.lower() == "connection"
        ]
        for key in headers_to_remove:
            del response.headers[key]
        # Set keep-alive
        response.headers["Connection"] = "keep-alive"
        # Also set it in the raw headers if available
        if hasattr(response, "raw_headers"):
            response.raw_headers = [
                (k, v) for k, v in response.raw_headers if k.lower() != b"connection"
            ] + [(b"connection", b"keep-alive")]
    return response


@app.get("/ping")
async def ping():
    """Health check endpoint required by RunPod Serverless load balancer."""
    return {"status": "healthy"}


@app.get("/reserve")
async def reserve():
    """
    Reserve endpoint that starts the API server, returns RunPod environment variables,
    and keeps connection open.
    Returns JSON with RUNPOD_PUBLIC_IP and RUNPOD_TCP_PORT_8000, then streams to keep connection alive.
    """
    global _api_server_thread, _api_server_running

    # Start the API server if not already running
    if not _api_server_running:

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
        await asyncio.sleep(0.5)

    # Get RunPod environment variables
    public_ip = os.getenv("RUNPOD_PUBLIC_IP", "")
    tcp_port = os.getenv("RUNPOD_TCP_PORT_8080", "")

    # Create JSON response
    response_data = {
        "host": public_ip,
        "port": tcp_port,
    }

    # Convert to JSON string
    json_str = json.dumps(response_data) + "\n"

    async def stream_response():
        """Stream the JSON response and then keep connection alive."""
        # Send the JSON response first
        yield json_str.encode("utf-8")

        # Keep connection alive by periodically sending keepalive data
        # This will continue until the client closes the connection
        # Use shorter intervals to ensure connection stays alive
        try:
            while True:
                await asyncio.sleep(0.5)  # Send keepalive every 0.5 seconds
                yield b": keepalive\n\n"  # SSE format: comment line followed by double newline
        except asyncio.CancelledError:
            # Connection was closed by client
            pass
        except Exception:
            # Handle any other exceptions gracefully
            pass

    # Use StreamingResponse with proper headers for HTTP event streaming
    class KeepAliveStreamingResponse(StreamingResponse):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Ensure Connection header is set to keep-alive
            self.headers["Connection"] = "keep-alive"
            # Remove connection header if it exists with different casing
            for key in list(self.headers.keys()):
                if key.lower() == "connection" and key != "Connection":
                    del self.headers[key]

    return KeepAliveStreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering in nginx
            "Keep-Alive": "timeout=300",  # Keep connection alive for 5 minutes
        },
    )


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
            timeout_keep_alive=300,  # Keep connections alive for 5 minutes
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
            timeout_keep_alive=300,  # Keep connections alive for 5 minutes
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
            timeout_keep_alive=300,  # Keep connections alive for 5 minutes
        )
    else:
        # Normal mode - use main() with argument parsing
        main()

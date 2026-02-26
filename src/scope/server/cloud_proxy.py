"""Cloud proxy helpers: run logic locally or proxy to cloud when connected.

Use the @cloud_proxy decorator on route handlers that should proxy to the
cloud-hosted scope backend when cloud_manager.is_connected is True. The
decorator forwards the request (path, query, body) to the cloud and returns
the cloud response; otherwise it runs the wrapped handler (local logic).

Handlers that need custom cloud behavior (e.g. WebRTC relay, ICE servers)
should not use this decorator and keep explicit if cloud_manager.is_connected
branches. Recording download is supported via a path callable and
_base64_content response handling.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import Response

from .cloud_connection import CloudConnectionManager
from .models_config import get_assets_dir
from .schema import AssetFileInfo, HardwareInfoResponse

logger = logging.getLogger(__name__)


async def _proxy_to_cloud(
    cloud_manager: CloudConnectionManager,
    http_request: Request,
    path: str,
    method: str,
    timeout: float,
    error_detail: str,
) -> Any:
    """Forward request to cloud and return parsed response or raise HTTPException."""
    path_with_query = (
        f"{path}?{http_request.url.query}" if http_request.url.query else path
    )
    body: dict | None = None
    if method in ("POST", "PATCH", "PUT"):
        try:
            body = await http_request.json()
        except Exception:
            body = None

    logger.info(f"Proxying to cloud: {method} {path_with_query}")
    try:
        response = await cloud_manager.api_request(
            method=method,
            path=path_with_query,
            body=body,
            timeout=timeout,
        )
    except Exception as e:
        logger.error(f"Cloud proxy request failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"{error_detail}: {e}",
        ) from e

    status = response.get("status", 200)
    if status >= 400:
        raise HTTPException(
            status_code=status,
            detail=response.get("error", error_detail),
        )

    # Binary response: cloud returned base64-encoded content (e.g. recording download)
    if "_base64_content" in response:
        content = base64.b64decode(response["_base64_content"])
        media_type = response.get("media_type", "video/mp4")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = response.get("filename", f"recording-{timestamp}.mp4")
        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            },
        )

    return response.get("data", {})


PathResolver = Callable[[Request, CloudConnectionManager], str]

# When path is omitted, the request URL path is used when proxying (same as route).
CLOUD_REQUEST_FAILED = "Cloud request failed"


async def get_hardware_info_from_cloud(
    cloud_manager: CloudConnectionManager,
    spout_available: bool,
    ndi_available: bool = False,
    osc_enabled: bool = False,
    osc_port: int = 9000,
) -> HardwareInfoResponse:
    """Fetch hardware info from cloud and return with local output availability.

    Spout/NDI/OSC availability is taken from the caller (local) because output
    sink frames and OSC control flow through the local backend.
    """
    logger.info("Proxying hardware info request to cloud")
    try:
        response = await cloud_manager.api_request(
            method="GET",
            path="/api/v1/hardware/info",
            timeout=30.0,
        )
    except Exception as e:
        logger.error(f"Cloud proxy request failed: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"{CLOUD_REQUEST_FAILED}: {e}",
        ) from e

    status = response.get("status", 200)
    if status >= 400:
        raise HTTPException(
            status_code=status,
            detail=response.get("error", CLOUD_REQUEST_FAILED),
        )

    data = response.get("data", {})
    return HardwareInfoResponse(
        vram_gb=data.get("vram_gb"),
        spout_available=spout_available,
        ndi_available=ndi_available,
        osc_enabled=osc_enabled,
        osc_port=osc_port,
    )


async def upload_asset_to_cloud(
    cloud_manager: CloudConnectionManager,
    content: bytes,
    filename: str,
    content_type: str,
    asset_type: str,
) -> AssetFileInfo:
    """Upload asset to cloud (POST with base64 body) and save a local copy for thumbnails.

    Call when cloud_manager.is_connected; raises HTTPException on cloud errors.
    """
    logger.info(
        f"upload_asset: Uploading {asset_type} to cloud and locally: {filename}"
    )

    # Save locally for thumbnail serving
    assets_dir = get_assets_dir()
    assets_dir.mkdir(parents=True, exist_ok=True)
    local_file_path = assets_dir / filename
    local_file_path.write_bytes(content)
    logger.info(f"upload_asset: Saved local copy for thumbnails: {local_file_path}")

    base64_content = base64.b64encode(content).decode("utf-8")
    response = await cloud_manager.api_request(
        method="POST",
        path=f"/api/v1/assets?filename={filename}",
        body={
            "_base64_content": base64_content,
            "_content_type": content_type,
        },
        timeout=60.0,
    )

    data = response.get("data", {})
    cloud_path = data.get("path", "")
    logger.info(f"upload_asset: Uploaded to cloud: {cloud_path}")

    size_mb = round(len(content) / (1024 * 1024), 2)
    return AssetFileInfo(
        name=data.get("name", filename),
        path=cloud_path,
        size_mb=data.get("size_mb", size_mb),
        folder=data.get("folder"),
        type=data.get("type", asset_type),
        created_at=data.get("created_at", time.time()),
    )


def recording_download_cloud_path(
    _http_request: Request,
    cloud_manager: CloudConnectionManager,
) -> str:
    """Resolve cloud path for recording download (uses cloud session ID)."""
    webrtc_client = getattr(cloud_manager, "_webrtc_client", None)
    session_id = getattr(webrtc_client, "session_id", None) if webrtc_client else None
    if not session_id:
        raise HTTPException(
            status_code=404,
            detail="No active cloud session for recording download",
        )
    return f"/api/v1/recordings/{session_id}"


def cloud_proxy(
    path: str | PathResolver | None = None,
    *,
    timeout: float = 30.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: when cloud is connected, proxy the request to cloud; else run the handler.

    The wrapped handler must accept `http_request: Request` and
    `cloud_manager: CloudConnectionManager` (e.g. via Depends) so the
    decorator can forward the request and check connection. Do not consume
    the request body before the decorator runs (the decorator only reads
    body when proxying).

    Args:
        path: API path to call on cloud. If omitted, the request's URL path is
              used (same as the route). If a string, that path is used. If a
              callable (http_request, cloud_manager) -> str, it is used for
              dynamic paths (e.g. recording download). Query string is always
              appended from the incoming request.
        timeout: Timeout in seconds for the cloud request (default 30.0).
    """

    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(f)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cloud_manager = kwargs.get("cloud_manager")
            http_request = kwargs.get("http_request")

            if not isinstance(cloud_manager, CloudConnectionManager):
                return await f(*args, **kwargs)
            if not cloud_manager.is_connected:
                return await f(*args, **kwargs)
            if not isinstance(http_request, Request):
                logger.warning(
                    "cloud_proxy: http_request not in handler kwargs, running locally"
                )
                return await f(*args, **kwargs)

            if path is None:
                actual_path = http_request.url.path
            elif callable(path):
                actual_path = path(http_request, cloud_manager)
            else:
                actual_path = path
            actual_method = http_request.method.upper()
            return await _proxy_to_cloud(
                cloud_manager,
                http_request,
                actual_path,
                actual_method,
                timeout,
                CLOUD_REQUEST_FAILED,
            )

        return wrapper

    return decorator

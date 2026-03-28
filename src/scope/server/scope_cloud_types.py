from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager
    from .livepeer import LivepeerConnection

    type ScopeCloudBackend = CloudConnectionManager | LivepeerConnection
else:
    ScopeCloudBackend = Any

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .cloud_connection import CloudConnectionManager
    from .livepeer import LivepeerConnection

    ScopeCloudBackend: TypeAlias = CloudConnectionManager | LivepeerConnection
else:
    ScopeCloudBackend = Any

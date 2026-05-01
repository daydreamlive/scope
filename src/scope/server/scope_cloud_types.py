from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .livepeer import LivepeerConnection

    type ScopeCloudBackend = LivepeerConnection
else:
    ScopeCloudBackend = Any

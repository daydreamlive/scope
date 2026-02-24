"""Shared utilities for the workflows package."""

from __future__ import annotations

from typing import Any


def get_plugin_list(plugin_manager: Any) -> list[dict[str, Any]]:
    """Return the full plugin list from *plugin_manager*."""
    return plugin_manager.list_plugins_sync()


def find_plugin_info(
    plugin_list: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    """Find and return the plugin-info dict whose ``name`` matches *name*."""
    for info in plugin_list:
        if info.get("name") == name:
            return info
    return None

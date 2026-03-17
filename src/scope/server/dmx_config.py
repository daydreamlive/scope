"""File-backed DMX mapping configuration.

Stores the global DMX config (mappings, port preference) at
``~/.daydream-scope/dmx-config.json``.  The config survives server restarts
and can be exported/imported as JSON by the frontend.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "dmx-config.json"


def _config_dir() -> Path:
    return Path.home() / ".daydream-scope"


def _config_path() -> Path:
    return _config_dir() / _CONFIG_FILENAME


def _default_config() -> dict[str, Any]:
    return {
        "enabled": False,
        "preferred_port": 6454,
        "log_all_messages": False,
        "mappings": [],
    }


def _validate_mapping(m: dict) -> dict | None:
    """Return a cleaned mapping dict or None if invalid."""
    try:
        universe = int(m.get("universe", 0))
        channel = int(m.get("channel", 0))
        key = str(m.get("key", ""))
        if not key or channel < 1 or channel > 512 or universe < 0:
            return None
        return {
            "universe": universe,
            "channel": channel,
            "key": key,
        }
    except (TypeError, ValueError):
        return None


def load_config() -> dict[str, Any]:
    """Load DMX config from disk, returning defaults on any error."""
    path = _config_path()
    if not path.exists():
        return _default_config()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = _default_config()
        if isinstance(raw.get("enabled"), bool):
            cfg["enabled"] = raw["enabled"]
        if isinstance(raw.get("preferred_port"), int):
            cfg["preferred_port"] = raw["preferred_port"]
        if isinstance(raw.get("log_all_messages"), bool):
            cfg["log_all_messages"] = raw["log_all_messages"]
        if isinstance(raw.get("mappings"), list):
            cleaned: list[dict] = []
            for m in raw["mappings"]:
                v = _validate_mapping(m)
                if v is not None:
                    cleaned.append(v)
            cfg["mappings"] = cleaned
        return cfg
    except Exception:
        logger.exception("Failed to load DMX config from %s", path)
        return _default_config()


def save_config(cfg: dict[str, Any]) -> None:
    """Persist DMX config to disk."""
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        logger.exception("Failed to save DMX config to %s", path)
        raise


def mappings_to_dict(
    mappings: list[dict],
) -> dict[tuple[int, int], str]:
    """Convert the JSON mapping list to the (universe, channel)->key dict
    used by DMXServer at runtime."""
    result: dict[tuple[int, int], str] = {}
    for m in mappings:
        v = _validate_mapping(m)
        if v:
            result[(v["universe"], v["channel"])] = v["key"]
    return result

"""Dynamic OSC path inventory and HTML docs generator.

Builds the list of available OSC paths from:
- Runtime parameter schema (Parameters model in schema.py)
- Pipeline config schemas from the registry
- Currently active pipeline chain state from PipelineManager
"""

import html
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

# Runtime params that are useful to control via OSC (from schema.Parameters).
# We curate this list to exclude structural params like pipeline_ids.
_RUNTIME_PARAMS: list[dict[str, Any]] = [
    {
        "key": "noise_scale",
        "type": "float",
        "description": "Noise scale (0.0-1.0)",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "key": "noise_controller",
        "type": "bool",
        "description": "Enable automatic noise scale adjustment based on motion detection",
    },
    {
        "key": "kv_cache_attention_bias",
        "type": "float",
        "description": "KV-cache attention bias (0.01-1.0). Lower = less reliance on past frames",
        "min": 0.01,
        "max": 1.0,
    },
    {
        "key": "manage_cache",
        "type": "bool",
        "description": "Enable automatic cache management for parameter updates",
    },
    {
        "key": "reset_cache",
        "type": "bool",
        "description": "Trigger a one-shot cache reset (send true)",
    },
    {
        "key": "vace_context_scale",
        "type": "float",
        "description": "VACE hint injection scale (0.0-2.0)",
        "min": 0.0,
        "max": 2.0,
    },
    {
        "key": "paused",
        "type": "bool",
        "description": "Pause / resume generation",
    },
]


def _extract_osc_paths_from_schema(
    config_schema: dict,
    pipeline_id: str,
) -> list[dict[str, Any]]:
    """Extract OSC-controllable paths from a pipeline's config_schema JSON."""
    paths: list[dict[str, Any]] = []
    properties = config_schema.get("properties", {})
    for key, prop in properties.items():
        ui = prop.get("ui", {})
        is_load_param = ui.get("is_load_param", True)
        if is_load_param:
            continue

        entry: dict[str, Any] = {
            "key": key,
            "type": prop.get("type", "any"),
            "description": prop.get("description", ""),
            "pipeline_id": pipeline_id,
        }
        if "minimum" in prop:
            entry["min"] = prop["minimum"]
        if "maximum" in prop:
            entry["max"] = prop["maximum"]
        if "enum" in prop:
            entry["enum"] = prop["enum"]
        paths.append(entry)
    return paths


def get_osc_paths(
    pipeline_manager: "PipelineManager | None",
) -> dict[str, Any]:
    """Build the full OSC path inventory split into *active* and *available* sections."""
    from scope.core.pipelines.registry import PipelineRegistry

    active_pipeline_ids: list[str] = []
    if pipeline_manager:
        status = pipeline_manager.get_status_info()
        pid = status.get("pipeline_id")
        if pid:
            active_pipeline_ids.append(pid)

    # Collect per-pipeline runtime paths
    pipeline_paths: dict[str, list[dict[str, Any]]] = {}
    for pid in PipelineRegistry.list_pipelines():
        config_class = PipelineRegistry.get_config_class(pid)
        if not config_class:
            continue
        schema_data = config_class.get_schema_with_metadata()
        config_schema = schema_data.get("config_schema", {})
        paths = _extract_osc_paths_from_schema(config_schema, pid)
        if paths:
            pipeline_paths[pid] = paths

    # Build active / available split
    active_paths: list[dict[str, Any]] = []
    available_paths: list[dict[str, Any]] = []

    # Runtime params are always "active"
    for p in _RUNTIME_PARAMS:
        active_paths.append({**p, "osc_address": f"/scope/{p['key']}"})

    for pid, paths in pipeline_paths.items():
        target = active_paths if pid in active_pipeline_ids else available_paths
        for p in paths:
            target.append({**p, "osc_address": f"/scope/{p['key']}"})

    return {
        "active": active_paths,
        "available": available_paths,
        "active_pipeline_ids": active_pipeline_ids,
    }


def _path_row_html(path: dict[str, Any]) -> str:
    addr = html.escape(path["osc_address"])
    ptype = html.escape(str(path.get("type", "")))
    desc = html.escape(path.get("description", ""))
    constraints: list[str] = []
    if "min" in path:
        constraints.append(f"min: {path['min']}")
    if "max" in path:
        constraints.append(f"max: {path['max']}")
    if "enum" in path:
        constraints.append(f"values: {path['enum']}")
    constraint_str = html.escape(", ".join(constraints)) if constraints else ""
    pid = html.escape(path.get("pipeline_id", "runtime"))
    return (
        f"<tr>"
        f"<td class='addr'><code>{addr}</code></td>"
        f"<td>{ptype}</td>"
        f"<td>{desc}</td>"
        f"<td>{constraint_str}</td>"
        f"<td>{pid}</td>"
        f"</tr>"
    )


def render_osc_docs_html(
    pipeline_manager: "PipelineManager | None",
    osc_port: int,
) -> str:
    """Render a self-contained HTML page documenting all current OSC paths."""
    data = get_osc_paths(pipeline_manager)
    active_rows = "\n".join(_path_row_html(p) for p in data["active"])
    available_rows = "\n".join(_path_row_html(p) for p in data["available"])
    active_ids = ", ".join(data["active_pipeline_ids"]) or "none"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daydream Scope &mdash; OSC Reference</title>
<style>
  :root {{
    --bg: #0f0f12;
    --surface: #1a1a22;
    --border: #2a2a35;
    --text: #e4e4ec;
    --muted: #8888a0;
    --accent: #6366f1;
    --accent-soft: #6366f122;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    line-height: 1.5;
  }}
  h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
  .subtitle {{ color: var(--muted); margin-bottom: 1.5rem; font-size: .9rem; }}
  h2 {{
    font-size: 1.1rem;
    margin: 2rem 0 .75rem;
    padding-bottom: .4rem;
    border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: .85rem;
  }}
  th, td {{
    text-align: left;
    padding: .5rem .75rem;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--muted); font-weight: 600; }}
  td.addr code {{
    background: var(--accent-soft);
    color: var(--accent);
    padding: .15rem .4rem;
    border-radius: 3px;
    font-size: .85em;
  }}
  .info {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    font-size: .85rem;
    color: var(--muted);
  }}
  .info code {{ color: var(--text); }}
  .empty {{ color: var(--muted); font-style: italic; padding: 1rem 0; }}
  pre {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    overflow-x: auto;
    font-size: .82rem;
  }}
</style>
</head>
<body>
<h1>Daydream Scope &mdash; OSC Reference</h1>
<p class="subtitle">Auto-generated from current app state. Refresh to update.</p>

<div class="info">
  OSC UDP port: <code>{osc_port}</code> &middot;
  Active pipeline(s): <code>{html.escape(active_ids)}</code>
</div>

<h2>Quick Start</h2>
<pre>
pip install python-osc

from pythonosc.udp_client import SimpleUDPClient
client = SimpleUDPClient("127.0.0.1", {osc_port})
client.send_message("/scope/noise_scale", 0.5)
</pre>

<h2>Active Now</h2>
<p class="subtitle">Controls for the currently loaded pipeline chain and global runtime parameters.</p>
{"<table><thead><tr><th>OSC Address</th><th>Type</th><th>Description</th><th>Constraints</th><th>Source</th></tr></thead><tbody>" + active_rows + "</tbody></table>" if active_rows else '<p class="empty">No active paths (no pipeline loaded).</p>'}

<h2>Available</h2>
<p class="subtitle">Controls for other installed pipelines / plugins (not currently active).</p>
{"<table><thead><tr><th>OSC Address</th><th>Type</th><th>Description</th><th>Constraints</th><th>Source</th></tr></thead><tbody>" + available_rows + "</tbody></table>" if available_rows else '<p class="empty">No additional paths available.</p>'}

</body>
</html>"""

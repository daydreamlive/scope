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
    # -- Input & Controls --
    {
        "key": "prompt",
        "type": "string",
        "description": "Set the prompt text (creates a single prompt with weight 1.0)",
    },
    {
        "key": "input_mode",
        "type": "string",
        "description": "Switch input mode",
        "enum": ["text", "video"],
    },
    {
        "key": "transition_steps",
        "type": "integer",
        "description": "Number of generation steps to transition between prompts (0 = instant)",
        "min": 0,
    },
    {
        "key": "interpolation_method",
        "type": "string",
        "description": "Spatial interpolation method for blending multiple prompts",
        "enum": ["linear", "slerp"],
    },
    {
        "key": "temporal_interpolation_method",
        "type": "string",
        "description": "Temporal interpolation method for transitions between prompts",
        "enum": ["linear", "slerp"],
    },
    # -- Generation controls --
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

# Maps OSC type strings to the python-osc argument types accepted for validation
_TYPE_VALIDATORS: dict[str, set[type]] = {
    "float": {int, float},
    "number": {int, float},
    "integer": {int},
    "bool": {bool, int, float},
    "boolean": {bool, int, float},
    "string": {str},
}


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


def _collect_pipeline_paths() -> dict[str, list[dict[str, Any]]]:
    """Collect per-pipeline runtime paths from the registry."""
    from scope.core.pipelines.registry import PipelineRegistry

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
    return pipeline_paths


def get_osc_paths(
    pipeline_manager: "PipelineManager | None",
) -> dict[str, Any]:
    """Build the full OSC path inventory split into *active* and *available* sections.

    Each section is a dict mapping source names to lists of path entries.
    """
    active_pipeline_ids: list[str] = []
    if pipeline_manager:
        active_pipeline_ids = pipeline_manager.get_loaded_pipeline_ids()

    pipeline_paths = _collect_pipeline_paths()

    # Build active / available split, grouped by source
    active_groups: dict[str, list[dict[str, Any]]] = {}
    available_groups: dict[str, list[dict[str, Any]]] = {}

    # Runtime params are always "active"
    runtime_list: list[dict[str, Any]] = []
    for p in _RUNTIME_PARAMS:
        runtime_list.append({**p, "osc_address": f"/scope/{p['key']}"})
    active_groups["Runtime"] = runtime_list

    for pid, paths in pipeline_paths.items():
        target = active_groups if pid in active_pipeline_ids else available_groups
        group: list[dict[str, Any]] = []
        for p in paths:
            group.append({**p, "osc_address": f"/scope/{p['key']}"})
        target[pid] = group

    return {
        "active": active_groups,
        "available": available_groups,
        "active_pipeline_ids": active_pipeline_ids,
    }


def get_all_known_paths(
    pipeline_manager: "PipelineManager | None",
) -> dict[str, dict[str, Any]]:
    """Return a flat dict mapping every known OSC key to its path metadata.

    Used by the OSC server for validating incoming messages.
    """
    data = get_osc_paths(pipeline_manager)
    result: dict[str, dict[str, Any]] = {}
    for groups in (data["active"], data["available"]):
        for paths in groups.values():
            for p in paths:
                result[p["key"]] = p
    return result


def validate_osc_value(path_info: dict[str, Any], value: Any) -> str | None:
    """Check whether *value* is acceptable for the given path.

    Returns None on success, or a human-readable reason string on failure.
    """
    expected_type = path_info.get("type", "any")
    accepted = _TYPE_VALIDATORS.get(expected_type)
    if accepted and type(value) not in accepted:
        return f"type mismatch: expected {expected_type}, got {type(value).__name__}"

    if isinstance(value, (int, float)):
        lo = path_info.get("min")
        hi = path_info.get("max")
        if lo is not None and value < lo:
            return f"value {value} below minimum {lo}"
        if hi is not None and value > hi:
            return f"value {value} above maximum {hi}"

    enum_values = path_info.get("enum")
    if enum_values is not None and value not in enum_values:
        return f"value {value!r} not in allowed values {enum_values}"

    return None


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _example_value(path: dict[str, Any]) -> str:
    """Pick a representative example value for Python code snippets."""
    ptype = path.get("type", "any")
    enum_vals = path.get("enum")
    if enum_vals:
        return repr(enum_vals[0])
    if ptype in ("float", "number"):
        lo = path.get("min", 0.0)
        hi = path.get("max", 1.0)
        mid = round((lo + hi) / 2, 2)
        return str(mid)
    if ptype in ("bool", "boolean"):
        return "True"
    if ptype == "integer":
        lo = path.get("min", 0)
        hi = path.get("max", 100)
        return str((lo + hi) // 2)
    if ptype == "string":
        key = path.get("key", "")
        if key == "prompt":
            return '"a beautiful sunset over the ocean"'
        return '"example"'
    return "0.5"


def _path_row_html(path: dict[str, Any], osc_port: int, row_id: str) -> str:
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

    example_val = _example_value(path)
    example_code = html.escape(
        f"from pythonosc.udp_client import SimpleUDPClient\n\n"
        f'client = SimpleUDPClient("127.0.0.1", {osc_port})\n'
        f'client.send_message("{path["osc_address"]}", {example_val})'
    )

    return (
        f'<tr class="path-row" onclick="toggle(\'{row_id}\')">'
        f"<td class='addr'><code>{addr}</code></td>"
        f"<td>{ptype}</td>"
        f"<td>{desc}</td>"
        f"<td>{constraint_str}</td>"
        f'<td class="chevron">&#9654;</td>'
        f"</tr>\n"
        f'<tr id="{row_id}" class="example-row" style="display:none">'
        f'<td colspan="5"><pre>{example_code}</pre></td>'
        f"</tr>"
    )


def _render_source_group(
    source_name: str,
    paths: list[dict[str, Any]],
    osc_port: int,
    id_prefix: str,
) -> str:
    """Render a source header + table for a group of paths."""
    rows = "\n".join(
        _path_row_html(p, osc_port, f"{id_prefix}-{i}") for i, p in enumerate(paths)
    )
    escaped_name = html.escape(source_name)
    return (
        f'<h3 class="source-header">{escaped_name}</h3>\n'
        f"<table><thead><tr>"
        f"<th>OSC Address</th><th>Type</th><th>Description</th>"
        f"<th>Constraints</th><th></th>"
        f"</tr></thead><tbody>\n{rows}\n</tbody></table>"
    )


def render_osc_docs_html(
    pipeline_manager: "PipelineManager | None",
    osc_port: int,
) -> str:
    """Render a self-contained HTML page documenting all current OSC paths."""
    data = get_osc_paths(pipeline_manager)
    active_groups: dict[str, list] = data["active"]
    available_groups: dict[str, list] = data["available"]

    active_html_parts: list[str] = []
    for i, (source, paths) in enumerate(active_groups.items()):
        active_html_parts.append(_render_source_group(source, paths, osc_port, f"a{i}"))
    active_html = (
        "\n".join(active_html_parts)
        if active_html_parts
        else ('<p class="empty">No active paths (no pipeline loaded).</p>')
    )

    available_html_parts: list[str] = []
    for i, (source, paths) in enumerate(available_groups.items()):
        available_html_parts.append(
            _render_source_group(source, paths, osc_port, f"v{i}")
        )
    available_html = (
        "\n".join(available_html_parts)
        if available_html_parts
        else ('<p class="empty">No additional paths available.</p>')
    )

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
  h3.source-header {{
    font-size: .95rem;
    color: var(--accent);
    margin: 1.25rem 0 .5rem;
    font-weight: 600;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: .85rem;
    margin-bottom: 1rem;
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
  .path-row {{
    cursor: pointer;
    transition: background .15s;
  }}
  .path-row:hover {{
    background: var(--surface);
  }}
  .path-row td.chevron {{
    color: var(--muted);
    font-size: .7rem;
    text-align: center;
    width: 2rem;
    transition: transform .2s;
  }}
  .path-row.expanded td.chevron {{
    transform: rotate(90deg);
  }}
  .example-row td {{
    padding: 0 .75rem .75rem;
    border-bottom: 1px solid var(--border);
  }}
  .example-row pre {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .75rem 1rem;
    overflow-x: auto;
    font-size: .8rem;
    margin: 0;
    color: var(--text);
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
<script>
function toggle(id) {{
  var row = document.getElementById(id);
  var trigger = row.previousElementSibling;
  if (row.style.display === "none") {{
    row.style.display = "table-row";
    trigger.classList.add("expanded");
  }} else {{
    row.style.display = "none";
    trigger.classList.remove("expanded");
  }}
}}
</script>
</head>
<body>
<h1>Daydream Scope &mdash; OSC Reference</h1>
<p class="subtitle">Auto-generated from current app state. Refresh to update.</p>

<div class="info">
  OSC UDP port: <code>{osc_port}</code>
</div>

<p class="info" style="margin-top:0.5em;">
  <strong>Note:</strong> OSC commands are forwarded to the active controller
  session. A session is automatically registered as the controller when a
  stream starts in the Scope UI.
</p>

<h2>Quick Start</h2>
<pre>
pip install python-osc

from pythonosc.udp_client import SimpleUDPClient
client = SimpleUDPClient("127.0.0.1", {osc_port})

# Set a prompt
client.send_message("/scope/prompt", "a beautiful sunset over the ocean")

# Adjust noise scale
client.send_message("/scope/noise_scale", 0.5)
</pre>

<h2>Active Now</h2>
<p class="subtitle">Controls for the currently loaded pipeline chain and global runtime parameters. Click a row to see example code.</p>
{active_html}

<h2>Available</h2>
<p class="subtitle">Controls for other installed pipelines / plugins (not currently active). Click a row to see example code.</p>
{available_html}

</body>
</html>"""

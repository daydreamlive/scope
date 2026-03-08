"""DMX documentation and HTML reference page generator.

Generates a self-contained HTML page documenting DMX channel mapping
for Daydream Scope.
"""

import html
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline_manager import PipelineManager


def render_dmx_docs_html(
    pipeline_manager: "PipelineManager | None",
    dmx_port: int,
) -> str:
    """Render a self-contained HTML page documenting DMX control."""
    # Get available parameters from the OSC docs module (same params apply)
    from .osc_docs import get_osc_paths

    data = get_osc_paths(pipeline_manager)

    # Collect all available parameters
    all_params: list[dict[str, Any]] = []
    for groups in (data["active"], data["available"]):
        for paths in groups.values():
            all_params.extend(paths)

    # Generate parameter table rows
    param_rows = ""
    for p in all_params:
        key = html.escape(p.get("key", ""))
        ptype = html.escape(str(p.get("type", "")))
        desc = html.escape(p.get("description", ""))
        constraints = []
        if "min" in p:
            constraints.append(f"min: {p['min']}")
        if "max" in p:
            constraints.append(f"max: {p['max']}")
        constraint_str = html.escape(", ".join(constraints)) if constraints else "-"

        param_rows += f"""
            <tr>
                <td><code>{key}</code></td>
                <td>{ptype}</td>
                <td>{desc}</td>
                <td>{constraint_str}</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Daydream Scope &mdash; DMX Reference</title>
<style>
  :root {{
    --bg: #0f0f12;
    --surface: #1a1a22;
    --border: #2a2a35;
    --text: #e4e4ec;
    --muted: #8888a0;
    --accent: #22c55e;
    --accent-soft: #22c55e22;
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
  h3 {{
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
  td code {{
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
  pre {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem;
    overflow-x: auto;
    font-size: .82rem;
  }}
  ul {{ padding-left: 1.5rem; margin-bottom: 1rem; }}
  li {{ margin-bottom: 0.5rem; }}
  .highlight {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>Daydream Scope &mdash; DMX Reference</h1>
<p class="subtitle">Control Scope parameters via Art-Net DMX.</p>

<div class="info">
  <strong>Art-Net UDP Port:</strong> <code>{dmx_port}</code>
</div>

<h2>Overview</h2>
<p class="info">
  DMX (Digital Multiplex) control uses the <strong>Art-Net</strong> protocol over UDP.
  Unlike OSC, DMX channels are numeric (1-512) with 8-bit values (0-255).
  You must configure a <strong>mapping</strong> to connect a DMX channel to a Scope parameter.
</p>

<h2>How It Works</h2>
<ul>
  <li>Scope listens for Art-Net packets on UDP port <code>{dmx_port}</code></li>
  <li>Each DMX universe has 512 channels with values 0-255</li>
  <li>Configure channel mappings in <strong>Settings → DMX</strong></li>
  <li>DMX value 0 maps to the parameter's minimum value</li>
  <li>DMX value 255 maps to the parameter's maximum value</li>
</ul>

<h2>Quick Start</h2>
<pre>
# Python example using stupidArtnet
from stupidArtnet import StupidArtnet

# Connect to Scope on the same machine
artnet = StupidArtnet("127.0.0.1", 0)  # Universe 0
artnet.start()

# Set a DMX packet (512 channels)
packet = [0] * 512

# If you've mapped channel 1 → noise_scale:
# Set channel 1 to 128 (50% = 0.5 noise_scale)
packet[0] = 128

artnet.set(packet)
artnet.show()
</pre>

<h2>Example Mapping</h2>
<div class="info">
  <p>In Settings → DMX, create a mapping:</p>
  <ul>
    <li><strong>Universe:</strong> 0</li>
    <li><strong>Channel:</strong> 1</li>
    <li><strong>Parameter:</strong> noise_scale</li>
    <li><strong>Min Value:</strong> 0.0</li>
    <li><strong>Max Value:</strong> 1.0</li>
  </ul>
  <p>Now DMX channel 1 controls noise_scale: value 0 → 0.0, value 255 → 1.0</p>
</div>

<h2>Compatible Software</h2>
<ul>
  <li><strong>QLC+</strong> — Free, open-source lighting control</li>
  <li><strong>MagicQ</strong> — Professional lighting console (free PC version)</li>
  <li><strong>TouchDesigner</strong> — Art-Net output via DMX Out CHOP</li>
  <li><strong>Resolume Arena</strong> — Art-Net output for fixture control</li>
  <li><strong>MA Lighting grandMA</strong> — Professional consoles</li>
  <li><strong>Chamsys MagicQ</strong> — Professional consoles</li>
</ul>

<h2>Available Parameters</h2>
<p class="subtitle">These parameters can be mapped to DMX channels.</p>

<table>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Type</th>
      <th>Description</th>
      <th>Range</th>
    </tr>
  </thead>
  <tbody>
    {param_rows}
  </tbody>
</table>

<h2>Troubleshooting</h2>
<ul>
  <li><strong>No response:</strong> Ensure your Art-Net sender targets port {dmx_port} and the correct IP</li>
  <li><strong>Wrong values:</strong> Check your mapping's min/max range matches the parameter</li>
  <li><strong>Multiple universes:</strong> Each mapping specifies which universe it responds to</li>
  <li><strong>Firewall:</strong> Ensure UDP port {dmx_port} is open for incoming traffic</li>
</ul>

</body>
</html>"""

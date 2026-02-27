# OSC (Open Sound Control) Integration

Scope includes an optional OSC server that allows external applications to control pipeline parameters in real-time over UDP. This enables integration with creative tools like TouchDesigner, Resolume, Max/MSP, Ableton Live, and hardware MIDI controllers.

## Prerequisites

Install the `python-osc` dependency:

```bash
uv pip install python-osc
```

Or install the optional group:

```bash
uv sync --extra osc
```

## Enabling the OSC Server

### CLI Flag

```bash
uv run daydream-scope --osc
```

With a custom port (default is 9000):

```bash
uv run daydream-scope --osc --osc-port 8000
```

### Environment Variables

```bash
DAYDREAM_SCOPE_OSC=1 uv run daydream-scope
DAYDREAM_SCOPE_OSC=1 DAYDREAM_SCOPE_OSC_PORT=8000 uv run daydream-scope
```

### Runtime API

You can also start/stop the OSC server at runtime without restarting Scope:

```bash
# Start
curl -X POST http://localhost:8000/api/v1/osc/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "port": 9000}'

# Check status
curl http://localhost:8000/api/v1/osc/status

# Stop
curl -X POST http://localhost:8000/api/v1/osc/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

## OSC Address Map

All addresses are under the `/scope/` namespace.

| Address | Arguments | Description |
|---|---|---|
| `/scope/prompt` | `<string>` | Set the first prompt text |
| `/scope/prompt/weight` | `<float 0-100>` | Set first prompt weight |
| `/scope/noise` | `<float 0.0-1.0>` | Set noise scale |
| `/scope/denoise` | `<int> [<int>...]` | Set denoising step list |
| `/scope/cache/reset` | *(none)* | Trigger a cache reset |
| `/scope/cache/bias` | `<float 0.01-1.0>` | Set KV cache attention bias |
| `/scope/output/<type>/enable` | `[<string name>]` | Enable an output sink (e.g., `/scope/output/ndi/enable`) |
| `/scope/output/<type>/disable` | *(none)* | Disable an output sink |
| `/scope/param/<key>` | `<value>` | Generic parameter passthrough |

### Generic Parameters

The `/scope/param/<key>` address is a catch-all that lets you set any pipeline parameter by name. The `<key>` becomes the parameter name and the argument becomes the value. This means new parameters are automatically controllable via OSC without code changes.

## Examples

### Python (python-osc)

```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 9000)

# Change the prompt
client.send_message("/scope/prompt", "A cinematic shot of a futuristic city at night")

# Adjust noise
client.send_message("/scope/noise", 0.5)

# Set denoising steps
client.send_message("/scope/denoise", [700, 500])

# Reset cache
client.send_message("/scope/cache/reset", None)

# Enable NDI output
client.send_message("/scope/output/ndi/enable", "ScopeOut")

# Generic parameter
client.send_message("/scope/param/noise_controller", True)
```

### TouchDesigner

1. Add a **CHOP Execute** or **Script** DAT
2. Use the `td` module with OSC Out CHOP, or use `python-osc` directly:

```python
# In a Script DAT or CHOP Execute callback
import socket, struct

def send_osc(address, value, host="127.0.0.1", port=9000):
    """Minimal OSC sender for TouchDesigner."""
    # For full OSC support, use the python-osc library or TD's built-in OSC Out CHOP
    pass

# Easier: use TD's OSC Out CHOP
# 1. Create an OSC Out CHOP
# 2. Set Network Address to 127.0.0.1, Port to 9000
# 3. Map channels to OSC addresses:
#    - /scope/noise -> noise_scale slider
#    - /scope/cache/bias -> bias knob
```

Alternatively, use TD's built-in **OSC Out CHOP**:
- Set **Network Address** to `127.0.0.1`
- Set **Port** to `9000`
- Create channels with names matching the OSC addresses (e.g., `scope/noise`)

### Max/MSP

```
[udpsend 127.0.0.1 9000]
    |
[prepend /scope/prompt]
    |
[message "A forest with glowing mushrooms"]
```

### Resolume

Resolume Arena supports sending OSC natively. In the Output menu:
1. Add an OSC output target pointing to `127.0.0.1:9000`
2. Map parameters to the `/scope/` addresses above

## Architecture

The OSC server runs as a background UDP listener within the Scope server process. Incoming messages are dispatched to handlers that translate OSC addresses into pipeline parameter updates, which are pushed to all active WebRTC sessions via the same `update_parameters()` path used by the frontend.

```
External App ──UDP──> OSC Server ──> Address Mapper ──> FrameProcessor.update_parameters()
                                                              │
                                                              ▼
                                                         Pipeline
```

## Troubleshooting

**OSC server not starting:**
- Ensure `python-osc` is installed: `uv pip install python-osc`
- Check if the port is already in use: `lsof -i :9000`
- Try a different port: `--osc-port 9001`

**Messages not having effect:**
- Verify a stream is active (OSC messages require an active WebRTC session)
- Check server logs for `OSC message received but no active sessions`
- Verify the address format matches the table above (addresses are case-sensitive)

**Testing connectivity:**
```python
# Quick test script
from pythonosc.udp_client import SimpleUDPClient
client = SimpleUDPClient("127.0.0.1", 9000)
client.send_message("/scope/noise", 0.3)
# Check Scope server logs for: "OSC noise scale: 0.3"
```

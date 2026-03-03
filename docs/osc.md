# OSC (Open Sound Control) Integration

Scope includes an OSC server that allows external applications to control pipeline parameters in real-time over UDP. This enables integration with creative tools like TouchDesigner, Resolume, Max/MSP, Ableton Live, and hardware MIDI controllers.

## Enabling the OSC Server

The OSC server is toggled on/off at runtime via the HTTP API. It binds to the same host and port as the main Scope HTTP server (UDP and TCP can share a port).

```bash
# Start
curl -X POST http://localhost:8000/api/v1/osc/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Check status
curl http://localhost:8000/api/v1/osc/status

# Stop
curl -X POST http://localhost:8000/api/v1/osc/config \
  -H "Content-Type: application/json" \
  -d '{"enabled": false}'
```

You can also toggle OSC on/off from the Scope UI in the Settings panel.

## OSC Address Format

All OSC messages use the `/scope/<key>` address pattern, where `<key>` is the pipeline parameter name. The key can contain slashes for nested paths. The value argument(s) are forwarded directly as the parameter value.

| Address | Arguments | Description |
|---|---|---|
| `/scope/<key>` | `<value> [<value>...]` | Set any pipeline parameter by name |

A single argument is passed as a scalar value; multiple arguments are passed as a list.

### Examples

| Address | Arguments | Effect |
|---|---|---|
| `/scope/prompt` | `"A cinematic city"` | Set the prompt text |
| `/scope/noise_scale` | `0.5` | Set noise scale |
| `/scope/denoising_step_list` | `700 500` | Set denoising steps (multi-arg becomes list) |
| `/scope/reset_cache` | `1` | Trigger a cache reset |
| `/scope/kv_cache_attention_bias` | `0.3` | Set KV cache attention bias |
| `/scope/noise_controller` | `1` | Enable noise controller |

Messages sent to addresses outside the `/scope/` namespace are silently ignored.

## Client Examples

### Python (python-osc)

```python
from pythonosc.udp_client import SimpleUDPClient

client = SimpleUDPClient("127.0.0.1", 8000)

# Change the prompt
client.send_message("/scope/prompt", "A cinematic shot of a futuristic city at night")

# Adjust noise
client.send_message("/scope/noise_scale", 0.5)

# Set denoising steps
client.send_message("/scope/denoising_step_list", [700, 500])

# Reset cache
client.send_message("/scope/reset_cache", True)

# Generic parameter
client.send_message("/scope/noise_controller", True)
```

### TouchDesigner

1. Add a **CHOP Execute** or **Script** DAT
2. Use the `td` module with OSC Out CHOP, or use `python-osc` directly

Alternatively, use TD's built-in **OSC Out CHOP**:
- Set **Network Address** to `127.0.0.1`
- Set **Port** to `8000`
- Create channels with names matching the OSC addresses (e.g., `scope/noise_scale`)

### Max/MSP

```
[udpsend 127.0.0.1 8000]
    |
[prepend /scope/prompt]
    |
[message "A forest with glowing mushrooms"]
```

### Resolume

Resolume Arena supports sending OSC natively. In the Output menu:
1. Add an OSC output target pointing to `127.0.0.1:8000`
2. Map parameters to the `/scope/<key>` addresses

## Architecture

The OSC server runs as a background UDP listener within the Scope server process. Incoming messages are dispatched to a generic handler that translates OSC addresses into pipeline parameter updates, which are pushed to all active WebRTC sessions via the same `update_parameters()` path used by the frontend.

```
External App ──UDP──> OSC Server ──> Generic Handler ──> FrameProcessor.update_parameters()
                                                              │
                                                              ▼
                                                         Pipeline
```

## Troubleshooting

**OSC server not starting:**
- Check if the UDP port is already in use: `lsof -i :8000`
- Check server logs for error messages

**Messages not having effect:**
- Verify a stream is active (OSC messages require an active WebRTC session)
- Check server logs for `OSC message received but no active sessions`
- Verify the address starts with `/scope/` (addresses are case-sensitive)

**Testing connectivity:**
```python
# Quick test script
from pythonosc.udp_client import SimpleUDPClient
client = SimpleUDPClient("127.0.0.1", 8000)
client.send_message("/scope/noise_scale", 0.3)
# Check Scope server logs for: "OSC: noise_scale = 0.3"
```

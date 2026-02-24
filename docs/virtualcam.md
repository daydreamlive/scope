# Using Virtual Camera Output

Scope can send its processed video output to a virtual camera device, making it appear as a standard webcam in any application.

## Supported Applications

Any application that supports webcam input works with Scope's virtual camera:

- **Video conferencing**: Zoom, Google Meet, Microsoft Teams, Discord
- **Streaming**: OBS Studio, Streamlabs
- **Recording**: Any screen recorder with webcam support
- **Creative tools**: After Effects, Premiere Pro (as live input)

## Prerequisites

### Windows

Install [OBS Studio](https://obsproject.com/) (version 26.0 or later). The OBS Virtual Camera is included automatically.

### macOS

Install [OBS Studio](https://obsproject.com/) (version 26.0 or later). The OBS Virtual Camera is included automatically.

### Linux

Install the v4l2loopback kernel module:

```bash
# Ubuntu/Debian
sudo apt install v4l2loopback-dkms

# Load the module
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Scope" exclusive_caps=1
```

To make the module load automatically on boot, add `v4l2loopback` to `/etc/modules-load.d/v4l2loopback.conf`.

### Python Package

The virtual camera feature requires the `pyvirtualcam` package. Install it with:

```bash
uv sync --extra virtualcam
```

Or add it to your installation:

```bash
pip install pyvirtualcam
```

## Usage

1. In the **Settings** panel on the right side of the Scope UI, scroll down to find **Virtual Camera** under the Output section.
2. Toggle it **ON**.
3. Open your target application (Zoom, OBS, etc.).
4. Select **OBS Virtual Camera** (Windows/macOS) or the v4l2loopback device (Linux) as your camera source.
5. Start streaming in Scope — the output will appear in your application.

## Troubleshooting

### "Virtual Camera" toggle not visible

- Ensure OBS Studio is installed (Windows/macOS)
- Ensure v4l2loopback is loaded (Linux): `lsmod | grep v4l2loopback`
- Ensure `pyvirtualcam` is installed: `pip show pyvirtualcam`
- Restart Scope after installation

### Camera shows black/no video

- Make sure Scope is actively streaming (press Start)
- Check that no other application has exclusive access to the virtual camera

### OBS shows "Camera in use"

- The OBS Virtual Camera can only be used by one sender at a time
- Close OBS if you want to send from Scope
- Alternatively, use [NDI](ndi.md) or [Spout](spout.md) for OBS integration (these support multiple receivers)

### Linux: Device not found

Ensure the v4l2loopback module is loaded with the correct parameters:

```bash
# Check if module is loaded
lsmod | grep v4l2loopback

# Load with explicit device number
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="Scope" exclusive_caps=1

# Verify device exists
ls -la /dev/video*
```

### Resolution mismatch

The virtual camera is created with a fixed resolution matching Scope's pipeline dimensions. If your target application expects a different resolution, it should automatically scale the video. If you see stretching or letterboxing:

1. Change Scope's pipeline resolution to match your target (e.g., 1280x720 for 720p)
2. Or adjust your application's camera settings to match Scope's output

## Technical Details

Scope uses [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) to interface with platform-specific virtual camera backends:

| Platform | Backend | Device Name |
|----------|---------|-------------|
| Windows | OBS Virtual Camera | `OBS Virtual Camera` |
| macOS | OBS Virtual Camera | `OBS Virtual Camera` |
| Linux | v4l2loopback | `/dev/video<n>` |

### Limitations

1. **Single instance**: The OBS Virtual Camera only supports one sender at a time. If OBS is running with its virtual camera active, Scope cannot use it.

2. **Video only**: Virtual camera output is video-only. Audio must be handled separately through your system's audio routing.

3. **Fixed resolution at start**: The virtual camera resolution is set when streaming starts. To change resolution, stop and restart the stream.

## Comparison with Other Outputs

| Feature | Virtual Camera | NDI | Spout |
|---------|---------------|-----|-------|
| Platform | All | All | Windows |
| Requires app | OBS (Win/Mac) or v4l2loopback (Linux) | NDI Tools | None (built-in) |
| Network sharing | No (local only) | Yes | No (local only) |
| Multiple receivers | No | Yes | Yes |
| Target apps | Any webcam app | NDI-compatible only | Spout-compatible only |
| Best for | Zoom, Discord, Meet | TouchDesigner, Resolume, OBS | TouchDesigner, Resolume |

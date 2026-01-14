# Scope Layout Control Preprocessor

Interactive layout control preprocessor for VACE conditioning. Generates white background + black circle contour frames based on keyboard/mouse input for point-based subject control.

## Installation

```bash
daydream-scope install -e plugins/scope-layout-control
```

## Usage

1. Select a VACE-enabled pipeline (e.g., LongLive, Krea Realtime Video)
2. Enable VACE in the settings panel
3. Select "Layout Control (WASD)" from the preprocessor dropdown
4. Start streaming
5. Click the video output area to enable pointer lock
6. Use WASD or arrow keys to move the circle position
7. The model will follow the circle for subject positioning

## Controls

- **W / ArrowUp**: Move circle up
- **S / ArrowDown**: Move circle down
- **A / ArrowLeft**: Move circle left
- **D / ArrowRight**: Move circle right
- **Mouse movement**: Fine-grained position control
- **Escape**: Release pointer lock

## Configuration

The preprocessor can be configured with:

- `radius`: Circle radius in pixels (default: 80)
- `move_speed`: Movement speed per frame for keys (default: 0.02)
- `mouse_sensitivity`: Mouse movement multiplier (default: 0.002)
- `initial_x/y`: Starting position (default: center-top)

## How It Works

The preprocessor maintains internal state for the circle position. On each pipeline call:

1. Position is updated based on currently pressed keys and accumulated mouse delta
2. Frames are generated with interpolation from previous to current position
3. Output is white background with black circle contour in VACE format

This creates smooth motion even when generating multiple frames per chunk.

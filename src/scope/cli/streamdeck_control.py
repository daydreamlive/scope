#!/usr/bin/env python3
"""Stream Deck controller for Scope - sends style commands to remote server.

Usage:
    VIDEO_API_URL=http://your-gpu-server:8000 uv run python -m scope.cli.streamdeck_control

Or:
    uv run python -m scope.cli.streamdeck_control --url http://your-gpu-server:8000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from io import BytesIO

import httpx
from PIL import Image, ImageDraw, ImageFont

# Button layout (15-key Stream Deck, 3 rows x 5 cols)
# Key indices go left-to-right, top-to-bottom:
#   [0]  [1]  [2]  [3]  [4]    Row 0: HIDARI YETI TMNT RAT KAIJU
#   [5]  [6]  [7]  [8]  [9]    Row 1: [empty row]
#   [10] [11] [12] [13] [14]   Row 2: STEP HARD SOFT PLAY [empty]

STYLES = ["hidari", "yeti", "tmnt", "rat", "kaiju"]

# Key index mapping (0-14, left-to-right, top-to-bottom)
STYLE_KEYS = {0: "hidari", 1: "yeti", 2: "tmnt", 3: "rat", 4: "kaiju"}
ACTION_KEYS = {
    10: "step",       # Bottom row, first
    11: "hard_cut",   # Bottom row, second
    12: "soft_cut",   # Bottom row, third
    13: "play_pause", # Bottom row, fourth
}


def create_button_image(
    deck, text: str, bg_color: str = "#1a1a2e", text_color: str = "#ffffff", active: bool = False
) -> bytes:
    """Create a button image with text."""
    # Get the button size for this deck
    image_format = deck.key_image_format()
    size = (image_format["size"][0], image_format["size"][1])

    # Create image
    if active:
        bg_color = "#4a9eff"  # Highlight active style
    img = Image.new("RGB", size, bg_color)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default
    font_size = size[0] // 5
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Center the text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    draw.text((x, y), text, font=font, fill=text_color)

    # Rotate 180° - Stream Deck Original has flipped orientation
    img = img.rotate(180)

    # Convert to the format the deck expects
    img_bytes = BytesIO()
    img.save(img_bytes, format="JPEG")
    return img_bytes.getvalue()


class StreamDeckController:
    """Controls Scope via Stream Deck button presses."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")
        self.client = httpx.Client(timeout=5.0)
        self.deck = None
        self.current_style: str | None = None
        self.is_paused: bool = False

    def connect(self) -> bool:
        """Connect to Stream Deck."""
        from StreamDeck.DeviceManager import DeviceManager

        decks = DeviceManager().enumerate()
        if not decks:
            print("No Stream Deck found!")
            return False

        self.deck = decks[0]
        self.deck.open()
        try:
            self.deck.reset()
        except Exception as e:
            print(f"Warning: Could not reset deck ({e}), continuing anyway...")
        print(f"Connected: {self.deck.deck_type()} ({self.deck.key_count()} keys)")
        return True

    def update_buttons(self):
        """Update all button images."""
        if not self.deck:
            return

        # Style buttons (keys 0-3)
        for key, style in STYLE_KEYS.items():
            active = style == self.current_style
            img = create_button_image(self.deck, style[:6].upper(), active=active)
            self.deck.set_key_image(key, img)

        # Action buttons (bottom row: 10, 11, 12, 13)
        self.deck.set_key_image(10, create_button_image(self.deck, "STEP", bg_color="#2d3436"))
        self.deck.set_key_image(11, create_button_image(self.deck, "HARD", bg_color="#d63031"))
        self.deck.set_key_image(12, create_button_image(self.deck, "SOFT", bg_color="#fdcb6e", text_color="#000000"))
        self.deck.set_key_image(13, create_button_image(self.deck, "PLAY" if self.is_paused else "PAUSE", bg_color="#2d3436"))

        # Clear unused keys
        for key in range(15):
            if key not in STYLE_KEYS and key not in [10, 11, 12, 13]:
                self.deck.set_key_image(key, create_button_image(self.deck, "", bg_color="#0d0d0d"))

    def fetch_state(self):
        """Fetch current state from server."""
        try:
            r = self.client.get(f"{self.api_url}/api/v1/realtime/state")
            if r.status_code == 200:
                state = r.json()
                self.current_style = state.get("active_style")
                self.is_paused = state.get("paused", False)
                return True
        except httpx.RequestError as e:
            print(f"Failed to fetch state: {e}")
        return False

    def set_style(self, style: str):
        """Set the active style."""
        try:
            r = self.client.put(f"{self.api_url}/api/v1/realtime/style", json={"name": style})
            if r.status_code == 200:
                print(f"Style: {style}")
                self.current_style = style
                self.update_buttons()
            else:
                print(f"Failed to set style: {r.status_code}")
        except httpx.RequestError as e:
            print(f"Error: {e}")

    def toggle_pause(self):
        """Toggle pause/play."""
        try:
            endpoint = "/api/v1/realtime/run" if self.is_paused else "/api/v1/realtime/pause"
            r = self.client.post(f"{self.api_url}{endpoint}")
            if r.status_code == 200:
                self.is_paused = not self.is_paused
                print("Paused" if self.is_paused else "Running")
                self.update_buttons()
        except httpx.RequestError as e:
            print(f"Error: {e}")

    def step(self):
        """Step one frame."""
        try:
            r = self.client.post(f"{self.api_url}/api/v1/realtime/step")
            if r.status_code == 200:
                print("Stepped")
        except httpx.RequestError as e:
            print(f"Error: {e}")

    def hard_cut(self):
        """Trigger hard cut (reset cache)."""
        try:
            r = self.client.post(f"{self.api_url}/api/v1/realtime/hard-cut")
            if r.status_code == 200:
                print("Hard cut!")
        except httpx.RequestError as e:
            print(f"Error: {e}")

    def soft_cut(self):
        """Trigger soft cut."""
        try:
            r = self.client.post(f"{self.api_url}/api/v1/realtime/soft-cut")
            if r.status_code == 200:
                print("Soft cut")
        except httpx.RequestError as e:
            print(f"Error: {e}")

    def on_key(self, deck, key: int, pressed: bool):
        """Handle key press."""
        if not pressed:  # Only act on press, not release
            return

        if key in STYLE_KEYS:
            self.set_style(STYLE_KEYS[key])
        elif key == 10:
            self.step()
        elif key == 11:
            self.hard_cut()
        elif key == 12:
            self.soft_cut()
        elif key == 13:
            self.toggle_pause()

    def run(self):
        """Main loop."""
        if not self.connect():
            return 1

        # Fetch initial state
        if self.fetch_state():
            print(f"Current style: {self.current_style}, Paused: {self.is_paused}")
        else:
            print("Warning: Could not fetch initial state (server may be offline)")

        self.update_buttons()
        self.deck.set_key_callback(self.on_key)

        print("\nStream Deck ready! Press Ctrl+C to exit.")
        print("  Row 1: HIDARI | YETI | TMNT | RAT | KAIJU")
        print("  Row 3: STEP | HARD | SOFT | PLAY/PAUSE")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            if self.deck:
                try:
                    self.deck.reset()
                except Exception:
                    pass  # Ignore reset errors during cleanup
                self.deck.close()

        return 0


def main():
    parser = argparse.ArgumentParser(description="Stream Deck controller for Scope")
    parser.add_argument(
        "--url",
        default=os.environ.get("VIDEO_API_URL", "http://localhost:8000"),
        help="Scope server URL (default: VIDEO_API_URL env or http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"Connecting to: {args.url}")
    controller = StreamDeckController(args.url)
    sys.exit(controller.run())


if __name__ == "__main__":
    main()

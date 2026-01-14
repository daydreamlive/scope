"""
Point-Based Subject Control Demo - Real-time Interactive Application.

A minimal, high-performance demo showcasing point-based subject control with LongLive + VACE.

Left panel: Draggable circle (control signal)
Right panel: Real-time LongLive output following the circle

Usage:
    python -m scope.core.pipelines.longlive.demo_point_control
"""

import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pygame
import torch
from omegaconf import OmegaConf

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    "subject_image": "frontend/public/assets/woman2.jpg",
    "prompt": "",
    "height": 512,
    "width": 512,
    "frames_per_chunk": 12,
    "layout_radius": 80,
    "vace_context_scale": 1.5,
    "vae_type": "tae",
    "window_width": 1024,  # 512 left + 512 right
    "window_height": 512,
    "target_fps": 60,  # Display refresh rate
}

# =========================================================================


class PointControlDemo:
    def __init__(self):
        self.config = CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Shared state (thread-safe)
        self.circle_pos = [0.5, 0.35]  # Normalized [x, y]
        self.pos_lock = threading.Lock()

        # Frame queue for display
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

        # Control flags
        self.running = True
        self.pipeline_ready = threading.Event()
        self.first_chunk_done = threading.Event()

        # Display state
        self.current_frame: np.ndarray | None = None
        self.last_frame_time = 0.0

        # Dynamic FPS tracking (matches FrameProcessor approach)
        self.playback_fps = 30.0  # Default, will be updated from pipeline performance
        self.min_fps = 1.0
        self.max_fps = 60.0
        self.fps_lock = threading.Lock()

        # Mouse state
        self.dragging = False

        # Playback control
        self.paused = False

        # Previous position for interpolation
        self.prev_pos = [0.5, 0.35]

    def get_circle_pos(self) -> tuple[float, float]:
        with self.pos_lock:
            return tuple(self.circle_pos)

    def update_pipeline_fps(self, num_frames: int, processing_time: float):
        """Update playback FPS based on actual pipeline performance."""
        if processing_time > 0:
            estimated_fps = num_frames / processing_time
            clamped_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
            with self.fps_lock:
                self.playback_fps = clamped_fps

    def get_playback_fps(self) -> float:
        with self.fps_lock:
            return self.playback_fps

    def set_circle_pos(self, x: float, y: float):
        with self.pos_lock:
            self.circle_pos = [
                max(0.1, min(0.9, x)),
                max(0.1, min(0.9, y)),
            ]

    def create_layout_frame(self, x_norm: float, y_norm: float) -> np.ndarray:
        """Create a single layout control frame (white bg + black contour)."""
        h, w = self.config["height"], self.config["width"]
        frame = np.ones((h, w, 3), dtype=np.uint8) * 255

        px = int(x_norm * w)
        py = int(y_norm * h)
        radius = self.config["layout_radius"]
        px = max(radius, min(w - radius, px))
        py = max(radius, min(h - radius, py))

        cv2.circle(frame, (px, py), radius, (0, 0, 0), thickness=3)
        return frame

    def create_layout_chunk(
        self, x_norm: float, y_norm: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create layout control tensors for a chunk, interpolating from previous position."""
        n_frames = self.config["frames_per_chunk"]
        h, w = self.config["height"], self.config["width"]

        # Get previous position for smooth interpolation
        prev_x, prev_y = self.prev_pos

        # Interpolate positions across frames for smooth motion
        frames = []
        for i in range(n_frames):
            t = i / max(n_frames - 1, 1)  # 0 to 1
            interp_x = prev_x + (x_norm - prev_x) * t
            interp_y = prev_y + (y_norm - prev_y) * t
            frames.append(self.create_layout_frame(interp_x, interp_y))

        frames = np.stack(frames)
        masks = np.ones((n_frames, h, w), dtype=np.float32)

        # Update previous position for next chunk
        self.prev_pos = [x_norm, y_norm]

        # Convert to tensors: [1, C, F, H, W]
        frames_t = torch.from_numpy(frames).float() / 255.0 * 2.0 - 1.0
        frames_t = frames_t.permute(0, 3, 1, 2).unsqueeze(0).permute(0, 2, 1, 3, 4)

        masks_t = torch.from_numpy(masks).float()
        masks_t = masks_t.unsqueeze(1).unsqueeze(0).permute(0, 2, 1, 3, 4)

        return frames_t.to(self.device), masks_t.to(self.device)

    def pipeline_thread(self):
        """Background thread that continuously generates chunks."""
        print("Initializing pipeline...")

        # Resolve paths
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent.parent.parent.parent
        subject_path = project_root / self.config["subject_image"]

        if not subject_path.exists():
            print(f"ERROR: Subject image not found: {subject_path}")
            self.running = False
            return

        # Initialize pipeline
        vace_path = str(
            get_model_file_path("Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors")
        )

        pipeline_config = OmegaConf.create(
            {
                "model_dir": str(get_models_dir()),
                "generator_path": str(
                    get_model_file_path("LongLive-1.3B/models/longlive_base.pt")
                ),
                "lora_path": str(get_model_file_path("LongLive-1.3B/models/lora.pt")),
                "vace_path": vace_path,
                "text_encoder_path": str(
                    get_model_file_path(
                        "WanVideo_comfy/umt5-xxl-enc-fp8_e4m3fn.safetensors"
                    )
                ),
                "tokenizer_path": str(
                    get_model_file_path("Wan2.1-T2V-1.3B/google/umt5-xxl")
                ),
                "model_config": OmegaConf.load(script_dir / "model.yaml"),
                "height": self.config["height"],
                "width": self.config["width"],
                "vae_type": self.config["vae_type"],
            }
        )

        pipeline_config.model_config.base_model_kwargs = (
            pipeline_config.model_config.base_model_kwargs or {}
        )
        pipeline_config.model_config.base_model_kwargs["vace_in_dim"] = 96

        pipeline = LongLivePipeline(
            pipeline_config, device=self.device, dtype=torch.bfloat16
        )
        print("Pipeline ready!")
        self.pipeline_ready.set()

        chunk_idx = 0

        while self.running:
            # Wait while paused
            while self.paused and self.running:
                time.sleep(0.1)

            start_time = time.time()
            x, y = self.get_circle_pos()

            if chunk_idx == 0:
                # First chunk: extension mode to establish subject identity
                kwargs = {
                    "prompts": [{"text": self.config["prompt"], "weight": 100}],
                    "first_frame_image": str(subject_path),
                }
                mode = "extension"
            else:
                # Subsequent chunks: layout control
                frames_t, masks_t = self.create_layout_chunk(x, y)
                kwargs = {
                    "prompts": [{"text": self.config["prompt"], "weight": 100}],
                    "vace_context_scale": self.config["vace_context_scale"],
                    "vace_input_frames": frames_t,
                    "vace_input_masks": masks_t,
                }
                mode = "layout"

            # Generate
            output = pipeline(**kwargs)

            latency = time.time() - start_time
            num_frames = output.shape[0]
            fps = num_frames / latency
            print(
                f"Chunk {chunk_idx} ({mode}): {num_frames} frames, {latency:.2f}s, {fps:.1f} fps, pos=({x:.2f}, {y:.2f})"
            )

            # Update playback FPS based on actual pipeline performance
            self.update_pipeline_fps(num_frames, latency)

            # Convert to display format and queue frames
            frames_np = output.detach().cpu().numpy()
            frames_np = np.clip(frames_np * 255, 0, 255).astype(np.uint8)

            # Skip chunk 0 frames for display - they establish identity but don't follow circle
            # Only display chunks 1+ where layout control is active
            if chunk_idx == 0:
                self.first_chunk_done.set()
                print("  (chunk 0 frames not displayed - identity only)")
            else:
                for frame in frames_np:
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                    except queue.Full:
                        pass  # Drop frame if queue full

            chunk_idx += 1

        print("Pipeline thread exiting")

    def run(self):
        """Main GUI loop."""
        pygame.init()
        screen = pygame.display.set_mode(
            (self.config["window_width"], self.config["window_height"])
        )
        pygame.display.set_caption("Point-Based Subject Control Demo")
        clock = pygame.time.Clock()

        # Start pipeline thread
        pipeline_thread = threading.Thread(target=self.pipeline_thread, daemon=True)
        pipeline_thread.start()

        # Fonts
        font = pygame.font.Font(None, 24)

        print("Waiting for pipeline to initialize...")

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        mx, my = event.pos
                        if mx < self.config["width"]:  # Left panel
                            self.dragging = True
                            self.set_circle_pos(
                                mx / self.config["width"],
                                my / self.config["height"],
                            )
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        mx, my = event.pos
                        self.set_circle_pos(
                            mx / self.config["width"],
                            my / self.config["height"],
                        )
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print(f"{'Paused' if self.paused else 'Resumed'}")

            # Clear screen
            screen.fill((30, 30, 30))

            # === LEFT PANEL: Control circle ===
            left_surface = pygame.Surface((self.config["width"], self.config["height"]))
            left_surface.fill((255, 255, 255))  # White background

            x, y = self.get_circle_pos()
            px = int(x * self.config["width"])
            py = int(y * self.config["height"])
            radius = self.config["layout_radius"]

            # Draw black contour circle - exactly matches layout control sent to model
            pygame.draw.circle(left_surface, (0, 0, 0), (px, py), radius, 3)

            screen.blit(left_surface, (0, 0))

            # === RIGHT PANEL: Output frames ===
            # Advance frame at dynamic playback_fps rate (calculated from pipeline performance)
            current_time = time.time()
            current_fps = self.get_playback_fps()
            frame_interval = 1.0 / current_fps

            if (
                not self.paused
                and current_time - self.last_frame_time >= frame_interval
            ):
                # Time to show next frame - grab from queue if available
                try:
                    self.current_frame = self.frame_queue.get_nowait()
                    self.last_frame_time = current_time
                except queue.Empty:
                    # No new frame available, keep showing current
                    pass

            # Display current frame
            if self.current_frame is not None:
                # Convert RGB to pygame surface
                surface = pygame.surfarray.make_surface(
                    self.current_frame.swapaxes(0, 1)
                )
                screen.blit(surface, (self.config["width"], 0))
            else:
                # Show loading message
                if not self.pipeline_ready.is_set():
                    text = font.render("Loading pipeline...", True, (255, 255, 255))
                else:
                    text = font.render(
                        "Generating first chunk...", True, (255, 255, 255)
                    )
                screen.blit(
                    text, (self.config["width"] + 20, self.config["height"] // 2)
                )

            # Draw divider line
            pygame.draw.line(
                screen,
                (100, 100, 100),
                (self.config["width"], 0),
                (self.config["width"], self.config["height"]),
                2,
            )

            # Status text
            queue_size = self.frame_queue.qsize()
            paused_str = " | PAUSED" if self.paused else ""
            status = f"Pos: ({x:.2f}, {y:.2f}) | Queue: {queue_size} | FPS: {current_fps:.1f}{paused_str}"
            text = font.render(status, True, (200, 200, 200))
            screen.blit(text, (10, self.config["height"] - 30))

            pygame.display.flip()
            clock.tick(self.config["target_fps"])

        pygame.quit()
        print("Demo finished")


def main():
    demo = PointControlDemo()
    demo.run()


if __name__ == "__main__":
    main()

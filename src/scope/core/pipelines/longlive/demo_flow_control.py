"""
Optical Flow Control Demo - Real-time Interactive Application.

A demo showcasing optical flow-based control with LongLive + VACE.

Left panel: Flow map (paintable or from input video)
Right panel: Real-time LongLive output following the flow

Controls:
    - Left drag: Paint flow vectors (disabled when input_video is set)
    - Right click: Clear flow map
    - Space: Pause/resume
    - R: Reset flow map / restart video
    - T: Edit text prompt
    - +/-: Adjust brush size
    - Mouse wheel: Adjust flow strength

Usage:
    python -m scope.core.pipelines.longlive.demo_flow_control
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
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from torchvision.utils import flow_to_image

from scope.core.config import get_model_file_path, get_models_dir

from .pipeline import LongLivePipeline

# ============================= CONFIGURATION =============================

CONFIG = {
    # Subject image path (optional - set to None or "" to use text-only generation)
    "subject_image": None,
    # Input video for flow extraction (optional - set to None for painting mode)
    "input_video": "frontend/public/assets/test.mp4",
    # Text prompt for generation
    "prompt": "a fox sitting in the grass looking around",
    "height": 512,
    "width": 512,
    "frames_per_chunk": 12,
    "vace_context_scale": 1.5,
    "vae_type": "tae",
    "window_width": 1024,  # 512 left + 512 right
    "window_height": 512,
    "target_fps": 60,  # Display refresh rate
    # Flow painting settings (only used when input_video is None)
    "brush_radius": 40,
    "flow_strength": 15.0,  # Max flow magnitude
    "flow_decay": 0.98,  # Per-frame decay (1.0 = no decay)
    "flow_blur_sigma": 5,  # Gaussian blur for smoother flow
}

# =========================================================================


class FlowControlDemo:
    def __init__(self):
        self.config = CONFIG
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Flow map state [H, W, 2] - stores (u, v) flow vectors
        h, w = self.config["height"], self.config["width"]
        self.flow_map = np.zeros((h, w, 2), dtype=np.float32)
        self.flow_lock = threading.Lock()

        # Frame queue for display
        self.frame_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

        # Control flags
        self.running = True
        self.pipeline_ready = threading.Event()
        self.first_chunk_done = threading.Event()

        # Display state
        self.current_frame: np.ndarray | None = None
        self.last_frame_time = 0.0

        # Dynamic FPS tracking
        self.playback_fps = 30.0
        self.min_fps = 1.0
        self.max_fps = 60.0
        self.fps_lock = threading.Lock()

        # Mouse state
        self.dragging = False
        self.last_mouse_pos: tuple[int, int] | None = None

        # Brush settings
        self.brush_radius = self.config["brush_radius"]
        self.flow_strength = self.config["flow_strength"]

        # Playback control
        self.paused = False

        # Previous flow for interpolation
        self.prev_flow_map: np.ndarray | None = None

        # Text prompt (thread-safe)
        self.prompt = self.config["prompt"]
        self.prompt_lock = threading.Lock()
        self.editing_prompt = False
        self.prompt_input = ""

        # Video input mode
        self.video_mode = False
        self.video_flow_frames: list[np.ndarray] = []  # Pre-computed flow RGB frames
        self.video_frame_idx = 0
        self.video_flow_idx = 0  # For pipeline consumption
        self.video_flow_lock = threading.Lock()

        # Load input video if provided
        if self.config["input_video"]:
            self._load_input_video()

    def _load_input_video(self):
        """Load input video and pre-compute optical flow frames."""
        video_path = self.config["input_video"]
        print(f"Loading input video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            return

        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            print("ERROR: Video must have at least 2 frames")
            return

        print(f"Loaded {len(frames)} frames, computing optical flow...")

        # Initialize RAFT model for flow computation
        weights = Raft_Small_Weights.DEFAULT
        raft_model = raft_small(weights=weights).to(self.device).eval()
        transforms = weights.transforms()

        h, w = self.config["height"], self.config["width"]

        # Compute optical flow between consecutive frames
        flow_frames = []
        with torch.no_grad():
            for i in range(len(frames) - 1):
                # Prepare frames
                frame1 = frames[i]
                frame2 = frames[i + 1]

                # Resize to target size
                frame1 = cv2.resize(frame1, (w, h))
                frame2 = cv2.resize(frame2, (w, h))

                # Convert to tensors [1, 3, H, W]
                t1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0)
                t2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0)

                # Normalize to [0, 1]
                t1 = t1 / 255.0
                t2 = t2 / 255.0

                # Apply RAFT transforms
                t1, t2 = transforms(t1.to(self.device), t2.to(self.device))

                # Compute flow
                flow = raft_model(t1, t2)[-1]  # Take final flow prediction

                # Convert to RGB visualization
                flow_rgb = flow_to_image(flow[0])  # [3, H, W] uint8
                flow_rgb = flow_rgb.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
                flow_frames.append(flow_rgb)

                if (i + 1) % 10 == 0:
                    print(f"  Processed {i + 1}/{len(frames) - 1} frame pairs")

        # Duplicate first frame (VACE convention)
        self.video_flow_frames = [flow_frames[0]] + flow_frames
        self.video_mode = True
        print(f"Flow extraction complete: {len(self.video_flow_frames)} flow frames")

    def get_flow_map_copy(self) -> np.ndarray:
        with self.flow_lock:
            return self.flow_map.copy()

    def get_prompt(self) -> str:
        with self.prompt_lock:
            return self.prompt

    def set_prompt(self, prompt: str):
        with self.prompt_lock:
            self.prompt = prompt

    def update_pipeline_fps(self, num_frames: int, processing_time: float):
        if processing_time > 0:
            estimated_fps = num_frames / processing_time
            clamped_fps = max(self.min_fps, min(self.max_fps, estimated_fps))
            with self.fps_lock:
                self.playback_fps = clamped_fps

    def get_playback_fps(self) -> float:
        with self.fps_lock:
            return self.playback_fps

    def paint_flow(self, x: int, y: int, dx: float, dy: float):
        """Paint flow vectors at position (x, y) with direction (dx, dy)."""
        h, w = self.config["height"], self.config["width"]

        # Clamp position
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))

        # Normalize direction
        mag = np.sqrt(dx**2 + dy**2)
        if mag < 0.1:
            return

        # Scale flow by strength
        flow_u = (dx / mag) * min(mag, self.flow_strength)
        flow_v = (dy / mag) * min(mag, self.flow_strength)

        # Create brush mask (soft circular brush)
        y_coords, x_coords = np.ogrid[:h, :w]
        dist = np.sqrt((x_coords - x) ** 2 + (y_coords - y) ** 2)
        brush = np.clip(1.0 - dist / self.brush_radius, 0, 1)
        brush = brush**2  # Soft falloff

        with self.flow_lock:
            # Add flow weighted by brush
            self.flow_map[..., 0] += brush * flow_u
            self.flow_map[..., 1] += brush * flow_v

            # Clamp magnitude
            mag = np.sqrt(self.flow_map[..., 0] ** 2 + self.flow_map[..., 1] ** 2)
            mask = mag > self.flow_strength
            if np.any(mask):
                scale = np.where(mask, self.flow_strength / (mag + 1e-8), 1.0)
                self.flow_map[..., 0] *= scale
                self.flow_map[..., 1] *= scale

    def clear_flow(self):
        """Clear the flow map."""
        with self.flow_lock:
            self.flow_map.fill(0)

    def apply_flow_decay(self):
        """Apply decay to flow map (call each frame)."""
        decay = self.config["flow_decay"]
        if decay < 1.0:
            with self.flow_lock:
                self.flow_map *= decay

    def get_flow_rgb(self) -> np.ndarray:
        """Get current flow map as RGB visualization."""
        if self.video_mode and self.video_flow_frames:
            # Return current video flow frame
            return self.video_flow_frames[self.video_frame_idx]

        # Paint mode: compute from flow map
        flow = self.get_flow_map_copy()

        # Apply gaussian blur for smoother flow
        sigma = self.config["flow_blur_sigma"]
        if sigma > 0:
            flow[..., 0] = cv2.GaussianBlur(flow[..., 0], (0, 0), sigma)
            flow[..., 1] = cv2.GaussianBlur(flow[..., 1], (0, 0), sigma)

        # Convert to RGB using torchvision (same as VACE preprocessing)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).float()  # [2, H, W]
        rgb_tensor = flow_to_image(flow_tensor)  # [3, H, W] uint8
        rgb = rgb_tensor.permute(1, 2, 0).numpy()  # [H, W, 3]

        return rgb

    def advance_video_frame(self):
        """Advance to next video flow frame (for display)."""
        if self.video_mode and self.video_flow_frames:
            self.video_frame_idx = (self.video_frame_idx + 1) % len(
                self.video_flow_frames
            )

    def reset_video(self):
        """Reset video playback to beginning."""
        self.video_frame_idx = 0
        with self.video_flow_lock:
            self.video_flow_idx = 0

    def create_flow_chunk(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create flow control tensors for a chunk."""
        n_frames = self.config["frames_per_chunk"]
        h, w = self.config["height"], self.config["width"]

        if self.video_mode and self.video_flow_frames:
            # Video mode: get consecutive frames from pre-computed flow
            frames = []
            with self.video_flow_lock:
                for i in range(n_frames):
                    idx = (self.video_flow_idx + i) % len(self.video_flow_frames)
                    frames.append(self.video_flow_frames[idx])
                # Advance index for next chunk
                self.video_flow_idx = (self.video_flow_idx + n_frames) % len(
                    self.video_flow_frames
                )
            frames = np.stack(frames)
        else:
            # Paint mode: interpolate from previous flow map
            current_flow = self.get_flow_map_copy()

            # Apply blur
            sigma = self.config["flow_blur_sigma"]
            if sigma > 0:
                current_flow[..., 0] = cv2.GaussianBlur(
                    current_flow[..., 0], (0, 0), sigma
                )
                current_flow[..., 1] = cv2.GaussianBlur(
                    current_flow[..., 1], (0, 0), sigma
                )

            if self.prev_flow_map is None:
                self.prev_flow_map = current_flow.copy()

            # Interpolate flow maps across frames
            frames = []
            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                interp_flow = self.prev_flow_map * (1 - t) + current_flow * t

                # Convert to RGB
                flow_tensor = torch.from_numpy(interp_flow).permute(2, 0, 1).float()
                rgb_tensor = flow_to_image(flow_tensor)
                rgb = rgb_tensor.permute(1, 2, 0).numpy()
                frames.append(rgb)

            self.prev_flow_map = current_flow.copy()
            frames = np.stack(frames)

        masks = np.ones((n_frames, h, w), dtype=np.float32)

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

        # Check for optional subject image
        subject_path = None
        if self.config["subject_image"]:
            subject_path = project_root / self.config["subject_image"]
            if not subject_path.exists():
                print(f"WARNING: Subject image not found: {subject_path}")
                print("Proceeding with text-only generation...")
                subject_path = None

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
            current_prompt = self.get_prompt()

            if chunk_idx == 0:
                # First chunk: establish identity (with or without subject image)
                kwargs = {
                    "prompts": [{"text": current_prompt, "weight": 100}],
                }
                if subject_path:
                    kwargs["first_frame_image"] = str(subject_path)
                    mode = "extension"
                else:
                    mode = "text-only"
            else:
                # Subsequent chunks: flow control
                frames_t, masks_t = self.create_flow_chunk()
                kwargs = {
                    "prompts": [{"text": current_prompt, "weight": 100}],
                    "vace_context_scale": self.config["vace_context_scale"],
                    "vace_input_frames": frames_t,
                    "vace_input_masks": masks_t,
                }
                mode = "flow"

            # Generate
            output = pipeline(**kwargs)

            latency = time.time() - start_time
            num_frames = output.shape[0]
            fps = num_frames / latency
            print(
                f"Chunk {chunk_idx} ({mode}): {num_frames} frames, {latency:.2f}s, {fps:.1f} fps"
            )

            # Update playback FPS
            self.update_pipeline_fps(num_frames, latency)

            # Convert to display format and queue frames
            frames_np = output.detach().cpu().numpy()
            frames_np = np.clip(frames_np * 255, 0, 255).astype(np.uint8)

            if chunk_idx == 0:
                self.first_chunk_done.set()
                print("  (chunk 0 displayed for text-only, identity-establishing)")

            # Queue all frames for display
            for frame in frames_np:
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass

            chunk_idx += 1

        print("Pipeline thread exiting")

    def run(self):
        """Main GUI loop."""
        pygame.init()
        screen = pygame.display.set_mode(
            (self.config["window_width"], self.config["window_height"])
        )
        pygame.display.set_caption("Optical Flow Control Demo")
        clock = pygame.time.Clock()

        # Start pipeline thread
        pipeline_thread = threading.Thread(target=self.pipeline_thread, daemon=True)
        pipeline_thread.start()

        # Fonts
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 20)

        print("Waiting for pipeline to initialize...")
        if self.video_mode:
            print("VIDEO MODE: R=restart video, T=edit prompt, Space=pause")
        else:
            print(
                "PAINT MODE: Left-drag=paint flow, Right-click=clear, R=reset, T=edit prompt, +/-=brush size, Scroll=strength"
            )

        # Video playback timing
        video_frame_time = 0.0
        video_fps = 24.0  # Display video flow at this rate

        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.editing_prompt:
                        # Click outside text area cancels editing
                        self.editing_prompt = False
                    elif not self.video_mode:
                        # Paint mode only
                        if event.button == 1:  # Left click
                            mx, my = event.pos
                            if mx < self.config["width"]:  # Left panel
                                self.dragging = True
                                self.last_mouse_pos = (mx, my)
                        elif event.button == 3:  # Right click
                            self.clear_flow()
                        elif event.button == 4:  # Scroll up
                            self.flow_strength = min(50.0, self.flow_strength + 1.0)
                        elif event.button == 5:  # Scroll down
                            self.flow_strength = max(1.0, self.flow_strength - 1.0)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False
                        self.last_mouse_pos = None
                elif event.type == pygame.MOUSEMOTION:
                    if (
                        not self.video_mode
                        and self.dragging
                        and self.last_mouse_pos is not None
                    ):
                        mx, my = event.pos
                        if mx < self.config["width"]:
                            lx, ly = self.last_mouse_pos
                            dx = mx - lx
                            dy = my - ly
                            # Paint along the drag path
                            self.paint_flow(mx, my, dx * 2, dy * 2)
                            self.last_mouse_pos = (mx, my)
                elif event.type == pygame.KEYDOWN:
                    if self.editing_prompt:
                        # Text input mode
                        if event.key == pygame.K_RETURN:
                            # Confirm prompt
                            self.set_prompt(self.prompt_input)
                            self.editing_prompt = False
                            print(f"Prompt updated: {self.prompt_input}")
                        elif event.key == pygame.K_ESCAPE:
                            # Cancel editing
                            self.editing_prompt = False
                        elif event.key == pygame.K_BACKSPACE:
                            self.prompt_input = self.prompt_input[:-1]
                        else:
                            # Add character
                            if event.unicode and event.unicode.isprintable():
                                self.prompt_input += event.unicode
                    else:
                        # Normal mode
                        if event.key == pygame.K_ESCAPE:
                            self.running = False
                        elif event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                            print(f"{'Paused' if self.paused else 'Resumed'}")
                        elif event.key == pygame.K_r:
                            if self.video_mode:
                                self.reset_video()
                                print("Video restarted")
                            else:
                                self.clear_flow()
                                print("Flow map reset")
                        elif event.key == pygame.K_t:
                            # Start editing prompt
                            self.editing_prompt = True
                            self.prompt_input = self.get_prompt()
                        elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                            if not self.video_mode:
                                self.brush_radius = min(100, self.brush_radius + 5)
                        elif event.key == pygame.K_MINUS:
                            if not self.video_mode:
                                self.brush_radius = max(5, self.brush_radius - 5)

            # Apply flow decay (paint mode only)
            if not self.video_mode:
                self.apply_flow_decay()

            # Advance video frame (video mode)
            if self.video_mode and not self.paused:
                current_time = time.time()
                if current_time - video_frame_time >= 1.0 / video_fps:
                    self.advance_video_frame()
                    video_frame_time = current_time

            # Clear screen
            screen.fill((30, 30, 30))

            # === LEFT PANEL: Flow visualization ===
            flow_rgb = self.get_flow_rgb()
            flow_surface = pygame.surfarray.make_surface(flow_rgb.swapaxes(0, 1))
            screen.blit(flow_surface, (0, 0))

            # Draw brush cursor if mouse in left panel (and not editing)
            if not self.editing_prompt:
                mouse_pos = pygame.mouse.get_pos()
                if mouse_pos[0] < self.config["width"]:
                    pygame.draw.circle(
                        screen,
                        (0, 0, 0),
                        mouse_pos,
                        self.brush_radius,
                        1,
                    )

            # === RIGHT PANEL: Output frames ===
            current_time = time.time()
            current_fps = self.get_playback_fps()
            frame_interval = 1.0 / current_fps

            if (
                not self.paused
                and current_time - self.last_frame_time >= frame_interval
            ):
                try:
                    self.current_frame = self.frame_queue.get_nowait()
                    self.last_frame_time = current_time
                except queue.Empty:
                    pass

            if self.current_frame is not None:
                surface = pygame.surfarray.make_surface(
                    self.current_frame.swapaxes(0, 1)
                )
                screen.blit(surface, (self.config["width"], 0))
            else:
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

            # === PROMPT DISPLAY / EDITOR ===
            if self.editing_prompt:
                # Draw text input box
                input_rect = pygame.Rect(10, 10, self.config["width"] - 20, 30)
                pygame.draw.rect(screen, (50, 50, 50), input_rect)
                pygame.draw.rect(screen, (100, 150, 255), input_rect, 2)

                # Draw input text with cursor
                cursor = "|" if int(time.time() * 2) % 2 == 0 else ""
                input_text = font.render(
                    self.prompt_input + cursor, True, (255, 255, 255)
                )
                screen.blit(input_text, (15, 17))

                # Instructions
                edit_help = font.render(
                    "Enter=confirm | Esc=cancel", True, (100, 150, 255)
                )
                screen.blit(edit_help, (10, 45))
            else:
                # Show current prompt (truncated if too long)
                current_prompt = self.get_prompt()
                display_prompt = (
                    current_prompt[:50] + "..."
                    if len(current_prompt) > 50
                    else current_prompt
                )
                prompt_text = small_font.render(
                    f"Prompt: {display_prompt}", True, (180, 180, 180)
                )
                screen.blit(prompt_text, (10, 10))

            # Status text
            queue_size = self.frame_queue.qsize()
            paused_str = " | PAUSED" if self.paused else ""
            editing_str = " | EDITING" if self.editing_prompt else ""
            if self.video_mode:
                video_info = (
                    f"Frame: {self.video_frame_idx}/{len(self.video_flow_frames)}"
                )
                status = f"VIDEO | {video_info} | Queue: {queue_size} | FPS: {current_fps:.1f}{paused_str}{editing_str}"
            else:
                status = f"PAINT | Queue: {queue_size} | FPS: {current_fps:.1f} | Brush: {self.brush_radius} | Strength: {self.flow_strength:.1f}{paused_str}{editing_str}"
            text = font.render(status, True, (200, 200, 200))
            screen.blit(text, (10, self.config["height"] - 30))

            # Help text
            if self.video_mode:
                help_text = "R=restart | T=prompt | Space=pause"
            else:
                help_text = "Drag=paint | RClick=clear | R=reset | T=prompt | +/-=brush | Scroll=strength"
            help_surf = small_font.render(help_text, True, (150, 150, 150))
            screen.blit(help_surf, (10, self.config["height"] - 50))

            pygame.display.flip()
            clock.tick(self.config["target_fps"])

        pygame.quit()
        print("Demo finished")


def main():
    demo = FlowControlDemo()
    demo.run()


if __name__ == "__main__":
    main()

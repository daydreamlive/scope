"""Tempo-synced visual effects block.

Replaces the single-purpose BeatNoiseModulationBlock with a comprehensive
effects system supporting multiple independently toggleable effect modules:

  1. Noise Breathing  – oscillates noise_scale and kv_cache_attention_bias
  2. Prompt Cycling   – switches prompts on beat boundaries
  3. Ref Image Switch – cycles reference images on beat boundaries
  4. Denoising Modulation – varies denoising step count with the beat
  5. VACE Context Pulse   – pulses vace_context_scale on beat

All effects share a common beat-tracking core (onset detection, envelope
generation, accent patterns) and are configured via a single ``tempo_effects``
dict parameter sent from the frontend.

Design constraint:
  The pipeline runs at ~2-7 FPS.  At 120 BPM a beat is 500 ms.  We rely on
  beat_count change detection so beat onsets are never missed even when the
  pipeline call rate is close to the beat rate.
"""

import logging
import math
import random
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise breathing defaults
# ---------------------------------------------------------------------------
# In text mode noise_scale is None so we modulate denoising_step_list[0]
# directly.  The swing is in timestep units around the base value.
# At intensity=1.0 on a downbeat, the first step rises by STEP_SWING_MAX.
STEP_SWING_MAX = 200  # timestep units (e.g. 700 → 900 on full beat)
STEP_FLOOR = 200
STEP_CEILING = 950

# In video mode we nudge noise_scale (which also controls latent blend).
NS_VIDEO_SWING_MAX = 0.15
NS_VIDEO_FLOOR = 0.30
NS_VIDEO_CEILING = 0.85

# KV cache attention bias: 1.0 = full coherence, lower = more novel.
KV_BIAS_SWING_MAX = 0.50
KV_BIAS_FLOOR = 0.30

# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

def _envelope(phase: float, shape: str) -> float:
    """Return a 0-1 envelope value for the given beat *phase* (0-1)."""
    if shape == "exponential":
        return math.exp(-4.0 * phase)
    if shape == "square":
        return 1.0 if phase < 0.25 else 0.0
    # default: cosine
    return (1.0 + math.cos(phase * 2.0 * math.pi)) / 2.0


def _accent(bar_position: float, pattern: str) -> float:
    """Return an accent multiplier (0-1) based on bar position."""
    bar_beat = int(bar_position) % 4
    is_downbeat = bar_position < 1.0

    if pattern == "all_equal":
        return 1.0
    if pattern == "backbeat":
        return 1.0 if bar_beat in (1, 3) else 0.6

    # default: downbeat emphasis
    if is_downbeat:
        return 1.0
    if bar_beat == 2:
        return 0.85
    return 0.7


# ---------------------------------------------------------------------------
# Effect helpers
# ---------------------------------------------------------------------------

def _get_effect_cfg(tempo_effects: dict | None, key: str) -> dict | None:
    """Safely extract an effect sub-dict, returning None if disabled."""
    if tempo_effects is None:
        return None
    cfg = tempo_effects.get(key)
    if cfg is None or not isinstance(cfg, dict):
        return None
    if not cfg.get("enabled", False):
        return None
    return cfg


class TempoEffectsBlock(ModularPipelineBlocks):
    """Multi-effect tempo-synced modulation block.

    Reads a ``tempo_effects`` configuration dict from pipeline state and
    applies all enabled effects on each pipeline call.  When no config is
    present, falls back to the legacy noise-breathing behaviour so existing
    setups keep working.
    """

    model_name = "Wan2.1"

    def __init__(self):
        super().__init__()
        # Beat tracking
        self._last_beat_count: int | None = None
        self._modulation: float = 0.0

        # Prompt cycling state
        self._prompt_index: int = 0
        self._prompt_direction: int = 1  # for pingpong
        self._last_prompt_switch_beat: int = 0

        # Ref image cycling state
        self._ref_image_index: int = 0
        self._ref_image_direction: int = 1
        self._last_ref_switch_beat: int = 0

    @property
    def description(self) -> str:
        return "Multi-effect tempo-synced modulation for beat-locked visuals"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("bpm", type_hint=float, description="Current BPM"),
            InputParam("beat_phase", type_hint=float, description="Phase within current beat (0.0-1.0)"),
            InputParam("bar_position", type_hint=float, description="Position within bar (0 to beats_per_bar)"),
            InputParam("beat_count", type_hint=int, description="Total beat count since start"),
            InputParam("is_playing", type_hint=bool, description="Whether tempo transport is active"),
            InputParam("noise_scale", type_hint=float, default=0.7, description="Base noise scale"),
            InputParam("kv_cache_attention_bias", type_hint=float, default=1.0, description="KV cache attention bias"),
            InputParam("video", type_hint=Any, description="Video input (presence indicates video mode)"),
            InputParam("denoising_step_list", type_hint=list[int] | torch.Tensor, description="Current denoising step schedule"),
            InputParam("prompts", type_hint=Any, description="Current prompt list"),
            InputParam("vace_ref_images", type_hint=Any, description="Current reference images"),
            InputParam("first_frame_image", type_hint=Any, description="First frame reference image"),
            InputParam("vace_context_scale", type_hint=float, default=1.0, description="VACE context conditioning scale"),
            InputParam("tempo_effects", type_hint=dict, description="Effect configuration from frontend"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam("noise_scale", type_hint=float, description="Beat-modulated noise scale"),
            OutputParam("kv_cache_attention_bias", type_hint=float, description="Beat-modulated KV cache attention bias"),
            OutputParam("denoising_step_list", type_hint=list[int] | torch.Tensor, description="Beat-modulated denoising step schedule"),
            OutputParam("prompts", type_hint=Any, description="Beat-switched prompts"),
            OutputParam("vace_ref_images", type_hint=Any, description="Beat-switched reference images"),
            OutputParam("first_frame_image", type_hint=Any, description="Beat-switched first frame image"),
            OutputParam("vace_context_scale", type_hint=float, description="Beat-modulated VACE context scale"),
        ]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._last_beat_count = None
        self._modulation = 0.0
        self._prompt_index = 0
        self._prompt_direction = 1
        self._last_prompt_switch_beat = 0
        self._ref_image_index = 0
        self._ref_image_direction = 1
        self._last_ref_switch_beat = 0
        if hasattr(self, "_base_first_step"):
            del self._base_first_step

    # ------------------------------------------------------------------
    # Core beat tracking
    # ------------------------------------------------------------------

    def _update_beat(self, beat_count: int, beat_phase: float, envelope_shape: str) -> bool:
        """Update internal beat state.  Returns True on beat onset.

        At ~2-7 FPS the pipeline produces one chunk every ~1.5 s, so we
        can only sample beat state once per chunk.  A smooth envelope is
        impossible to track at this rate.  Instead we use a simple
        binary scheme: if one or more beats landed since the last chunk,
        modulation = 1.0 (strong); otherwise it decays toward 0.
        """
        beat_hit = False
        if self._last_beat_count is None:
            self._last_beat_count = beat_count
        elif beat_count != self._last_beat_count:
            beat_hit = True
            self._last_beat_count = beat_count

        if beat_hit:
            self._modulation = 1.0
        else:
            # Gentle decay so the "off-beat" chunk is clearly weaker.
            # At ~2 chunks/sec this gives roughly 1.0 → 0.4 → 0.16 ...
            self._modulation *= 0.4

        return beat_hit

    # ------------------------------------------------------------------
    # Individual effects
    # ------------------------------------------------------------------

    def _apply_noise_breathing(
        self, block_state: Any, modulation: float, is_video_mode: bool
    ) -> None:
        if is_video_mode and block_state.noise_scale is not None:
            # Video mode: nudge noise_scale (also controls latent blend).
            base_ns = float(block_state.noise_scale)
            swing = NS_VIDEO_SWING_MAX * modulation
            block_state.noise_scale = min(
                max(base_ns + swing, NS_VIDEO_FLOOR), NS_VIDEO_CEILING
            )
        else:
            # Text mode: modulate denoising_step_list[0] directly.
            # This is the lever that produces visible rhythmic change
            # without the destructive side-effects of setting noise_scale.
            self._modulate_first_step(block_state, modulation)

        # KV cache attention bias works in both modes.
        kv_swing = KV_BIAS_SWING_MAX * modulation
        block_state.kv_cache_attention_bias = max(1.0 - kv_swing, KV_BIAS_FLOOR)

    def _modulate_first_step(
        self, block_state: Any, modulation: float
    ) -> None:
        """Swing denoising_step_list[0] up on beat for visible variation."""
        step_list = block_state.denoising_step_list
        if step_list is None:
            return

        if isinstance(step_list, torch.Tensor):
            steps = step_list.tolist()
        else:
            steps = list(step_list)

        if not steps:
            return

        if not hasattr(self, "_base_first_step"):
            self._base_first_step = steps[0]

        boosted = self._base_first_step + int(STEP_SWING_MAX * modulation)
        steps[0] = max(STEP_FLOOR, min(STEP_CEILING, boosted))
        block_state.denoising_step_list = steps

    def _apply_prompt_cycling(
        self, block_state: Any, beat_count: int, beat_hit: bool, cfg: dict
    ) -> None:
        prompts_list = cfg.get("prompts")
        if not prompts_list or not isinstance(prompts_list, list) or len(prompts_list) < 2:
            return

        interval = max(1, int(cfg.get("beat_interval", 4)))
        mode = cfg.get("mode", "sequential")

        beats_since_switch = beat_count - self._last_prompt_switch_beat
        if beat_hit and beats_since_switch >= interval:
            self._last_prompt_switch_beat = beat_count
            n = len(prompts_list)

            if mode == "random":
                self._prompt_index = random.randint(0, n - 1)
            elif mode == "pingpong":
                self._prompt_index += self._prompt_direction
                if self._prompt_index >= n:
                    self._prompt_index = n - 2
                    self._prompt_direction = -1
                elif self._prompt_index < 0:
                    self._prompt_index = 1
                    self._prompt_direction = 1
            else:  # sequential
                self._prompt_index = (self._prompt_index + 1) % n

        idx = max(0, min(self._prompt_index, len(prompts_list) - 1))
        selected = prompts_list[idx]

        if isinstance(selected, str):
            block_state.prompts = [selected]
        elif isinstance(selected, list):
            block_state.prompts = selected
        else:
            block_state.prompts = [str(selected)]

    def _apply_ref_image_switching(
        self, block_state: Any, beat_count: int, beat_hit: bool, cfg: dict
    ) -> None:
        images = cfg.get("images")
        if not images or not isinstance(images, list) or len(images) < 2:
            return

        interval = max(1, int(cfg.get("beat_interval", 8)))
        mode = cfg.get("mode", "sequential")
        target = cfg.get("target", "vace_ref_images")

        beats_since_switch = beat_count - self._last_ref_switch_beat
        if beat_hit and beats_since_switch >= interval:
            self._last_ref_switch_beat = beat_count
            n = len(images)

            if mode == "random":
                self._ref_image_index = random.randint(0, n - 1)
            elif mode == "pingpong":
                self._ref_image_index += self._ref_image_direction
                if self._ref_image_index >= n:
                    self._ref_image_index = n - 2
                    self._ref_image_direction = -1
                elif self._ref_image_index < 0:
                    self._ref_image_index = 1
                    self._ref_image_direction = 1
            else:
                self._ref_image_index = (self._ref_image_index + 1) % n

        idx = max(0, min(self._ref_image_index, len(images) - 1))
        selected_image = images[idx]

        if target == "first_frame_image":
            block_state.first_frame_image = selected_image
        else:
            if isinstance(selected_image, list):
                block_state.vace_ref_images = selected_image
            else:
                block_state.vace_ref_images = [selected_image]

    def _apply_denoising_modulation(
        self, block_state: Any, modulation: float, cfg: dict
    ) -> None:
        """Additive boost on top of whatever step0 is currently set to."""
        step_list = block_state.denoising_step_list
        if step_list is None:
            return

        intensity = float(cfg.get("intensity", 0.5))
        mod = modulation * intensity

        if isinstance(step_list, torch.Tensor):
            steps = step_list.tolist()
        else:
            steps = list(step_list)

        if not steps:
            return

        current = steps[0]
        boosted = current + int(STEP_SWING_MAX * mod)
        steps[0] = max(STEP_FLOOR, min(STEP_CEILING, boosted))
        block_state.denoising_step_list = steps

    def _apply_vace_context_pulse(
        self, block_state: Any, modulation: float, cfg: dict
    ) -> None:
        min_scale = float(cfg.get("min_scale", 0.3))
        max_scale = float(cfg.get("max_scale", 1.0))
        block_state.vace_context_scale = min_scale + (max_scale - min_scale) * modulation

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        is_playing = block_state.is_playing
        if not is_playing:
            self._reset()
            self.set_block_state(state, block_state)
            return components, state

        beat_count = int(block_state.beat_count or 0)
        beat_phase = float(block_state.beat_phase or 0.0)
        bar_position = float(block_state.bar_position or 0.0)
        is_video_mode = block_state.video is not None

        tempo_effects = block_state.tempo_effects
        if tempo_effects is None or not isinstance(tempo_effects, dict):
            tempo_effects = None

        # Determine envelope shape from noise breathing config (used for beat tracking)
        noise_cfg = _get_effect_cfg(tempo_effects, "noise_breathing")
        envelope_shape = "cosine"
        if noise_cfg:
            envelope_shape = noise_cfg.get("envelope", "cosine")

        beat_hit = self._update_beat(beat_count, beat_phase, envelope_shape)

        # --- Determine if we're in legacy mode (no tempo_effects config) ---
        legacy_mode = tempo_effects is None

        # --- Noise Breathing ---
        if legacy_mode:
            # Backward-compatible: use old defaults
            accent_val = _accent(bar_position, "downbeat")
            intensity = 0.8
            modulation = min(self._modulation * intensity * accent_val, 1.0)
            self._apply_noise_breathing(block_state, modulation, is_video_mode)
        elif noise_cfg:
            intensity = float(noise_cfg.get("intensity", 0.8))
            accent_pattern = noise_cfg.get("accent", "downbeat")
            accent_val = _accent(bar_position, accent_pattern)
            modulation = min(self._modulation * intensity * accent_val, 1.0)
            self._apply_noise_breathing(block_state, modulation, is_video_mode)
        else:
            modulation = 0.0

        # --- Prompt Cycling ---
        prompt_cfg = _get_effect_cfg(tempo_effects, "prompt_cycling")
        if prompt_cfg:
            self._apply_prompt_cycling(block_state, beat_count, beat_hit, prompt_cfg)

        # --- Reference Image Switching ---
        ref_cfg = _get_effect_cfg(tempo_effects, "ref_image_switching")
        if ref_cfg:
            self._apply_ref_image_switching(block_state, beat_count, beat_hit, ref_cfg)

        # --- Denoising Step Modulation ---
        denoise_cfg = _get_effect_cfg(tempo_effects, "denoising_modulation")
        if denoise_cfg:
            envelope_shape_d = denoise_cfg.get("envelope", "cosine")
            env_d = _envelope(beat_phase, envelope_shape_d)
            accent_d = _accent(bar_position, "downbeat")
            mod_d = min(env_d * float(denoise_cfg.get("intensity", 0.5)) * accent_d, 1.0)
            if beat_hit:
                mod_d = float(denoise_cfg.get("intensity", 0.5))
            self._apply_denoising_modulation(block_state, mod_d, denoise_cfg)

        # --- VACE Context Pulse ---
        vace_cfg = _get_effect_cfg(tempo_effects, "vace_context_pulse")
        if vace_cfg:
            envelope_shape_v = vace_cfg.get("envelope", "cosine")
            env_v = _envelope(beat_phase, envelope_shape_v)
            accent_v = _accent(bar_position, "downbeat")
            mod_v = min(env_v * accent_v, 1.0)
            if beat_hit:
                mod_v = 1.0
            self._apply_vace_context_pulse(block_state, mod_v, vace_cfg)

        # --- Logging ---
        active_effects = []
        if legacy_mode or noise_cfg:
            active_effects.append("noise")
        if prompt_cfg:
            active_effects.append(f"prompt[{self._prompt_index}]")
        if ref_cfg:
            active_effects.append(f"ref_img[{self._ref_image_index}]")
        if denoise_cfg:
            active_effects.append("denoise")
        if vace_cfg:
            active_effects.append("vace")

        ns = float(block_state.noise_scale) if block_state.noise_scale is not None else -1
        kv = float(block_state.kv_cache_attention_bias) if block_state.kv_cache_attention_bias is not None else 1.0
        step0 = -1
        sl = block_state.denoising_step_list
        if sl is not None:
            if isinstance(sl, torch.Tensor):
                step0 = int(sl[0].item())
            elif isinstance(sl, list) and sl:
                step0 = int(sl[0])

        logger.info(
            "tempo_fx: beat=%d hit=%s phase=%.2f bar=%.2f mod=%.3f | "
            "step0=%d ns=%.2f kv=%.3f effects=[%s]",
            beat_count,
            beat_hit,
            beat_phase,
            bar_position,
            modulation,
            step0,
            ns,
            kv,
            ",".join(active_effects),
        )

        self.set_block_state(state, block_state)
        return components, state

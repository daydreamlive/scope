"""Beat-reactive noise modulation block.

Creates a rhythmic "breathing" effect synced to the tempo clock by oscillating
noise_scale and kv_cache_attention_bias with the beat.

How the two levers work:

  noise_scale (the primary lever):
    DenoiseBlock overrides the first denoising timestep as:
      denoising_step_list[0] = int(1000 * noise_scale) - 100
    So noise_scale=0.9 → timestep 800 (chaotic, novel), noise_scale=0.4 →
    timestep 300 (stable, coherent). In video mode it also controls the
    latent blend: latents = noise * scale + input * (1 - scale).
    This is the lever that produces the most dramatic visual change.

  kv_cache_attention_bias (secondary lever):
    Applied as log(bias) to attention scores for past-frame tokens.
    bias=1.0 → no effect, bias=0.15 → log(0.15)=-1.9 → strong suppression
    of past-frame attention, producing more novel content.

  Together: on beats, high noise_scale + low kv_bias = "inhale" (break from
  the past, chaotic). Between beats, low noise_scale + high kv_bias =
  "exhale" (return to coherence).

Design constraint:
  The pipeline runs at ~2-7 FPS. At 120 BPM a beat is 500ms. We use
  beat_count change detection to reliably detect beats even when the
  pipeline call rate is close to the beat rate.
"""

import logging
import math
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam

logger = logging.getLogger(__name__)

# noise_scale range: controls first denoising timestep via int(1000*ns)-100
# and latent blend in video mode. Must stay in [0.2, 0.95] for valid timesteps.
NS_BEAT_HIGH = 0.90  # On beat: timestep 800, very chaotic
NS_BEAT_LOW = 0.45  # Between beats: timestep 350, stable
NS_VIDEO_BEAT_HIGH = 0.88  # Video mode ceiling (also affects latent blend)
NS_VIDEO_BEAT_LOW = 0.50  # Video mode floor

# kv_cache_attention_bias range
KV_BIAS_BEAT_HIGH = 1.0  # Between beats: full coherence with past
KV_BIAS_BEAT_LOW = 0.15  # On beat: log(0.15)=-1.9, strong past suppression


class BeatNoiseModulationBlock(ModularPipelineBlocks):
    """Rhythmic "breathing" modulation synced to beat state.

    On every pipeline call, oscillates noise_scale and kv_cache_attention_bias
    based on the current beat state. noise_scale is the primary lever —
    it directly controls the first denoising timestep in both text and video
    mode, producing dramatic visual variation on beat onsets.
    """

    model_name = "Wan2.1"

    def __init__(self):
        super().__init__()
        self._last_beat_count: int | None = None
        self._modulation: float = 0.0

    @property
    def description(self) -> str:
        return "Beat-reactive breathing modulation for tempo-synced visual effects"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("bpm", type_hint=float, description="Current BPM"),
            InputParam(
                "beat_phase",
                type_hint=float,
                description="Phase within current beat (0.0-1.0)",
            ),
            InputParam(
                "bar_position",
                type_hint=float,
                description="Position within bar (0 to beats_per_bar)",
            ),
            InputParam(
                "beat_count", type_hint=int, description="Total beat count since start"
            ),
            InputParam(
                "is_playing",
                type_hint=bool,
                description="Whether tempo transport is active",
            ),
            InputParam(
                "noise_scale",
                type_hint=float,
                default=0.7,
                description="Base noise scale before beat modulation",
            ),
            InputParam(
                "noise_controller",
                type_hint=bool,
                default=True,
                description="Motion-aware noise control flag",
            ),
            InputParam(
                "kv_cache_attention_bias",
                type_hint=float,
                default=1.0,
                description="KV cache attention bias",
            ),
            InputParam(
                "video",
                type_hint=Any,
                description="Video input (presence indicates video mode)",
            ),
            InputParam(
                "denoising_step_list",
                type_hint=list[int] | torch.Tensor,
                description="Current denoising step schedule",
            ),
            InputParam(
                "beat_noise_intensity",
                type_hint=float,
                default=0.8,
                description="How aggressive the beat modulation is (0.0-1.0)",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "noise_scale",
                type_hint=float,
                description="Beat-modulated noise scale",
            ),
            OutputParam(
                "noise_controller",
                type_hint=bool,
                description="Noise controller flag (disabled during beat sync)",
            ),
            OutputParam(
                "kv_cache_attention_bias",
                type_hint=float,
                description="Beat-modulated KV cache attention bias",
            ),
            OutputParam(
                "denoising_step_list",
                type_hint=list[int] | torch.Tensor,
                description="Beat-modulated denoising step schedule",
            ),
        ]

    def _reset(self) -> None:
        self._last_beat_count = None
        self._modulation = 0.0

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        is_playing = block_state.is_playing
        if not is_playing:
            self._reset()
            self.set_block_state(state, block_state)
            return components, state

        beat_count = int(block_state.beat_count or 0)
        beat_phase = float(block_state.beat_phase or 0.0)
        bar_position = float(block_state.bar_position or 0.0)
        intensity = float(block_state.beat_noise_intensity or 0.8)
        is_video_mode = block_state.video is not None

        # --- Beat onset detection ---
        beat_hit = False
        if self._last_beat_count is None:
            self._last_beat_count = beat_count
        elif beat_count != self._last_beat_count:
            beat_hit = True
            self._last_beat_count = beat_count

        # --- Breathing envelope ---
        # Cosine curve: 1.0 at beat onset (phase=0), 0.0 at mid-beat (phase=0.5)
        cosine_envelope = (1.0 + math.cos(beat_phase * 2.0 * math.pi)) / 2.0

        if beat_hit:
            self._modulation = 1.0
        else:
            self._modulation = max(cosine_envelope, self._modulation * 0.25)

        # Downbeat emphasis
        is_downbeat = bar_position < 1.0
        bar_beat = int(bar_position) % 4
        if is_downbeat:
            accent = 1.0
        elif bar_beat == 2:
            accent = 0.85
        else:
            accent = 0.7

        modulation = min(self._modulation * intensity * accent, 1.0)

        # --- noise_scale (PRIMARY LEVER — works in both text and video mode) ---
        # This is what actually produces visible change. DenoiseBlock uses it as:
        #   denoising_step_list[0] = int(1000 * noise_scale) - 100
        # High noise_scale on beat = high first timestep = chaotic/novel output.
        # Low noise_scale between beats = low first timestep = stable/coherent.
        if is_video_mode:
            ns_high = NS_VIDEO_BEAT_HIGH
            ns_low = NS_VIDEO_BEAT_LOW
        else:
            ns_high = NS_BEAT_HIGH
            ns_low = NS_BEAT_LOW

        modulated_noise = ns_low + (ns_high - ns_low) * modulation
        block_state.noise_scale = modulated_noise
        block_state.noise_controller = False

        # --- kv_cache_attention_bias (secondary lever) ---
        modulated_bias = KV_BIAS_BEAT_HIGH - (KV_BIAS_BEAT_HIGH - KV_BIAS_BEAT_LOW) * modulation
        modulated_bias = max(modulated_bias, 0.05)
        block_state.kv_cache_attention_bias = modulated_bias

        logger.info(
            "beat_mod: beat=%d hit=%s phase=%.2f bar=%.2f mod=%.3f | "
            "noise=%.3f (ts=%d) kv_bias=%.3f video=%s",
            beat_count,
            beat_hit,
            beat_phase,
            bar_position,
            modulation,
            modulated_noise,
            int(1000 * modulated_noise) - 100,
            modulated_bias,
            is_video_mode,
        )

        self.set_block_state(state, block_state)
        return components, state

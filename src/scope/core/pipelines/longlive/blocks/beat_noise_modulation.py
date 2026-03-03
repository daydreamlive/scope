"""Beat-reactive noise modulation block.

Modulates noise_scale and denoising steps based on tempo sync beat state,
creating VJ-style pulsing visuals synced to music.
"""

import math
from typing import Any

import torch
from diffusers.modular_pipelines import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam


class BeatNoiseModulationBlock(ModularPipelineBlocks):
    """Modulates generation parameters based on beat/tempo state.

    When tempo sync is active (is_playing=True), this block overrides:
    - noise_scale: Spikes on beat onsets, decays between beats. Downbeats hit harder.
    - noise_controller: Forced off so beat modulation takes precedence over motion-aware control.
    - denoising_step_list: In text mode (no video input), adds/removes steps on beats
      for a grittier-on-beat / cleaner-off-beat effect.
    - kv_cache_attention_bias: Briefly weakened on strong beats for more visual novelty.

    The modulation curve uses an exponential decay from beat onset (beat_phase=0)
    with configurable intensity. Bar downbeats (bar_position near 0) get a stronger hit.
    """

    model_name = "Wan2.1"

    @property
    def description(self) -> str:
        return "Beat-reactive noise modulation for tempo-synced visual effects"

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

    @torch.no_grad()
    def __call__(self, components, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = self.get_block_state(state)

        is_playing = block_state.is_playing
        if not is_playing:
            self.set_block_state(state, block_state)
            return components, state

        beat_phase = float(block_state.beat_phase or 0.0)
        bar_position = float(block_state.bar_position or 0.0)
        intensity = float(block_state.beat_noise_intensity or 0.8)
        base_noise_scale = float(block_state.noise_scale or 0.7)
        is_video_mode = block_state.video is not None

        # --- Beat envelope ---
        # Exponential decay from beat onset: strongest at phase=0, fades to ~0 by phase=1
        # decay_rate controls how fast the hit fades (higher = sharper transient)
        decay_rate = 6.0
        beat_envelope = math.exp(-decay_rate * beat_phase)

        # Downbeat emphasis: bar_position near 0 (or near beats_per_bar) gets a boost
        beat_in_bar = bar_position % 1.0
        is_downbeat = bar_position < 1.0
        downbeat_boost = 1.3 if is_downbeat else 1.0

        # Accent on beats 1 and 3 (backbeat)
        bar_beat = int(bar_position) % 4
        accent = 1.15 if bar_beat in (0, 2) else 1.0

        # Combined modulation factor: 0 (no beat effect) to ~1.3 (strong downbeat hit)
        modulation = beat_envelope * intensity * downbeat_boost * accent
        modulation = min(modulation, 1.0)

        if is_video_mode:
            # Video mode: modulate noise_scale
            # On beat: push noise_scale toward 0.95 (aggressive reimagining)
            # Off beat: let it settle to a lower base (more input preservation)
            noise_floor = max(0.2, base_noise_scale * 0.4)
            noise_ceiling = 0.95
            modulated_noise = noise_floor + (noise_ceiling - noise_floor) * modulation

            block_state.noise_scale = modulated_noise
            block_state.noise_controller = False
        else:
            # Text mode: modulate denoising steps for quality variation
            # On beat: fewer steps = rawer, grittier generation
            # Off beat: more steps = cleaner, more refined
            base_steps = block_state.denoising_step_list
            if base_steps is not None:
                if isinstance(base_steps, torch.Tensor):
                    steps_list = base_steps.tolist()
                else:
                    steps_list = list(base_steps)

                if len(steps_list) >= 2 and modulation > 0.3:
                    # On strong beats, drop to fewer denoising steps
                    # This gives a rawer, more chaotic look
                    reduced = steps_list[: max(2, len(steps_list) - 1)]
                    block_state.denoising_step_list = reduced

        # KV cache attention bias: weaken on strong beats for more visual novelty
        # Normal: 1.0 (full reliance on cached context)
        # On beat: drops toward 0.5 (less temporal coherence = more surprise)
        base_bias = float(block_state.kv_cache_attention_bias or 1.0)
        bias_floor = 0.5
        modulated_bias = base_bias - (base_bias - bias_floor) * modulation * 0.6
        block_state.kv_cache_attention_bias = modulated_bias

        self.set_block_state(state, block_state)
        return components, state

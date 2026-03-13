"""Schema for Modulation Oscilloscope pipeline."""

from ..base_schema import BasePipelineConfig, ModeDefaults


class ModScopeConfig(BasePipelineConfig):
    """Configuration for the Modulation Oscilloscope pipeline.

    Renders real-time oscilloscope traces for modulatable parameters
    (noise_scale, vace_context_scale, kv_cache_attention_bias).
    Use with Tempo Sync + Modulation to visualize wave shapes.
    No model downloads required.
    """

    pipeline_id = "mod-scope"
    pipeline_name = "Modulation Scope"
    pipeline_description = (
        "Oscilloscope-style visualizer for beat-synced parameter modulation. "
        "Shows rolling waveform traces, level meters, and beat state. "
        "No model downloads required."
    )

    supports_prompts = False

    noise_scale: float = 0.5
    vace_context_scale: float = 1.0
    kv_cache_attention_bias: float = 0.3

    modes = {"text": ModeDefaults(default=True)}

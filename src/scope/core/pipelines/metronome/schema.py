from pydantic import Field

from ..base_schema import BasePipelineConfig, ModeDefaults, UsageType, ui_field_config


class MetronomeConfig(BasePipelineConfig):
    """Configuration for the Metronome test pipeline.

    Renders a visual metronome for testing beat-sync latency compensation.
    Displays current beat/bar position and provides toggle controls whose
    colors mix additively so parameter scheduling can be verified visually.
    """

    pipeline_id = "metronome"
    pipeline_name = "Metronome"
    pipeline_description = (
        "Visual metronome for testing beat-sync parameter scheduling. "
        "Shows beat/bar position with toggleable color overlays and "
        "adjustable artificial latency."
    )
    supports_prompts = False
    modified = True
    usage = [UsageType.PREPROCESSOR]

    modes = {"video": ModeDefaults(default=True)}

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=2000.0,
        description="Artificial pipeline latency in milliseconds (simulates slow model)",
        json_schema_extra=ui_field_config(
            order=1,
            label="Artificial Latency (ms)",
        ),
    )

    layer_a: bool = Field(
        default=False,
        description="Toggle magenta color overlay",
        json_schema_extra=ui_field_config(
            order=2,
            label="Layer A (Magenta)",
        ),
    )

    layer_b: bool = Field(
        default=False,
        description="Toggle cyan color overlay",
        json_schema_extra=ui_field_config(
            order=3,
            label="Layer B (Cyan)",
        ),
    )

    layer_c: bool = Field(
        default=False,
        description="Toggle gold color overlay",
        json_schema_extra=ui_field_config(
            order=4,
            label="Layer C (Gold)",
        ),
    )

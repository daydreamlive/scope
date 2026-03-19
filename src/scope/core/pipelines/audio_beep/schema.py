from pydantic import Field

from ..base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class AudioBeepConfig(BasePipelineConfig):
    """Configuration for the Audio Beep pipeline."""

    pipeline_id = "audio-beep"
    pipeline_name = "Audio Beep"
    pipeline_description = (
        "Audio-only pipeline that generates periodic beep tones. "
        "Useful for testing audio streaming without a GPU."
    )

    produces_video = False
    produces_audio = True
    supports_prompts = False

    modes = {"text": ModeDefaults(default=True)}

    frequency: float = Field(
        default=440.0,
        ge=20.0,
        le=20000.0,
        description="Beep frequency in Hz",
        json_schema_extra=ui_field_config(order=1, label="Frequency (Hz)"),
    )
    beep_duration: float = Field(
        default=0.1,
        ge=0.01,
        le=2.0,
        description="Duration of each beep in seconds",
        json_schema_extra=ui_field_config(order=2, label="Beep Duration (s)"),
    )
    beep_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Time between beep starts in seconds",
        json_schema_extra=ui_field_config(order=3, label="Beep Interval (s)"),
    )
    volume: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Beep volume (0.0 to 1.0)",
        json_schema_extra=ui_field_config(order=4, label="Volume"),
    )

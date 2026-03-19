from pydantic import Field

from ..base_schema import (
    BasePipelineConfig,
    ModeDefaults,
    ui_field_config,
)


class AudioVideoTestConfig(BasePipelineConfig):
    """Configuration for the Audio Video Test pipeline."""

    pipeline_id = "audio-video-test"
    pipeline_name = "Audio Video Test"
    pipeline_description = (
        "Generates periodic beep tones with a flashing video frame. "
        "Useful for testing audio+video streaming and observing A/V sync."
    )

    supports_prompts = False
    produces_audio = True

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
        description="Duration of each beep/flash in seconds",
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

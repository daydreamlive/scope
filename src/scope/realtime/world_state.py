"""
WorldState - Domain-agnostic representation of what's happening.

WorldState captures the "truth" of the scene without any style/LoRA knowledge.
It's the input to prompt compilation, not the output.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class BeatType(str, Enum):
    """Narrative beat types that affect pacing and framing."""

    SETUP = "setup"
    ESCALATION = "escalation"
    CLIMAX = "climax"
    PAYOFF = "payoff"
    RESET = "reset"
    TRANSITION = "transition"


class CameraIntent(str, Enum):
    """Abstract camera intentions (style-agnostic)."""

    ESTABLISHING = "establishing"
    CLOSE_UP = "close_up"
    MEDIUM = "medium"
    WIDE = "wide"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    TRACKING = "tracking"
    STATIC = "static"


class CharacterState(BaseModel):
    """Internal state of a character."""

    name: str
    emotion: str = "neutral"
    action: str = "idle"
    intensity: float = Field(default=0.5, ge=0.0, le=1.0)

    # What the character knows/believes (for future narrative logic)
    knowledge: dict[str, Any] = Field(default_factory=dict)

    # Relationships to other characters
    relationships: dict[str, str] = Field(default_factory=dict)


class PropState(BaseModel):
    """State of a prop in the scene."""

    name: str
    location: str = ""
    visible: bool = True
    material: str | None = None
    state: str = "normal"  # e.g., "broken", "glowing", "wet"


class WorldState(BaseModel):
    """
    Complete state of the world at a moment in time.

    This is style-agnostic - it describes WHAT is happening,
    not HOW it should be rendered. The StyleManifest + PromptCompiler
    translate this into style-specific prompt tokens.
    """

    # Scene context
    scene_description: str = ""
    location: str = ""
    time_of_day: str = ""
    weather: str = ""

    # Narrative state
    beat: BeatType = BeatType.SETUP
    tension: float = Field(default=0.5, ge=0.0, le=1.0)
    pacing: float = Field(default=0.5, ge=0.0, le=1.0)  # 0=slow, 1=frantic

    # Camera
    camera: CameraIntent = CameraIntent.MEDIUM
    focus_target: str = ""  # What/who the camera focuses on

    # Characters
    characters: list[CharacterState] = Field(default_factory=list)

    # Props
    props: list[PropState] = Field(default_factory=list)

    # Current action/event description
    action: str = ""

    # Abstract mood/atmosphere (0-1 scales)
    mood: dict[str, float] = Field(default_factory=dict)
    # e.g., {"comedy": 0.8, "tension": 0.2, "warmth": 0.6}

    # Free-form tags for additional context
    tags: list[str] = Field(default_factory=list)

    # Custom fields for domain-specific data
    custom: dict[str, Any] = Field(default_factory=dict)

    def get_character(self, name: str) -> CharacterState | None:
        """Get a character by name."""
        for char in self.characters:
            if char.name == name:
                return char
        return None

    def get_prop(self, name: str) -> PropState | None:
        """Get a prop by name."""
        for prop in self.props:
            if prop.name == name:
                return prop
        return None

    def is_empty(self) -> bool:
        """Check if this WorldState has no meaningful content.

        An empty WorldState means no scene has been described - just defaults.
        In performance mode, style switches should preserve the current prompt
        rather than recompiling an empty WorldState to just trigger words.
        """
        return (
            not self.scene_description
            and not self.action
            and not self.characters
            and not self.focus_target
            and not self.location
            and not self.tags
            and not self.mood
            and not self.custom
        )

    def get_mood(self, key: str, default: float = 0.5) -> float:
        """Get a mood value."""
        return self.mood.get(key, default)

    def set_mood(self, key: str, value: float) -> None:
        """Set a mood value (clamped to 0-1)."""
        self.mood[key] = max(0.0, min(1.0, value))

    def to_context_dict(self) -> dict[str, Any]:
        """
        Convert to a flat dictionary suitable for LLM context.

        This is what gets passed to the PromptCompiler/LLM.
        """
        context = {
            "scene": self.scene_description,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "weather": self.weather,
            "beat": self.beat.value,
            "tension": self.tension,
            "pacing": self.pacing,
            "camera": self.camera.value,
            "focus": self.focus_target,
            "action": self.action,
            "tags": self.tags,
        }

        # Add characters
        for i, char in enumerate(self.characters):
            prefix = f"char_{i}"
            context[f"{prefix}_name"] = char.name
            context[f"{prefix}_emotion"] = char.emotion
            context[f"{prefix}_action"] = char.action
            context[f"{prefix}_intensity"] = char.intensity

        # Add props
        for i, prop in enumerate(self.props):
            prefix = f"prop_{i}"
            context[f"{prefix}_name"] = prop.name
            context[f"{prefix}_location"] = prop.location
            context[f"{prefix}_visible"] = prop.visible
            context[f"{prefix}_state"] = prop.state
            if prop.material:
                context[f"{prefix}_material"] = prop.material

        # Add mood values
        for key, value in self.mood.items():
            context[f"mood_{key}"] = value

        # Add custom fields
        context.update(self.custom)

        return context


# Convenience factory functions


def create_simple_world(
    action: str,
    emotion: str = "neutral",
    camera: CameraIntent = CameraIntent.MEDIUM,
    tension: float = 0.5,
) -> WorldState:
    """Create a simple WorldState with just the basics."""
    return WorldState(
        action=action,
        camera=camera,
        tension=tension,
        characters=[CharacterState(name="subject", emotion=emotion, action=action)],
    )


def create_character_scene(
    character_name: str,
    emotion: str,
    action: str,
    location: str = "",
    beat: BeatType = BeatType.SETUP,
) -> WorldState:
    """Create a WorldState focused on a single character."""
    return WorldState(
        location=location,
        beat=beat,
        action=action,
        focus_target=character_name,
        characters=[
            CharacterState(
                name=character_name,
                emotion=emotion,
                action=action,
            )
        ],
    )

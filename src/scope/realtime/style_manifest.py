"""
StyleManifest - LoRA-specific vocabulary and metadata.

A StyleManifest captures everything needed to translate abstract world concepts
into effective prompt tokens for a specific LoRA/style.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Style Directory Resolution
# =============================================================================


def get_style_dirs() -> list[Path]:
    """
    Return style directories in precedence order (later wins on name conflicts).

    Priority:
    1. SCOPE_STYLES_DIRS env var (os.pathsep-separated: ":" on Linux, ";" on Windows)
    2. Default: ./styles (repo built-ins), ~/.daydream-scope/styles (user overrides)
    """
    if custom := os.environ.get("SCOPE_STYLES_DIRS"):
        return [Path(p).expanduser().resolve() for p in custom.split(os.pathsep) if p]

    return [
        Path("styles").resolve(),  # repo/dev built-ins
        Path.home() / ".daydream-scope" / "styles",  # user overrides
    ]


# =============================================================================
# LoRA Path Canonicalization
# =============================================================================


def get_lora_dir() -> Path:
    """Get the canonical LoRA directory path."""
    from scope.server.models_config import get_models_dir

    return get_models_dir() / "lora"


def canonicalize_lora_path(raw: str | None) -> str | None:
    """
    Canonicalize a LoRA path for consistent matching.

    Resolution rules:
    - None/empty → None
    - Already absolute → resolve and return
    - Bare filename (no /) → resolve under models/lora
    - Relative path with / → resolve under models/lora

    This ensures the same canonical string is used for:
    - Pipeline preload: loras=[{"path": <canonical>, ...}]
    - Runtime updates: lora_scales=[{"path": <canonical>, ...}]

    Args:
        raw: Raw path from manifest (e.g., "rat_21_step5500.safetensors")

    Returns:
        Canonical absolute path string, or None if input was empty
    """
    if not raw:
        return None

    p = Path(raw).expanduser()

    # Already absolute
    if p.is_absolute():
        return str(p.resolve())

    # Relative path: resolve under lora directory
    lora_dir = get_lora_dir()
    return str((lora_dir / p).resolve())


class StyleManifest(BaseModel):
    """
    Metadata and vocabulary for a specific LoRA/style.

    The vocab dictionaries map abstract concepts to effective prompt tokens
    discovered through experimentation with this specific LoRA.
    """

    # Identity
    name: str
    description: str = ""

    # LoRA configuration
    lora_path: str | None = None
    lora_default_scale: float = 0.85

    # Trigger words (always included in prompt)
    trigger_words: list[str] = Field(default_factory=list)

    # Vocabulary mappings: abstract concept → effective tokens
    # These are populated from your prompt experiments
    material_vocab: dict[str, str] = Field(default_factory=dict)
    motion_vocab: dict[str, str] = Field(default_factory=dict)
    camera_vocab: dict[str, str] = Field(default_factory=dict)
    lighting_vocab: dict[str, str] = Field(default_factory=dict)
    emotion_vocab: dict[str, str] = Field(default_factory=dict)
    beat_vocab: dict[str, str] = Field(default_factory=dict)

    # Custom vocab categories (extensible)
    custom_vocab: dict[str, dict[str, str]] = Field(default_factory=dict)

    # Prompt constraints
    default_negative: str = ""
    max_prompt_tokens: int = 77

    # Priority order for token budget allocation
    # Earlier items are kept when truncating
    priority_order: list[str] = Field(
        default_factory=lambda: [
            "trigger",
            "action",
            "material",
            "camera",
            "mood",
        ]
    )

    # Path to instruction sheet (markdown/text with LLM instructions)
    instruction_sheet_path: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    def get_vocab(self, category: str, key: str, default: str | None = None) -> str:
        """
        Look up a vocab term by category and key.

        Args:
            category: Vocab category (material, motion, camera, etc.)
            key: The abstract term to look up
            default: Fallback if not found

        Returns:
            The effective prompt tokens, or default/key if not found
        """
        vocab_dict = getattr(self, f"{category}_vocab", None)
        if vocab_dict is None:
            vocab_dict = self.custom_vocab.get(category, {})

        result = vocab_dict.get(key)
        if result is not None:
            return result

        # Check for "default" key in vocab
        if "default" in vocab_dict:
            return vocab_dict["default"]

        return default if default is not None else key

    def get_all_vocab(self) -> dict[str, dict[str, str]]:
        """Return all vocab dictionaries merged."""
        all_vocab = {
            "material": self.material_vocab,
            "motion": self.motion_vocab,
            "camera": self.camera_vocab,
            "lighting": self.lighting_vocab,
            "emotion": self.emotion_vocab,
            "beat": self.beat_vocab,
        }
        all_vocab.update(self.custom_vocab)
        return {k: v for k, v in all_vocab.items() if v}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StyleManifest":
        """Load a StyleManifest from a YAML file."""
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save this StyleManifest to a YAML file."""
        path = Path(path)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)


class StyleRegistry:
    """
    Registry for loading and caching StyleManifests.

    Manifests can be loaded from:
    - Individual YAML files
    - A directory of manifests
    - Programmatic registration
    """

    def __init__(self):
        self._manifests: dict[str, StyleManifest] = {}
        self._default_style: str | None = None

    def register(self, manifest: StyleManifest) -> None:
        """Register a manifest by name."""
        self._manifests[manifest.name] = manifest
        if self._default_style is None:
            self._default_style = manifest.name

    def load_from_file(self, path: str | Path) -> StyleManifest:
        """Load and register a manifest from a YAML file."""
        manifest = StyleManifest.from_yaml(path)
        self.register(manifest)
        return manifest

    def load_from_directory(self, directory: str | Path) -> list[StyleManifest]:
        """
        Load all manifest.yaml files from a directory tree.

        Manifests are loaded in sorted path order for deterministic behavior.
        """
        directory = Path(directory)
        if not directory.exists():
            return []

        manifests = []
        # Sort for deterministic ordering
        for manifest_path in sorted(directory.rglob("manifest.yaml")):
            try:
                manifest = self.load_from_file(manifest_path)
                manifests.append(manifest)
            except Exception as e:
                # Log but don't fail on individual manifest errors
                logger.warning(f"Failed to load {manifest_path}: {e}")
        return manifests

    def load_from_style_dirs(self) -> list[StyleManifest]:
        """
        Load styles from all configured style directories.

        Later directories win on name conflicts (user overrides repo).
        """
        all_manifests = []
        for style_dir in get_style_dirs():
            manifests = self.load_from_directory(style_dir)
            all_manifests.extend(manifests)
            if manifests:
                logger.info(f"Loaded {len(manifests)} styles from {style_dir}")
        return all_manifests

    def get_all_lora_paths(self, skip_missing: bool = True) -> list[str]:
        """
        Get canonical LoRA paths for all registered styles.

        Args:
            skip_missing: If True, skip LoRAs that don't exist on disk (with warning)

        Returns:
            List of unique canonical LoRA paths
        """
        seen: dict[str, str] = {}  # canonical_path -> style_name (for logging)

        for style_name in self.list_styles():
            manifest = self.get(style_name)
            if not manifest or not manifest.lora_path:
                continue

            canonical = canonicalize_lora_path(manifest.lora_path)
            if not canonical:
                continue

            # Check if file exists
            if skip_missing and not Path(canonical).exists():
                logger.warning(
                    f"Style '{style_name}': LoRA not found at {canonical}, skipping"
                )
                continue

            if canonical in seen:
                logger.debug(
                    f"Style '{style_name}' shares LoRA with '{seen[canonical]}': {canonical}"
                )
            else:
                seen[canonical] = style_name

        return list(seen.keys())

    def build_lora_scales_for_style(
        self, active_style: str | None
    ) -> list[dict[str, Any]]:
        """
        Build lora_scales list for a style switch.

        Sets the active style's LoRA to its default scale, all others to 0.0.
        Uses canonical paths for consistent matching with preloaded LoRAs.

        Args:
            active_style: Name of the style to activate (None = all at 0.0)

        Returns:
            List of {"path": <canonical>, "scale": <float>} dicts
        """
        scales_by_path: dict[str, float] = {}

        # First, set all known style LoRA paths to 0.0 (deduped by canonical path).
        for style_name in self.list_styles():
            manifest = self.get(style_name)
            if not manifest or not manifest.lora_path:
                continue

            canonical = canonicalize_lora_path(manifest.lora_path)
            if not canonical:
                continue

            scales_by_path.setdefault(canonical, 0.0)

        # Then, ensure the active style wins even if multiple styles share a path.
        if active_style:
            manifest = self.get(active_style)
            if manifest and manifest.lora_path:
                canonical = canonicalize_lora_path(manifest.lora_path)
                if canonical:
                    scales_by_path[canonical] = manifest.lora_default_scale

        return [{"path": p, "scale": s} for p, s in scales_by_path.items()]

    def get(self, name: str) -> StyleManifest | None:
        """Get a manifest by name."""
        return self._manifests.get(name)

    def get_trigger_word(self, style_name: str) -> str | None:
        """Get the primary trigger word for a style.

        Args:
            style_name: Name of the style

        Returns:
            First trigger word, or None if style not found or has no triggers
        """
        manifest = self.get(style_name)
        if manifest and manifest.trigger_words:
            return manifest.trigger_words[0]
        return None

    def get_default(self) -> StyleManifest | None:
        """Get the default style manifest."""
        if self._default_style:
            return self._manifests.get(self._default_style)
        return None

    def set_default(self, name: str) -> None:
        """Set the default style by name."""
        if name not in self._manifests:
            raise ValueError(f"Style '{name}' not found in registry")
        self._default_style = name

    def list_styles(self) -> list[str]:
        """List all registered style names."""
        return list(self._manifests.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._manifests

    def __len__(self) -> int:
        return len(self._manifests)

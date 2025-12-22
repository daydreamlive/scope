"""Utility functions for PersonaLive pipeline."""

from pathlib import Path

from omegaconf import OmegaConf


def load_model_config(config, pipeline_file_path: str | Path) -> OmegaConf:
    """Load model configuration from config or auto-load from model.yaml.

    Args:
        config: Configuration object that may contain a model_config attribute
        pipeline_file_path: Path to the pipeline's __file__ (used to locate model.yaml)

    Returns:
        OmegaConf: The model configuration, either from config or loaded from model.yaml
    """
    model_config = getattr(config, "model_config", None)
    if not model_config:
        model_yaml_path = Path(pipeline_file_path).parent / "model.yaml"
        model_config = OmegaConf.load(model_yaml_path)
    return model_config

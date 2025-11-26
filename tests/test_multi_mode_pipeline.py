"""Tests for MultiModePipeline base class and self-configuring blocks.

These tests validate the new declarative pipeline architecture.
"""

import pytest
import torch
from diffusers.modular_pipelines import (
    AutoPipelineBlocks,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)

from scope.core.pipelines.components import ComponentsManager
from scope.core.pipelines.defaults import GENERATION_MODE_TEXT, GENERATION_MODE_VIDEO
from scope.core.pipelines.helpers import build_pipeline_schema
from scope.core.pipelines.interface import Requirements
from scope.core.pipelines.multi_mode import MultiModePipeline
from scope.core.pipelines.multi_mode_blocks import (
    ConfigureForModeBlock,
    LoadComponentsBlock,
)


class MockGenerator:
    """Mock generator for testing."""

    def get_scheduler(self):
        return "mock_scheduler"


class MockTextEncoder:
    """Mock text encoder for testing."""

    pass


class MockVAE:
    """Mock VAE for testing."""

    def __init__(self, strategy: str, **kwargs):
        self.strategy = strategy

    def to(self, device=None, dtype=None):
        return self


class MockOutputBlock(ModularPipelineBlocks):
    """Mock block that produces output_video for testing."""

    def __call__(self, components, state):
        state.set("output_video", torch.zeros(1, 3, 4, 4))
        return components, state


class MockTextWorkflow(SequentialPipelineBlocks):
    """Mock text workflow for testing."""

    block_classes = [
        ConfigureForModeBlock,
        LoadComponentsBlock,
        MockOutputBlock,
    ]
    block_names = ["configure", "load_components", "output"]


class MockVideoWorkflow(SequentialPipelineBlocks):
    """Mock video workflow for testing."""

    block_classes = [
        ConfigureForModeBlock,
        LoadComponentsBlock,
        MockOutputBlock,
    ]
    block_names = ["configure", "load_components", "output"]


class MockAutoBlocks(AutoPipelineBlocks):
    """Mock AutoPipelineBlocks for testing."""

    block_classes = [MockVideoWorkflow, MockTextWorkflow]
    block_names = ["video_mode", "text_mode"]
    block_trigger_inputs = ["video", None]


class MockConfig:
    """Mock config for testing."""

    def __init__(self):
        self.height = 512
        self.width = 512
        self.seed = 42


class MockPipeline(MultiModePipeline):
    """Test pipeline implementation."""

    @classmethod
    def get_schema(cls):
        return build_pipeline_schema(
            pipeline_id="test-pipeline",
            name="Test Pipeline",
            description="Test pipeline for unit tests",
            native_mode=GENERATION_MODE_TEXT,
            shared={"manage_cache": True, "base_seed": 42},
            text_overrides={
                "denoising_steps": [1000, 750],
                "resolution": {"height": 512, "width": 512},
            },
            video_overrides={
                "denoising_steps": [750],
                "resolution": {"height": 256, "width": 256},
                "input_size": 4,
                "vae_strategy": "video_vae",
            },
        )

    @classmethod
    def get_blocks(cls):
        return MockAutoBlocks()

    @classmethod
    def get_components(cls):
        return {
            "generator": MockGenerator,
            "text_encoder": MockTextEncoder,
            "vae": {
                "text": {"strategy": "text_vae"},
                "video": {"strategy": "video_vae"},
            },
        }

    @classmethod
    def get_defaults(cls):
        return {
            "text": {
                "denoising_steps": [1000, 750],
                "resolution": {"height": 512, "width": 512},
            },
            "video": {
                "denoising_steps": [750],
                "resolution": {"height": 256, "width": 256},
                "input_size": 4,
            },
        }


class TestMultiModePipeline:
    """Test suite for MultiModePipeline base class."""

    def test_initialization(self):
        """Test pipeline initializes correctly."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert pipeline.components is not None
        assert pipeline.blocks is not None
        assert pipeline.state is not None
        assert pipeline.first_call is True

    def test_prepare_text_mode(self):
        """Test prepare() returns None for text mode."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        requirements = pipeline.prepare(generation_mode="text")
        assert requirements is None

    def test_prepare_video_mode(self):
        """Test prepare() returns Requirements for video mode."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        requirements = pipeline.prepare(generation_mode="video")
        assert requirements is not None
        assert isinstance(requirements, Requirements)
        assert requirements.input_size == 4

    def test_infer_mode_from_kwargs(self):
        """Test mode inference from kwargs."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Test explicit mode
        mode = pipeline._infer_mode_from_kwargs({"generation_mode": "video"})
        assert mode == GENERATION_MODE_VIDEO

        # Test default mode (native)
        mode = pipeline._infer_mode_from_kwargs({})
        assert mode == GENERATION_MODE_TEXT

    def test_declarative_methods(self):
        """Test all declarative methods return expected types."""
        assert MockPipeline.get_schema() is not None
        assert isinstance(MockPipeline.get_schema(), dict)

        assert MockPipeline.get_blocks() is not None
        assert isinstance(MockPipeline.get_blocks(), AutoPipelineBlocks)

        assert MockPipeline.get_components() is not None
        assert isinstance(MockPipeline.get_components(), dict)

        assert MockPipeline.get_defaults() is not None
        assert isinstance(MockPipeline.get_defaults(), dict)

    def test_components_config_includes_pipeline_class(self):
        """Test components config includes pipeline_class for LoadComponentsBlock."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert "pipeline_class" in pipeline.components.config
        assert pipeline.components.config["pipeline_class"] == MockPipeline

    def test_components_config_includes_vae_init_kwargs(self):
        """Test components config includes vae_init_kwargs."""
        config = MockConfig()
        generator = MockGenerator()
        text_encoder = MockTextEncoder()
        model_config = {"base_model_name": "test"}
        vae_kwargs = {"model_dir": "/test/path"}

        pipeline = MockPipeline(
            config=config,
            generator=generator,
            text_encoder=text_encoder,
            model_config=model_config,
            device=torch.device("cpu"),
            dtype=torch.float32,
            vae_init_kwargs=vae_kwargs,
        )

        assert "vae_init_kwargs" in pipeline.components.config
        assert pipeline.components.config["vae_init_kwargs"] == vae_kwargs


class TestConfigureForModeBlock:
    """Test suite for ConfigureForModeBlock."""

    def test_configure_with_detected_mode(self):
        """Test block with _detected_mode in state."""
        block = ConfigureForModeBlock()
        components = ComponentsManager({"device": "cpu"})
        state = PipelineState()
        state.set("_detected_mode", "video")

        components_out, state_out = block(components, state)

        assert components_out is components
        assert state_out is state
        assert state.get("_detected_mode") == "video"

    def test_configure_without_detected_mode(self):
        """Test block without _detected_mode sets default."""
        block = ConfigureForModeBlock()
        components = ComponentsManager({"device": "cpu"})
        state = PipelineState()

        components_out, state_out = block(components, state)

        assert components_out is components
        assert state_out is state
        assert state.get("_detected_mode") == "text"


class TestLoadComponentsBlock:
    """Test suite for LoadComponentsBlock."""

    def test_load_components_without_pipeline_class(self):
        """Test block handles missing pipeline_class gracefully."""
        block = LoadComponentsBlock()
        components = ComponentsManager({"device": "cpu"})
        state = PipelineState()
        state.set("_detected_mode", "video")

        components_out, state_out = block(components, state)

        assert components_out is components
        assert state_out is state

    def test_load_components_with_mode_specific_vae(self):
        """Test block loads mode-specific VAE."""
        import scope.core.pipelines.base.vae as vae_module

        def mock_create_vae(strategy, pipeline_name, **kwargs):
            return MockVAE(strategy=strategy)

        original_create_vae = getattr(vae_module, "create_vae", None)

        try:
            # Monkey patch
            vae_module.create_vae = mock_create_vae

            block = LoadComponentsBlock()
            components = ComponentsManager(
                {
                    "device": torch.device("cpu"),
                    "dtype": torch.float32,
                    "pipeline_class": MockPipeline,
                    "pipeline_name": "test-pipeline",
                    "vae_init_kwargs": {},
                }
            )
            state = PipelineState()
            state.set("_detected_mode", "video")

            components_out, state_out = block(components, state)

            assert components_out is components
            assert state_out is state
            assert "vae" in components_out._components
            vae = components_out._components["vae"]
            assert isinstance(vae, MockVAE)
            assert vae.strategy == "video_vae"

        finally:
            # Restore original
            if original_create_vae is not None:
                vae_module.create_vae = original_create_vae

    def test_vae_caching(self):
        """Test VAE is cached and reused."""
        import scope.core.pipelines.base.vae as vae_module

        call_count = 0

        def mock_create_vae(strategy, pipeline_name, **kwargs):
            nonlocal call_count
            call_count += 1
            return MockVAE(strategy=strategy)

        original_create_vae = getattr(vae_module, "create_vae", None)

        try:
            vae_module.create_vae = mock_create_vae

            block = LoadComponentsBlock()
            components = ComponentsManager(
                {
                    "device": torch.device("cpu"),
                    "dtype": torch.float32,
                    "pipeline_class": MockPipeline,
                    "pipeline_name": "test-pipeline",
                    "vae_init_kwargs": {},
                }
            )
            state = PipelineState()
            state.set("_detected_mode", "video")

            # First call
            block(components, state)
            assert call_count == 1

            # Second call should use cache
            block(components, state)
            assert call_count == 1  # Should not increment

        finally:
            if original_create_vae:
                vae_module.create_vae = original_create_vae


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

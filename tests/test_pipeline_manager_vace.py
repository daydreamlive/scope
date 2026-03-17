"""Tests for VACE configuration in PipelineManager."""

from unittest.mock import MagicMock, patch


class TestConfigureVace:
    """Tests for _configure_vace helper."""

    def _make_manager(self):
        from scope.server.pipeline_manager import PipelineManager

        return PipelineManager()

    def test_sets_vace_context_scale_from_load_params(self):
        """vace_context_scale should be extracted from load_params."""
        manager = self._make_manager()
        config = {}
        load_params = {"vace_context_scale": 0.5}

        with patch.object(manager, "_get_vace_checkpoint_path", return_value="/fake"):
            manager._configure_vace(config, load_params)

        assert config["vace_context_scale"] == 0.5

    def test_vace_context_scale_defaults_to_1(self):
        """vace_context_scale should default to 1.0 when load_params has other keys."""
        manager = self._make_manager()
        config = {}
        load_params = {"some_other_param": True}

        with patch.object(manager, "_get_vace_checkpoint_path", return_value="/fake"):
            manager._configure_vace(config, load_params)

        assert config["vace_context_scale"] == 1.0

    def test_sets_ref_images_from_load_params(self):
        """ref_images should be extracted from load_params when present."""
        manager = self._make_manager()
        config = {}
        load_params = {"ref_images": ["img1.png", "img2.png"]}

        with patch.object(manager, "_get_vace_checkpoint_path", return_value="/fake"):
            manager._configure_vace(config, load_params)

        assert config["ref_images"] == ["img1.png", "img2.png"]

    def test_no_ref_images_when_empty(self):
        """ref_images should not be set when load_params has empty list."""
        manager = self._make_manager()
        config = {}
        load_params = {"ref_images": []}

        with patch.object(manager, "_get_vace_checkpoint_path", return_value="/fake"):
            manager._configure_vace(config, load_params)

        assert "ref_images" not in config

    def test_no_load_params(self):
        """With no load_params, only vace_path should be set."""
        manager = self._make_manager()
        config = {}

        with patch.object(manager, "_get_vace_checkpoint_path", return_value="/fake"):
            manager._configure_vace(config, None)

        assert config["vace_path"] == "/fake"
        assert "vace_context_scale" not in config
        assert "ref_images" not in config


class TestKreaVaceConfig:
    """Tests that the Krea pipeline branch propagates VACE params like _configure_vace."""

    def _load_krea_config(self, load_params):
        """Run the Krea branch of _load_pipeline_implementation and return the config.

        Mocks all heavy dependencies (model loading, torch, pipeline constructor)
        so only the config-building logic executes.
        """
        from scope.server.pipeline_manager import PipelineManager

        manager = PipelineManager()

        mock_pipeline_instance = MagicMock()
        captured_config = {}

        def capture_config(config, **kwargs):
            captured_config.update(config)
            return mock_pipeline_instance

        with (
            patch(
                "scope.core.pipelines.registry.PipelineRegistry.get",
                return_value=None,
            ),
            patch("scope.server.pipeline_manager.torch") as mock_torch,
            patch(
                "scope.server.models_config.get_model_file_path",
                side_effect=lambda p: f"/models/{p}",
            ),
            patch(
                "scope.server.models_config.get_models_dir",
                return_value="/models",
            ),
            patch(
                "scope.core.pipelines.KreaRealtimeVideoPipeline",
                side_effect=capture_config,
            ),
        ):
            mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 4090"
            mock_torch.device.return_value = "cuda"
            mock_torch.bfloat16 = "bfloat16"

            manager._load_pipeline_implementation(
                "krea-realtime-video",
                load_params=load_params,
            )

        return captured_config

    def test_vace_context_scale_propagated(self):
        """Krea branch should propagate vace_context_scale from load_params."""
        config = self._load_krea_config({"vace_context_scale": 0.5})
        assert config["vace_context_scale"] == 0.5

    def test_vace_context_scale_defaults_to_1(self):
        """Krea branch should default vace_context_scale to 1.0."""
        config = self._load_krea_config({"some_other_param": True})
        assert config["vace_context_scale"] == 1.0

    def test_ref_images_propagated(self):
        """Krea branch should propagate ref_images from load_params."""
        config = self._load_krea_config({"ref_images": ["img.png"]})
        assert config["ref_images"] == ["img.png"]

    def test_ref_images_not_set_when_empty(self):
        """Krea branch should not set ref_images when list is empty."""
        config = self._load_krea_config({"ref_images": []})
        assert "ref_images" not in config

    def test_uses_14b_vace_checkpoint(self):
        """Krea branch should use the 14B VACE checkpoint, not the default 1.3B."""
        config = self._load_krea_config({"some_other_param": True})
        assert "14B" in config["vace_path"]
        assert "1_3B" not in config["vace_path"]

    def test_vace_disabled(self):
        """Krea branch should not set VACE config when vace_enabled is False."""
        config = self._load_krea_config({"vace_enabled": False})
        assert "vace_path" not in config
        assert "vace_context_scale" not in config

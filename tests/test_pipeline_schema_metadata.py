"""Tests for frames_per_chunk and temporal_downsample_factor in pipeline schema metadata."""

from scope.core.pipelines.base_schema import BasePipelineConfig


class TestSchemaMetadataFrameChunk:
    """Tests for frames_per_chunk and temporal_downsample_factor in get_schema_with_metadata()."""

    def test_base_config_returns_none_for_frame_chunk_fields(self):
        """BasePipelineConfig should return None for both fields by default."""
        metadata = BasePipelineConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] is None
        assert metadata["temporal_downsample_factor"] is None

    def test_custom_config_computes_frames_per_chunk(self):
        """Config with num_frame_per_block and vae_temporal_downsample_factor should compute correctly."""

        class TestConfig(BasePipelineConfig):
            pipeline_id = "test-frames"
            num_frame_per_block = 3
            vae_temporal_downsample_factor = 4

        metadata = TestConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 12
        assert metadata["temporal_downsample_factor"] == 4

    def test_partial_config_returns_none(self):
        """Config with only one of the two fields set should return None for both."""

        class OnlyBlockConfig(BasePipelineConfig):
            pipeline_id = "test-partial-block"
            num_frame_per_block = 3

        metadata = OnlyBlockConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] is None
        assert metadata["temporal_downsample_factor"] is None

        class OnlyVaeConfig(BasePipelineConfig):
            pipeline_id = "test-partial-vae"
            vae_temporal_downsample_factor = 4

        metadata = OnlyVaeConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] is None
        assert metadata["temporal_downsample_factor"] is None

    def test_longlive_config_values(self):
        """LongLiveConfig should have frames_per_chunk=12, temporal_downsample_factor=4."""
        from scope.core.pipelines.longlive.schema import LongLiveConfig

        metadata = LongLiveConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 12
        assert metadata["temporal_downsample_factor"] == 4

    def test_streamdiffusionv2_config_values(self):
        """StreamDiffusionV2Config should have frames_per_chunk=4, temporal_downsample_factor=4."""
        from scope.core.pipelines.streamdiffusionv2.schema import (
            StreamDiffusionV2Config,
        )

        metadata = StreamDiffusionV2Config.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 4
        assert metadata["temporal_downsample_factor"] == 4

    def test_krea_realtime_video_config_values(self):
        """KreaRealtimeVideoConfig should have frames_per_chunk=12, temporal_downsample_factor=4."""
        from scope.core.pipelines.krea_realtime_video.schema import (
            KreaRealtimeVideoConfig,
        )

        metadata = KreaRealtimeVideoConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 12
        assert metadata["temporal_downsample_factor"] == 4

    def test_reward_forcing_config_values(self):
        """RewardForcingConfig should have frames_per_chunk=12, temporal_downsample_factor=4."""
        from scope.core.pipelines.reward_forcing.schema import RewardForcingConfig

        metadata = RewardForcingConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 12
        assert metadata["temporal_downsample_factor"] == 4

    def test_memflow_config_values(self):
        """MemFlowConfig should have frames_per_chunk=12, temporal_downsample_factor=4."""
        from scope.core.pipelines.memflow.schema import MemFlowConfig

        metadata = MemFlowConfig.get_schema_with_metadata()
        assert metadata["frames_per_chunk"] == 12
        assert metadata["temporal_downsample_factor"] == 4

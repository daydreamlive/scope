"""Pipelines package."""


def __getattr__(name):
    """Lazy import for pipeline and config classes to avoid triggering heavy imports."""
    # Pipeline classes
    if name == "LongLivePipeline":
        from .longlive.pipeline import LongLivePipeline

        return LongLivePipeline
    elif name == "KreaRealtimeVideoPipeline":
        from .krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

        return KreaRealtimeVideoPipeline
    elif name == "RewardForcingPipeline":
        from .reward_forcing.pipeline import RewardForcingPipeline

        return RewardForcingPipeline
    elif name == "StreamDiffusionV2Pipeline":
        from .streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        return StreamDiffusionV2Pipeline
    elif name == "PassthroughPipeline":
        from .passthrough.pipeline import PassthroughPipeline

        return PassthroughPipeline
    elif name == "MemFlowPipeline":
        from .memflow.pipeline import MemFlowPipeline

        return MemFlowPipeline
    elif name == "VideoDepthAnythingPipeline":
        from .video_depth_anything.pipeline import VideoDepthAnythingPipeline

        return VideoDepthAnythingPipeline
    elif name == "ControllerVisualizerPipeline":
        from .controller_viz.pipeline import ControllerVisualizerPipeline

        return ControllerVisualizerPipeline
    elif name == "RIFEPipeline":
        from .rife.pipeline import RIFEPipeline

        return RIFEPipeline
    # Config classes
    elif name == "BasePipelineConfig":
        from .base_schema import BasePipelineConfig

        return BasePipelineConfig
    elif name == "LongLiveConfig":
        from .longlive.schema import LongLiveConfig

        return LongLiveConfig
    elif name == "StreamDiffusionV2Config":
        from .streamdiffusionv2.schema import StreamDiffusionV2Config

        return StreamDiffusionV2Config
    elif name == "KreaRealtimeVideoConfig":
        from .krea_realtime_video.schema import KreaRealtimeVideoConfig

        return KreaRealtimeVideoConfig
    elif name == "PassthroughConfig":
        from .passthrough.schema import PassthroughConfig

        return PassthroughConfig
    elif name == "RewardForcingConfig":
        from .reward_forcing.schema import RewardForcingConfig

        return RewardForcingConfig
    elif name == "MemFlowConfig":
        from .memflow.schema import MemFlowConfig

        return MemFlowConfig
    elif name == "VideoDepthAnythingConfig":
        from .video_depth_anything.schema import VideoDepthAnythingConfig

        return VideoDepthAnythingConfig
    elif name == "RIFEConfig":
        from .rife.schema import RIFEConfig

        return RIFEConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Pipeline classes
    "LongLivePipeline",
    "KreaRealtimeVideoPipeline",
    "RewardForcingPipeline",
    "StreamDiffusionV2Pipeline",
    "PassthroughPipeline",
    "MemFlowPipeline",
    "VideoDepthAnythingPipeline",
    "ControllerVisualizerPipeline",
    "RIFEPipeline",
    # Config classes
    "BasePipelineConfig",
    "LongLiveConfig",
    "StreamDiffusionV2Config",
    "KreaRealtimeVideoConfig",
    "PassthroughConfig",
    "RewardForcingConfig",
    "MemFlowConfig",
    "VideoDepthAnythingConfig",
    "RIFEConfig",
]

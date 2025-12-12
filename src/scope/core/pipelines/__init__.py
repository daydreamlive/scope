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
    elif name == "DecartApiPipeline":
        from .decart_api.pipeline import DecartApiPipeline

        return DecartApiPipeline
    # Config classes
    elif name == "BasePipelineConfig":
        from .schema import BasePipelineConfig

        return BasePipelineConfig
    elif name == "LongLiveConfig":
        from .schema import LongLiveConfig

        return LongLiveConfig
    elif name == "StreamDiffusionV2Config":
        from .schema import StreamDiffusionV2Config

        return StreamDiffusionV2Config
    elif name == "KreaRealtimeVideoConfig":
        from .schema import KreaRealtimeVideoConfig

        return KreaRealtimeVideoConfig
    elif name == "PassthroughConfig":
        from .schema import PassthroughConfig

        return PassthroughConfig
    elif name == "DecartApiConfig":
        from .schema import DecartApiConfig

        return DecartApiConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Pipeline classes
    "LongLivePipeline",
    "KreaRealtimeVideoPipeline",
    "RewardForcingPipeline",
    "StreamDiffusionV2Pipeline",
    "PassthroughPipeline",
    "DecartApiPipeline",
    # Config classes
    "BasePipelineConfig",
    "LongLiveConfig",
    "StreamDiffusionV2Config",
    "KreaRealtimeVideoConfig",
    "PassthroughConfig",
    "DecartApiConfig",
]

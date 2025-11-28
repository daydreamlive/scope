"""Pipelines package."""


def __getattr__(name):
    """Lazy import for pipeline classes to avoid triggering heavy imports."""
    if name == "LongLivePipeline":
        from .longlive.pipeline import LongLivePipeline

        return LongLivePipeline
    elif name == "KreaRealtimeVideoPipeline":
        from .krea_realtime_video.pipeline import KreaRealtimeVideoPipeline

        return KreaRealtimeVideoPipeline
    elif name == "StreamDiffusionV2Pipeline":
        from .streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline

        return StreamDiffusionV2Pipeline
    elif name == "PassthroughPipeline":
        from .passthrough.pipeline import PassthroughPipeline

        return PassthroughPipeline
    elif name == "MultiModePipeline":
        from .multi_mode import MultiModePipeline

        return MultiModePipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LongLivePipeline",
    "KreaRealtimeVideoPipeline",
    "StreamDiffusionV2Pipeline",
    "PassthroughPipeline",
    "MultiModePipeline",
]

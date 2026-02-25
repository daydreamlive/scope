"""Moondream vision language model plugin for Daydream Scope."""

from scope.core.plugins.hookspecs import hookimpl


@hookimpl
def register_pipelines(register):
    """Register the Moondream pipeline with Scope."""
    from .pipeline import MoondreamPipeline

    register(MoondreamPipeline)

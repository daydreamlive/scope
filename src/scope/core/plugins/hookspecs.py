"""Hook specifications for the Scope plugin system."""

import pluggy

hookspec = pluggy.HookspecMarker("scope")
hookimpl = pluggy.HookimplMarker("scope")


class ScopeHookSpec:
    """Hook specifications for Scope plugins."""

    @hookspec
    def register_pipelines(self, register):
        """Register custom pipeline implementations.

        Args:
            register: Callback to register pipeline classes.
                     Usage: register(PipelineClass)

        Example:
            @scope.core.hookimpl
            def register_pipelines(register):
                register(MyPipeline)
        """

    @hookspec
    def register_input_sources(self, register):
        """Register custom input source implementations.

        Input sources provide video frames to Scope from external sources
        like NDI, capture cards, RTMP streams, etc.

        Args:
            register: Callback to register input source classes.
                     Usage: register(InputSourceClass)

        Example:
            @scope.core.hookimpl
            def register_input_sources(register):
                from .input import NDIInputSource
                register(NDIInputSource)
        """

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
            @hookimpl
            def register_pipelines(self, register):
                register(MyPipeline)
        """

    @hookspec
    def register_event_processors(self, register):
        """Register event processors for discrete async operations.

        Event processors handle triggered operations (as opposed to continuous
        frame processing). They run asynchronously and don't block pipelines.

        Args:
            register: Callback to register processors.
                     Usage: register(name: str, processor: EventProcessor)

        Example:
            @hookimpl
            def register_event_processors(self, register):
                register("prompt_enhancer", MyPromptEnhancer())
        """

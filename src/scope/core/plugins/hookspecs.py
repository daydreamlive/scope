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

        Note:
            Preprocessors are pipelines that implement the Pipeline interface
            and can be used to preprocess video input. They are registered
            through the same hook as regular pipelines.

        Example:
            @scope.core.hookimpl
            def register_pipelines(register):
                register(MyPipeline)
                register(MyPreprocessor)  # Preprocessors are also pipelines
        """

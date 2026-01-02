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
                     Usage: register(PipelineClass) or register(PipelineClass, also_preprocessor=True)

        Note:
            Preprocessors are pipelines that implement the Pipeline interface
            and can be used to preprocess video input. They are registered
            through the same hook as regular pipelines. To make a pipeline
            also available as a preprocessor, pass also_preprocessor=True.

        Example:
            @scope.core.hookimpl
            def register_pipelines(register):
                register(MyPipeline)  # Regular pipeline
                register(MyPreprocessor, also_preprocessor=True)  # Pipeline that's also a preprocessor
        """

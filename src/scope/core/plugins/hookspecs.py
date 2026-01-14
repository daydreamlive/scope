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
            def register_pipelines(register):
                register(MyPipeline)
        """

    @hookspec
    def register_preprocessors(self, register):
        """Register video preprocessors for VACE conditioning.

        Args:
            register: Callback to register preprocessor classes.
                     Usage: register(id, name, PreprocessorClass)

        Example:
            @hookimpl
            def register_preprocessors(register):
                register("pose", "Pose", PoseTensorRTPreprocessor)
        """

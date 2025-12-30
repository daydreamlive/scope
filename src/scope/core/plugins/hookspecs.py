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

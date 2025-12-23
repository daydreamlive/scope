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
    def register_artifacts(self, register):
        """Register model artifacts for download.

        Allows plugins to declare which model files need to be downloaded
        for their pipelines. These artifacts will be downloaded when the
        user requests model download for the pipeline.

        Args:
            register: Callback to register artifacts.
                     Usage: register(pipeline_id, [Artifact, ...])

        Example:
            from scope.server.artifacts import HuggingfaceRepoArtifact

            @scope.core.hookimpl
            def register_artifacts(register):
                register("my-pipeline", [
                    HuggingfaceRepoArtifact(
                        repo_id="user/model-repo",
                        files=["model.safetensors"],
                    ),
                ])
        """

    @hookspec
    def register_routes(self, app):
        """Register custom API routes with the FastAPI application.

        Allows plugins to add custom HTTP endpoints for pipeline-specific
        functionality (e.g., uploading reference images, custom configuration).

        Args:
            app: FastAPI application instance

        Example:
            from fastapi import HTTPException

            @scope.core.hookimpl
            def register_routes(app):
                @app.post("/api/v1/my-pipeline/custom-endpoint")
                async def my_custom_endpoint():
                    return {"status": "ok"}
        """

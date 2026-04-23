"""Verification scenarios for the Node/Pipeline unification.

`Pipeline` used to be a distinct subclass of `BaseNode`. After the
unification it is just an alias for :class:`Node`, and a node is
treated as a "pipeline" (config-driven, chunked-style) purely by
whether ``get_config_class()`` returns a non-None config class.

The tests below cover the user-visible invariants of that
unification:

1. Name aliases: ``Pipeline is Node`` and ``BaseNode is Node``.
2. Existing pipeline subclasses still register, resolve, and produce
   valid ``NodeDefinition`` projections from their config class.
3. Plain event-style nodes (no config class) are not surfaced by the
   pipeline registry even though they share storage with pipelines.
4. Shared ``NodeRegistry`` storage: pipelines and plain nodes live in
   the same dict and can be enumerated together.
5. A minimal plain node (only ``node_type_id`` + static
   ``get_definition`` + ``execute``) registers and executes.
6. A minimal config-driven node (only ``get_config_class`` +
   ``__call__``) registers, projects into a ``NodeDefinition``, and
   has its ``pipeline_meta.config_schema`` populated.
"""

from typing import ClassVar

# NOTE: PipelineRegistry is imported lazily inside each test function that
# needs it. Importing it at module load time triggers the registry's
# side-effecting initializer, which imports every built-in pipeline
# (LongLive → diffusers, etc.). That breaks freezegun-based tests in
# ``test_logs_config`` and cascades into metaclass conflicts in every
# downstream test that imports ``scope.server.app``. Same rationale as
# ``test_pipeline_registry.py``.
from scope.core.nodes import (
    BaseNode,
    Node,
    NodeDefinition,
    NodePort,
    NodeRegistry,
    Requirements,
)
from scope.core.pipelines.base_schema import BasePipelineConfig
from scope.core.pipelines.interface import Pipeline
from scope.core.pipelines.interface import Requirements as RequirementsFromInterface


class TestAliases:
    """Scenario 1 — name aliases survive the unification."""

    def test_pipeline_is_node(self):
        assert Pipeline is Node

    def test_basenode_is_node(self):
        assert BaseNode is Node

    def test_requirements_re_export(self):
        assert RequirementsFromInterface is Requirements


class TestExistingPipelinesStillWork:
    """Scenario 2 — every registered pipeline projects a valid definition."""

    def test_all_pipelines_subclass_node(self):
        from scope.core.pipelines.registry import PipelineRegistry

        for pid in PipelineRegistry.list_pipelines():
            cls = PipelineRegistry.get(pid)
            assert cls is not None, f"PipelineRegistry.get({pid!r}) returned None"
            assert issubclass(cls, Node), f"{pid}: {cls} is not a Node subclass"

    def test_all_pipelines_have_matching_config_id(self):
        from scope.core.pipelines.registry import PipelineRegistry

        for pid in PipelineRegistry.list_pipelines():
            cfg = PipelineRegistry.get_config_class(pid)
            assert cfg is not None, f"{pid}: config class missing"
            assert cfg.pipeline_id == pid, (
                f"{pid}: config.pipeline_id={cfg.pipeline_id!r} mismatch"
            )

    def test_every_pipeline_projects_definition(self):
        from scope.core.pipelines.registry import PipelineRegistry

        for pid in PipelineRegistry.list_pipelines():
            cls = PipelineRegistry.get(pid)
            defn = cls.get_definition()
            assert isinstance(defn, NodeDefinition)
            assert defn.node_type_id == pid
            assert defn.pipeline_meta is not None
            assert "config_schema" in defn.pipeline_meta, (
                f"{pid}: pipeline_meta is missing config_schema"
            )


class TestPlainNodesNotPipelines:
    """Scenario 3 — plain event-style nodes are not in the pipeline registry."""

    def test_scheduler_registered_as_node(self):
        assert NodeRegistry.is_registered("scheduler")

    def test_scheduler_not_a_pipeline(self):
        from scope.core.pipelines.registry import PipelineRegistry

        assert not PipelineRegistry.is_registered("scheduler")

    def test_scheduler_has_no_config_class(self):
        scheduler_cls = NodeRegistry.get("scheduler")
        assert scheduler_cls is not None
        assert scheduler_cls.get_config_class() is None


class TestSharedRegistryStorage:
    """Scenario 4 — pipelines and plain nodes share ``NodeRegistry`` storage."""

    def test_pipelines_visible_in_node_registry(self):
        from scope.core.pipelines.registry import PipelineRegistry

        all_nodes = set(NodeRegistry.list_node_types())
        pipelines = set(PipelineRegistry.list_pipelines())
        missing = pipelines - all_nodes
        assert not missing, (
            f"Pipelines {missing!r} missing from NodeRegistry — storage is not unified."
        )

    def test_plain_nodes_not_in_pipeline_registry(self):
        from scope.core.pipelines.registry import PipelineRegistry

        all_nodes = set(NodeRegistry.list_node_types())
        pipelines = set(PipelineRegistry.list_pipelines())
        # Scheduler is the canonical plain node; it must not leak into
        # the pipeline view.
        assert "scheduler" in all_nodes
        assert "scheduler" not in pipelines


class TestMinimalPlainNode:
    """Scenario 5 — a minimal event-style Node subclass works end-to-end."""

    def test_register_and_execute(self):
        from scope.core.pipelines.registry import PipelineRegistry

        class _DoubleNode(Node):
            node_type_id: ClassVar[str] = "_test-double-node"

            @classmethod
            def get_definition(cls) -> NodeDefinition:
                return NodeDefinition(
                    node_type_id=cls.node_type_id,
                    display_name="Double",
                    inputs=[NodePort(name="x", port_type="number")],
                    outputs=[NodePort(name="y", port_type="number")],
                )

            def execute(self, inputs, **kwargs):
                return {"y": inputs["x"] * 2}

        try:
            NodeRegistry.register(_DoubleNode)

            # Ends up in the node registry, not the pipeline registry.
            assert NodeRegistry.is_registered("_test-double-node")
            assert not PipelineRegistry.is_registered("_test-double-node")

            # Execution contract works.
            node = _DoubleNode(node_id="d1")
            assert node.execute({"x": 21}) == {"y": 42}

            # Definition projects correctly.
            defn = _DoubleNode.get_definition()
            assert defn.node_type_id == "_test-double-node"
            assert defn.pipeline_meta is None
        finally:
            NodeRegistry.unregister("_test-double-node")


class TestMinimalConfigDrivenNode:
    """Scenario 6 — a minimal config-driven ("pipeline") Node works end-to-end."""

    def test_register_execute_and_project_definition(self):
        from scope.core.pipelines.registry import PipelineRegistry

        class _MiniConfig(BasePipelineConfig):
            pipeline_id: ClassVar[str] = "_test-mini-pipeline"
            pipeline_name: ClassVar[str] = "Mini"
            pipeline_description: ClassVar[str] = "Minimal config-driven node"
            inputs: ClassVar[list[str]] = ["video"]
            outputs: ClassVar[list[str]] = ["video"]

        class _MiniPipeline(Node):
            @classmethod
            def get_config_class(cls):
                return _MiniConfig

            def prepare(self, **kwargs):
                return Requirements(input_size=1)

            def __call__(self, **kwargs):
                return {"video": kwargs.get("video")}

        try:
            PipelineRegistry.register("_test-mini-pipeline", _MiniPipeline)

            # Registered under both views of the storage.
            assert PipelineRegistry.is_registered("_test-mini-pipeline")
            assert NodeRegistry.is_registered("_test-mini-pipeline")

            # Config class resolves through the pipeline-registry API.
            assert (
                PipelineRegistry.get_config_class("_test-mini-pipeline") is _MiniConfig
            )

            # Definition is derived from the config class — no duplicate
            # declaration of ports, name, etc.
            defn = _MiniPipeline.get_definition()
            assert defn.node_type_id == "_test-mini-pipeline"
            assert defn.display_name == "Mini"
            assert [p.name for p in defn.inputs] == ["video"]
            assert [p.name for p in defn.outputs] == ["video"]
            assert defn.pipeline_meta is not None
            assert "config_schema" in defn.pipeline_meta

            # Chunked execution contract works.
            out = _MiniPipeline()(video=[1, 2, 3])
            assert out == {"video": [1, 2, 3]}

            # Prepare returns a Requirements instance.
            req = _MiniPipeline().prepare()
            assert isinstance(req, Requirements)
            assert req.input_size == 1
        finally:
            PipelineRegistry.unregister("_test-mini-pipeline")


class TestRegistrationIdDerivation:
    """Scenario 7 — registration derives the id from either source."""

    def test_derives_from_node_type_id(self):
        class _Plain(Node):
            node_type_id: ClassVar[str] = "_test-plain-derive"

            @classmethod
            def get_definition(cls):
                return NodeDefinition(
                    node_type_id=cls.node_type_id, display_name="Plain"
                )

            def execute(self, inputs, **kwargs):
                return {}

        try:
            NodeRegistry.register(_Plain)
            assert NodeRegistry.get("_test-plain-derive") is _Plain
        finally:
            NodeRegistry.unregister("_test-plain-derive")

    def test_derives_from_config_pipeline_id(self):
        class _Config(BasePipelineConfig):
            pipeline_id: ClassVar[str] = "_test-config-derive"

        class _FromConfig(Node):
            @classmethod
            def get_config_class(cls):
                return _Config

            def __call__(self, **kwargs):
                return {}

        try:
            NodeRegistry.register(_FromConfig)
            assert NodeRegistry.get("_test-config-derive") is _FromConfig
        finally:
            NodeRegistry.unregister("_test-config-derive")

    def test_rejects_class_without_id(self):
        import pytest

        class _Orphan(Node):
            def execute(self, inputs, **kwargs):
                return {}

        with pytest.raises(ValueError, match="Cannot determine node_type_id"):
            NodeRegistry.register(_Orphan)


class TestArtifacts:
    """Scenario 8 — any Node can declare artifacts, not just pipelines."""

    def test_plain_node_defaults_to_empty_artifacts(self):
        class _Plain(Node):
            node_type_id: ClassVar[str] = "_test-plain-no-artifacts"

            @classmethod
            def get_definition(cls):
                return NodeDefinition(
                    node_type_id=cls.node_type_id, display_name="Plain"
                )

            def execute(self, inputs, **kwargs):
                return {}

        assert _Plain.get_artifacts() == []

    def test_plain_node_can_override_artifacts(self):
        from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact

        weights = HuggingfaceRepoArtifact(repo_id="foo/bar", files=["model.bin"])

        class _Heavy(Node):
            node_type_id: ClassVar[str] = "_test-heavy-node"

            @classmethod
            def get_definition(cls):
                return NodeDefinition(
                    node_type_id=cls.node_type_id, display_name="Heavy"
                )

            @classmethod
            def get_artifacts(cls):
                return [weights]

            def execute(self, inputs, **kwargs):
                return {}

        assert _Heavy.get_artifacts() == [weights]

    def test_config_driven_node_delegates_to_config_artifacts(self):
        from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact

        weights = HuggingfaceRepoArtifact(repo_id="foo/config-bar", files=["model.bin"])

        class _Config(BasePipelineConfig):
            pipeline_id: ClassVar[str] = "_test-artifacts-config"
            artifacts: ClassVar[list] = [weights]

        class _ConfigDriven(Node):
            @classmethod
            def get_config_class(cls):
                return _Config

            def __call__(self, **kwargs):
                return {}

        assert _ConfigDriven.get_artifacts() == [weights]

    def test_base_node_has_no_artifacts(self):
        assert Node.get_artifacts() == []

    def test_artifact_registry_resolves_plain_node(self):
        """The server-side artifact_registry should find a plain node's weights."""
        from scope.core.pipelines.artifacts import HuggingfaceRepoArtifact
        from scope.server.artifact_registry import get_artifacts_for_pipeline

        weights = HuggingfaceRepoArtifact(
            repo_id="foo/integration", files=["model.bin"]
        )

        class _PlainWithArtifacts(Node):
            node_type_id: ClassVar[str] = "_test-plain-with-artifacts"

            @classmethod
            def get_definition(cls):
                return NodeDefinition(
                    node_type_id=cls.node_type_id, display_name="Plain+weights"
                )

            @classmethod
            def get_artifacts(cls):
                return [weights]

            def execute(self, inputs, **kwargs):
                return {}

        try:
            NodeRegistry.register(_PlainWithArtifacts)
            assert get_artifacts_for_pipeline("_test-plain-with-artifacts") == [weights]
        finally:
            NodeRegistry.unregister("_test-plain-with-artifacts")

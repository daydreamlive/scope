"""Unified node abstraction for the Scope graph.

Everything on the graph is a :class:`Node`. A Node declares its input
and output ports, optional editable parameters, optional artifacts,
and optional lifecycle hooks. There are two invocation styles:

- Event-style (``execute``): scheduled by :class:`NodeProcessor` — a
  call per tick with the current input port values. Used by plain
  utility nodes (scheduler, math, etc.).
- Chunked-style (``__call__``): scheduled by
  :class:`PipelineProcessor` — driven by buffered video frames via
  queues, driven by a ``prepare()`` that returns how many frames to
  batch together. Used by video pipelines.

The same class hierarchy backs both: ``Pipeline`` is just an alias for
``Node``. A node is treated as "config-driven" (i.e. what we used to
call a pipeline) iff ``get_config_class()`` returns a non-None
:class:`BasePipelineConfig`, which drives parameter-panel rendering,
JSON-schema generation, and chunked execution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from scope.core.pipelines.artifacts import Artifact
    from scope.core.pipelines.base_schema import BasePipelineConfig

logger = logging.getLogger(__name__)


class NodePort(BaseModel):
    """Describes an input or output port on a node."""

    name: str = Field(..., description="Port identifier (used in edge wiring)")
    port_type: str = Field(
        ...,
        description=(
            "Type of data carried by this port. Built-in types: "
            "'audio', 'video', 'number', 'string', 'boolean', 'trigger'. "
            "Plugins may define custom types (e.g. 'latent', 'model')."
        ),
    )
    required: bool = Field(default=True, description="Whether this input is required")
    description: str = Field(default="", description="Human-readable description")
    default_value: Any = Field(default=None, description="Default value for inputs")


class NodeParam(BaseModel):
    """Describes an editable parameter (widget) on a node.

    Parameters are user-configurable values that live on the node card.
    Like ComfyUI widgets, a parameter may be overridden by connecting
    an incoming wire to the corresponding input port — the widget then
    becomes an input and the default value is ignored.

    Widget-specific hints (number min/max/step, select options, etc.)
    go into the free-form ``ui`` dict so the base schema doesn't grow
    as new widget kinds are added. The frontend renderer dispatches on
    ``param_type`` and reads whichever ``ui`` keys apply.
    """

    name: str = Field(..., description="Parameter identifier")
    param_type: Literal["number", "string", "boolean", "select"] = Field(
        ..., description="Widget type for the frontend"
    )
    default: Any = Field(default=None, description="Default value")
    description: str = Field(default="", description="Human-readable label")
    ui: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Widget-specific hints consumed by the frontend renderer. "
            "Number widgets read ``min``/``max``/``step``; select "
            "widgets read ``options``; plugin-defined widget kinds may "
            "use any keys they like."
        ),
    )
    convertible_to_input: bool = Field(
        default=True,
        description=(
            "If True, this parameter can be overridden by connecting an "
            "input wire (ComfyUI-style widget-to-input conversion)."
        ),
    )


class NodeDefinition(BaseModel):
    """Static metadata describing a node type."""

    node_type_id: str = Field(..., description="Unique node type identifier")
    display_name: str = Field(..., description="Human-readable name")
    category: str = Field(default="general", description="Category for grouping")
    description: str = Field(default="", description="What this node does")
    inputs: list[NodePort] = Field(default_factory=list)
    outputs: list[NodePort] = Field(default_factory=list)
    params: list[NodeParam] = Field(
        default_factory=list,
        description="Editable parameters (widgets) displayed on the node card.",
    )
    continuous: bool = Field(
        default=False,
        description=(
            "If True, source nodes (no inputs) re-execute continuously "
            "instead of executing once. Useful for streaming generators."
        ),
    )
    pipeline_meta: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Rich metadata (config_schema, mode_defaults, supports_lora, "
            "supports_vace, etc.) for nodes with a Pydantic config class. "
            "``None`` for plain nodes. Populated by the default "
            ":meth:`Node.get_definition` from the config class's "
            "``get_schema_with_metadata()``."
        ),
    )
    plugin_name: str | None = Field(
        default=None,
        description=(
            "Python package name of the plugin that provides this node, or "
            "``None`` for built-ins. Enriched by the discovery endpoint "
            "from :class:`PluginManager`'s plugin mapping; not set by the "
            "node class itself."
        ),
    )


class Requirements(BaseModel):
    """Runtime chunking requirements returned by :meth:`Node.prepare`.

    Chunked-style nodes tell the runtime how many frames to batch per
    ``__call__`` invocation. Event-style nodes return ``None``.
    """

    input_size: int


class Node:
    """Unified base class for every graph node.

    Subclasses pick exactly one invocation style:

    - **Event-style**: set ``node_type_id``, override :meth:`get_definition`
      with static metadata, and implement :meth:`execute`. Scheduled by
      :class:`NodeProcessor`.
    - **Chunked-style**: override :meth:`get_config_class` to return a
      :class:`BasePipelineConfig` subclass, implement :meth:`__call__`,
      and optionally :meth:`prepare` for multi-frame batching.
      Scheduled by :class:`PipelineProcessor`. The default
      :meth:`get_definition` projects the config class into a
      :class:`NodeDefinition` so nothing has to be declared twice.
    """

    node_type_id: ClassVar[str] = ""

    def __init__(self, node_id: str = "", config: dict[str, Any] | None = None):
        self.node_id = node_id
        self.config = config or {}

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------

    @classmethod
    def get_config_class(cls) -> type[BasePipelineConfig] | None:
        """Return the Pydantic config class, or ``None`` for plain nodes.

        A non-None return marks this node as config-driven (a "pipeline"
        in the historical sense): it gets its identity, ports, and
        parameter-panel schema from the config class rather than from
        static ``node_type_id`` + :meth:`get_definition` declarations.
        """
        return None

    @classmethod
    def get_definition(cls) -> NodeDefinition:
        """Return static metadata for this node type.

        When :meth:`get_config_class` returns a class, the definition
        is derived from it automatically. Plain nodes must override
        this method.
        """
        config = cls.get_config_class()
        if config is None:
            raise NotImplementedError(
                f"{cls.__name__} must either override get_definition() or "
                "return a config class from get_config_class()."
            )
        return _definition_from_config(cls, config)

    @classmethod
    def get_dynamic_output_ports(cls, params: dict[str, Any]) -> set[str]:
        """Return output port names that depend on runtime params.

        Used by the graph executor to accept edges from ports that are
        not declared statically in :meth:`get_definition` — e.g. the
        scheduler node derives one output port per user-configured
        trigger. The default returns an empty set.
        """
        return set()

    @classmethod
    def get_artifacts(cls) -> list[Artifact]:
        """Return the model artifacts this node depends on.

        Artifacts are downloadable resources (HuggingFace repos, Google
        Drive files, etc.) that must be present before the node can run.
        The UI download flow and the :class:`PipelineManager` use this
        list to decide whether the node needs lifecycle-managed loading.

        The default delegates to ``get_config_class().artifacts`` so
        config-driven nodes (historical "pipelines") keep declaring
        artifacts on the config class. Plain event-style nodes that need
        model weights override this method directly.
        """
        config = cls.get_config_class()
        if config is None:
            return []
        return list(getattr(config, "artifacts", []) or [])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self) -> None:  # noqa: B027 — intentional no-op hook
        """Release any resources held by the node.

        Called by the processor when the graph is torn down. Nodes
        with background threads or OS handles override.
        """

    def prepare(self, **kwargs) -> Requirements | None:  # noqa: B027 — optional hook
        """Chunked-style pre-execution hook.

        Called once per tick by :class:`PipelineProcessor` before
        gathering input frames. Return a :class:`Requirements` with
        the number of frames the next ``__call__`` expects, or
        ``None`` if the node is idle / chunking is not applicable.
        """
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Event-style execution — one call per tick.

        ``inputs`` maps input port names to the latest values on those
        ports. Returns a dict mapping output port names to values,
        which :class:`NodeProcessor` fans out to downstream edges.
        Override in plain nodes; chunked-style nodes use
        :meth:`__call__` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override execute() to be invoked "
            "via NodeProcessor (event-style), or override __call__() to be "
            "invoked via PipelineProcessor (chunked-style)."
        )

    def __call__(self, **kwargs) -> dict[str, Any]:
        """Chunked-style execution — one call per batch of frames.

        ``kwargs`` carries pipeline parameters plus per-port frame
        lists (e.g. ``video=[tensor, ...]``). Returns a dict whose
        keys match declared output ports (``video``, ``audio``, etc.).
        Override in chunked-style nodes; event-style nodes use
        :meth:`execute` instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override __call__() to be invoked "
            "via PipelineProcessor (chunked-style), or override execute() "
            "to be invoked via NodeProcessor (event-style)."
        )


# ---------------------------------------------------------------------------
# Backward-compatibility aliases.
#
# ``BaseNode`` was the old name of the base class; ``Pipeline`` used to be a
# separate subclass. Now both are just ``Node`` so every call site, subclass,
# and plugin that imported either continues to work unchanged.
# ---------------------------------------------------------------------------

BaseNode = Node


def _definition_from_config(
    cls: type[Node], config: type[BasePipelineConfig]
) -> NodeDefinition:
    """Project a :class:`BasePipelineConfig` class into a :class:`NodeDefinition`.

    Populates the compact catalog fields (id, ports, etc.) and stores the full
    ``get_schema_with_metadata()`` payload in ``pipeline_meta`` for the
    frontend's parameter panel. ``params`` stays empty because the Pydantic
    schema is too structured to flatten into ``NodeParam`` widgets.
    """
    try:
        pipeline_meta = config.get_schema_with_metadata()
    except Exception as e:
        # Degrade gracefully: log the failure but keep the node visible
        # in the catalog so a single bad plugin can't 500 the whole
        # definitions endpoint.
        logger.error(
            "Failed to build pipeline_meta for %s: %s. Exposing degraded "
            "payload (identity fields only, no config_schema).",
            getattr(config, "pipeline_id", cls.__name__),
            e,
            exc_info=True,
        )
        pipeline_meta = {
            "id": getattr(config, "pipeline_id", cls.__name__),
            "name": getattr(config, "pipeline_name", cls.__name__),
            "description": getattr(config, "pipeline_description", "") or "",
            "version": getattr(config, "pipeline_version", ""),
            "schema_error": str(e),
        }
    inputs = _as_ports(getattr(config, "inputs", ["video"]) or ["video"])
    # Every prompt-capable pipeline (the BasePipelineConfig default) gets a
    # `prompts` string input for free so a backend node — e.g. a Prompt
    # Enhancer — can feed the same handle the UI exposes as "Enter prompt…"
    # without the pipeline having to declare the port itself.
    if getattr(config, "supports_prompts", False) and not any(
        p.name == "prompts" for p in inputs
    ):
        inputs.append(NodePort(name="prompts", port_type="string"))
    return NodeDefinition(
        node_type_id=config.pipeline_id,
        display_name=getattr(config, "pipeline_name", config.pipeline_id),
        category="pipeline",
        description=getattr(config, "pipeline_description", "") or "",
        inputs=inputs,
        outputs=_as_ports(getattr(config, "outputs", ["video"]) or ["video"]),
        params=[],
        continuous=False,
        pipeline_meta=pipeline_meta,
    )


def _as_ports(entries: list[Any]) -> list[NodePort]:
    """Normalise a config ``inputs``/``outputs`` list into :class:`NodePort`.

    Supports mixed lists: a plain string is treated as a ``"video"`` port
    (matches the original per-pipeline convention), while a :class:`NodePort`
    entry passes through unchanged so pipelines can declare non-video ports
    the same way scheduler-style nodes do.
    """
    ports: list[NodePort] = []
    for entry in entries:
        if isinstance(entry, NodePort):
            ports.append(entry)
        else:
            ports.append(NodePort(name=str(entry), port_type="video"))
    return ports

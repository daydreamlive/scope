"""Registry for pipeline block nodes.

Maps block_id -> BlockNode and tracks which pipelines expose blocks,
enabling the API to serve block-level schemas to the frontend.
"""

from __future__ import annotations

import logging
from typing import Any

from .block_node import BlockNode, BlockNodeSchema, BlockParameter

logger = logging.getLogger(__name__)


class BlockRegistry:
    """Registry for block node schemas, keyed by block_id."""

    _blocks: dict[str, BlockNode] = {}
    _pipeline_blocks: dict[str, list[str]] = {}

    @classmethod
    def register_pipeline_blocks(
        cls,
        pipeline_id: str,
        blocks_dict: dict,
        block_parameters: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Register all blocks from a pipeline's block dictionary.

        Args:
            pipeline_id: The pipeline these blocks belong to.
            blocks_dict: Ordered dict mapping block_name -> block_class.
            block_parameters: Optional dict mapping block_name to parameter
                config with "overrides" and/or "additional" keys.
        """
        block_parameters = block_parameters or {}
        block_ids: list[str] = []
        for block_name, block_class in blocks_dict.items():
            block_id = f"{pipeline_id}.{block_name}"
            try:
                param_config = block_parameters.get(block_name, {})
                overrides = param_config.get("overrides", {})
                additional_dicts = param_config.get("additional", [])
                additional = [BlockParameter(**d) for d in additional_dicts]

                node = BlockNode(
                    block_id,
                    block_name,
                    block_class,
                    pipeline_id,
                    parameter_overrides=overrides,
                    additional_parameters=additional,
                )
                cls._blocks[block_id] = node
                block_ids.append(block_id)
            except Exception:
                logger.warning(
                    "Failed to register block %s for pipeline %s",
                    block_name,
                    pipeline_id,
                    exc_info=True,
                )
        cls._pipeline_blocks[pipeline_id] = block_ids
        logger.debug(
            "Registered %d blocks for pipeline %s",
            len(block_ids),
            pipeline_id,
        )

    @classmethod
    def get_pipeline_block_schemas(
        cls,
        pipeline_id: str,
    ) -> list[BlockNodeSchema] | None:
        """Get ordered block schemas for a pipeline.

        Returns:
            List of BlockNodeSchema in execution order, or None if pipeline
            has no registered blocks.
        """
        block_ids = cls._pipeline_blocks.get(pipeline_id)
        if block_ids is None:
            return None
        return [cls._blocks[bid].get_schema() for bid in block_ids]

    @classmethod
    def get_all_schemas(cls) -> dict[str, BlockNodeSchema]:
        """Get all registered block schemas."""
        return {bid: node.get_schema() for bid, node in cls._blocks.items()}

    @classmethod
    def get_pipeline_block_ids(cls) -> dict[str, list[str]]:
        """Get mapping of pipeline_id -> ordered block_ids."""
        return dict(cls._pipeline_blocks)

    @classmethod
    def has_blocks(cls, pipeline_id: str) -> bool:
        """Check if a pipeline has registered blocks."""
        return pipeline_id in cls._pipeline_blocks

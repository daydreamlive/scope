/**
 * Auto-generates a subgraph structure from pipeline block schemas.
 *
 * When a pipeline with blocks (e.g., longlive) is added to the graph,
 * this factory creates the inner block nodes and edges that form the
 * subgraph the user can "zoom into".
 */

import type {
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import type {
  BlockSchemasResponse,
  BlockNodeSchema,
  PipelineSchemaInfo,
} from "../../../lib/api";

/** Horizontal spacing between block nodes inside the subgraph. */
const BLOCK_SPACING_X = 280;
/** Default block node width. */
const BLOCK_WIDTH = 220;

/**
 * Determine whether a port should be treated as a "stream" (video/tensor)
 * or "param" (scalar/config) for subgraph boundary mapping.
 */
function portHandleKind(
  typeHint: string,
  name: string
): "stream" | "param" {
  const hint = typeHint.toLowerCase();
  if (
    hint.includes("tensor") ||
    name === "video" ||
    name === "output_video"
  ) {
    return "stream";
  }
  return "param";
}

/**
 * Map a type_hint string to a SubgraphPort paramType.
 */
function toParamType(
  typeHint: string
): "string" | "number" | "boolean" | "list_number" | undefined {
  const hint = typeHint.toLowerCase();
  if (hint.includes("bool")) return "boolean";
  if (hint.includes("int") || hint.includes("float")) return "number";
  if (hint.includes("str")) return "string";
  if (hint.includes("list") && hint.includes("number")) return "list_number";
  return undefined;
}

/**
 * Create the inner subgraph structure for a pipeline with blocks.
 *
 * @returns The inner nodes, edges, and boundary port definitions, or null
 *          if the pipeline has no block definitions.
 */
export function createBlockSubgraph(
  pipelineId: string,
  blockSchemas: BlockSchemasResponse,
  pipelineSchema: PipelineSchemaInfo
): {
  innerNodes: SerializedSubgraphNode[];
  innerEdges: SerializedSubgraphEdge[];
  subgraphInputs: SubgraphPort[];
  subgraphOutputs: SubgraphPort[];
} | null {
  const blockIds = blockSchemas.pipeline_blocks[pipelineId];
  if (!blockIds || blockIds.length === 0) return null;

  const blocks: BlockNodeSchema[] = blockIds
    .map(id => blockSchemas.blocks[id])
    .filter(Boolean);
  if (blocks.length === 0) return null;

  // ── Create block nodes ──
  const innerNodes: SerializedSubgraphNode[] = blocks.map((block, i) => ({
    id: block.block_id,
    type: "block",
    position: { x: i * BLOCK_SPACING_X, y: 0 },
    width: BLOCK_WIDTH,
    data: {
      label: formatBlockName(block.block_name),
      nodeType: "block",
      blockId: block.block_id,
      blockSchema: block,
    },
  }));

  // ── Wire edges between consecutive blocks ──
  // Blocks communicate via PipelineState: if block N outputs "X" and block M
  // (M > N) inputs "X", they share that state variable. We wire edges from
  // the first producer to all downstream consumers.
  const innerEdges: SerializedSubgraphEdge[] = [];
  const producerOf: Record<string, { blockId: string; kind: "stream" | "param" }> = {};

  for (const block of blocks) {
    // First, wire edges from known producers to this block's inputs
    for (const input of block.inputs) {
      const producer = producerOf[input.name];
      if (producer) {
        const inputKind = portHandleKind(input.type_hint, input.name);
        innerEdges.push({
          id: `e-${producer.blockId}-${block.block_id}-${input.name}`,
          source: producer.blockId,
          sourceHandle: buildHandleId(producer.kind, input.name),
          target: block.block_id,
          targetHandle: buildHandleId(inputKind, input.name),
        });
      }
    }

    // Then register this block's outputs as producers
    for (const output of block.outputs) {
      const kind = portHandleKind(output.type_hint, output.name);
      producerOf[output.name] = { blockId: block.block_id, kind };
    }
  }

  // ── Determine external (subgraph boundary) ports ──
  // External inputs: inputs on blocks that have no internal producer
  // AND match the pipeline's declared stream inputs, or are important params
  const pipelineInputs = new Set(pipelineSchema.inputs ?? ["video"]);
  const pipelineOutputs = new Set(pipelineSchema.outputs ?? ["video"]);

  const subgraphInputs: SubgraphPort[] = [];
  const seenInputNames = new Set<string>();

  // Map pipeline-declared stream inputs to the first block that consumes them
  for (const block of blocks) {
    for (const input of block.inputs) {
      if (seenInputNames.has(input.name)) continue;
      if (pipelineInputs.has(input.name)) {
        const kind = portHandleKind(input.type_hint, input.name);
        subgraphInputs.push({
          name: input.name,
          portType: kind,
          paramType: kind === "param" ? toParamType(input.type_hint) : undefined,
          innerNodeId: block.block_id,
          innerHandleId: buildHandleId(kind, input.name),
        });
        seenInputNames.add(input.name);
      }
    }
  }

  // External outputs: the pipeline's declared outputs mapped to the last
  // block that produces them
  const subgraphOutputs: SubgraphPort[] = [];
  // output_video is the actual state key produced by the decode block,
  // but the pipeline's external port is "video"
  const outputMapping: Record<string, string> = {
    output_video: "video",
  };

  for (const outputName of pipelineOutputs) {
    // Find the last block that produces this output (or its mapped name)
    for (let i = blocks.length - 1; i >= 0; i--) {
      const block = blocks[i];
      const matchingOutput = block.outputs.find(
        o => o.name === outputName || outputMapping[o.name] === outputName
      );
      if (matchingOutput) {
        const kind = portHandleKind(matchingOutput.type_hint, matchingOutput.name);
        subgraphOutputs.push({
          name: outputName,
          portType: kind,
          paramType: kind === "param" ? toParamType(matchingOutput.type_hint) : undefined,
          innerNodeId: block.block_id,
          innerHandleId: buildHandleId(kind, matchingOutput.name),
        });
        break;
      }
    }
  }

  return { innerNodes, innerEdges, subgraphInputs, subgraphOutputs };
}

/**
 * Convert a snake_case block name to Title Case for display.
 */
function formatBlockName(name: string): string {
  return name
    .split("_")
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

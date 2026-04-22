/**
 * Pure-function connection-validation logic.
 *
 * Extracted from useConnectionLogic so it can be unit-tested independently
 * and keeps the hook thin.
 */

import type { Connection, Edge, Node } from "@xyflow/react";
import {
  parseHandleId,
  stripCustomNodeDirection,
} from "../../../lib/graphUtils";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { resolveSourceType } from "./typeResolution";
import { BOUNDARY_INPUT_ID, BOUNDARY_OUTPUT_ID } from "./subgraphSerialization";

// ---------------------------------------------------------------------------
// Target-node rules (param → param).  Each entry maps a target nodeType (or
// a special key) to a predicate that decides whether a given sourceType is
// acceptable for that target handle.
// ---------------------------------------------------------------------------

type TargetRule = (ctx: {
  sourceType: string;
  targetParsedName: string;
  targetNode: Node<FlowNodeData>;
}) => boolean | undefined; // undefined = "this rule doesn't apply, fall through"

/**
 * Ordered list of rules evaluated when both handles are "param".
 * The first rule that returns a boolean wins.  `undefined` means "skip".
 */
const TARGET_RULES: TargetRule[] = [
  // __vace param – only vace type
  ({ sourceType, targetParsedName }) => {
    if (targetParsedName === "__vace") return sourceType === "vace";
    return undefined;
  },
  // vace source can only go to __vace
  ({ sourceType, targetParsedName }) => {
    if (sourceType === "vace") return targetParsedName === "__vace";
    return undefined;
  },
  // __loras param – only lora type
  ({ sourceType, targetParsedName }) => {
    if (targetParsedName === "__loras") return sourceType === "lora";
    return undefined;
  },
  // lora source can only go to __loras
  ({ sourceType, targetParsedName }) => {
    if (sourceType === "lora") return targetParsedName === "__loras";
    return undefined;
  },
  // vace node targets
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "vace") return undefined;
    if (["ref_image", "first_frame", "last_frame"].includes(targetParsedName)) {
      return sourceType === "string";
    }
    return false;
  },
  // __prompt param – only string
  ({ sourceType, targetParsedName }) => {
    if (targetParsedName === "__prompt") return sourceType === "string";
    return undefined;
  },
  // math node – number for a / b
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "math") return undefined;
    return (
      sourceType === "number" &&
      (targetParsedName === "a" || targetParsedName === "b")
    );
  },
  // bool node – number or trigger for input; trigger handle accepts trigger/boolean/number
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "bool") return undefined;
    if (targetParsedName === "trigger")
      return (
        sourceType === "trigger" ||
        sourceType === "boolean" ||
        sourceType === "number"
      );
    return (
      (sourceType === "number" || sourceType === "trigger") &&
      targetParsedName === "input"
    );
  },
  // string-control in switch mode
  ({ sourceType, targetParsedName, targetNode }) => {
    if (
      targetNode.data.nodeType !== "control" ||
      targetNode.data.controlType !== "string" ||
      targetNode.data.controlMode !== "switch"
    )
      return undefined;
    if (targetParsedName.startsWith("item_")) return sourceType === "number";
    if (targetParsedName.startsWith("str_")) return sourceType === "string";
    if (targetParsedName === "trigger")
      return (
        sourceType === "trigger" ||
        sourceType === "boolean" ||
        sourceType === "number"
      );
    return false;
  },
  // slider / knobs / xypad – number only
  ({ sourceType, targetNode }) => {
    if (
      targetNode.data.nodeType === "slider" ||
      targetNode.data.nodeType === "knobs" ||
      targetNode.data.nodeType === "xypad"
    )
      return sourceType === "number";
    return undefined;
  },
  // tuple – per-row inputs accept number, value input accepts list_number
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "tuple") return undefined;
    if (targetParsedName === "value") return sourceType === "list_number";
    if (targetParsedName.startsWith("row_")) return sourceType === "number";
    return false;
  },
  // record – trigger input accepts trigger/boolean/number
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "record") return undefined;
    if (targetParsedName === "trigger")
      return (
        sourceType === "trigger" ||
        sourceType === "boolean" ||
        sourceType === "number"
      );
    return false;
  },
  // prompt_list – cycle accepts number, trigger accepts trigger/boolean/number
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "prompt_list") return undefined;
    if (targetParsedName === "cycle") return sourceType === "number";
    if (targetParsedName === "trigger")
      return (
        sourceType === "trigger" ||
        sourceType === "boolean" ||
        sourceType === "number"
      );
    return false;
  },
  // prompt_blend – weight_N accepts number, prompt_N accepts string
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "prompt_blend") return undefined;
    if (targetParsedName.startsWith("weight_")) return sourceType === "number";
    if (targetParsedName.startsWith("prompt_")) return sourceType === "string";
    return false;
  },
  // scheduler – start/reset accept trigger/boolean/number
  ({ sourceType, targetParsedName, targetNode }) => {
    if (targetNode.data.nodeType !== "scheduler") return undefined;
    if (targetParsedName === "start" || targetParsedName === "reset")
      return (
        sourceType === "trigger" ||
        sourceType === "boolean" ||
        sourceType === "number"
      );
    return false;
  },
];

// ---------------------------------------------------------------------------
// Subgraph / boundary helpers
// ---------------------------------------------------------------------------

function checkSubgraphSource(
  sourceNode: Node<FlowNodeData>,
  sourceParsedName: string,
  targetNode: Node<FlowNodeData>,
  targetParsedName: string
): boolean | undefined {
  if (sourceNode.data.nodeType !== "subgraph") return undefined;
  const port = sourceNode.data.subgraphOutputs?.find(
    p => p.name === sourceParsedName
  );
  if (!port) return undefined;
  if (port.paramType && targetNode.data.nodeType === "pipeline") {
    const targetParam = targetNode.data.parameterInputs?.find(
      p => p.name === targetParsedName
    );
    return targetParam ? port.paramType === targetParam.type : false;
  }
  return true;
}

function checkSubgraphTarget(
  sourceNode: Node<FlowNodeData>,
  targetNode: Node<FlowNodeData>,
  targetParsedName: string,
  nodes: Node<FlowNodeData>[],
  sourceHandleId?: string | null
): boolean | undefined {
  if (targetNode.data.nodeType !== "subgraph") return undefined;
  const port = targetNode.data.subgraphInputs?.find(
    p => p.name === targetParsedName
  );
  if (!port) return undefined;
  const srcType = resolveSourceType(
    sourceNode,
    nodes,
    [],
    new Set(),
    sourceHandleId
  );
  if (!srcType) return true;
  if (!port.paramType) return true;
  return srcType === port.paramType;
}

function checkSubgraphInputSource(
  sourceNode: Node<FlowNodeData>,
  sourceParsedName: string,
  targetNode: Node<FlowNodeData>,
  targetParsedName: string
): boolean | undefined {
  if (sourceNode.data.nodeType !== "subgraph_input") return undefined;
  const port = sourceNode.data.subgraphInputs?.find(
    p => p.name === sourceParsedName
  );
  if (!port) return undefined;
  if (port.paramType && targetNode.data.nodeType === "pipeline") {
    const targetParam = targetNode.data.parameterInputs?.find(
      p => p.name === targetParsedName
    );
    return targetParam ? port.paramType === targetParam.type : false;
  }
  return true;
}

function checkSubgraphOutputTarget(
  sourceNode: Node<FlowNodeData>,
  targetNode: Node<FlowNodeData>,
  targetParsedName: string,
  nodes: Node<FlowNodeData>[],
  sourceHandleId?: string | null
): boolean | undefined {
  if (targetNode.data.nodeType !== "subgraph_output") return undefined;
  const port = targetNode.data.subgraphOutputs?.find(
    p => p.name === targetParsedName
  );
  if (!port) return undefined;
  const srcType = resolveSourceType(
    sourceNode,
    nodes,
    [],
    new Set(),
    sourceHandleId
  );
  if (!srcType) return true;
  if (!port.paramType) return true;
  return srcType === port.paramType;
}

// ---------------------------------------------------------------------------
// Main exported validator
// ---------------------------------------------------------------------------

/**
 * Returns `true` when the proposed connection should be allowed.
 * Pure function – no React dependencies.
 */
export function validateConnection(
  edgeOrConnection: Edge | Connection,
  nodes: Node<FlowNodeData>[]
): boolean {
  let connection: Connection;
  if ("source" in edgeOrConnection && "target" in edgeOrConnection) {
    connection = edgeOrConnection as Connection;
  } else {
    const edge = edgeOrConnection as Edge;
    connection = {
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourceHandle ?? null,
      targetHandle: edge.targetHandle ?? null,
    };
  }

  if (
    !connection.source ||
    !connection.target ||
    !connection.sourceHandle ||
    !connection.targetHandle
  ) {
    return true;
  }

  const sourceParsed = parseHandleId(connection.sourceHandle);
  const targetParsed = parseHandleId(connection.targetHandle);

  if (!sourceParsed || !targetParsed) return true;

  // Boundary "add" ports always valid
  if (
    (connection.source === BOUNDARY_INPUT_ID &&
      sourceParsed.name === "__add__") ||
    (connection.target === BOUNDARY_OUTPUT_ID &&
      targetParsed.name === "__add__")
  ) {
    return true;
  }

  // Stream ↔ stream: for custom_node edges, enforce port-type matching
  // against the node's declared inputs/outputs. For built-in source /
  // pipeline / sink nodes, streams are untyped (video) and always ok.
  if (sourceParsed.kind === "stream" && targetParsed.kind === "stream") {
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);
    if (!sourceNode || !targetNode) return true;
    const srcIsCustom = sourceNode.data.nodeType === "custom_node";
    const tgtIsCustom = targetNode.data.nodeType === "custom_node";
    if (!srcIsCustom && !tgtIsCustom) return true;

    // For each custom endpoint, look up the declared port. We need to
    // distinguish "ports not hydrated yet" (allow, optimistic) from
    // "ports hydrated but this port is missing" (reject, stale wire
    // or typo'd port id). `customNodeInputs`/`customNodeOutputs` is
    // set to an array — possibly empty — exactly when the node has
    // been hydrated from /api/v1/nodes/definitions.
    // CustomNode handles are namespaced as stream:in:<name> / stream:out:<name>
    // so port names in handle IDs carry an in:/out: prefix; strip it before
    // matching against the declared port list.
    let srcType: string | undefined;
    if (srcIsCustom) {
      const outputs = sourceNode.data.customNodeOutputs;
      if (outputs === undefined) {
        // Not hydrated yet — fall through to the compatible path.
      } else {
        const portName = stripCustomNodeDirection(sourceParsed.name);
        const port = outputs.find(p => p.name === portName);
        if (!port) return false;
        srcType = port.port_type;
      }
    }

    let tgtType: string | undefined;
    if (tgtIsCustom) {
      const inputs = targetNode.data.customNodeInputs;
      if (inputs === undefined) {
        // Not hydrated yet — fall through to the compatible path.
      } else {
        const portName = stripCustomNodeDirection(targetParsed.name);
        const port = inputs.find(p => p.name === portName);
        if (!port) return false;
        tgtType = port.port_type;
      }
    }

    // One side is a built-in (untyped stream) or not yet hydrated —
    // we can't compare types, so allow.
    if (!srcType || !tgtType) return true;
    return srcType === tgtType;
  }

  // Param ↔ param
  if (sourceParsed.kind === "param" && targetParsed.kind === "param") {
    const sourceNode = nodes.find(n => n.id === connection.source);
    const targetNode = nodes.find(n => n.id === connection.target);
    if (!sourceNode || !targetNode) return false;

    // Primitive can connect to anything (as source or target)
    if (sourceNode.data.nodeType === "primitive") return true;
    if (targetNode.data.nodeType === "primitive") return true;

    // Reroute source
    if (sourceNode.data.nodeType === "reroute") {
      if (!sourceNode.data.valueType) return true;
      if (targetNode.data.nodeType === "reroute") {
        return (
          !targetNode.data.valueType ||
          targetNode.data.valueType === sourceNode.data.valueType
        );
      }
    }

    // Reroute target
    if (targetNode.data.nodeType === "reroute") {
      if (!targetNode.data.valueType) return true;
      const srcType = resolveSourceType(
        sourceNode,
        nodes,
        [],
        new Set(),
        connection.sourceHandle
      );
      if (!srcType) return true;
      return srcType === targetNode.data.valueType;
    }

    const sourceType = resolveSourceType(
      sourceNode,
      nodes,
      [],
      new Set(),
      connection.sourceHandle
    );
    if (!sourceType) return false;

    // Run table-driven target rules
    for (const rule of TARGET_RULES) {
      const result = rule({
        sourceType,
        targetParsedName: targetParsed.name,
        targetNode,
      });
      if (result !== undefined) return result;
    }

    // Subgraph / boundary node checks
    const sgSrc = checkSubgraphSource(
      sourceNode,
      sourceParsed.name,
      targetNode,
      targetParsed.name
    );
    if (sgSrc !== undefined) return sgSrc;

    const sgTgt = checkSubgraphTarget(
      sourceNode,
      targetNode,
      targetParsed.name,
      nodes,
      connection.sourceHandle
    );
    if (sgTgt !== undefined) return sgTgt;

    const sgInSrc = checkSubgraphInputSource(
      sourceNode,
      sourceParsed.name,
      targetNode,
      targetParsed.name
    );
    if (sgInSrc !== undefined) return sgInSrc;

    const sgOutTgt = checkSubgraphOutputTarget(
      sourceNode,
      targetNode,
      targetParsed.name,
      nodes,
      connection.sourceHandle
    );
    if (sgOutTgt !== undefined) return sgOutTgt;

    // Default: look up the target's parameterInputs
    const targetParam = targetNode.data.parameterInputs?.find(
      p => p.name === targetParsed.name
    );
    if (!targetParam) return false;

    if (targetParam.type === "list_number" && sourceType === "number")
      return true;
    if (targetParam.type === "list_number" && sourceType === "list_number")
      return true;
    if (targetParam.type === "string" && sourceType === "video_path")
      return true;
    if (targetParam.type === "string" && sourceType === "audio_path")
      return true;

    return sourceType === targetParam.type;
  }

  return false;
}

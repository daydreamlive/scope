/**
 * useSubgraphEval
 *
 * Headless evaluator that runs at any depth.  For every `subgraph` node
 * visible on the current canvas it:
 *   1. reads live input values from connected source nodes
 *   2. topologically evaluates the serialized inner graph
 *   3. writes the computed output values as `portValues` on the subgraph node
 *
 * This makes the SubgraphNode card show live output values even when the
 * user is outside the subgraph.
 */

import { useEffect, useRef, useCallback } from "react";
import type { Edge, Node } from "@xyflow/react";
import type {
  FlowNodeData,
  SubgraphPort,
  SerializedSubgraphNode,
  SerializedSubgraphEdge,
} from "../../../../lib/graphUtils";
import { buildHandleId, parseHandleId } from "../../../../lib/graphUtils";
import { getAnyValueFromNode } from "../../utils/getValueFromNode";
import { computeResult } from "../../utils/computeResult";

/* ── Types ────────────────────────────────────────────────────────────────── */

type SetNodes = (
  updater: (nds: Node<FlowNodeData>[]) => Node<FlowNodeData>[]
) => void;

/* ── Pure evaluator ───────────────────────────────────────────────────────── */

/**
 * Evaluate the inner graph of a subgraph given input port values.
 * Returns a map of output port name → computed value.
 */
export function evaluateInnerGraph(
  innerNodes: SerializedSubgraphNode[],
  innerEdges: SerializedSubgraphEdge[],
  subgraphInputs: SubgraphPort[],
  subgraphOutputs: SubgraphPort[],
  inputPortValues: Record<string, unknown>,
  persistentState?: Map<string, unknown>,
  allComputed?: Map<string, unknown>
): Record<string, unknown> {
  // Build a map of nodeId → node data for quick access
  const nodeMap = new Map<string, Record<string, unknown>>();
  for (const n of innerNodes) {
    nodeMap.set(n.id, { ...n.data, __nodeType: n.type });
  }

  // Build adjacency: for each node, which edges feed INTO it (target side)
  // and which edges come OUT of it (source side)
  const incomingEdges = new Map<string, SerializedSubgraphEdge[]>();
  const outgoingEdges = new Map<string, SerializedSubgraphEdge[]>();
  for (const e of innerEdges) {
    if (!incomingEdges.has(e.target)) incomingEdges.set(e.target, []);
    incomingEdges.get(e.target)!.push(e);
    if (!outgoingEdges.has(e.source)) outgoingEdges.set(e.source, []);
    outgoingEdges.get(e.source)!.push(e);
  }

  // Computed values: nodeId:handleName → value
  const computed = new Map<string, unknown>();

  // Seed input boundary values
  for (const port of subgraphInputs) {
    if (port.portType !== "param") continue;
    const val = inputPortValues[port.name];
    // The inner node connected to this input port receives the value
    // We store it keyed by innerNodeId:innerHandleName
    const parsed = parseHandleId(port.innerHandleId);
    if (parsed) {
      computed.set(`${port.innerNodeId}:input:${parsed.name}`, val ?? null);
    }
  }

  // Also seed from inner edges that originate from the input boundary
  // (boundary edges connect EVAL_INPUT to inner nodes, but since the
  //  boundary nodes are virtual and not in innerNodes, we use SubgraphPort
  //  mappings directly above)

  // Topological sort via Kahn's algorithm
  const allNodeIds = innerNodes.map(n => n.id);
  const inDegree = new Map<string, number>();
  for (const nid of allNodeIds) inDegree.set(nid, 0);
  for (const e of innerEdges) {
    // Only count param edges (stream edges don't carry scalar values)
    const parsedSrc = parseHandleId(e.sourceHandle);
    if (parsedSrc && parsedSrc.kind === "stream") continue;
    inDegree.set(e.target, (inDegree.get(e.target) ?? 0) + 1);
  }

  const queue: string[] = [];
  for (const [nid, deg] of inDegree) {
    if (deg === 0) queue.push(nid);
  }

  const order: string[] = [];
  while (queue.length > 0) {
    const nid = queue.shift()!;
    order.push(nid);
    for (const e of outgoingEdges.get(nid) ?? []) {
      const parsedSrc = parseHandleId(e.sourceHandle);
      if (parsedSrc && parsedSrc.kind === "stream") continue;
      const newDeg = (inDegree.get(e.target) ?? 1) - 1;
      inDegree.set(e.target, newDeg);
      if (newDeg === 0) queue.push(e.target);
    }
  }

  // Add any nodes not reached (cycles or disconnected) at the end
  for (const nid of allNodeIds) {
    if (!order.includes(nid)) order.push(nid);
  }

  // Evaluate each node in topological order
  for (const nid of order) {
    const data = nodeMap.get(nid);
    if (!data) continue;
    const nodeType = data.__nodeType as string;

    // Gather input values for this node from edges
    const inputs = new Map<string, unknown>();
    for (const e of incomingEdges.get(nid) ?? []) {
      const parsedSrc = parseHandleId(e.sourceHandle);
      const parsedTgt = parseHandleId(e.targetHandle);
      if (!parsedTgt) continue;
      if (parsedSrc && parsedSrc.kind === "stream") continue;

      // Check if we have a computed output for the source
      const srcKey = `${e.source}:output:${parsedSrc?.name ?? "value"}`;
      if (computed.has(srcKey)) {
        inputs.set(parsedTgt.name, computed.get(srcKey));
      }
      // Also check if there's a direct input seed (from boundary)
      const directKey = `${nid}:input:${parsedTgt.name}`;
      if (computed.has(directKey) && !inputs.has(parsedTgt.name)) {
        inputs.set(parsedTgt.name, computed.get(directKey));
      }
    }

    // Also pull any direct seeds (from boundary ports)
    for (const [key, val] of computed) {
      if (key.startsWith(`${nid}:input:`)) {
        const paramName = key.slice(`${nid}:input:`.length);
        if (!inputs.has(paramName)) {
          inputs.set(paramName, val);
        }
      }
    }

    // Compute output(s) based on node type
    const evalState = persistentState ?? new Map<string, unknown>();
    const outputValues = evaluateNode(nodeType, data, inputs, nid, evalState);

    // Store computed outputs
    for (const [handleName, val] of outputValues) {
      computed.set(`${nid}:output:${handleName}`, val);
    }

    // Propagate outputs via edges to downstream nodes' inputs
    for (const e of outgoingEdges.get(nid) ?? []) {
      const parsedSrc = parseHandleId(e.sourceHandle);
      const parsedTgt = parseHandleId(e.targetHandle);
      if (!parsedSrc || !parsedTgt) continue;
      if (parsedSrc.kind === "stream") continue;

      const srcKey = `${nid}:output:${parsedSrc.name}`;
      if (computed.has(srcKey)) {
        computed.set(
          `${e.target}:input:${parsedTgt.name}`,
          computed.get(srcKey)
        );
      }
    }
  }

  if (allComputed) {
    for (const [k, v] of computed) {
      allComputed.set(k, v);
    }
  }

  // Collect output boundary values
  const result: Record<string, unknown> = {};
  for (const port of subgraphOutputs) {
    if (port.portType !== "param") continue;
    // The output boundary port maps to an inner node's output handle
    const parsed = parseHandleId(port.innerHandleId);
    if (!parsed) continue;
    const key = `${port.innerNodeId}:output:${parsed.name}`;
    if (computed.has(key)) {
      result[port.name] = computed.get(key);
    }
  }

  return result;
}

/**
 * Evaluate a single node, returning a map of output handle name → value.
 */
function evaluateNode(
  nodeType: string,
  data: Record<string, unknown>,
  inputs: Map<string, unknown>,
  nodeId: string,
  state: Map<string, unknown>
): Map<string, unknown> {
  const out = new Map<string, unknown>();

  switch (nodeType) {
    case "math": {
      const op = (data.mathOp as string) ?? "add";
      const a = toNumber(inputs.get("a"));
      const b = toNumber(inputs.get("b"));
      let result = computeResult(op, a, b);
      const outputType = data.mathOutputType as string | undefined;
      if (result !== null && outputType) {
        if (outputType === "int") result = Math.trunc(result);
      }
      out.set("value", result);
      break;
    }
    case "bool": {
      const mode = (data.boolMode as string) ?? "gate";
      const threshold = (data.boolThreshold as number) ?? 0;
      const input = toNumber(inputs.get("input"));
      if (mode === "gate") {
        out.set("value", input !== null && input > threshold ? 1 : 0);
      } else {
        // Toggle requires state — use stored value as best guess
        out.set("value", data.value ? 1 : 0);
      }
      break;
    }
    case "trigger": {
      out.set("value", data.value ? 1 : 0);
      break;
    }
    case "primitive":
    case "reroute": {
      const val = data.value ?? null;
      out.set("value", val);
      // Also pass through any input
      if (inputs.has("value") && inputs.get("value") !== undefined) {
        out.set("value", inputs.get("value"));
      }
      break;
    }
    case "slider": {
      let sliderVal: number | null = (data.value as number) ?? null;
      if (inputs.has("value") && inputs.get("value") !== undefined) {
        sliderVal = inputs.get("value") as number;
      }
      if (typeof sliderVal === "number") {
        const sMin = (data.sliderMin as number) ?? 0;
        const sMax = (data.sliderMax as number) ?? 1;
        sliderVal = Math.min(Math.max(sliderVal, sMin), sMax);
      }
      out.set("value", sliderVal);
      break;
    }
    case "control": {
      const ctrlType = (data.controlType as string) ?? "float";
      const ctrlMode = (data.controlMode as string) ?? "animated";

      if (ctrlType === "string" && ctrlMode === "switch") {
        const items = (data.controlItems as string[]) ?? [];
        let bestIdx = -1;
        let bestVal = 0;
        for (let i = 0; i < items.length; i++) {
          const trigVal = toNumber(inputs.get(`item_${i}`));
          if (trigVal !== null && trigVal > bestVal) {
            bestVal = trigVal;
            bestIdx = i;
          }
        }
        const stateKey = `${nodeId}:switch_value`;
        if (bestIdx >= 0) {
          const strVal = inputs.get(`str_${bestIdx}`);
          const selected = strVal ?? items[bestIdx] ?? null;
          state.set(stateKey, selected);
          out.set("value", selected);
        } else {
          const latched = state.get(stateKey);
          out.set("value", latched ?? data.currentValue ?? null);
        }
      } else {
        out.set("value", data.currentValue ?? null);
        if (inputs.has("value") && inputs.get("value") !== undefined) {
          out.set("value", inputs.get("value"));
        }
      }
      break;
    }
    case "knobs": {
      const knobs = data.knobs as { value: number }[] | undefined;
      if (knobs) {
        for (let i = 0; i < knobs.length; i++) {
          const inputKey = `knob_${i}`;
          const inputVal = inputs.get(inputKey);
          out.set(
            inputKey,
            typeof inputVal === "number" ? inputVal : knobs[i].value
          );
        }
      }
      break;
    }
    case "tuple": {
      let vals = data.tupleValues as number[] | undefined;
      if (inputs.has("value") && Array.isArray(inputs.get("value"))) {
        vals = inputs.get("value") as number[];
      }
      const tMin = (data.tupleMin as number) ?? 0;
      const tMax = (data.tupleMax as number) ?? 1000;
      if (vals) {
        for (const [inputName, inputVal] of inputs) {
          if (inputName.startsWith("row_") && typeof inputVal === "number") {
            const idx = parseInt(inputName.replace("row_", ""), 10);
            if (!isNaN(idx) && idx < vals.length) {
              vals = [...vals];
              vals[idx] = Math.min(Math.max(inputVal, tMin), tMax);
            }
          }
        }
      }
      out.set("value", vals ?? null);
      break;
    }
    case "subgraph": {
      // Nested subgraph — recursively evaluate
      const nestedInputs = (data.subgraphInputs ?? []) as SubgraphPort[];
      const nestedOutputs = (data.subgraphOutputs ?? []) as SubgraphPort[];
      const nestedNodes = (data.subgraphNodes ??
        []) as SerializedSubgraphNode[];
      const nestedEdges = (data.subgraphEdges ??
        []) as SerializedSubgraphEdge[];

      // Build input values for the nested subgraph from our inputs
      const nestedInputVals: Record<string, unknown> = {};
      for (const port of nestedInputs) {
        if (port.portType !== "param") continue;
        if (inputs.has(port.name)) {
          nestedInputVals[port.name] = inputs.get(port.name);
        }
      }

      const nestedResult = evaluateInnerGraph(
        nestedNodes,
        nestedEdges,
        nestedInputs,
        nestedOutputs,
        nestedInputVals
      );

      for (const [k, v] of Object.entries(nestedResult)) {
        out.set(k, v);
      }
      break;
    }
    default:
      // Unknown node types — pass through any input as "value" output
      if (inputs.has("value")) {
        out.set("value", inputs.get("value"));
      }
      break;
  }

  return out;
}

function toNumber(val: unknown): number | null {
  if (typeof val === "number") return val;
  if (typeof val === "boolean") return val ? 1 : 0;
  return null;
}

/* ── Hook ─────────────────────────────────────────────────────────────────── */

export function useSubgraphEval(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  setNodes: SetNodes
) {
  const setNodesRef = useRef(setNodes);
  setNodesRef.current = setNodes;

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const edgesRef = useRef(edges);
  edgesRef.current = edges;

  // Track last computed outputs per subgraph node to avoid unnecessary updates
  const lastOutputsRef = useRef<Map<string, string>>(new Map());

  // Persistent state per subgraph (for latching switch selections, etc.)
  const evalStateRef = useRef<Map<string, Map<string, unknown>>>(new Map());

  const rafHandle = useRef<number | null>(null);

  const evaluate = useCallback(() => {
    const currentNodes = nodesRef.current;
    const currentEdges = edgesRef.current;

    const sgNodes = currentNodes.filter(n => n.data.nodeType === "subgraph");
    if (sgNodes.length === 0) {
      rafHandle.current = requestAnimationFrame(evaluate);
      return;
    }

    const updates: { nodeId: string; portValues: Record<string, unknown> }[] =
      [];

    for (const sg of sgNodes) {
      const sgInputs: SubgraphPort[] = sg.data.subgraphInputs ?? [];
      const sgOutputs: SubgraphPort[] = sg.data.subgraphOutputs ?? [];
      const innerNodes = (sg.data.subgraphNodes ??
        []) as SerializedSubgraphNode[];
      const innerEdges = (sg.data.subgraphEdges ??
        []) as SerializedSubgraphEdge[];

      if (innerNodes.length === 0) continue;

      // Read live input values from connected source nodes on current canvas
      const inputPortValues: Record<string, unknown> = {};
      for (const port of sgInputs) {
        if (port.portType !== "param") continue;
        const handleId = buildHandleId("param", port.name);
        const edge = currentEdges.find(
          e => e.target === sg.id && e.targetHandle === handleId
        );
        if (!edge) continue;
        const srcNode = currentNodes.find(n => n.id === edge.source);
        if (!srcNode) continue;
        inputPortValues[port.name] = getAnyValueFromNode(
          srcNode,
          edge.sourceHandle
        );
      }

      // Evaluate the inner graph (with persistent state for latching)
      if (!evalStateRef.current.has(sg.id)) {
        evalStateRef.current.set(sg.id, new Map());
      }
      const sgState = evalStateRef.current.get(sg.id)!;
      const outputValues = evaluateInnerGraph(
        innerNodes,
        innerEdges,
        sgInputs,
        sgOutputs,
        inputPortValues,
        sgState
      );

      // Merge input values into portValues too (so inputs are also readable from portValues)
      const merged: Record<string, unknown> = {
        ...inputPortValues,
        ...outputValues,
      };

      // Check if anything changed (shallow JSON compare)
      const key = JSON.stringify(merged);
      if (lastOutputsRef.current.get(sg.id) === key) continue;
      lastOutputsRef.current.set(sg.id, key);

      updates.push({ nodeId: sg.id, portValues: merged });
    }

    if (updates.length > 0) {
      setNodesRef.current(nds =>
        nds.map(n => {
          const upd = updates.find(u => u.nodeId === n.id);
          if (!upd) return n;
          return {
            ...n,
            data: { ...n.data, portValues: upd.portValues },
          };
        })
      );
    }

    rafHandle.current = requestAnimationFrame(evaluate);
  }, []);

  useEffect(() => {
    rafHandle.current = requestAnimationFrame(evaluate);
    return () => {
      if (rafHandle.current !== null) cancelAnimationFrame(rafHandle.current);
    };
  }, [evaluate]);
}

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

/** Per-trigger_action node state that persists across rAF evaluation ticks. */
interface TriggerActionEvalState {
  lastTriggerInput: number;
  currentValue: unknown;
  // animate_number fields
  animStartTime?: number;
  animFrom?: number;
  animTo?: number;
  animDuration?: number;
  animCurve?: string;
  // toggle_bool / cycle_strings state
  toggleState?: boolean;
  cycleIndex?: number;
}

/** Keyed by inner node id. */
export type EvalStateStore = Map<string, TriggerActionEvalState>;

/* ── Easing functions ────────────────────────────────────────────────────── */

function easeLinear(t: number): number {
  return t;
}
function easeIn(t: number): number {
  return t * t;
}
function easeOut(t: number): number {
  return t * (2 - t);
}
function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

const EASING_FNS: Record<string, (t: number) => number> = {
  linear: easeLinear,
  ease_in: easeIn,
  ease_out: easeOut,
  ease_in_out: easeInOut,
};

/** Interpolate y from sorted breakpoints (both axes normalized 0-1). */
function interpolateCurvePoints(
  points: Array<{ x: number; y: number }>,
  t: number
): number {
  if (points.length === 0) return t;
  if (points.length === 1) return points[0].y;
  if (t <= points[0].x) return points[0].y;
  if (t >= points[points.length - 1].x) return points[points.length - 1].y;
  for (let i = 0; i < points.length - 1; i++) {
    if (t >= points[i].x && t <= points[i + 1].x) {
      const segLen = points[i + 1].x - points[i].x;
      const localT = segLen > 0 ? (t - points[i].x) / segLen : 0;
      return points[i].y + localT * (points[i + 1].y - points[i].y);
    }
  }
  return points[points.length - 1].y;
}

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
  stateStore?: EvalStateStore
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
    const outputValues = evaluateNode(
      nodeType, nid, data, inputs, stateStore,
      nodeMap, incomingEdges.get(nid)
    );

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
 * An optional `stateStore` provides persistent per-node state across ticks
 * (used for trigger_action rising-edge detection, value latching, and animation).
 */
function evaluateNode(
  nodeType: string,
  nodeId: string,
  data: Record<string, unknown>,
  inputs: Map<string, unknown>,
  stateStore?: EvalStateStore,
  /** Optional: map of all inner node data (keyed by node id) for cross-node lookups */
  nodeMap?: Map<string, Record<string, unknown>>,
  /** Optional: incoming edges for this node, for looking up source nodes */
  incomingEdgesForNode?: SerializedSubgraphEdge[]
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
      out.set("value", (data.value as number) ?? null);
      break;
    }
    case "control": {
      out.set("value", data.currentValue ?? null);
      break;
    }
    case "knobs": {
      const knobs = data.knobs as { value: number }[] | undefined;
      if (knobs) {
        for (let i = 0; i < knobs.length; i++) {
          out.set(`knob_${i}`, knobs[i].value);
        }
      }
      break;
    }
    case "timeline": {
      // Output each trigger's current value from triggerValues
      const triggerValues = (data.triggerValues ?? {}) as Record<
        string,
        number
      >;
      const triggers = (data.timelineTriggers ?? []) as Array<{
        id: string;
        time: number;
      }>;
      for (const trigger of triggers) {
        out.set(`trigger_${trigger.id}`, triggerValues[trigger.id] ?? 0);
      }
      break;
    }
    case "curve": {
      // Curve node: shape data is read directly by trigger_action; output is placeholder
      out.set("value", 0);
      break;
    }
    case "trigger_action": {
      const actionType = (data.triggerActionType as string) ?? "set_number";
      const triggerInput = toNumber(inputs.get("trigger")) ?? 0;

      // Get or create persistent state for this node
      let state = stateStore?.get(nodeId);
      if (!state) {
        state = {
          lastTriggerInput: 0,
          currentValue: data.currentValue ?? 0,
          toggleState: (data.triggerToggleState as boolean) ?? false,
          cycleIndex: (data.triggerCycleIndex as number) ?? 0,
        };
        stateStore?.set(nodeId, state);
      }

      // Detect rising edge: previous <= 0 and current > 0
      const risingEdge = state.lastTriggerInput <= 0 && triggerInput > 0;
      state.lastTriggerInput = triggerInput;

      if (risingEdge) {
        switch (actionType) {
          case "set_number":
            state.currentValue = Number(data.triggerSetValue) || 0;
            break;
          case "set_string":
            state.currentValue = String(data.triggerSetValue ?? "");
            break;
          case "set_bool":
            state.currentValue = data.triggerSetValue ? 1 : 0;
            break;
          case "animate_number": {
            // Start animation: record start time and params
            state.animStartTime = Date.now();
            state.animFrom = (data.triggerAnimateFrom as number) ?? 0;
            state.animTo = (data.triggerAnimateTo as number) ?? 1;
            state.animDuration = (data.triggerAnimateDuration as number) ?? 1;
            state.animCurve = (data.triggerAnimateCurve as string) ?? "linear";
            state.currentValue = state.animFrom;
            break;
          }
          case "toggle_bool": {
            state.toggleState = !state.toggleState;
            state.currentValue = state.toggleState ? 1 : 0;
            break;
          }
          case "cycle_strings": {
            const items = (data.triggerCycleItems as string[]) ?? [];
            if (items.length > 0) {
              state.cycleIndex = ((state.cycleIndex ?? 0) + 1) % items.length;
              state.currentValue = items[state.cycleIndex] ?? "";
            }
            break;
          }
        }
      }

      // For animate_number, interpolate if animation is in progress
      if (
        actionType === "animate_number" &&
        state.animStartTime !== undefined &&
        state.animDuration !== undefined &&
        state.animFrom !== undefined &&
        state.animTo !== undefined
      ) {
        const elapsed = (Date.now() - state.animStartTime) / 1000;
        const progress = Math.min(elapsed / state.animDuration, 1);

        // Check for connected curve node to use custom curve shape + value trajectory
        let curvePoints: Array<{ x: number; y: number }> | undefined;
        let curveMin = 0;
        let curveMax = 1;
        if (nodeMap && incomingEdgesForNode) {
          const curveHandleId = buildHandleId("param", "curve");
          for (const e of incomingEdgesForNode) {
            if (e.targetHandle === curveHandleId && e.source) {
              const srcData = nodeMap.get(e.source);
              if (srcData && srcData.__nodeType === "curve") {
                curvePoints = srcData.curvePoints as Array<{ x: number; y: number }> | undefined;
                curveMin = (srcData.curveMin as number) ?? 0;
                curveMax = (srcData.curveMax as number) ?? 1;
              }
              break;
            }
          }
        }

        if (curvePoints && curvePoints.length >= 2) {
          // Curve defines full value trajectory: y (0-1) maps to [curveMin, curveMax]
          const curveY = interpolateCurvePoints(curvePoints, progress);
          state.currentValue = curveMin + (curveMax - curveMin) * curveY;
        } else {
          // Preset easing between From and To
          const easeFn = EASING_FNS[state.animCurve ?? "linear"] ?? easeLinear;
          const eased = easeFn(progress);
          state.currentValue =
            state.animFrom + (state.animTo - state.animFrom) * eased;
        }
        // Clear animation state once complete
        if (progress >= 1) {
          state.animStartTime = undefined;
        }
      }

      out.set("value", state.currentValue ?? 0);
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

  // Persistent state stores for stateful inner nodes (keyed by subgraph node id)
  const stateStoresRef = useRef<Map<string, EvalStateStore>>(new Map());

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

      // Get or create state store for this subgraph
      if (!stateStoresRef.current.has(sg.id)) {
        stateStoresRef.current.set(sg.id, new Map());
      }
      const store = stateStoresRef.current.get(sg.id)!;

      // Evaluate the inner graph
      const outputValues = evaluateInnerGraph(
        innerNodes,
        innerEdges,
        sgInputs,
        sgOutputs,
        inputPortValues,
        store
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

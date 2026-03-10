import { useEffect, useRef } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { parseHandleId } from "../../../lib/graphUtils";

/** Shallow-compare two values, handling arrays element-by-element. */
function valuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!valuesEqual(a[i], b[i])) return false;
    }
    return true;
  }
  // Handle knob objects: {label, min, max, value}
  if (a && b && typeof a === "object" && typeof b === "object") {
    const ka = Object.keys(a as Record<string, unknown>);
    const kb = Object.keys(b as Record<string, unknown>);
    if (ka.length !== kb.length) return false;
    for (const k of ka) {
      if (
        !valuesEqual(
          (a as Record<string, unknown>)[k],
          (b as Record<string, unknown>)[k]
        )
      )
        return false;
    }
    return true;
  }
  return false;
}

/** Node types that produce parameter output values. */
const PRODUCER_TYPES = new Set<FlowNodeData["nodeType"]>([
  "primitive",
  "control",
  "math",
  "slider",
  "knobs",
  "xypad",
  "tuple",
  "reroute",
  "image",
  "vace",
]);

/** Node types that can receive values on their input handles. */
const UI_INPUT_TYPES = new Set<FlowNodeData["nodeType"]>([
  "slider",
  "knobs",
  "xypad",
  "tuple",
  "reroute",
  "vace",
]);

export function useValueForwarding(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  findConnectedPipelineParams: (
    sourceNodeId: string,
    edges: Edge[],
    nodes: Node<FlowNodeData>[]
  ) => Array<{ nodeId: string; paramName: string }>,
  resolveBackendId: (nodeId: string) => string,
  isStreamingRef: React.RefObject<boolean>,
  onNodeParamChangeRef: React.RefObject<
    ((nodeId: string, key: string, value: unknown) => void) | undefined
  >,
  setNodes?: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>
) {
  const lastForwardTimeRef = useRef<Record<string, number>>({});

  // Output forwarding
  useEffect(() => {
    if (!isStreamingRef.current || !onNodeParamChangeRef.current) return;

    const throttleMs = 100;

    for (const node of nodes) {
      if (!PRODUCER_TYPES.has(node.data.nodeType)) continue;

      const connected = findConnectedPipelineParams(node.id, edges, nodes);
      if (connected.length === 0) continue;

      // Get values to forward
      const valuesToForward: Array<{
        handleName: string | null;
        value: unknown;
      }> = [];

      if (node.data.nodeType === "primitive") {
        valuesToForward.push({ handleName: null, value: node.data.value });
      } else if (node.data.nodeType === "reroute") {
        valuesToForward.push({ handleName: null, value: node.data.value });
      } else if (
        node.data.nodeType === "control" ||
        node.data.nodeType === "math"
      ) {
        valuesToForward.push({
          handleName: null,
          value: node.data.currentValue,
        });
      } else if (node.data.nodeType === "slider") {
        valuesToForward.push({ handleName: "value", value: node.data.value });
      } else if (node.data.nodeType === "knobs") {
        const knobs = node.data.knobs;
        if (knobs) {
          for (let i = 0; i < knobs.length; i++) {
            valuesToForward.push({
              handleName: `knob_${i}`,
              value: knobs[i].value,
            });
          }
        }
      } else if (node.data.nodeType === "xypad") {
        valuesToForward.push({ handleName: "x", value: node.data.padX });
        valuesToForward.push({ handleName: "y", value: node.data.padY });
      } else if (node.data.nodeType === "tuple") {
        valuesToForward.push({
          handleName: "value",
          value: node.data.tupleValues,
        });
      } else if (node.data.nodeType === "image") {
        valuesToForward.push({
          handleName: "value",
          value: node.data.imagePath || "",
        });
      }

      // Throttle animated nodes
      const isAnimated =
        node.data.nodeType === "control" || node.data.nodeType === "math";
      if (isAnimated) {
        const now = Date.now();
        const lastTime = lastForwardTimeRef.current[node.id] || 0;
        if (now - lastTime < throttleMs) continue;
        lastForwardTimeRef.current[node.id] = now;
      }

      // Forward to connected pipeline params
      for (const edge of edges) {
        if (edge.source !== node.id) continue;
        const sourceParsed = parseHandleId(edge.sourceHandle);
        const targetParsed = parseHandleId(edge.targetHandle);
        if (!sourceParsed || sourceParsed.kind !== "param") continue;
        if (!targetParsed || targetParsed.kind !== "param") continue;

        // Find target node
        const targetNode = nodes.find(n => n.id === edge.target);
        if (!targetNode) continue;

        // Only to pipeline nodes
        if (targetNode.data.nodeType !== "pipeline") continue;

        // VACE compound connection: expand into individual parameters
        if (
          node.data.nodeType === "vace" &&
          sourceParsed.name === "__vace" &&
          targetParsed.name === "__vace"
        ) {
          const backendId = resolveBackendId(edge.target);
          const ctxScale =
            typeof node.data.vaceContextScale === "number"
              ? node.data.vaceContextScale
              : 1.0;
          onNodeParamChangeRef.current(
            backendId,
            "vace_context_scale",
            ctxScale
          );
          const refImg = (node.data.vaceRefImage as string) || "";
          if (refImg) {
            onNodeParamChangeRef.current(backendId, "vace_ref_images", [
              refImg,
            ]);
          }
          const firstFrame = (node.data.vaceFirstFrame as string) || "";
          if (firstFrame) {
            onNodeParamChangeRef.current(
              backendId,
              "first_frame_image",
              firstFrame
            );
          }
          const lastFrame = (node.data.vaceLastFrame as string) || "";
          if (lastFrame) {
            onNodeParamChangeRef.current(
              backendId,
              "last_frame_image",
              lastFrame
            );
          }
          continue;
        }

        // Find matching value
        const entry = valuesToForward.find(v => {
          if (v.handleName === null) return true; // value/control/math: single output
          return v.handleName === sourceParsed.name;
        });
        if (!entry || entry.value === undefined) continue;

        const backendId = resolveBackendId(edge.target);
        if (targetParsed.name === "__prompt") {
          onNodeParamChangeRef.current(backendId, "prompts", [
            { text: String(entry.value), weight: 100 },
          ]);
        } else {
          onNodeParamChangeRef.current(
            backendId,
            targetParsed.name,
            entry.value
          );
        }
      }
    }
  }, [
    nodes,
    edges,
    findConnectedPipelineParams,
    resolveBackendId,
    isStreamingRef,
    onNodeParamChangeRef,
  ]);

  // Input consumption
  useEffect(() => {
    if (!setNodes) return;

    // Build UI node updates map
    const updates = new Map<string, Record<string, unknown>>();

    for (const edge of edges) {
      const targetNode = nodes.find(n => n.id === edge.target);
      if (!targetNode || !UI_INPUT_TYPES.has(targetNode.data.nodeType))
        continue;

      const targetParsed = parseHandleId(edge.targetHandle);
      if (!targetParsed || targetParsed.kind !== "param") continue;

      // Find source node
      const sourceNode = nodes.find(n => n.id === edge.source);
      if (!sourceNode) continue;

      let sourceValue: unknown;
      const sourceParsed = parseHandleId(edge.sourceHandle);
      if (!sourceParsed || sourceParsed.kind !== "param") continue;

      if (sourceNode.data.nodeType === "primitive") {
        sourceValue = sourceNode.data.value;
      } else if (sourceNode.data.nodeType === "reroute") {
        sourceValue = sourceNode.data.value;
      } else if (
        sourceNode.data.nodeType === "control" ||
        sourceNode.data.nodeType === "math"
      ) {
        sourceValue = sourceNode.data.currentValue;
      } else if (sourceNode.data.nodeType === "slider") {
        sourceValue = sourceNode.data.value;
      } else if (sourceNode.data.nodeType === "knobs") {
        const idx = parseInt(sourceParsed.name.replace("knob_", ""), 10);
        const knobs = sourceNode.data.knobs;
        if (knobs && !isNaN(idx) && idx < knobs.length) {
          sourceValue = knobs[idx].value;
        }
      } else if (sourceNode.data.nodeType === "xypad") {
        if (sourceParsed.name === "x") sourceValue = sourceNode.data.padX;
        else if (sourceParsed.name === "y") sourceValue = sourceNode.data.padY;
      } else if (sourceNode.data.nodeType === "tuple") {
        sourceValue = sourceNode.data.tupleValues;
      }

      if (sourceValue === undefined) continue;

      // Determine field to update
      const nodeUpdates = updates.get(edge.target) ?? {};

      if (
        targetNode.data.nodeType === "slider" &&
        targetParsed.name === "value"
      ) {
        const min = targetNode.data.sliderMin ?? 0;
        const max = targetNode.data.sliderMax ?? 1;
        const clamped = Math.min(Math.max(Number(sourceValue), min), max);
        nodeUpdates["value"] = clamped;
      } else if (targetNode.data.nodeType === "knobs") {
        const idx = parseInt(targetParsed.name.replace("knob_", ""), 10);
        const knobs = targetNode.data.knobs;
        if (knobs && !isNaN(idx) && idx < knobs.length) {
          const knob = knobs[idx];
          const clamped = Math.min(
            Math.max(Number(sourceValue), knob.min),
            knob.max
          );
          // Merge knobs array
          const existingKnobs = (nodeUpdates["knobs"] as typeof knobs) ?? [
            ...knobs,
          ];
          existingKnobs[idx] = { ...existingKnobs[idx], value: clamped };
          nodeUpdates["knobs"] = existingKnobs;
        }
      } else if (targetNode.data.nodeType === "xypad") {
        if (targetParsed.name === "x") {
          const min = targetNode.data.padMinX ?? 0;
          const max = targetNode.data.padMaxX ?? 1;
          nodeUpdates["padX"] = Math.min(
            Math.max(Number(sourceValue), min),
            max
          );
        } else if (targetParsed.name === "y") {
          const min = targetNode.data.padMinY ?? 0;
          const max = targetNode.data.padMaxY ?? 1;
          nodeUpdates["padY"] = Math.min(
            Math.max(Number(sourceValue), min),
            max
          );
        }
      } else if (targetNode.data.nodeType === "tuple") {
        if (targetParsed.name === "value" && Array.isArray(sourceValue)) {
          nodeUpdates["tupleValues"] = sourceValue;
        } else if (
          targetParsed.name.startsWith("row_") &&
          typeof sourceValue === "number"
        ) {
          const rowIdx = parseInt(targetParsed.name.replace("row_", ""), 10);
          const tupleValues = targetNode.data.tupleValues;
          if (tupleValues && !isNaN(rowIdx) && rowIdx < tupleValues.length) {
            const existingValues = (nodeUpdates["tupleValues"] as number[]) ?? [
              ...tupleValues,
            ];
            const clamped = Math.min(
              Math.max(sourceValue, targetNode.data.tupleMin ?? 0),
              targetNode.data.tupleMax ?? 1000
            );
            existingValues[rowIdx] = clamped;
            nodeUpdates["tupleValues"] = existingValues;
          }
        }
      } else if (targetNode.data.nodeType === "vace") {
        // Map image input handles to VACE node data fields
        if (targetParsed.name === "ref_image") {
          nodeUpdates["vaceRefImage"] = String(sourceValue);
        } else if (targetParsed.name === "first_frame") {
          nodeUpdates["vaceFirstFrame"] = String(sourceValue);
        } else if (targetParsed.name === "last_frame") {
          nodeUpdates["vaceLastFrame"] = String(sourceValue);
        }
      } else if (targetNode.data.nodeType === "reroute") {
        // Pass source value through to the reroute node
        nodeUpdates["value"] = sourceValue;
      }

      if (Object.keys(nodeUpdates).length > 0) {
        updates.set(edge.target, nodeUpdates);
      }
    }

    if (updates.size === 0) return;

    // Apply updates (return original if unchanged to avoid loop)
    setNodes(nds => {
      let anyNodeChanged = false;
      const result = nds.map(n => {
        const upd = updates.get(n.id);
        if (!upd) return n;

        // Check if values changed (deep compare)
        let changed = false;
        for (const [key, val] of Object.entries(upd)) {
          if (!valuesEqual(n.data[key], val)) {
            changed = true;
            break;
          }
        }
        if (!changed) return n;

        anyNodeChanged = true;
        return { ...n, data: { ...n.data, ...upd } };
      });
      return anyNodeChanged ? result : nds;
    });
  }, [nodes, edges, setNodes]);
}

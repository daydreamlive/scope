import { useEffect, useRef, useMemo } from "react";
import type { Edge, Node } from "@xyflow/react";
import type { FlowNodeData, SubgraphPort } from "../../../../lib/graphUtils";
import { parseHandleId } from "../../../../lib/graphUtils";
import { getAnyValueFromNode } from "../../utils/getValueFromNode";

function valuesEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!valuesEqual(a[i], b[i])) return false;
    }
    return true;
  }
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
  "midi",
  "bool",
  "trigger",
  "subgraph_input",
  "subgraph",
  "tempo",
  "prompt_list",
  "prompt_blend",
]);

const UI_INPUT_TYPES = new Set<FlowNodeData["nodeType"]>([
  "primitive",
  "slider",
  "knobs",
  "xypad",
  "tuple",
  "reroute",
  "vace",
  "pipeline",
  "record",
  "prompt_list",
  "prompt_blend",
]);

function resolveSubgraphTarget(
  sgNode: Node<FlowNodeData>,
  portName: string,
  allNodes: Node<FlowNodeData>[],
  prefix: string
): { backendId: string; paramName: string } | null {
  const ports: SubgraphPort[] = sgNode.data.subgraphInputs ?? [];
  const port = ports.find(p => p.name === portName);
  if (!port) return null;

  const sgPrefix = prefix ? `${prefix}${sgNode.id}:` : `${sgNode.id}:`;
  const innerHandleParsed = parseHandleId(port.innerHandleId);
  if (!innerHandleParsed) return null;
  const innerParamName = innerHandleParsed.name;

  const innerNodes = sgNode.data.subgraphNodes ?? [];
  const innerNodeData = innerNodes.find(n => n.id === port.innerNodeId);

  if (!innerNodeData) {
    return {
      backendId: sgPrefix + port.innerNodeId,
      paramName: innerParamName,
    };
  }

  const innerType =
    innerNodeData.type || (innerNodeData.data.nodeType as string);

  if (innerType === "subgraph") {
    const nestedSgNode: Node<FlowNodeData> = {
      id: port.innerNodeId,
      type: "subgraph",
      position: { x: 0, y: 0 },
      data: innerNodeData.data as FlowNodeData,
    };
    return resolveSubgraphTarget(
      nestedSgNode,
      innerParamName,
      allNodes,
      sgPrefix
    );
  }

  if (innerType === "pipeline") {
    return {
      backendId: sgPrefix + port.innerNodeId,
      paramName: innerParamName,
    };
  }

  return null;
}

export function useValueForwarding(
  nodes: Node<FlowNodeData>[],
  edges: Edge[],
  findConnectedPipelineParams: (
    sourceNodeId: string,
    edges: Edge[],
    nodes: Node<FlowNodeData>[]
  ) => Array<{ nodeId: string; paramName: string }>,
  resolveBackendId: (nodeId: string) => string,
  isStreaming: boolean,
  onNodeParamChangeRef: React.RefObject<
    ((nodeId: string, key: string, value: unknown) => void) | undefined
  >,
  setNodes?: React.Dispatch<React.SetStateAction<Node<FlowNodeData>[]>>
) {
  const lastForwardTimeRef = useRef<Record<string, number>>({});

  // Track previous node data to skip backend sends when only positions changed
  const prevNodeDataRef = useRef<Map<string, FlowNodeData>>(new Map());

  // Dedup: tracks last value sent per (backendId:paramName) to prevent
  // duplicate sends when the RAF loop's setNodes triggers a re-render
  // that re-fires this effect with identical producer values.
  const lastSentRef = useRef<Map<string, unknown>>(new Map());

  // Detect streaming session start (false→true) and clear dedup state so the
  // new backend session receives all parameter values, even if they haven't
  // changed since the previous session.
  const wasStreamingRef = useRef(false);
  const sessionTick = useMemo(() => {
    if (isStreaming && !wasStreamingRef.current) {
      wasStreamingRef.current = true;
      return Date.now();
    }
    if (!isStreaming) wasStreamingRef.current = false;
    return 0;
  }, [isStreaming]);

  useEffect(() => {
    if (!isStreaming || !onNodeParamChangeRef.current) return;

    // On new session start, clear dedup state so all params are re-sent.
    if (sessionTick > 0) {
      lastSentRef.current.clear();
      prevNodeDataRef.current.clear();
    }

    const sendParam = (
      backendId: string,
      paramName: string,
      value: unknown
    ) => {
      const dedupKey = `${backendId}\0${paramName}`;
      if (valuesEqual(lastSentRef.current.get(dedupKey), value)) return;
      lastSentRef.current.set(dedupKey, value);
      onNodeParamChangeRef.current!(backendId, paramName, value);
    };

    // Check if any producer node data actually changed (not just positions)
    let anyDataChanged = false;
    const prevData = prevNodeDataRef.current;
    const nextData = new Map<string, FlowNodeData>();
    for (const node of nodes) {
      if (!PRODUCER_TYPES.has(node.data.nodeType)) continue;
      nextData.set(node.id, node.data);
      if (prevData.get(node.id) !== node.data) {
        anyDataChanged = true;
      }
    }
    // Also detect removed nodes
    if (nextData.size !== prevData.size) anyDataChanged = true;
    prevNodeDataRef.current = nextData;
    if (!anyDataChanged) return;

    const throttleMs = 100;

    for (const node of nodes) {
      if (!PRODUCER_TYPES.has(node.data.nodeType)) continue;

      const connected = findConnectedPipelineParams(node.id, edges, nodes);
      if (connected.length === 0) continue;

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
        const mediaHandleName =
          node.data.mediaType === "video" ? "video_value" : "value";
        valuesToForward.push({
          handleName: mediaHandleName,
          value: node.data.imagePath || "",
        });
      } else if (node.data.nodeType === "midi") {
        const midiChannels = node.data.midiChannels;
        if (midiChannels) {
          for (let i = 0; i < midiChannels.length; i++) {
            valuesToForward.push({
              handleName: `midi_${i}`,
              value: midiChannels[i].value,
            });
          }
        }
      } else if (
        node.data.nodeType === "bool" ||
        node.data.nodeType === "trigger"
      ) {
        valuesToForward.push({ handleName: "value", value: node.data.value });
      } else if (node.data.nodeType === "tempo") {
        valuesToForward.push({
          handleName: "bpm",
          value: node.data.tempoBpm ?? 0,
        });
        valuesToForward.push({
          handleName: "beat_phase",
          value: node.data.tempoBeatPhase ?? 0,
        });
        valuesToForward.push({
          handleName: "beat_count",
          value:
            ((node.data.tempoBeatCount as number) ?? 0) -
            ((node.data.tempoBeatCountOffset as number) ?? 0),
        });
        valuesToForward.push({
          handleName: "bar_position",
          value: node.data.tempoBarPosition ?? 0,
        });
        valuesToForward.push({
          handleName: "is_playing",
          value: (node.data.tempoIsPlaying as boolean) ? 1 : 0,
        });
      } else if (node.data.nodeType === "prompt_list") {
        valuesToForward.push({
          handleName: "prompt",
          value: node.data.promptListActiveText ?? "",
        });
      } else if (node.data.nodeType === "prompt_blend") {
        valuesToForward.push({
          handleName: "prompts",
          value: node.data.promptBlendItems ?? [],
        });
      } else if (
        node.data.nodeType === "subgraph_input" ||
        node.data.nodeType === "subgraph"
      ) {
        const pv = (node.data.portValues ?? {}) as Record<string, unknown>;
        for (const [key, val] of Object.entries(pv)) {
          valuesToForward.push({ handleName: key, value: val });
        }
      }

      const isAnimated =
        node.data.nodeType === "control" ||
        node.data.nodeType === "math" ||
        node.data.nodeType === "tempo";
      if (isAnimated) {
        const now = Date.now();
        const lastTime = lastForwardTimeRef.current[node.id] || 0;
        if (now - lastTime < throttleMs) continue;
        lastForwardTimeRef.current[node.id] = now;
      }

      for (const edge of edges) {
        if (edge.source !== node.id) continue;
        const sourceParsed = parseHandleId(edge.sourceHandle);
        const targetParsed = parseHandleId(edge.targetHandle);
        if (!sourceParsed || sourceParsed.kind !== "param") continue;
        if (!targetParsed || targetParsed.kind !== "param") continue;

        const targetNode = nodes.find(n => n.id === edge.target);
        if (!targetNode) continue;

        let resolvedBackendId: string | null = null;
        let resolvedParamName: string | null = null;

        if (targetNode.data.nodeType === "subgraph") {
          const result = resolveSubgraphTarget(
            targetNode,
            targetParsed.name,
            nodes,
            ""
          );
          if (result) {
            resolvedBackendId = result.backendId;
            resolvedParamName = result.paramName;
          } else continue;
        } else if (targetNode.data.nodeType === "pipeline") {
          resolvedBackendId = resolveBackendId(edge.target);
          resolvedParamName = targetParsed.name;
        } else continue;

        if (
          node.data.nodeType === "vace" &&
          sourceParsed.name === "__vace" &&
          targetParsed.name === "__vace"
        ) {
          const backendId = resolvedBackendId;
          const ctxScale =
            typeof node.data.vaceContextScale === "number"
              ? node.data.vaceContextScale
              : 1.0;
          sendParam(backendId, "vace_context_scale", ctxScale);

          sendParam(backendId, "vace_use_input_video", false);
          const refImg = (node.data.vaceRefImage as string) || "";
          if (refImg) sendParam(backendId, "vace_ref_images", [refImg]);
          const firstFrame = (node.data.vaceFirstFrame as string) || "";
          if (firstFrame) sendParam(backendId, "first_frame_image", firstFrame);
          const lastFrame = (node.data.vaceLastFrame as string) || "";
          if (lastFrame) sendParam(backendId, "last_frame_image", lastFrame);
          continue;
        }

        const entry = valuesToForward.find(v => {
          if (v.handleName === null) return true; // single-output node
          return v.handleName === sourceParsed.name;
        });
        if (!entry || entry.value === undefined) continue;

        if (resolvedParamName === "__prompt") {
          if (Array.isArray(entry.value)) {
            sendParam(resolvedBackendId, "prompts", entry.value);
            const srcNode = nodes.find(n => n.id === edge.source);
            if (srcNode?.data.nodeType === "prompt_blend") {
              sendParam(
                resolvedBackendId,
                "prompt_interpolation_method",
                srcNode.data.promptBlendMethod ?? "linear"
              );
            }
          } else {
            sendParam(resolvedBackendId, "prompts", [
              { text: String(entry.value), weight: 100 },
            ]);
          }
        } else {
          sendParam(resolvedBackendId, resolvedParamName, entry.value);
        }

        // Auto-forward tempo meta-settings to connected pipelines
        if (
          node.data.nodeType === "tempo" &&
          (targetNode.data.nodeType === "pipeline" ||
            targetNode.data.nodeType === "subgraph")
        ) {
          sendParam(
            resolvedBackendId,
            "quantize_mode",
            node.data.tempoQuantizeMode ?? "none"
          );
          sendParam(
            resolvedBackendId,
            "lookahead_ms",
            node.data.tempoLookaheadMs ?? 0
          );
          sendParam(
            resolvedBackendId,
            "beat_cache_reset_rate",
            node.data.tempoBeatResetRate ?? "none"
          );
        }
      }
    }
  }, [
    nodes,
    edges,
    findConnectedPipelineParams,
    resolveBackendId,
    isStreaming,
    sessionTick,
    onNodeParamChangeRef,
  ]);

  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  useEffect(() => {
    if (!setNodes) return;

    let handle = 0;
    const tick = () => {
      const currentNodes = nodesRef.current;
      const currentEdges = edgesRef.current;

      const updates = new Map<string, Record<string, unknown>>();

      for (const edge of currentEdges) {
        const targetNode = currentNodes.find(n => n.id === edge.target);
        if (!targetNode || !UI_INPUT_TYPES.has(targetNode.data.nodeType))
          continue;

        const targetParsed = parseHandleId(edge.targetHandle);
        if (!targetParsed || targetParsed.kind !== "param") continue;

        const sourceNode = currentNodes.find(n => n.id === edge.source);
        if (!sourceNode) continue;

        const sourceParsed = parseHandleId(edge.sourceHandle);
        if (!sourceParsed || sourceParsed.kind !== "param") continue;

        const sourceValue = getAnyValueFromNode(sourceNode, edge.sourceHandle);
        if (sourceValue === undefined || sourceValue === null) continue;

        const nodeUpdates = updates.get(edge.target) ?? {};

        if (
          targetNode.data.nodeType === "primitive" &&
          targetParsed.name === "value"
        ) {
          nodeUpdates["value"] = sourceValue;
        } else if (
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
            nodeUpdates["tupleValues"] = sourceValue as number[];
          } else if (
            targetParsed.name.startsWith("row_") &&
            typeof sourceValue === "number"
          ) {
            const rowIdx = parseInt(targetParsed.name.replace("row_", ""), 10);
            const tupleValues = targetNode.data.tupleValues;
            if (tupleValues && !isNaN(rowIdx) && rowIdx < tupleValues.length) {
              const clamped = Math.min(
                Math.max(sourceValue, targetNode.data.tupleMin ?? 0),
                targetNode.data.tupleMax ?? 1000
              );
              const currentValue = tupleValues[rowIdx];
              if (Math.abs(clamped - currentValue) > 0.0001) {
                const existingValues = (nodeUpdates[
                  "tupleValues"
                ] as number[]) ?? [...tupleValues];
                existingValues[rowIdx] = clamped;
                nodeUpdates["tupleValues"] = existingValues;
              }
            }
          }
        } else if (targetNode.data.nodeType === "vace") {
          if (targetParsed.name === "ref_image") {
            nodeUpdates["vaceRefImage"] = String(sourceValue);
          } else if (targetParsed.name === "first_frame") {
            nodeUpdates["vaceFirstFrame"] = String(sourceValue);
          } else if (targetParsed.name === "last_frame") {
            nodeUpdates["vaceLastFrame"] = String(sourceValue);
          }
        } else if (targetNode.data.nodeType === "reroute") {
          nodeUpdates["value"] = sourceValue;
        } else if (
          targetNode.data.nodeType === "record" &&
          targetParsed.name === "trigger"
        ) {
          nodeUpdates["triggerValue"] = Boolean(sourceValue);
        } else if (
          targetNode.data.nodeType === "prompt_list" &&
          targetParsed.name === "cycle"
        ) {
          const newIntVal = Math.floor(Number(sourceValue));
          const prevIntVal = Math.floor(
            Number(targetNode.data.promptListCycleValue ?? -1)
          );
          if (newIntVal !== prevIntVal && prevIntVal >= 0) {
            const items = (targetNode.data.promptListItems as string[]) ?? [""];
            const nextIdx =
              (((targetNode.data.promptListActiveIndex as number) ?? 0) + 1) %
              items.length;
            nodeUpdates["promptListActiveIndex"] = nextIdx;
            nodeUpdates["promptListActiveText"] = items[nextIdx] ?? "";
          }
          nodeUpdates["promptListCycleValue"] = newIntVal;
        } else if (targetNode.data.nodeType === "prompt_blend") {
          const items = [
            ...((targetNode.data.promptBlendItems as Array<{
              text: string;
              weight: number;
            }>) ?? []),
          ];
          if (targetParsed.name.startsWith("prompt_")) {
            const idx = parseInt(targetParsed.name.replace("prompt_", ""), 10);
            if (!isNaN(idx) && idx < items.length) {
              items[idx] = { ...items[idx], text: String(sourceValue) };
              nodeUpdates["promptBlendItems"] = items;
            }
          } else if (targetParsed.name.startsWith("weight_")) {
            const idx = parseInt(targetParsed.name.replace("weight_", ""), 10);
            if (!isNaN(idx) && idx < items.length) {
              const clamped = Math.max(
                0,
                Math.min(100, Math.round(Number(sourceValue) || 0))
              );
              const remaining = 100 - clamped;
              const otherSum = items.reduce(
                (sum, p, j) => (j === idx ? sum : sum + p.weight),
                0
              );
              items[idx] = { ...items[idx], weight: clamped };
              if (otherSum > 0) {
                for (let j = 0; j < items.length; j++) {
                  if (j !== idx) {
                    items[j] = {
                      ...items[j],
                      weight: Math.round(
                        (items[j].weight / otherSum) * remaining
                      ),
                    };
                  }
                }
              } else if (items.length > 1) {
                const even = Math.round(remaining / (items.length - 1));
                for (let j = 0; j < items.length; j++) {
                  if (j !== idx) {
                    items[j] = { ...items[j], weight: even };
                  }
                }
              }
              nodeUpdates["promptBlendItems"] = items;
            }
          }
        } else if (targetNode.data.nodeType === "pipeline") {
          // Update parameterValues so the greyed-out pill shows live values
          const prevParams = (nodeUpdates["parameterValues"] as Record<
            string,
            unknown
          >) ?? {
            ...((targetNode.data.parameterValues as Record<string, unknown>) ??
              {}),
          };
          prevParams[targetParsed.name] = sourceValue;
          nodeUpdates["parameterValues"] = prevParams;
          if (targetParsed.name === "__prompt") {
            nodeUpdates["promptText"] = String(sourceValue);
          }
        }

        if (Object.keys(nodeUpdates).length > 0) {
          updates.set(edge.target, nodeUpdates);
        }
      }

      const vaceHandleFields: Record<string, string> = {
        ref_image: "vaceRefImage",
        first_frame: "vaceFirstFrame",
        last_frame: "vaceLastFrame",
      };
      for (const node of currentNodes) {
        if (node.data.nodeType !== "vace") continue;
        for (const [handleName, dataField] of Object.entries(
          vaceHandleFields
        )) {
          const handleId = `param:${handleName}`;
          const hasEdge = currentEdges.some(
            e => e.target === node.id && e.targetHandle === handleId
          );
          if (!hasEdge && node.data[dataField]) {
            const nodeUpdates = updates.get(node.id) ?? {};
            nodeUpdates[dataField] = "";
            updates.set(node.id, nodeUpdates);
          }
        }
      }

      if (updates.size > 0) {
        setNodes(nds => {
          let anyNodeChanged = false;
          const result = nds.map(n => {
            const upd = updates.get(n.id);
            if (!upd) return n;

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
      }

      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);

    return () => cancelAnimationFrame(handle);
  }, [setNodes]);
}

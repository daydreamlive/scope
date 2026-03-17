import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId, parseHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useHandlePositions } from "../hooks/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePillInput,
  NodePill,
  NODE_TOKENS,
} from "../ui";

type BoolNodeType = Node<FlowNodeData, "bool">;

const COLOR = "#34d399"; // emerald-400

const MODE_OPTIONS = [
  { value: "gate", label: "Gate" },
  { value: "toggle", label: "Toggle" },
];

function getValueFromNode(
  node: Node<FlowNodeData>,
  sourceHandleId?: string | null
): number | null {
  if (node.data.nodeType === "primitive" || node.data.nodeType === "reroute") {
    const val = node.data.value;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "control" || node.data.nodeType === "math") {
    const val = node.data.currentValue;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "slider") {
    const val = node.data.value;
    if (typeof val === "number") return val;
    return null;
  }
  if (node.data.nodeType === "knobs") {
    const knobs = node.data.knobs;
    if (!knobs || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("knob_", ""), 10);
    if (isNaN(idx) || idx >= knobs.length) return null;
    return knobs[idx].value;
  }
  if (node.data.nodeType === "xypad") {
    if (!sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    if (parsed.name === "x") return node.data.padX ?? null;
    if (parsed.name === "y") return node.data.padY ?? null;
    return null;
  }
  if (node.data.nodeType === "midi") {
    const channels = node.data.midiChannels;
    if (!channels || !sourceHandleId) return null;
    const parsed = parseHandleId(sourceHandleId);
    if (!parsed) return null;
    const idx = parseInt(parsed.name.replace("midi_", ""), 10);
    if (isNaN(idx) || idx >= channels.length) return null;
    return channels[idx].value;
  }
  return null;
}

export function BoolNode({ id, data, selected }: NodeProps<BoolNodeType>) {
  const { updateData } = useNodeData(id);
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const mode = data.boolMode || "gate";
  const threshold = data.boolThreshold ?? 0;
  const currentOutput = Boolean(data.value);

  // Find the input edge
  const inputEdge = edges.find(
    e => e.target === id && e.targetHandle === buildHandleId("param", "input")
  );

  // Get source node & value
  const sourceNode = inputEdge
    ? allNodes.find(n => n.id === inputEdge.source)
    : null;
  const inputValue = sourceNode
    ? getValueFromNode(sourceNode, inputEdge?.sourceHandle)
    : null;

  const isAboveThreshold = inputValue !== null && inputValue > threshold;

  // Track previous "above threshold" state for toggle rising-edge detection
  const prevAboveRef = useRef(false);
  // Track the stored toggle state separately so it persists across renders
  const toggleStateRef = useRef(currentOutput);

  useEffect(() => {
    let newOutput: boolean;

    if (mode === "gate") {
      // Gate: output follows input directly
      newOutput = isAboveThreshold;
    } else {
      // Toggle: flip on rising edge (transition from below to above threshold)
      const wasAbove = prevAboveRef.current;
      if (isAboveThreshold && !wasAbove) {
        // Rising edge — flip
        toggleStateRef.current = !toggleStateRef.current;
      }
      prevAboveRef.current = isAboveThreshold;
      newOutput = toggleStateRef.current;
    }

    if (newOutput !== currentOutput) {
      updateData({ value: newOutput });
    }
  }, [mode, isAboveThreshold, currentOutput, updateData]);

  // Keep toggle ref in sync if data.value is changed externally
  useEffect(() => {
    toggleStateRef.current = Boolean(data.value);
  }, [data.value]);

  // Keep prevAboveRef in sync for gate mode too
  useEffect(() => {
    if (mode === "gate") {
      prevAboveRef.current = isAboveThreshold;
    }
  }, [mode, isAboveThreshold]);

  const { setRowRef, rowPositions } = useHandlePositions([mode]);

  return (
    <NodeCard selected={selected}>
      <NodeHeader
        title={data.customTitle || "Bool"}
        dotColor="bg-emerald-400"
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      <NodeBody withGap>
        <NodeParamRow label="Mode">
          <NodePillSelect
            value={mode}
            onChange={v => updateData({ boolMode: v as "gate" | "toggle" })}
            options={MODE_OPTIONS}
          />
        </NodeParamRow>
        <NodeParamRow label="Threshold">
          <NodePillInput
            type="number"
            value={threshold}
            onChange={v => updateData({ boolThreshold: Number(v) })}
          />
        </NodeParamRow>
        <div ref={setRowRef("input")} className={NODE_TOKENS.paramRow}>
          <span className={NODE_TOKENS.labelText}>In</span>
          <NodePill className="opacity-75">
            {inputValue !== null ? inputValue.toFixed(3) : "—"}
          </NodePill>
        </div>
        <div ref={setRowRef("output")} className={NODE_TOKENS.paramRow}>
          <span className={NODE_TOKENS.labelText}>Out</span>
          <div className="flex items-center gap-1.5">
            <div
              className="w-2.5 h-2.5 rounded-full transition-colors"
              style={{
                backgroundColor: currentOutput ? COLOR : "#333",
                boxShadow: currentOutput ? `0 0 6px ${COLOR}` : "none",
              }}
            />
            <span
              className={`text-[10px] font-medium ${currentOutput ? "text-emerald-400" : "text-[#666]"}`}
            >
              {currentOutput ? "true" : "false"}
            </span>
          </div>
        </div>
      </NodeBody>

      {/* Input handle (number) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "input")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["input"] ?? 78,
          left: 8,
          backgroundColor: "#38bdf8", // sky-400, number color
        }}
      />

      {/* Output handle (boolean) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{
          top: rowPositions["output"] ?? 100,
          right: 8,
          backgroundColor: COLOR,
        }}
      />
    </NodeCard>
  );
}

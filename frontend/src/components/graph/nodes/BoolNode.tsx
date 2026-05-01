import { Handle, Position, useEdges, useNodes } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useEffect, useRef } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getNumberFromNode } from "../utils/getValueFromNode";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import { useConnectedNumber } from "../hooks/node/useConnectedValue";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePillInput,
  NodePill,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_BOOLEAN as COLOR, COLOR_TRIGGER } from "../nodeColors";

type BoolNodeType = Node<FlowNodeData, "bool">;

const MODE_OPTIONS = [
  { value: "gate", label: "Gate" },
  { value: "toggle", label: "Toggle" },
];

export function BoolNode({ id, data, selected }: NodeProps<BoolNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const edges = useEdges();
  const allNodes = useNodes() as Node<FlowNodeData>[];

  const mode = data.boolMode || "gate";
  const threshold = data.boolThreshold ?? 0;
  const thresholdIn = useConnectedNumber(id, "threshold", threshold);
  const effectiveThreshold = thresholdIn.value;
  const currentOutput = Boolean(data.value);

  const inputEdge = edges.find(
    e => e.target === id && e.targetHandle === buildHandleId("param", "input")
  );

  const sourceNode = inputEdge
    ? allNodes.find(n => n.id === inputEdge.source)
    : null;
  const inputValue = sourceNode
    ? getNumberFromNode(sourceNode, inputEdge?.sourceHandle)
    : null;

  const isAboveThreshold =
    inputValue !== null && inputValue > effectiveThreshold;

  const prevAboveRef = useRef(false);
  const toggleStateRef = useRef(currentOutput);

  useEffect(() => {
    // Only drive value from the numeric input when it's connected.
    // When no numeric input, the trigger handle (via useValueForwarding) controls value.
    if (!inputEdge) return;

    let newOutput: boolean;

    if (mode === "gate") {
      newOutput = isAboveThreshold;
    } else {
      const wasAbove = prevAboveRef.current;
      if (isAboveThreshold && !wasAbove) {
        toggleStateRef.current = !toggleStateRef.current;
      }
      prevAboveRef.current = isAboveThreshold;
      newOutput = toggleStateRef.current;
    }

    if (newOutput !== currentOutput) {
      updateData({ value: newOutput });
    }
  }, [mode, isAboveThreshold, currentOutput, updateData, inputEdge]);

  // Sync toggle ref if data.value changes externally
  useEffect(() => {
    toggleStateRef.current = Boolean(data.value);
  }, [data.value]);

  // Sync prevAboveRef in gate mode
  useEffect(() => {
    if (mode === "gate") {
      prevAboveRef.current = isAboveThreshold;
    }
  }, [mode, isAboveThreshold]);

  const { setRowRef, rowPositions } = useHandlePositions([
    mode,
    thresholdIn.connected,
  ]);

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Bool"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          <NodeParamRow label="Mode">
            <NodePillSelect
              value={mode}
              onChange={v => updateData({ boolMode: v as "gate" | "toggle" })}
              options={MODE_OPTIONS}
            />
          </NodeParamRow>
          <div ref={setRowRef("threshold")}>
            <NodeParamRow label="Threshold">
              {thresholdIn.connected ? (
                <NodePill className="opacity-50">{effectiveThreshold}</NodePill>
              ) : (
                <NodePillInput
                  type="number"
                  value={threshold}
                  onChange={v => updateData({ boolThreshold: Number(v) })}
                />
              )}
            </NodeParamRow>
          </div>
          <div ref={setRowRef("trigger")} className={NODE_TOKENS.paramRow}>
            <span className={NODE_TOKENS.labelText}>Trigger</span>
            <NodePill className="opacity-50">—</NodePill>
          </div>
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
      )}

      {/* Threshold input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "threshold")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["threshold"] ?? 56,
                left: 0,
                backgroundColor: "#38bdf8",
              }
        }
      />

      {/* Trigger input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "trigger")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["trigger"] ?? 78,
                left: 0,
                backgroundColor: COLOR_TRIGGER,
              }
        }
      />

      {/* Input handle (number) */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "input")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["input"] ?? 100,
                left: 0,
                backgroundColor: "#38bdf8",
              }
        }
      />

      {/* Output handle (boolean) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["output"] ?? 100,
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}

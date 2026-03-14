import { useState, useRef, useEffect, useCallback } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData, SubgraphPort } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";

type SubgraphInputNodeType = Node<FlowNodeData, "subgraph_input">;

const PORT_TYPE_COLORS: Record<string, string> = {
  stream: "#eeeeee",
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
  list_number: "#38bdf8",
};

function getPortColor(port: SubgraphPort): string {
  if (port.portType === "stream") return PORT_TYPE_COLORS.stream;
  return PORT_TYPE_COLORS[port.paramType || "stream"] || "#9ca3af";
}

export const ADD_HANDLE_ID = buildHandleId("stream", "__add__");

const ROW_HEIGHT = 28;
const PAD_Y = 10;
const CURVE_R = 12;
const BRACKET_GAP = 14;
const DOT_COL_W = 14;

function rightBracketPath(height: number, depth: number): string {
  const r = Math.min(CURVE_R, height / 2);
  return [
    `M 0 0`,
    `Q ${depth} 0, ${depth} ${r}`,
    `L ${depth} ${height - r}`,
    `Q ${depth} ${height}, 0 ${height}`,
  ].join(" ");
}

/* ── Inline editable label ── */
function EditableLabel({
  value,
  onCommit,
}: {
  value: string;
  onCommit: (newValue: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = useCallback(() => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) onCommit(trimmed);
    else setDraft(value);
    setEditing(false);
  }, [draft, value, onCommit]);

  if (!editing) {
    return (
      <span
        className="text-[10px] text-[#999] select-none whitespace-nowrap pr-1.5 cursor-text"
        onDoubleClick={e => {
          e.stopPropagation();
          setDraft(value);
          setEditing(true);
        }}
        title="Double-click to rename"
      >
        {value}
      </span>
    );
  }

  return (
    <input
      ref={inputRef}
      className="text-[10px] text-[#ccc] bg-[#333] border border-[#555] rounded px-0.5 outline-none w-16"
      value={draft}
      onChange={e => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={e => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          setDraft(value);
          setEditing(false);
        }
      }}
    />
  );
}

export function SubgraphInputNode({ data }: NodeProps<SubgraphInputNodeType>) {
  const ports: SubgraphPort[] = data.subgraphInputs ?? [];

  const onPortRename = data.onPortRename as
    | ((oldName: string, newName: string, portType: string) => void)
    | undefined;

  const totalRows = ports.length + 1;
  const bracketHeight = totalRows * ROW_HEIGHT + PAD_Y;
  const bracketDepth = 10;
  const svgWidth = bracketDepth + 4;

  return (
    <div className="relative flex flex-row items-start">
      {/* Labels + dots column — right-aligned so dots pin to the right edge */}
      <div className="flex flex-col items-end" style={{ paddingTop: PAD_Y }}>
        {ports.map(port => {
          const isStream = port.portType === "stream";
          const dotSize = isStream ? 10 : 8;
          return (
            <div
              key={port.name}
              className="flex items-center"
              style={{ height: ROW_HEIGHT }}
            >
              <EditableLabel
                value={port.name}
                onCommit={newName =>
                  onPortRename?.(port.name, newName, port.portType)
                }
              />
              <span
                className="flex items-center justify-center shrink-0"
                style={{ width: DOT_COL_W }}
              >
                <span
                  className="inline-block rounded-full"
                  style={{
                    backgroundColor: getPortColor(port),
                    width: dotSize,
                    height: dotSize,
                  }}
                />
              </span>
            </div>
          );
        })}
        {/* "+" row */}
        <div
          className="flex items-center justify-end"
          style={{ height: ROW_HEIGHT }}
        >
          <span
            className="flex items-center justify-center shrink-0"
            style={{ width: DOT_COL_W }}
          >
            <span
              className="inline-flex items-center justify-center rounded-full"
              style={{
                width: 10,
                height: 10,
                border: "2px dashed #555",
              }}
            />
          </span>
        </div>
      </div>

      {/* Bracket gap */}
      <div style={{ width: BRACKET_GAP }} />

      {/* SVG bracket — overflow visible so curves don't clip */}
      <svg
        className="shrink-0"
        style={{
          marginTop: PAD_Y / 2,
          width: svgWidth,
          height: bracketHeight,
          overflow: "visible",
        }}
        viewBox={`0 0 ${svgWidth} ${bracketHeight}`}
        fill="none"
      >
        <path
          d={rightBracketPath(bracketHeight, bracketDepth)}
          stroke="rgba(255,255,255,0.35)"
          strokeWidth={2}
          fill="none"
        />
      </svg>

      {/* Invisible React Flow Handles overlaid on dots */}
      {ports.map((port, i) => {
        const handleId = buildHandleId(port.portType, port.name);
        const color = getPortColor(port);
        const isStream = port.portType === "stream";
        const dotSize = isStream ? 10 : 8;
        return (
          <Handle
            key={port.name}
            type="source"
            position={Position.Right}
            id={handleId}
            style={{
              position: "absolute",
              top: PAD_Y + i * ROW_HEIGHT + ROW_HEIGHT / 2,
              right: svgWidth + BRACKET_GAP + DOT_COL_W / 2 - dotSize / 2,
              backgroundColor: color,
              width: dotSize,
              height: dotSize,
              border: "none",
              opacity: 0,
            }}
          />
        );
      })}

      {/* "+" add handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={ADD_HANDLE_ID}
        style={{
          position: "absolute",
          top: PAD_Y + ports.length * ROW_HEIGHT + ROW_HEIGHT / 2,
          right: svgWidth + BRACKET_GAP + DOT_COL_W / 2 - 5,
          backgroundColor: "transparent",
          width: 10,
          height: 10,
          border: "none",
          opacity: 0,
        }}
      />
    </div>
  );
}

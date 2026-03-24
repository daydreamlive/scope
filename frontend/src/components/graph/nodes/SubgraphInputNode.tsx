import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData, SubgraphPort } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { EditableLabel } from "../ui/EditableLabel";
import { PARAM_TYPE_COLORS, COLOR_DEFAULT } from "../nodeColors";

type SubgraphInputNodeType = Node<FlowNodeData, "subgraph_input">;

function getPortColor(port: SubgraphPort): string {
  if (port.portType === "stream") return PARAM_TYPE_COLORS.stream;
  return PARAM_TYPE_COLORS[port.paramType || "stream"] || COLOR_DEFAULT;
}

const ADD_HANDLE_ID = buildHandleId("stream", "__add__");

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

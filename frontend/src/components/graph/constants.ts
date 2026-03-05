import type { Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { parseHandleId } from "../../lib/graphUtils";

export const HANDLE_COLORS: Record<string, string> = {
  video: "#ffffff",
  video2: "#ffffff",
  vace_input_frames: "#a78bfa",
  vace_input_masks: "#f472b6",
  source: "#4ade80",
  sink: "#fb923c",
};

export const PARAM_TYPE_COLORS: Record<string, string> = {
  string: "#fbbf24",
  number: "#38bdf8",
  boolean: "#34d399",
  float: "#a78bfa",
  int: "#a78bfa",
};

export function getEdgeColor(
  sourceNode: Node<FlowNodeData> | undefined,
  handleId: string | null | undefined
): string {
  if (!sourceNode || !handleId) return "#9ca3af";

  const parsed = parseHandleId(handleId);
  if (!parsed) return "#9ca3af";

  if (parsed.kind === "param") {
    if (sourceNode.data.nodeType === "value") {
      const valueType = sourceNode.data.valueType;
      return PARAM_TYPE_COLORS[valueType || "string"] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "control") {
      const controlType = sourceNode.data.controlType;
      const outputType = controlType === "string" ? "string" : "number";
      return PARAM_TYPE_COLORS[outputType] || "#9ca3af";
    }
    if (sourceNode.data.nodeType === "math") {
      return PARAM_TYPE_COLORS["number"] || "#9ca3af";
    }
    return "#9ca3af";
  }

  if (sourceNode.data.nodeType === "pipeline") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }

  if (sourceNode.data.nodeType === "source") {
    return HANDLE_COLORS[parsed.name] || HANDLE_COLORS.video;
  }
  if (sourceNode.data.nodeType === "sink") {
    return HANDLE_COLORS.sink;
  }

  return HANDLE_COLORS.video;
}


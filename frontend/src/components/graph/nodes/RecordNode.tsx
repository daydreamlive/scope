import { useCallback, useEffect, useRef, useState } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { NodeCard, NodeHeader, collapsedHandleStyle } from "../ui";

type RecordNodeType = Node<FlowNodeData, "record">;

const HEADER_H = 28;

function formatElapsed(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0)
    return `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return `${m}:${String(s).padStart(2, "0")}`;
}

export function RecordNode({ id, data, selected }: NodeProps<RecordNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const isStreaming = !!data.isStreaming;

  const onStartRecording = data.onStartRecording as (() => void) | undefined;
  const onStopRecording = data.onStopRecording as (() => void) | undefined;

  const [isRecording, setIsRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const startTimeRef = useRef<number | null>(null);

  // Reset recording state when stream stops
  useEffect(() => {
    if (!isStreaming) {
      setIsRecording(false);
      startTimeRef.current = null;
      setElapsed(0);
    }
  }, [isStreaming]);

  // Elapsed timer
  useEffect(() => {
    if (isRecording) {
      startTimeRef.current = Date.now();
      setElapsed(0);
      const interval = setInterval(() => {
        if (startTimeRef.current) {
          setElapsed(Math.floor((Date.now() - startTimeRef.current) / 1000));
        }
      }, 1000);
      return () => clearInterval(interval);
    } else {
      startTimeRef.current = null;
      setElapsed(0);
    }
  }, [isRecording]);

  const handleStart = useCallback(() => {
    onStartRecording?.();
    setIsRecording(true);
  }, [onStartRecording]);

  const handleStop = useCallback(() => {
    onStopRecording?.();
    setIsRecording(false);
  }, [onStopRecording]);

  // Rising-edge detection for external trigger input
  const triggerValue = Boolean(data.triggerValue);
  const prevTriggerRef = useRef(false);
  useEffect(() => {
    const prev = prevTriggerRef.current;
    prevTriggerRef.current = triggerValue;
    if (triggerValue && !prev) {
      if (isRecording) {
        handleStop();
      } else if (isStreaming) {
        handleStart();
      }
    }
  }, [triggerValue, isRecording, isStreaming, handleStart, handleStop]);

  const handleY = HEADER_H + 30;
  const triggerHandleY = handleY + 20;

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Record"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <div className="px-3 pb-3 pt-1 flex flex-col gap-2">
          {/* Status indicator */}
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2.5 h-2.5 rounded-full shrink-0 ${
                isRecording
                  ? "bg-red-500 animate-pulse"
                  : "bg-[#555] opacity-60"
              }`}
            />
            <span className="text-[11px] font-mono text-[#ccc] tabular-nums">
              {isRecording ? formatElapsed(elapsed) : "Idle"}
            </span>
          </div>

          {/* Start / Stop buttons */}
          <div className="flex gap-1.5">
            <button
              onClick={handleStart}
              disabled={!isStreaming || isRecording}
              className="flex-1 text-[11px] font-medium px-2 py-1 rounded
                bg-red-500/20 hover:bg-red-500/30 text-red-400
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-colors"
            >
              Start
            </button>
            <button
              onClick={handleStop}
              disabled={!isRecording}
              className="flex-1 text-[11px] font-medium px-2 py-1 rounded
                bg-white/10 hover:bg-white/15 text-[#ccc]
                disabled:opacity-30 disabled:cursor-not-allowed
                transition-colors"
            >
              Stop
            </button>
          </div>
        </div>
      )}
      <Handle
        type="target"
        position={Position.Left}
        id="stream:video"
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: handleY, left: 0, backgroundColor: "#ef4444" }
        }
      />
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "trigger")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("left")
            : { top: triggerHandleY, left: 0, backgroundColor: "#34d399" }
        }
      />
    </NodeCard>
  );
}

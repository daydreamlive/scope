import { useEffect, useRef } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { useNodeCollapse } from "../hooks/useNodeCollapse";
import { NodeCard, NodeHeader, collapsedHandleStyle } from "../ui";

type SinkNodeType = Node<FlowNodeData, "sink">;

const HEADER_H = 28;
const BODY_PAD = 6;
const PREVIEW_H = 120;

export function SinkNode({ id, data, selected }: NodeProps<SinkNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const remoteStream = data.remoteStream as MediaStream | null | undefined;
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && remoteStream instanceof MediaStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const handleY = HEADER_H + BODY_PAD + PREVIEW_H / 2;

  return (
    <NodeCard selected={selected} collapsed={collapsed}>
      <NodeHeader
        title={data.customTitle || "Sink"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <div className="p-2 flex-1 min-h-0 flex flex-col">
          <div className="relative rounded-md overflow-hidden bg-black/50 flex-1 min-h-[60px]">
            {remoteStream ? (
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />
            ) : (
              <div className="flex items-center justify-center h-full text-[10px] text-[#8c8c8d]">
                No output stream
              </div>
            )}
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
            : { top: handleY, left: 0, backgroundColor: "#ffffff" }
        }
      />
      <Handle
        type="source"
        position={Position.Right}
        id="stream:out"
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("right"), opacity: 0 }
            : { top: handleY, right: 0, backgroundColor: "#ffffff" }
        }
      />
    </NodeCard>
  );
}

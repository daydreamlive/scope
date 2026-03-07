import { useEffect, useRef } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { NodeCard, NodeHeader } from "../ui";

type SinkNodeType = Node<FlowNodeData, "sink">;

const HEADER_H = 28;
const BODY_PAD = 6;
const PREVIEW_H = 120;

export function SinkNode({ id, data, selected }: NodeProps<SinkNodeType>) {
  const { setNodes } = useReactFlow();
  const remoteStream = data.remoteStream as MediaStream | null | undefined;
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && remoteStream instanceof MediaStream) {
      videoRef.current.srcObject = remoteStream;
    }
  }, [remoteStream]);

  const handleY = HEADER_H + BODY_PAD + PREVIEW_H / 2;

  return (
    <NodeCard selected={selected}>
      <NodeHeader
        title={data.customTitle || "Sink"}
        dotColor="bg-orange-400"
        onTitleChange={newTitle =>
          setNodes(nds =>
            nds.map(n =>
              n.id === id
                ? { ...n, data: { ...n.data, customTitle: newTitle } }
                : n
            )
          )
        }
      />
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
      <Handle
        type="target"
        position={Position.Left}
        id="stream:video"
        className="!w-2 !h-2 !border-0"
        style={{ top: handleY, left: 8, backgroundColor: "#ffffff" }}
      />
      <Handle
        type="source"
        position={Position.Right}
        id="stream:out"
        className="!w-2 !h-2 !border-0"
        style={{ top: handleY, right: 8, backgroundColor: "#ffffff" }}
      />
    </NodeCard>
  );
}

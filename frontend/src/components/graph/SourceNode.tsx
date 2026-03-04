import { useEffect, useRef, useCallback } from "react";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../lib/graphUtils";
import { NodeCard, NodeHeader, NodeBody, NodeParamRow, NodePillSelect } from "./node-ui";

type SourceNodeType = Node<FlowNodeData, "source">;

const HEADER_H = 28;
const BODY_PAD = 6;
const SELECT_ROW_H = 20;
const PREVIEW_H = 120;

const SOURCE_MODE_OPTIONS = [
  { value: "video", label: "File" },
  { value: "camera", label: "Camera" },
  { value: "spout", label: "Spout" },
  { value: "ndi", label: "NDI" },
];

export function SourceNode({ id, data }: NodeProps<SourceNodeType>) {
  const { setNodes } = useReactFlow();
  const sourceMode = data.sourceMode || "video";
  const localStream = data.localStream as MediaStream | null | undefined;
  const onVideoFileUpload = data.onVideoFileUpload as ((file: File) => Promise<boolean>) | undefined;

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Attach local stream to <video> element
  useEffect(() => {
    if (videoRef.current && localStream) {
      videoRef.current.srcObject = localStream;
    }
  }, [localStream]);

  const handleSourceModeChange = (newMode: string) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            sourceMode: newMode as "video" | "camera" | "spout" | "ndi",
          },
        };
      })
    );
  };

  const handleFileClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file || !onVideoFileUpload) return;
      await onVideoFileUpload(file);
      // Reset file input so the same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = "";
    },
    [onVideoFileUpload]
  );

  const showPreview = sourceMode === "video" || sourceMode === "camera";
  const showFilePicker = sourceMode === "video";

  // Handle output position: after header + body content
  const handleY = HEADER_H + BODY_PAD + SELECT_ROW_H / 2;

  return (
    <NodeCard>
      <NodeHeader title="Source" dotColor="bg-green-400" />
      <NodeBody withGap>
        <NodeParamRow label="Source">
          <NodePillSelect
            value={sourceMode}
            onChange={handleSourceModeChange}
            options={SOURCE_MODE_OPTIONS}
          />
        </NodeParamRow>

        {showPreview && (
          <div className="relative rounded-md overflow-hidden bg-black/50" style={{ height: PREVIEW_H }}>
            {localStream ? (
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                autoPlay
                muted
                playsInline
              />
            ) : (
              <div className="flex items-center justify-center h-full text-[10px] text-[#8c8c8d]">
                {sourceMode === "camera" ? "Camera preview" : "No video loaded"}
              </div>
            )}

            {showFilePicker && (
              <>
                <button
                  type="button"
                  onClick={handleFileClick}
                  className="absolute bottom-1 right-1 bg-[#2a2a2a]/80 hover:bg-[#2a2a2a] text-[#fafafa] text-[9px] px-2 py-0.5 rounded border border-[rgba(119,119,119,0.35)] transition-colors"
                >
                  Choose file
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
              </>
            )}
          </div>
        )}

        {!showPreview && (
          <div
            className="flex items-center justify-center rounded-md bg-black/30 text-[10px] text-[#8c8c8d]"
            style={{ height: 40 }}
          >
            {sourceMode === "spout" ? "Spout input (server-side)" : "NDI input (server-side)"}
          </div>
        )}
      </NodeBody>
      <Handle
        type="source"
        position={Position.Right}
        id="stream:video"
        className="!w-2 !h-2 !border-0"
        style={{ top: handleY, right: 8, backgroundColor: "#ffffff" }}
      />
    </NodeCard>
  );
}

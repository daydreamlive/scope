import { useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { Handle, Position, useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { ImageIcon, Film, X } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { getAssetUrl } from "../../../lib/api";
import { MediaPicker, isVideoAsset } from "../../MediaPicker";
import { NodeCard, NodeHeader, NODE_TOKENS, collapsedHandleStyle } from "../ui";

type ImageNodeType = Node<FlowNodeData, "image">;

const IMAGE_COLOR = "#f472b6"; // pink-400
const VIDEO_COLOR = "#38bdf8"; // sky-400

export function ImageNode({ id, data, selected }: NodeProps<ImageNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const { setEdges } = useReactFlow<Node<FlowNodeData>>();
  const [isMediaPickerOpen, setIsMediaPickerOpen] = useState(false);

  const imagePath = (data.imagePath as string) || "";
  const mediaType = data.mediaType || "image";
  const isVideo = mediaType === "video";

  /** Remove outgoing edges from this node whose sourceHandle no longer matches. */
  const cleanupStaleEdges = useCallback(
    (newType: "image" | "video") => {
      const activeHandleId =
        newType === "video"
          ? buildHandleId("param", "video_value")
          : buildHandleId("param", "value");
      setEdges(eds =>
        eds.filter(e => e.source !== id || e.sourceHandle === activeHandleId)
      );
    },
    [id, setEdges]
  );

  const handleSelectMedia = useCallback(
    (path: string) => {
      const detectedType = isVideoAsset(path) ? "video" : "image";
      // Clean up edges from the old handle if type changed
      if (detectedType !== mediaType) {
        cleanupStaleEdges(detectedType);
      }
      updateData({ imagePath: path, mediaType: detectedType });
      setIsMediaPickerOpen(false); // Auto-close on selection
    },
    [updateData, mediaType, cleanupStaleEdges]
  );

  const handleRemoveMedia = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      updateData({ imagePath: "", mediaType: "image" });
      // Clean up edges since we're resetting to image (empty)
      if (mediaType === "video") {
        cleanupStaleEdges("image");
      }
    },
    [updateData, mediaType, cleanupStaleEdges]
  );

  const handleColor = isVideo ? VIDEO_COLOR : IMAGE_COLOR;
  const handleId = isVideo
    ? buildHandleId("param", "video_value")
    : buildHandleId("param", "value");

  return (
    <NodeCard
      selected={selected}
      minWidth={120}
      minHeight={80}
      className="!min-w-0"
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Media"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <>
          {/* Media area – uses absolute positioning so content never overflows */}
          <div className="flex-1 min-h-0 relative mx-2 my-1.5">
            {imagePath ? (
              <div
                className="absolute inset-0 rounded-lg overflow-hidden border border-[rgba(119,119,119,0.15)] group cursor-pointer"
                onClick={() => setIsMediaPickerOpen(true)}
              >
                {isVideo ? (
                  <div className="w-full h-full flex flex-col items-center justify-center bg-[#1a1a1a]">
                    <Film className="h-8 w-8 text-blue-400 mb-1" />
                    <span className="text-[9px] text-[#999]">Video</span>
                  </div>
                ) : (
                  <img
                    src={getAssetUrl(imagePath)}
                    alt="Selected"
                    className="w-full h-full object-contain"
                  />
                )}
                <button
                  onClick={handleRemoveMedia}
                  className="absolute top-1 right-1 bg-black/70 hover:bg-black text-white rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove media"
                >
                  <X className="h-3 w-3" />
                </button>
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-black/0 hover:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none">
                  <span className="text-[10px] text-white font-medium bg-black/60 px-2 py-1 rounded">
                    Replace
                  </span>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setIsMediaPickerOpen(true)}
                className="absolute inset-0 rounded-lg border-2 border-dashed border-[rgba(119,119,119,0.3)] flex flex-col items-center justify-center hover:border-[rgba(119,119,119,0.5)] hover:bg-[rgba(255,255,255,0.02)] transition-colors"
              >
                <ImageIcon className="h-4 w-4 mb-0.5 text-[#666]" />
                <span className="text-[10px] text-[#666]">Add Media</span>
              </button>
            )}
          </div>

          {/* Filename */}
          {imagePath && (
            <div className="flex justify-center px-2 pb-1.5 shrink-0">
              <span
                className={`${NODE_TOKENS.primaryText} truncate max-w-full`}
                title={imagePath}
              >
                {imagePath.split(/[/\\]/).pop() || imagePath}
              </span>
            </div>
          )}
        </>
      )}

      {/* Output handle (right) – changes color and ID based on media type */}
      <Handle
        type="source"
        position={Position.Right}
        id={handleId}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: "50%", right: 0, backgroundColor: handleColor }
        }
      />

      {/* Portal the MediaPicker to document.body so it escapes the React Flow transform */}
      {isMediaPickerOpen &&
        createPortal(
          <MediaPicker
            isOpen={isMediaPickerOpen}
            onClose={() => setIsMediaPickerOpen(false)}
            onSelectImage={handleSelectMedia}
            accept="all"
          />,
          document.body
        )}
    </NodeCard>
  );
}

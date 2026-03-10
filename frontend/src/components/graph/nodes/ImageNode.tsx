import { useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { ImageIcon, X } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { getAssetUrl } from "../../../lib/api";
import { MediaPicker } from "../../MediaPicker";
import { NodeCard, NodeHeader, NODE_TOKENS } from "../ui";

type ImageNodeType = Node<FlowNodeData, "image">;

const COLOR = "#f472b6"; // pink-400

export function ImageNode({ id, data, selected }: NodeProps<ImageNodeType>) {
  const { updateData } = useNodeData(id);
  const [isMediaPickerOpen, setIsMediaPickerOpen] = useState(false);

  const imagePath = (data.imagePath as string) || "";

  const handleSelectImage = useCallback(
    (path: string) => {
      updateData({ imagePath: path });
      setIsMediaPickerOpen(false); // Auto-close on selection
    },
    [updateData]
  );

  const handleRemoveImage = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      updateData({ imagePath: "" });
    },
    [updateData]
  );

  return (
    <NodeCard
      selected={selected}
      minWidth={120}
      minHeight={80}
      className="!min-w-0"
    >
      <NodeHeader
        title={data.customTitle || "Image"}
        dotColor="bg-pink-400"
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      {/* Image area – uses absolute positioning so the image never overflows */}
      <div className="flex-1 min-h-0 relative mx-2 my-1.5">
        {imagePath ? (
          <div
            className="absolute inset-0 rounded-lg overflow-hidden border border-[rgba(119,119,119,0.15)] group cursor-pointer"
            onClick={() => setIsMediaPickerOpen(true)}
          >
            <img
              src={getAssetUrl(imagePath)}
              alt="Selected"
              className="w-full h-full object-contain"
            />
            <button
              onClick={handleRemoveImage}
              className="absolute top-1 right-1 bg-black/70 hover:bg-black text-white rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity"
              title="Remove image"
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
            <span className="text-[10px] text-[#666]">Add Image</span>
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

      {/* Output handle (right) */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "value")}
        className="!w-2 !h-2 !border-0"
        style={{ top: "50%", right: 8, backgroundColor: COLOR }}
      />

      {/* Portal the MediaPicker to document.body so it escapes the React Flow transform */}
      {isMediaPickerOpen &&
        createPortal(
          <MediaPicker
            isOpen={isMediaPickerOpen}
            onClose={() => setIsMediaPickerOpen(false)}
            onSelectImage={handleSelectImage}
          />,
          document.body
        )}
    </NodeCard>
  );
}

import { useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { Music, X } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { MediaPicker } from "../../MediaPicker";
import { NodeCard, NodeHeader, NODE_TOKENS, collapsedHandleStyle } from "../ui";
import { COLOR_AUDIO } from "../nodeColors";

type AudioNodeType = Node<FlowNodeData, "audio">;

export function AudioNode({ id, data, selected }: NodeProps<AudioNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const [isPickerOpen, setIsPickerOpen] = useState(false);

  const audioPath = (data.audioPath as string) || "";
  const handleId = buildHandleId("param", "value");
  const fileName = audioPath ? audioPath.split(/[/\\]/).pop() || audioPath : "";

  const handleSelect = useCallback(
    (path: string) => {
      updateData({ audioPath: path });
      setIsPickerOpen(false);
    },
    [updateData]
  );

  const handleRemove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      updateData({ audioPath: "" });
    },
    [updateData]
  );

  return (
    <NodeCard
      selected={selected}
      minWidth={120}
      minHeight={60}
      className="!min-w-0"
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Audio"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <>
          <div className="flex-1 min-h-0 relative mx-2 my-1.5">
            {audioPath ? (
              <div
                className="absolute inset-0 rounded-lg overflow-hidden border border-[rgba(119,119,119,0.15)] group cursor-pointer flex flex-col items-center justify-center bg-[#1a1a1a]"
                onClick={() => setIsPickerOpen(true)}
              >
                <Music className="h-8 w-8 text-emerald-400 mb-1" />
                <button
                  onClick={handleRemove}
                  className="absolute top-1 right-1 bg-black/70 hover:bg-black text-white rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove audio"
                >
                  <X className="h-3 w-3" />
                </button>
                <div className="absolute inset-0 bg-black/0 hover:bg-black/30 transition-colors flex items-center justify-center opacity-0 group-hover:opacity-100 pointer-events-none">
                  <span className="text-[10px] text-white font-medium bg-black/60 px-2 py-1 rounded">
                    Replace
                  </span>
                </div>
              </div>
            ) : (
              <button
                onClick={() => setIsPickerOpen(true)}
                className="absolute inset-0 rounded-lg border-2 border-dashed border-[rgba(119,119,119,0.3)] flex flex-col items-center justify-center hover:border-[rgba(119,119,119,0.5)] hover:bg-[rgba(255,255,255,0.02)] transition-colors"
              >
                <Music className="h-4 w-4 mb-0.5 text-[#666]" />
                <span className="text-[10px] text-[#666]">Add Audio</span>
              </button>
            )}
          </div>

          {audioPath && (
            <div className="flex justify-center px-2 pb-1.5 shrink-0">
              <span
                className={`${NODE_TOKENS.primaryText} truncate max-w-full`}
                title={audioPath}
              >
                {fileName}
              </span>
            </div>
          )}
        </>
      )}

      <Handle
        type="source"
        position={Position.Right}
        id={handleId}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: "50%", right: 0, backgroundColor: COLOR_AUDIO }
        }
      />

      {isPickerOpen &&
        createPortal(
          <MediaPicker
            isOpen={isPickerOpen}
            onClose={() => setIsPickerOpen(false)}
            onSelectImage={handleSelect}
            accept="audio"
          />,
          document.body
        )}
    </NodeCard>
  );
}

import { useState, useCallback } from "react";
import { createPortal } from "react-dom";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { Music, X } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { getAssetUrl } from "../../../lib/api";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { MediaPicker } from "../../MediaPicker";
import { AudioPreviewPlaceholder } from "../../AudioPreviewPlaceholder";
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
      minWidth={160}
      minHeight={132}
      className={`border-emerald-400/45 shadow-[0_0_18px_rgba(52,211,153,0.12)] ${
        collapsed ? "" : "!min-w-[160px] min-h-[132px]"
      }`}
      collapsed={collapsed}
      autoMinHeight={false}
    >
      <NodeHeader
        title={data.customTitle || "Audio"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
        className="bg-emerald-400/10 border-b-emerald-400/20"
      />
      {!collapsed && (
        <>
          <div className="mx-2 my-1.5 min-h-[76px]">
            {audioPath ? (
              <div className="relative h-full min-h-[76px] rounded-lg border border-emerald-400/25 group overflow-hidden">
                <AudioPreviewPlaceholder
                  src={getAssetUrl(audioPath)}
                  fileName={fileName}
                />
                <button
                  onClick={() => setIsPickerOpen(true)}
                  className="absolute bottom-1 right-1 rounded bg-black/70 px-1.5 py-0.5 text-[9px] font-medium text-white opacity-0 transition-opacity hover:bg-black group-hover:opacity-100"
                  title="Replace audio"
                >
                  Replace
                </button>
                <button
                  onClick={handleRemove}
                  className="absolute top-1 right-1 bg-black/70 hover:bg-black text-white rounded p-1 opacity-0 group-hover:opacity-100 transition-opacity"
                  title="Remove audio"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setIsPickerOpen(true)}
                className="h-full min-h-[76px] w-full rounded-lg border-2 border-dashed border-emerald-400/35 flex flex-col items-center justify-center hover:border-emerald-400/60 hover:bg-emerald-400/[0.06] transition-colors"
              >
                <Music className="h-4 w-4 mb-0.5 text-emerald-300/80" />
                <span className="text-[10px] text-emerald-300/80">
                  Add Audio
                </span>
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

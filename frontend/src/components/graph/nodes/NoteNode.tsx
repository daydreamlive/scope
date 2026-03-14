import { useCallback, useRef, useEffect } from "react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/useNodeData";
import { NodeCard, NodeHeader } from "../ui";

type NoteNodeType = Node<FlowNodeData, "note">;

export function NoteNode({ id, data, selected }: NodeProps<NoteNodeType>) {
  const { updateData } = useNodeData(id);
  const noteText = (data.noteText as string) || "";
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleTextChange = useCallback(
    (newText: string) => {
      updateData({ noteText: newText });
    },
    [updateData]
  );

  // Auto-resize textarea to fit content
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [noteText]);

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
  };

  return (
    <NodeCard selected={selected} autoMinHeight minWidth={160} minHeight={40}>
      <NodeHeader
        title={data.customTitle || "Note"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
      />
      <textarea
        ref={textareaRef}
        value={noteText}
        onChange={e => handleTextChange(e.target.value)}
        onWheel={handleWheel}
        placeholder="Type a note…"
        className="flex-1 w-full resize-none bg-transparent border-none outline-none px-2 py-1.5 text-[#fafafa] text-[11px] leading-relaxed placeholder:text-[#555] nowheel nodrag"
        style={{ minHeight: 24, fieldSizing: "content" } as React.CSSProperties}
        spellCheck={false}
      />
    </NodeCard>
  );
}

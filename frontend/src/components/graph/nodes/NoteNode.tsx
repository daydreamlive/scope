import { useReactFlow } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { NodeCard, NodeHeader, NODE_TOKENS } from "../ui";

type NoteNodeType = Node<FlowNodeData, "note">;

export function NoteNode({ id, data, selected }: NodeProps<NoteNodeType>) {
  const { setNodes } = useReactFlow();
  const noteText = data.noteText || "";

  const handleTextChange = (newText: string) => {
    setNodes(nds =>
      nds.map(n => {
        if (n.id !== id) return n;
        return {
          ...n,
          data: {
            ...n.data,
            noteText: newText,
          },
        };
      })
    );
  };

  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
  };

  return (
    <NodeCard selected={selected}>
      <NodeHeader title="Note" dotColor="bg-amber-400" />
      <div className="flex-1 min-h-0 p-1.5">
        <textarea
          value={noteText}
          onChange={e => handleTextChange(e.target.value)}
          onWheel={handleWheel}
          placeholder="Add a note..."
          className={`${NODE_TOKENS.pillInput} !rounded-md w-full h-full resize-none text-left py-1.5 px-2 leading-relaxed nowheel`}
          style={{ minHeight: 40 }}
        />
      </div>
    </NodeCard>
  );
}

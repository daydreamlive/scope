import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useHandlePositions } from "../hooks/node/useHandlePositions";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodePillTextarea,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_STRING as COLOR, COLOR_NUMBER } from "../nodeColors";

type PromptBlendNodeType = Node<FlowNodeData, "prompt_blend">;

interface BlendItem {
  text: string;
  weight: number;
}

export function PromptBlendNode({
  id,
  data,
  selected,
}: NodeProps<PromptBlendNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const items: BlendItem[] =
    data.promptBlendItems && (data.promptBlendItems as BlendItem[]).length > 0
      ? (data.promptBlendItems as BlendItem[])
      : [{ text: "", weight: 100 }];
  const method = (data.promptBlendMethod as "linear" | "slerp") ?? "linear";

  const updateItems = useCallback(
    (newItems: BlendItem[]) => {
      updateData({ promptBlendItems: newItems });
    },
    [updateData]
  );

  const handleWeightChange = useCallback(
    (index: number, newWeight: number) => {
      const newItems = [...items];
      const clamped = Math.max(0, Math.min(100, newWeight));
      const remaining = 100 - clamped;

      const otherSum = items.reduce(
        (sum, p, i) => (i === index ? sum : sum + p.weight),
        0
      );

      newItems[index] = { ...newItems[index], weight: clamped };

      if (otherSum > 0) {
        newItems.forEach((_, i) => {
          if (i !== index) {
            const proportion = items[i].weight / otherSum;
            newItems[i] = {
              ...newItems[i],
              weight: Math.round(remaining * proportion),
            };
          }
        });
      } else if (items.length > 1) {
        const evenWeight = Math.round(remaining / (items.length - 1));
        newItems.forEach((_, i) => {
          if (i !== index) {
            newItems[i] = { ...newItems[i], weight: evenWeight };
          }
        });
      }

      updateItems(newItems);
    },
    [items, updateItems]
  );

  const updateTextAt = useCallback(
    (index: number, text: string) => {
      const newItems = [...items];
      newItems[index] = { ...newItems[index], text };
      updateItems(newItems);
    },
    [items, updateItems]
  );

  const addPrompt = useCallback(() => {
    const evenWeight = Math.round(100 / (items.length + 1));
    const newItems = items.map(item => ({ ...item, weight: evenWeight }));
    newItems.push({ text: "", weight: 100 - evenWeight * items.length });
    updateItems(newItems);
  }, [items, updateItems]);

  const removePrompt = useCallback(
    (index: number) => {
      if (items.length <= 1) return;
      const removed = items.filter((_, i) => i !== index);
      const totalWeight = removed.reduce((sum, p) => sum + p.weight, 0);
      const newItems =
        totalWeight > 0
          ? removed.map(p => ({
              ...p,
              weight: Math.round((p.weight / totalWeight) * 100),
            }))
          : removed.map(p => ({
              ...p,
              weight: Math.round(100 / removed.length),
            }));
      updateItems(newItems);
    },
    [items, updateItems]
  );

  const { setRowRef, rowPositions } = useHandlePositions([
    items.length,
    items.map(it => it.weight),
    method,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Prompt List"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-1">
            {/* Interpolation method selector */}
            {items.length >= 2 && (
              <div className="flex items-center justify-between gap-1">
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  Blend
                </span>
                <select
                  value={method}
                  onChange={e =>
                    updateData({
                      promptBlendMethod: e.target.value as "linear" | "slerp",
                    })
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1.5 py-0.5 focus:outline-none cursor-pointer`}
                >
                  <option value="linear">Linear</option>
                  <option value="slerp">SLERP</option>
                </select>
              </div>
            )}

            {/* Prompt entries */}
            <div className="flex flex-col gap-1.5">
              {items.map((item, i) => (
                <div key={i} className="flex flex-col gap-0.5">
                  {/* Weight row */}
                  <div
                    ref={setRowRef(`weight_${i}`)}
                    className="flex items-center gap-1"
                  >
                    <span
                      className={`${NODE_TOKENS.labelText} w-3 text-right shrink-0`}
                    >
                      {i}
                    </span>
                    {/* Weight slider */}
                    <div
                      className="relative flex-1 h-3.5 rounded-full cursor-pointer select-none"
                      style={{
                        background: "#1b1a1a",
                        border: "1px solid rgba(119,119,119,0.15)",
                      }}
                      onPointerDown={e => {
                        e.preventDefault();
                        e.stopPropagation();
                        const target = e.currentTarget as HTMLElement;
                        target.setPointerCapture(e.pointerId);
                        const setFromMouse = (clientX: number) => {
                          const rect = target.getBoundingClientRect();
                          let ratio = (clientX - rect.left) / rect.width;
                          ratio = Math.min(Math.max(ratio, 0), 1);
                          handleWeightChange(i, Math.round(ratio * 100));
                        };
                        setFromMouse(e.clientX);
                        const onMove = (ev: PointerEvent) =>
                          setFromMouse(ev.clientX);
                        const onUp = () => {
                          target.removeEventListener("pointermove", onMove);
                          target.removeEventListener("pointerup", onUp);
                        };
                        target.addEventListener("pointermove", onMove);
                        target.addEventListener("pointerup", onUp);
                      }}
                    >
                      <div
                        className="absolute left-0 top-0 h-full rounded-full pointer-events-none"
                        style={{
                          width: `${item.weight}%`,
                          background: COLOR,
                          opacity: 0.3,
                        }}
                      />
                      <div
                        className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full pointer-events-none"
                        style={{
                          left: `calc(${item.weight}% - 4px)`,
                          background: COLOR,
                        }}
                      />
                    </div>
                    <span className="text-[8px] font-mono tabular-nums text-[#aaa] w-[22px] text-right shrink-0">
                      {Math.round(item.weight)}
                    </span>
                    {items.length > 1 && (
                      <button
                        type="button"
                        className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0"
                        onClick={() => removePrompt(i)}
                        onPointerDown={e => e.stopPropagation()}
                      >
                        ×
                      </button>
                    )}
                  </div>
                  {/* Text input */}
                  <div ref={setRowRef(`prompt_${i}`)} className="ml-4">
                    <NodePillTextarea
                      value={item.text}
                      onChange={v => updateTextAt(i, v)}
                      placeholder={`Prompt ${i + 1}...`}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Add button */}
            <button
              type="button"
              className="w-full py-0.5 rounded bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-[#ccc] hover:border-[rgba(119,119,119,0.4)] text-[10px] transition-colors"
              onClick={addPrompt}
              onPointerDown={e => e.stopPropagation()}
            >
              + Add
            </button>
          </div>
        </NodeBody>
      )}

      {/* Per-row handles: blue on weight slider, yellow on prompt text */}
      {items.map((_, i) => (
        <Handle
          key={`weight-in-${i}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", `weight_${i}`)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`weight_${i}`] ?? 0,
                  left: 0,
                  backgroundColor: COLOR_NUMBER,
                }
          }
        />
      ))}
      {items.map((_, i) => (
        <Handle
          key={`prompt-in-${i}`}
          type="target"
          position={Position.Left}
          id={buildHandleId("param", `prompt_${i}`)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("left"), opacity: 0 }
              : {
                  top: rowPositions[`prompt_${i}`] ?? 0,
                  left: 0,
                  backgroundColor: COLOR,
                }
          }
        />
      ))}

      {/* Output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "prompts")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["weight_0"] ?? 0,
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}

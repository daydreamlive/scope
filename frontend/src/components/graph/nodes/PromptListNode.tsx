import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";
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
import { COLOR_STRING as COLOR } from "../nodeColors";

const COLOR_TRIGGER = "#f97316";

type PromptListNodeType = Node<FlowNodeData, "prompt_list">;

export function PromptListNode({
  id,
  data,
  selected,
}: NodeProps<PromptListNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const items: string[] =
    data.promptListItems && (data.promptListItems as string[]).length > 0
      ? (data.promptListItems as string[])
      : [""];
  const activeIndex = (data.promptListActiveIndex as number) ?? 0;

  const setActiveIndex = useCallback(
    (idx: number) => {
      const clamped = Math.max(0, Math.min(idx, items.length - 1));
      updateData({
        promptListActiveIndex: clamped,
        promptListActiveText: items[clamped] ?? "",
      });
    },
    [items, updateData]
  );

  const addPrompt = useCallback(() => {
    const newItems = [...items, ""];
    updateData({
      promptListItems: newItems,
      promptListActiveIndex: activeIndex,
      promptListActiveText: newItems[activeIndex] ?? "",
    });
  }, [items, activeIndex, updateData]);

  const removePrompt = useCallback(
    (index: number) => {
      if (items.length <= 1) return;
      const newItems = items.filter((_, i) => i !== index);
      const newIdx =
        activeIndex >= newItems.length ? newItems.length - 1 : activeIndex;
      updateData({
        promptListItems: newItems,
        promptListActiveIndex: newIdx,
        promptListActiveText: newItems[newIdx] ?? "",
      });
    },
    [items, activeIndex, updateData]
  );

  const updatePromptAt = useCallback(
    (index: number, text: string) => {
      const newItems = [...items];
      newItems[index] = text;
      updateData({
        promptListItems: newItems,
        ...(index === activeIndex ? { promptListActiveText: text } : {}),
      });
    },
    [items, activeIndex, updateData]
  );

  const { setRowRef, rowPositions } = useHandlePositions([
    items.length,
    activeIndex,
  ]);

  const [triggerFlash, setTriggerFlash] = useState(false);
  const triggerTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevCounterRef = useRef(0);
  const prevArmedRef = useRef(false);

  const triggerCounters =
    (data._promptTriggerCounters as Record<string, number>) ?? {};
  const triggerCounterSum = Object.values(triggerCounters).reduce(
    (a, b) => a + b,
    0
  );
  const triggerArmed = Boolean(data.promptListTriggerArmed);

  useEffect(() => {
    const counterChanged =
      triggerCounterSum > 0 && triggerCounterSum !== prevCounterRef.current;
    const armedRising = triggerArmed && !prevArmedRef.current;
    prevCounterRef.current = triggerCounterSum;
    prevArmedRef.current = triggerArmed;

    if (counterChanged || armedRising) {
      setTriggerFlash(true);
      if (triggerTimerRef.current) clearTimeout(triggerTimerRef.current);
      triggerTimerRef.current = setTimeout(() => setTriggerFlash(false), 200);
    }
  }, [triggerCounterSum, triggerArmed]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Prompt Cycle"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-1">
            {/* Navigation row */}
            <div
              ref={setRowRef("nav")}
              className="flex items-center justify-between"
            >
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={() =>
                    setActiveIndex(
                      (activeIndex - 1 + items.length) % items.length
                    )
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors !px-1.5`}
                >
                  ‹
                </button>
                <span className="text-[9px] font-mono tabular-nums text-[#fafafa] min-w-[24px] text-center">
                  {activeIndex + 1}/{items.length}
                </span>
                <button
                  type="button"
                  onClick={() =>
                    setActiveIndex((activeIndex + 1) % items.length)
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors !px-1.5`}
                >
                  ›
                </button>
              </div>
              <span
                className={`${NODE_TOKENS.labelText} text-[8px]`}
                style={{ color: "#38bdf8" }}
              >
                cycle →
              </span>
            </div>

            {/* Trigger input row */}
            <div ref={setRowRef("trigger")} className="flex items-center">
              <span
                className={`${NODE_TOKENS.labelText} text-[8px]`}
                style={{ color: COLOR_TRIGGER }}
              >
                ← next
              </span>
            </div>

            {/* Prompt entries */}
            <div className="flex flex-col gap-1">
              {items.map((text, i) => (
                <div
                  key={i}
                  className="flex items-start gap-1"
                  style={{
                    opacity: i === activeIndex ? 1 : 0.5,
                  }}
                >
                  <span
                    className={`${NODE_TOKENS.labelText} w-3 text-right shrink-0 mt-1 cursor-pointer`}
                    onClick={() => setActiveIndex(i)}
                    onPointerDown={e => e.stopPropagation()}
                  >
                    {i === activeIndex ? "▸" : i}
                  </span>
                  <div className="flex-1">
                    <NodePillTextarea
                      value={text}
                      onChange={v => updatePromptAt(i, v)}
                      placeholder={`Prompt ${i + 1}...`}
                    />
                  </div>
                  {items.length > 1 && (
                    <button
                      type="button"
                      className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0 mt-0.5"
                      onClick={() => removePrompt(i)}
                      onPointerDown={e => e.stopPropagation()}
                    >
                      ×
                    </button>
                  )}
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

      {/* Cycle input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "cycle")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["nav"] ?? 0,
                left: 0,
                backgroundColor: "#38bdf8",
              }
        }
      />

      {/* Trigger input handle */}
      <Handle
        type="target"
        position={Position.Left}
        id={buildHandleId("param", "trigger")}
        className={
          collapsed
            ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
            : "!w-2.5 !h-2.5 !border-0"
        }
        style={
          collapsed
            ? { ...collapsedHandleStyle("left"), opacity: 0 }
            : {
                top: rowPositions["trigger"] ?? 0,
                left: 0,
                backgroundColor: triggerFlash ? "#ffffff" : COLOR_TRIGGER,
                boxShadow: triggerFlash
                  ? "0 0 6px 2px rgba(255,255,255,0.6)"
                  : "none",
                transition: "background-color 200ms, box-shadow 200ms",
              }
        }
      />

      {/* Prompt output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "prompt")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : {
                top: rowPositions["nav"] ?? 0,
                right: 0,
                backgroundColor: COLOR,
              }
        }
      />
    </NodeCard>
  );
}

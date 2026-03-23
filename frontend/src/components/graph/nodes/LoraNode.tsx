import { useCallback, useEffect, useMemo, useRef } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { Plus, X } from "lucide-react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { resolveLoRAPath } from "../../../lib/workflowSettings";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import { useLoRAsContext } from "../../../contexts/LoRAsContext";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NodePillSearchableSelect,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";

type LoraNodeType = Node<FlowNodeData, "lora">;

const LORA_COLOR = "#f472b6"; // pink-400
const SCALE_MIN = -10;
const SCALE_MAX = 10;
const SCALE_STEP = 0.1;

interface LoraEntry {
  path: string;
  scale: number;
  mergeMode?: string;
}

export function LoraNode({ id, data, selected }: NodeProps<LoraNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const { loraFiles, refresh } = useLoRAsContext();

  // Refresh LoRA file list on mount so newly downloaded files are available
  useEffect(() => {
    refresh();
  }, [refresh]);

  // Auto-resolve bare filenames to full paths when loraFiles are available
  useEffect(() => {
    if (loraFiles.length === 0) return;
    const currentLoras: LoraEntry[] = Array.isArray(data.loras)
      ? data.loras
      : [];
    if (currentLoras.length === 0) return;

    let changed = false;
    const resolved = currentLoras.map(entry => {
      if (!entry.path || loraFiles.some(f => f.path === entry.path))
        return entry;
      const resolvedPath = resolveLoRAPath(entry.path, loraFiles);
      if (resolvedPath !== entry.path) {
        changed = true;
        return { ...entry, path: resolvedPath };
      }
      return entry;
    });

    if (changed) updateData({ loras: resolved });
  }, [loraFiles, data.loras, updateData]);

  const loras: LoraEntry[] = useMemo(
    () => (Array.isArray(data.loras) ? data.loras : []),
    [data.loras]
  );
  const mergeMode = (data.loraMergeMode as string) || "permanent_merge";

  const setLoras = useCallback(
    (newLoras: LoraEntry[]) => updateData({ loras: newLoras }),
    [updateData]
  );

  const handleAdd = useCallback(() => {
    setLoras([...loras, { path: "", scale: 1.0, mergeMode }]);
  }, [loras, mergeMode, setLoras]);

  const handleRemove = useCallback(
    (idx: number) => {
      setLoras(loras.filter((_, i) => i !== idx));
    },
    [loras, setLoras]
  );

  const handlePathChange = useCallback(
    (idx: number, path: string) => {
      const next = [...loras];
      next[idx] = { ...next[idx], path };
      setLoras(next);
    },
    [loras, setLoras]
  );

  const handleScaleChange = useCallback(
    (idx: number, scale: number) => {
      const next = [...loras];
      next[idx] = { ...next[idx], scale };
      setLoras(next);
    },
    [loras, setLoras]
  );

  const handleEntryMergeModeChange = useCallback(
    (idx: number, mode: string) => {
      const next = [...loras];
      next[idx] = { ...next[idx], mergeMode: mode };
      setLoras(next);
    },
    [loras, setLoras]
  );

  const fileOptions = loraFiles.map(f => ({
    value: f.path,
    label: f.name,
  }));

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "LoRA"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody withGap>
          {/* Global merge mode */}
          <NodeParamRow label="Strategy">
            <NodePillSelect
              value={mergeMode}
              onChange={val => updateData({ loraMergeMode: val })}
              options={[
                { value: "permanent_merge", label: "Permanent" },
                { value: "runtime_peft", label: "Runtime PEFT" },
              ]}
            />
          </NodeParamRow>

          {/* LoRA entries */}
          {loras.map((entry, idx) => (
            <LoraEntryRow
              key={idx}
              idx={idx}
              entry={entry}
              fileOptions={fileOptions}
              onPathChange={handlePathChange}
              onScaleChange={handleScaleChange}
              onMergeModeChange={handleEntryMergeModeChange}
              onRemove={handleRemove}
            />
          ))}

          {/* Add button */}
          <div>
            <button
              type="button"
              onClick={handleAdd}
              className={`${NODE_TOKENS.pill} flex items-center justify-center gap-1 w-full cursor-pointer hover:bg-[#2a2a2a] active:bg-[#333] transition-colors`}
            >
              <Plus className="h-3 w-3 text-[#fafafa]" />
              <span className={NODE_TOKENS.primaryText}>Add LoRA</span>
            </button>
          </div>
        </NodeBody>
      )}

      {/* Compound LoRA output handle */}
      <Handle
        type="source"
        position={Position.Right}
        id={buildHandleId("param", "__loras")}
        className="!w-2.5 !h-2.5 !border-0"
        style={
          collapsed
            ? collapsedHandleStyle("right")
            : { top: "50%", right: 0, backgroundColor: LORA_COLOR }
        }
      />
    </NodeCard>
  );
}

function LoraEntryRow({
  idx,
  entry,
  fileOptions,
  onPathChange,
  onScaleChange,
  onMergeModeChange,
  onRemove,
}: {
  idx: number;
  entry: LoraEntry;
  fileOptions: Array<{ value: string; label: string }>;
  onPathChange: (idx: number, path: string) => void;
  onScaleChange: (idx: number, scale: number) => void;
  onMergeModeChange: (idx: number, mode: string) => void;
  onRemove: (idx: number) => void;
}) {
  const sliderRef = useRef<HTMLDivElement>(null);

  const clampedScale = Math.min(Math.max(entry.scale, SCALE_MIN), SCALE_MAX);
  const pct = ((clampedScale - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)) * 100;

  const setValueFromMouse = useCallback(
    (clientX: number) => {
      if (!sliderRef.current) return;
      const rect = sliderRef.current.getBoundingClientRect();
      let ratio = (clientX - rect.left) / rect.width;
      ratio = Math.min(Math.max(ratio, 0), 1);
      let newVal = SCALE_MIN + ratio * (SCALE_MAX - SCALE_MIN);
      newVal =
        SCALE_MIN + Math.round((newVal - SCALE_MIN) / SCALE_STEP) * SCALE_STEP;
      newVal = Math.min(Math.max(newVal, SCALE_MIN), SCALE_MAX);
      onScaleChange(idx, parseFloat(newVal.toFixed(1)));
    },
    [idx, onScaleChange]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget as HTMLElement;
      target.setPointerCapture(e.pointerId);
      setValueFromMouse(e.clientX);

      const onMove = (ev: PointerEvent) => setValueFromMouse(ev.clientX);
      const onUp = () => {
        target.removeEventListener("pointermove", onMove);
        target.removeEventListener("pointerup", onUp);
      };
      target.addEventListener("pointermove", onMove);
      target.addEventListener("pointerup", onUp);
    },
    [setValueFromMouse]
  );

  return (
    <div
      className="flex flex-col gap-1 rounded-md p-1.5"
      style={{ border: "1px solid rgba(119,119,119,0.1)" }}
    >
      {/* File picker + remove */}
      <div className="flex items-center gap-1">
        <div className="flex-1 min-w-0">
          <NodePillSearchableSelect
            value={entry.path}
            onChange={val => onPathChange(idx, val)}
            options={[{ value: "", label: "Select LoRA..." }, ...fileOptions]}
            placeholder="Select LoRA..."
          />
        </div>
        <button
          type="button"
          onClick={() => onRemove(idx)}
          className="shrink-0 p-0.5 rounded hover:bg-[#333] transition-colors"
        >
          <X className="h-3 w-3 text-[#8c8c8d]" />
        </button>
      </div>

      {/* Scale slider */}
      <div className="flex flex-col gap-0.5">
        <p className={`${NODE_TOKENS.labelText} text-[9px]`}>
          Scale: {clampedScale.toFixed(1)}
        </p>
        <div
          ref={sliderRef}
          className="relative w-full h-4 rounded-full cursor-pointer select-none nodrag"
          style={{
            background: "#1b1a1a",
            border: "1px solid rgba(119,119,119,0.15)",
          }}
          onPointerDown={handlePointerDown}
        >
          <div
            className="absolute left-0 top-0 h-full rounded-full pointer-events-none"
            style={{
              width: `${pct}%`,
              background: LORA_COLOR,
              opacity: 0.35,
            }}
          />
          <div
            className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full pointer-events-none"
            style={{
              left: `calc(${pct}% - 5px)`,
              background: LORA_COLOR,
              boxShadow: `0 0 4px ${LORA_COLOR}`,
            }}
          />
        </div>
      </div>

      {/* Per-entry merge mode */}
      <NodeParamRow label="Mode">
        <NodePillSelect
          value={entry.mergeMode || "permanent_merge"}
          onChange={val => onMergeModeChange(idx, val)}
          options={[
            { value: "permanent_merge", label: "Permanent" },
            { value: "runtime_peft", label: "PEFT" },
          ]}
        />
      </NodeParamRow>
    </div>
  );
}

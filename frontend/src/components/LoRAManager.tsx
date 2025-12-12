import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { SliderWithInput } from "./ui/slider-with-input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Plus, X, RefreshCw } from "lucide-react";
import { LabelWithTooltip } from "./ui/label-with-tooltip";
import { PARAMETER_METADATA } from "../data/parameterMetadata";
import type { LoRAConfig, LoraMergeStrategy } from "../types";
import { listLoRAFiles, type LoRAFileInfo } from "../lib/api";
import { FilePicker } from "./ui/file-picker";

interface LoRAManagerProps {
  loras: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  loraMergeStrategy?: LoraMergeStrategy;
}

export function LoRAManager({
  loras,
  onLorasChange,
  disabled,
  isStreaming = false,
  loraMergeStrategy = "permanent_merge",
}: LoRAManagerProps) {
  const [availableLoRAs, setAvailableLoRAs] = useState<LoRAFileInfo[]>([]);
  const [isLoadingLoRAs, setIsLoadingLoRAs] = useState(false);
  const [localScales, setLocalScales] = useState<Record<string, number>>({});

  const loadAvailableLoRAs = async () => {
    setIsLoadingLoRAs(true);
    try {
      const response = await listLoRAFiles();
      setAvailableLoRAs(response.lora_files);
    } catch (error) {
      console.error("loadAvailableLoRAs: Failed to load LoRA files:", error);
    } finally {
      setIsLoadingLoRAs(false);
    }
  };

  useEffect(() => {
    loadAvailableLoRAs();
  }, []);

  // Sync localScales from loras prop when it changes from outside
  useEffect(() => {
    const newLocalScales: Record<string, number> = {};
    loras.forEach(lora => {
      newLocalScales[lora.id] = lora.scale;
    });
    setLocalScales(newLocalScales);
  }, [loras]);

  const handleAddLora = () => {
    const newLora: LoRAConfig = {
      id: crypto.randomUUID(),
      path: "",
      scale: 1.0,
      mergeMode: loraMergeStrategy,
    };
    onLorasChange([...loras, newLora]);
  };

  const handleRemoveLora = (id: string) => {
    onLorasChange(loras.filter(lora => lora.id !== id));
  };

  const handleLoraChange = (id: string, updates: Partial<LoRAConfig>) => {
    onLorasChange(
      loras.map(lora => (lora.id === id ? { ...lora, ...updates } : lora))
    );
  };

  const handleLocalScaleChange = (id: string, scale: number) => {
    setLocalScales(prev => ({ ...prev, [id]: scale }));
  };

  const handleScaleCommit = (id: string, scale: number) => {
    handleLoraChange(id, { scale });
  };

  const getScaleAdjustmentInfo = (lora: LoRAConfig) => {
    const effectiveMergeMode = lora.mergeMode || loraMergeStrategy;
    const isPermanentMerge = effectiveMergeMode === "permanent_merge";
    const isDisabled = disabled || (isStreaming && isPermanentMerge);
    const tooltipText =
      isStreaming && isPermanentMerge
        ? PARAMETER_METADATA.loraScaleDisabledDuringStream.tooltip
        : PARAMETER_METADATA.loraScale.tooltip;

    return { isDisabled, tooltipText };
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">LoRA Adapters</h3>
        <div className="flex gap-1">
          <Button
            size="sm"
            variant="outline"
            onClick={loadAvailableLoRAs}
            disabled={disabled || isLoadingLoRAs}
            className="h-6 px-2"
            title="Refresh LoRA list"
          >
            <RefreshCw
              className={`h-3 w-3 ${isLoadingLoRAs ? "animate-spin" : ""}`}
            />
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={handleAddLora}
            disabled={disabled || isStreaming}
            className="h-6 px-2"
            title={
              isStreaming ? "Cannot add LoRAs while streaming" : "Add LoRA"
            }
          >
            <Plus className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {loras.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No LoRA adapters configured. Follow the{" "}
          <a
            href="https://github.com/daydreamlive/scope/blob/main/docs/lora.md"
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            docs
          </a>{" "}
          to add LoRA files.
        </p>
      )}

      <div className="space-y-2">
        {loras.map(lora => (
          <div
            key={lora.id}
            className="rounded-lg border bg-card p-3 space-y-2"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex-1 min-w-0">
                <FilePicker
                  value={lora.path}
                  onChange={path => handleLoraChange(lora.id, { path })}
                  files={availableLoRAs}
                  disabled={disabled || isStreaming}
                  placeholder="Select LoRA file"
                  emptyMessage="No LoRA files found"
                />
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handleRemoveLora(lora.id)}
                disabled={disabled || isStreaming}
                className="h-6 w-6 p-0 shrink-0"
                title={
                  isStreaming
                    ? "Cannot remove LoRAs while streaming"
                    : "Remove LoRA"
                }
              >
                <X className="h-3 w-3" />
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <LabelWithTooltip
                label="Strategy:"
                tooltip={PARAMETER_METADATA.loraMergeStrategy.tooltip}
                className="text-xs text-muted-foreground w-16"
              />
              <Select
                value={lora.mergeMode || loraMergeStrategy}
                onValueChange={value => {
                  handleLoraChange(lora.id, {
                    mergeMode: value as LoraMergeStrategy,
                  });
                }}
                disabled={disabled || isStreaming}
              >
                <SelectTrigger className="h-7 flex-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="permanent_merge">
                    Permanent Merge
                  </SelectItem>
                  <SelectItem value="runtime_peft">Runtime PEFT</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="flex items-center gap-2">
              <LabelWithTooltip
                label="Scale:"
                tooltip={getScaleAdjustmentInfo(lora).tooltipText}
                className="text-xs text-muted-foreground w-16"
              />
              <div className="flex-1 min-w-0">
                <SliderWithInput
                  value={localScales[lora.id] ?? lora.scale}
                  onValueChange={value => {
                    handleLocalScaleChange(lora.id, value);
                  }}
                  onValueCommit={value => {
                    handleScaleCommit(lora.id, value);
                  }}
                  min={-10}
                  max={10}
                  step={0.1}
                  incrementAmount={0.1}
                  disabled={getScaleAdjustmentInfo(lora).isDisabled}
                  className="flex-1"
                  valueFormatter={v => Math.round(v * 10) / 10}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

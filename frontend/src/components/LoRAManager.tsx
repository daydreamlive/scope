import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { SliderWithInput } from "./ui/slider-with-input";
import { Plus, X, RefreshCw } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "./ui/tooltip";
import type { LoRAConfig } from "../types";
import { listLoRAFiles, type LoRAFileInfo } from "../lib/api";
import { FilePicker } from "./ui/file-picker";

interface LoRAManagerProps {
  loras: LoRAConfig[];
  onLorasChange: (loras: LoRAConfig[]) => void;
  disabled?: boolean;
}

export function LoRAManager({
  loras,
  onLorasChange,
  disabled,
}: LoRAManagerProps) {
  const [availableLoRAs, setAvailableLoRAs] = useState<LoRAFileInfo[]>([]);
  const [isLoadingLoRAs, setIsLoadingLoRAs] = useState(false);

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

  const handleAddLora = () => {
    const newLora: LoRAConfig = {
      id: crypto.randomUUID(),
      path: "",
      scale: 1.0,
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
            disabled={disabled}
            className="h-6 px-2"
            title="Add LoRA"
          >
            <Plus className="h-3 w-3" />
          </Button>
        </div>
      </div>

      {loras.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No LoRA adapters configured. Add LoRA files to models/lora directory.
        </p>
      )}

      <div className="space-y-2">
        {loras.map(lora => (
          <div
            key={lora.id}
            className="rounded-lg border bg-card p-3 space-y-2"
          >
            <div className="flex items-center justify-between gap-2">
              <div className="flex-1">
                <FilePicker
                  value={lora.path}
                  onChange={path => handleLoraChange(lora.id, { path })}
                  files={availableLoRAs}
                  disabled={disabled}
                  placeholder="Select LoRA file"
                  emptyMessage="No LoRA files found"
                />
              </div>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => handleRemoveLora(lora.id)}
                disabled={disabled}
                className="h-6 w-6 p-0"
              >
                <X className="h-3 w-3" />
              </Button>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground w-12">Scale:</span>
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className="flex-1">
                      <SliderWithInput
                        value={lora.scale}
                        onValueChange={value => {
                          handleLoraChange(lora.id, { scale: value });
                        }}
                        min={-10}
                        max={10}
                        step={0.1}
                        disabled={disabled}
                        className="flex-1"
                      />
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="text-xs">
                      Adjust LoRA strength in real-time. 0.0 = no effect, 1.0 =
                      full strength
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

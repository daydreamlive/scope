import { useState, useCallback, useEffect } from "react";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Toggle } from "../ui/toggle";

export interface ModulationConfig {
  enabled: boolean;
  shape: string;
  depth: number;
  rate: string;
  base_value: number;
}

export type ModulationsState = Record<string, ModulationConfig>;

const TARGETS = [
  { value: "noise_scale", label: "Noise Scale", defaultBase: 0.5 },
  {
    value: "vace_context_scale",
    label: "VACE Context Scale",
    defaultBase: 1.0,
  },
  {
    value: "kv_cache_attention_bias",
    label: "KV Cache Bias",
    defaultBase: 0.3,
  },
] as const;

const SHAPES = [
  { value: "sine", label: "Sine" },
  { value: "cosine", label: "Cosine" },
  { value: "triangle", label: "Triangle" },
  { value: "saw", label: "Saw" },
  { value: "square", label: "Square" },
  { value: "exp_decay", label: "Pulse Decay" },
] as const;

const RATES = [
  { value: "half_beat", label: "½ Beat" },
  { value: "beat", label: "Beat" },
  { value: "2_beat", label: "2 Beats" },
  { value: "bar", label: "Bar" },
  { value: "2_bar", label: "2 Bars" },
  { value: "4_bar", label: "4 Bars" },
] as const;

function defaultConfigFor(target: string): ModulationConfig {
  const t = TARGETS.find(t => t.value === target);
  return {
    enabled: true,
    shape: "cosine",
    depth: 0.3,
    rate: "bar",
    base_value: t?.defaultBase ?? 0.5,
  };
}

export function ModulationSection({
  modulations,
  onModulationsChange,
}: {
  modulations: ModulationsState;
  onModulationsChange: (modulations: ModulationsState) => void;
}) {
  const [selectedTarget, setSelectedTarget] = useState<string>(
    TARGETS[0].value
  );

  const config = modulations[selectedTarget];
  const isActive = config?.enabled ?? false;

  const updateConfig = useCallback(
    (target: string, patch: Partial<ModulationConfig>) => {
      const existing = modulations[target] ?? defaultConfigFor(target);
      const updated = { ...existing, ...patch };
      onModulationsChange({ ...modulations, [target]: updated });
    },
    [modulations, onModulationsChange]
  );

  const handleToggle = useCallback(
    (pressed: boolean) => {
      if (pressed) {
        const cfg = defaultConfigFor(selectedTarget);
        onModulationsChange({ ...modulations, [selectedTarget]: cfg });
      } else {
        const next = { ...modulations };
        delete next[selectedTarget];
        onModulationsChange(next);
      }
    },
    [selectedTarget, modulations, onModulationsChange]
  );

  // Keep selectedTarget in sync if it gets removed
  useEffect(() => {
    if (selectedTarget && !TARGETS.some(t => t.value === selectedTarget)) {
      setSelectedTarget(TARGETS[0].value);
    }
  }, [selectedTarget]);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label="Modulation"
          tooltip="Continuously modulate pipeline parameters in sync with the beat. Select a parameter target and configure the wave shape, depth, and rate."
          className="text-xs text-muted-foreground"
        />
        <Toggle
          pressed={isActive}
          onPressedChange={handleToggle}
          variant="outline"
          size="sm"
          className="h-6 text-[10px] px-2"
        >
          {isActive ? "ON" : "OFF"}
        </Toggle>
      </div>

      <div className="space-y-2">
        <Select value={selectedTarget} onValueChange={setSelectedTarget}>
          <SelectTrigger className="h-7 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {TARGETS.map(t => (
              <SelectItem key={t.value} value={t.value}>
                {t.label}
                {modulations[t.value]?.enabled ? " ●" : ""}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {isActive && config && (
        <div className="rounded-lg border bg-card/50 p-2 space-y-2">
          <div className="space-y-1">
            <LabelWithTooltip
              label="Shape"
              tooltip="Waveform shape for the modulation oscillator."
              className="text-[10px] text-muted-foreground"
            />
            <Select
              value={config.shape}
              onValueChange={shape => updateConfig(selectedTarget, { shape })}
            >
              <SelectTrigger className="h-7 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SHAPES.map(s => (
                  <SelectItem key={s.value} value={s.value}>
                    {s.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1">
            <LabelWithTooltip
              label="Depth"
              tooltip="How far the value swings from its base setting. Higher depth means stronger modulation effect."
              className="text-[10px] text-muted-foreground"
            />
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={100}
                step={1}
                value={Math.round(config.depth * 100)}
                onChange={e =>
                  updateConfig(selectedTarget, {
                    depth: Number(e.target.value) / 100,
                  })
                }
                className="flex-1 h-1.5 accent-foreground"
              />
              <span className="text-[10px] font-mono tabular-nums w-8 text-right text-muted-foreground">
                {Math.round(config.depth * 100)}%
              </span>
            </div>
          </div>

          <div className="space-y-1">
            <LabelWithTooltip
              label="Rate"
              tooltip="How often the modulation completes one cycle. 'Beat' = once per beat, 'Bar' = once per bar, etc."
              className="text-[10px] text-muted-foreground"
            />
            <Select
              value={config.rate}
              onValueChange={rate => updateConfig(selectedTarget, { rate })}
            >
              <SelectTrigger className="h-7 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {RATES.map(r => (
                  <SelectItem key={r.value} value={r.value}>
                    {r.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
      )}
    </div>
  );
}

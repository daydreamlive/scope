import { useState, useCallback, useEffect, useMemo } from "react";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Toggle } from "../ui/toggle";
import type { PipelineConfigSchema } from "../../lib/api";

export interface ModulationConfig {
  enabled: boolean;
  shape: string;
  depth: number;
  rate: string;
  base_value: number;
  min_value?: number;
  max_value?: number;
}

export type ModulationsState = Record<string, ModulationConfig>;

interface ModulationTarget {
  value: string;
  label: string;
  defaultBase: number;
  min: number;
  max: number;
  isList?: boolean;
}

function formatFieldName(key: string): string {
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase());
}

function extractModulatableTargets(
  configSchema: PipelineConfigSchema | undefined
): ModulationTarget[] {
  if (!configSchema?.properties) return [];

  const targets: ModulationTarget[] = [];
  for (const [key, prop] of Object.entries(configSchema.properties)) {
    if (!prop.ui?.modulatable) continue;

    // For list params (e.g. denoising_steps), base_value is unused by the engine
    // (it shifts elements additively), but we still need a placeholder.
    const isList = prop.type === "array" || Array.isArray(prop.default);
    const defaultBase = isList
      ? 0
      : typeof prop.default === "number"
        ? prop.default
        : 0.5;

    targets.push({
      value: key,
      label: prop.ui?.label || formatFieldName(key),
      defaultBase,
      min: prop.ui?.modulatable_min ?? prop.minimum ?? 0,
      max: prop.ui?.modulatable_max ?? prop.maximum ?? 1,
      isList,
    });
  }
  return targets;
}

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

function defaultConfigFor(target: ModulationTarget): ModulationConfig {
  return {
    enabled: true,
    shape: "sine",
    depth: target.isList ? 0.15 : 0.3,
    rate: target.isList ? "2_bar" : "bar",
    base_value: target.defaultBase,
    min_value: target.min,
    max_value: target.max,
  };
}

export function ModulationSection({
  modulations,
  onModulationsChange,
  configSchema,
}: {
  modulations: ModulationsState;
  onModulationsChange: (modulations: ModulationsState) => void;
  configSchema?: PipelineConfigSchema;
}) {
  const targets = useMemo(
    () => extractModulatableTargets(configSchema),
    [configSchema]
  );

  const [selectedTarget, setSelectedTarget] = useState<string>("");

  // Sync selectedTarget when targets change (pipeline switch)
  useEffect(() => {
    if (targets.length > 0 && !targets.some(t => t.value === selectedTarget)) {
      setSelectedTarget(targets[0].value);
    }
  }, [targets, selectedTarget]);

  const currentTarget = targets.find(t => t.value === selectedTarget);
  const config = modulations[selectedTarget];
  const isActive = config?.enabled ?? false;

  const updateConfig = useCallback(
    (target: string, patch: Partial<ModulationConfig>) => {
      const tgt = targets.find(t => t.value === target);
      const existing = modulations[target] ?? (tgt ? defaultConfigFor(tgt) : null);
      if (!existing) return;
      const updated = { ...existing, ...patch };
      onModulationsChange({ ...modulations, [target]: updated });
    },
    [modulations, onModulationsChange, targets]
  );

  const handleToggle = useCallback(
    (pressed: boolean) => {
      if (pressed && currentTarget) {
        const cfg = defaultConfigFor(currentTarget);
        onModulationsChange({ ...modulations, [selectedTarget]: cfg });
      } else {
        const next = { ...modulations };
        delete next[selectedTarget];
        onModulationsChange(next);
      }
    },
    [selectedTarget, currentTarget, modulations, onModulationsChange]
  );

  // Clear modulations for targets that no longer exist after pipeline switch
  useEffect(() => {
    const targetKeys = new Set(targets.map(t => t.value));
    const staleKeys = Object.keys(modulations).filter(k => !targetKeys.has(k));
    if (staleKeys.length > 0) {
      const cleaned = { ...modulations };
      for (const k of staleKeys) delete cleaned[k];
      onModulationsChange(cleaned);
    }
  }, [targets]); // eslint-disable-line react-hooks/exhaustive-deps

  if (targets.length === 0) return null;

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
            {targets.map(t => (
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

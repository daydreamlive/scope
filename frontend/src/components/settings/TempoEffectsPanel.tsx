import { useState, useCallback, useEffect, useRef } from "react";
import { Switch } from "../ui/switch";
import { Slider } from "../ui/slider";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { ChevronDown, ChevronRight } from "lucide-react";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface NoiseBreathingConfig {
  enabled: boolean;
  intensity: number;
  envelope: "cosine" | "exponential" | "square";
  accent: "downbeat" | "backbeat" | "all_equal";
}

interface PromptCyclingConfig {
  enabled: boolean;
  beat_interval: number;
  mode: "sequential" | "random" | "pingpong";
  prompts: string[];
}

interface RefImageSwitchingConfig {
  enabled: boolean;
  beat_interval: number;
  mode: "sequential" | "random" | "pingpong";
  target: "vace_ref_images" | "first_frame_image";
  images: string[];
}

interface DenoisingModulationConfig {
  enabled: boolean;
  intensity: number;
  envelope: "cosine" | "exponential" | "square";
}

interface VaceContextPulseConfig {
  enabled: boolean;
  min_scale: number;
  max_scale: number;
  envelope: "cosine" | "exponential" | "square";
}

export interface TempoEffectsConfig {
  noise_breathing: NoiseBreathingConfig;
  prompt_cycling: PromptCyclingConfig;
  ref_image_switching: RefImageSwitchingConfig;
  denoising_modulation: DenoisingModulationConfig;
  vace_context_pulse: VaceContextPulseConfig;
}

const DEFAULT_CONFIG: TempoEffectsConfig = {
  noise_breathing: {
    enabled: true,
    intensity: 0.8,
    envelope: "cosine",
    accent: "downbeat",
  },
  prompt_cycling: {
    enabled: false,
    beat_interval: 4,
    mode: "sequential",
    prompts: [],
  },
  ref_image_switching: {
    enabled: false,
    beat_interval: 8,
    mode: "sequential",
    target: "vace_ref_images",
    images: [],
  },
  denoising_modulation: {
    enabled: false,
    intensity: 0.5,
    envelope: "cosine",
  },
  vace_context_pulse: {
    enabled: false,
    min_scale: 0.3,
    max_scale: 1.0,
    envelope: "cosine",
  },
};

// ---------------------------------------------------------------------------
// Collapsible effect section
// ---------------------------------------------------------------------------

function EffectSection({
  title,
  tooltip,
  enabled,
  onToggle,
  children,
}: {
  title: string;
  tooltip: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-md border border-border/50 overflow-hidden">
      <button
        type="button"
        className="flex items-center justify-between w-full px-3 py-2 text-left hover:bg-muted/50 transition-colors"
        onClick={() => setOpen(!open)}
      >
        <div className="flex items-center gap-2">
          {open ? (
            <ChevronDown className="h-3 w-3 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          )}
          <LabelWithTooltip
            label={title}
            tooltip={tooltip}
            className="text-xs font-medium"
          />
        </div>
        <Switch
          checked={enabled}
          onCheckedChange={checked => onToggle(checked)}
          onClick={e => e.stopPropagation()}
          className="scale-75"
        />
      </button>
      {open && enabled && (
        <div className="px-3 pb-3 pt-1 space-y-3 border-t border-border/30">
          {children}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared controls
// ---------------------------------------------------------------------------

function EnvelopeSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: "cosine" | "exponential" | "square") => void;
}) {
  return (
    <div className="space-y-1">
      <LabelWithTooltip
        label="Envelope"
        tooltip="Shape of the beat modulation curve. Cosine is smooth, exponential is punchy, square is on/off."
        className="text-[11px] text-muted-foreground"
      />
      <Select
        value={value}
        onValueChange={v =>
          onChange(v as "cosine" | "exponential" | "square")
        }
      >
        <SelectTrigger className="h-7 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="cosine">Cosine</SelectItem>
          <SelectItem value="exponential">Exponential</SelectItem>
          <SelectItem value="square">Square</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

function CycleModeSelect({
  value,
  onChange,
}: {
  value: string;
  onChange: (v: "sequential" | "random" | "pingpong") => void;
}) {
  return (
    <div className="space-y-1">
      <LabelWithTooltip
        label="Cycle Mode"
        tooltip="How to cycle through items. Sequential loops in order, random picks randomly, pingpong bounces back and forth."
        className="text-[11px] text-muted-foreground"
      />
      <Select
        value={value}
        onValueChange={v =>
          onChange(v as "sequential" | "random" | "pingpong")
        }
      >
        <SelectTrigger className="h-7 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="sequential">Sequential</SelectItem>
          <SelectItem value="random">Random</SelectItem>
          <SelectItem value="pingpong">Ping-Pong</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

function IntensitySlider({
  value,
  onChange,
  label = "Intensity",
  tooltip = "How strong the effect is (0 = off, 1 = maximum).",
}: {
  value: number;
  onChange: (v: number) => void;
  label?: string;
  tooltip?: string;
}) {
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <LabelWithTooltip
          label={label}
          tooltip={tooltip}
          className="text-[11px] text-muted-foreground"
        />
        <span className="text-[11px] font-mono text-muted-foreground tabular-nums">
          {value.toFixed(2)}
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={v => onChange(v[0])}
        min={0}
        max={1}
        step={0.05}
      />
    </div>
  );
}

function BeatIntervalSelect({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-1">
      <LabelWithTooltip
        label="Beat Interval"
        tooltip="Switch every N beats."
        className="text-[11px] text-muted-foreground"
      />
      <Select
        value={String(value)}
        onValueChange={v => onChange(parseInt(v))}
      >
        <SelectTrigger className="h-7 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="1">Every beat</SelectItem>
          <SelectItem value="2">Every 2 beats</SelectItem>
          <SelectItem value="4">Every 4 beats (1 bar)</SelectItem>
          <SelectItem value="8">Every 8 beats (2 bars)</SelectItem>
          <SelectItem value="16">Every 16 beats (4 bars)</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Prompt list editor
// ---------------------------------------------------------------------------

function PromptListEditor({
  prompts,
  onChange,
}: {
  prompts: string[];
  onChange: (prompts: string[]) => void;
}) {
  const [text, setText] = useState(prompts.join("\n"));
  const textRef = useRef(text);
  textRef.current = text;

  useEffect(() => {
    const incoming = prompts.join("\n");
    if (incoming !== textRef.current) {
      setText(incoming);
    }
  }, [prompts]);

  const handleBlur = () => {
    const lines = text
      .split("\n")
      .map(l => l.trim())
      .filter(l => l.length > 0);
    onChange(lines);
  };

  return (
    <div className="space-y-1">
      <LabelWithTooltip
        label="Prompts"
        tooltip="One prompt per line. The effect will cycle through these on beat."
        className="text-[11px] text-muted-foreground"
      />
      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        onBlur={handleBlur}
        rows={3}
        placeholder={"a red forest\na blue ocean\na golden desert"}
        className="w-full text-xs bg-transparent border border-border/50 rounded-md p-2 resize-y focus:border-foreground outline-none placeholder:text-muted-foreground/50"
      />
      <p className="text-[10px] text-muted-foreground">
        {prompts.length} prompt{prompts.length !== 1 ? "s" : ""} loaded
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function TempoEffectsPanel({
  onConfigChange,
}: {
  onConfigChange: (config: TempoEffectsConfig) => void;
}) {
  const [config, setConfig] = useState<TempoEffectsConfig>(DEFAULT_CONFIG);
  const configRef = useRef(config);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const emitChange = useCallback(
    (newConfig: TempoEffectsConfig) => {
      configRef.current = newConfig;
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
      debounceTimer.current = setTimeout(() => {
        onConfigChange(configRef.current);
      }, 100);
    },
    [onConfigChange]
  );

  const update = useCallback(
    <K extends keyof TempoEffectsConfig>(
      key: K,
      patch: Partial<TempoEffectsConfig[K]>
    ) => {
      setConfig(prev => {
        const next = {
          ...prev,
          [key]: { ...prev[key], ...patch },
        };
        emitChange(next);
        return next;
      });
    },
    [emitChange]
  );

  // Send initial config on mount
  useEffect(() => {
    onConfigChange(config);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="space-y-2 pt-2">
      <p className="text-[11px] text-muted-foreground font-medium uppercase tracking-wider">
        Beat Effects
      </p>
      <p className="text-[10px] text-muted-foreground leading-relaxed">
        Effects apply per generation chunk (~1-2s). Visual changes sync to
        beats at the chunk boundary, so lower BPMs produce clearer rhythms.
      </p>

      {/* Noise Breathing */}
      <EffectSection
        title="Noise Breathing"
        tooltip="Oscillates noise scale and KV cache attention on beat for a rhythmic visual breathing effect."
        enabled={config.noise_breathing.enabled}
        onToggle={v => update("noise_breathing", { enabled: v })}
      >
        <IntensitySlider
          value={config.noise_breathing.intensity}
          onChange={v => update("noise_breathing", { intensity: v })}
        />
        <EnvelopeSelect
          value={config.noise_breathing.envelope}
          onChange={v => update("noise_breathing", { envelope: v })}
        />
        <div className="space-y-1">
          <LabelWithTooltip
            label="Accent Pattern"
            tooltip="Which beats get emphasized. Downbeat accents beat 1, backbeat accents beats 2 and 4."
            className="text-[11px] text-muted-foreground"
          />
          <Select
            value={config.noise_breathing.accent}
            onValueChange={v =>
              update("noise_breathing", {
                accent: v as "downbeat" | "backbeat" | "all_equal",
              })
            }
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="downbeat">Downbeat</SelectItem>
              <SelectItem value="backbeat">Backbeat</SelectItem>
              <SelectItem value="all_equal">All Equal</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </EffectSection>

      {/* Prompt Cycling */}
      <EffectSection
        title="Prompt Cycling"
        tooltip="Cycle through a list of prompts on beat boundaries for dramatic scene changes."
        enabled={config.prompt_cycling.enabled}
        onToggle={v => update("prompt_cycling", { enabled: v })}
      >
        <PromptListEditor
          prompts={config.prompt_cycling.prompts}
          onChange={v => update("prompt_cycling", { prompts: v })}
        />
        <BeatIntervalSelect
          value={config.prompt_cycling.beat_interval}
          onChange={v => update("prompt_cycling", { beat_interval: v })}
        />
        <CycleModeSelect
          value={config.prompt_cycling.mode}
          onChange={v => update("prompt_cycling", { mode: v })}
        />
      </EffectSection>

      {/* Reference Image Switching */}
      <EffectSection
        title="Ref Image Switching"
        tooltip="Cycle through reference images on beat boundaries. Requires VACE-enabled pipeline with reference images."
        enabled={config.ref_image_switching.enabled}
        onToggle={v => update("ref_image_switching", { enabled: v })}
      >
        <BeatIntervalSelect
          value={config.ref_image_switching.beat_interval}
          onChange={v => update("ref_image_switching", { beat_interval: v })}
        />
        <CycleModeSelect
          value={config.ref_image_switching.mode}
          onChange={v => update("ref_image_switching", { mode: v })}
        />
        <div className="space-y-1">
          <LabelWithTooltip
            label="Target"
            tooltip="Which image slot to cycle. VACE refs for style conditioning, first frame for image extension."
            className="text-[11px] text-muted-foreground"
          />
          <Select
            value={config.ref_image_switching.target}
            onValueChange={v =>
              update("ref_image_switching", {
                target: v as "vace_ref_images" | "first_frame_image",
              })
            }
          >
            <SelectTrigger className="h-7 text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="vace_ref_images">VACE Refs</SelectItem>
              <SelectItem value="first_frame_image">First Frame</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <p className="text-[10px] text-muted-foreground italic">
          Images are managed via the main reference image controls. Add multiple
          images there, then enable this effect to cycle through them on beat.
        </p>
      </EffectSection>

      {/* Denoising Step Modulation */}
      <EffectSection
        title="Denoising Modulation"
        tooltip="Varies the denoising step schedule on beat. More steps on beat = more detail, fewer between = smoother."
        enabled={config.denoising_modulation.enabled}
        onToggle={v => update("denoising_modulation", { enabled: v })}
      >
        <IntensitySlider
          value={config.denoising_modulation.intensity}
          onChange={v => update("denoising_modulation", { intensity: v })}
        />
        <EnvelopeSelect
          value={config.denoising_modulation.envelope}
          onChange={v => update("denoising_modulation", { envelope: v })}
        />
      </EffectSection>

      {/* VACE Context Pulse */}
      <EffectSection
        title="VACE Context Pulse"
        tooltip="Pulses the VACE context conditioning scale on beat. Higher = stronger reference influence, lower = more freedom."
        enabled={config.vace_context_pulse.enabled}
        onToggle={v => update("vace_context_pulse", { enabled: v })}
      >
        <IntensitySlider
          value={config.vace_context_pulse.min_scale}
          onChange={v => update("vace_context_pulse", { min_scale: v })}
          label="Min Scale"
          tooltip="Minimum VACE context scale (between beats)."
        />
        <IntensitySlider
          value={config.vace_context_pulse.max_scale}
          onChange={v => update("vace_context_pulse", { max_scale: v })}
          label="Max Scale"
          tooltip="Maximum VACE context scale (on beat)."
        />
        <EnvelopeSelect
          value={config.vace_context_pulse.envelope}
          onChange={v => update("vace_context_pulse", { envelope: v })}
        />
      </EffectSection>
    </div>
  );
}

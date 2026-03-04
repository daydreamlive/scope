import { useState, useEffect, useRef } from "react";
import { Toggle } from "../ui/toggle";
import { LabelWithTooltip } from "../ui/label-with-tooltip";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "../ui/select";
import { Button } from "../ui/button";
import type { TempoState } from "../../hooks/useTempoSync";
import type { TempoSourcesResponse, TempoEnableRequest } from "../../lib/api";
import {
  TempoEffectsPanel,
  type TempoEffectsConfig,
} from "./TempoEffectsPanel";

function BeatIndicator({
  beatPhase,
  barPosition,
  beatsPerBar,
}: {
  beatPhase: number;
  barPosition: number;
  beatsPerBar: number;
}) {
  const currentBeat = Math.floor(barPosition);
  const brightness = 1 - beatPhase;

  return (
    <div className="flex items-center gap-1.5">
      {Array.from({ length: beatsPerBar }, (_, i) => {
        const isActive = i === currentBeat;
        return (
          <div
            key={i}
            className="w-3 h-3 rounded-full border border-border transition-all duration-75"
            style={{
              backgroundColor: isActive
                ? `rgba(255, 255, 255, ${0.3 + brightness * 0.7})`
                : "rgba(255, 255, 255, 0.08)",
              boxShadow: isActive
                ? `0 0 ${4 + brightness * 6}px rgba(255, 255, 255, ${brightness * 0.5})`
                : "none",
            }}
          />
        );
      })}
    </div>
  );
}

export function TempoSyncSection({
  tempoState,
  sources,
  loading,
  error,
  onEnable,
  onDisable,
  onSetBpm,
  onRefreshSources,
  onTempoEffectsChange,
}: {
  tempoState: TempoState;
  sources: TempoSourcesResponse | null;
  loading: boolean;
  error: string | null;
  onEnable: (request: TempoEnableRequest) => void;
  onDisable: () => void;
  onSetBpm?: (bpm: number) => void;
  onRefreshSources: () => void;
  onTempoEffectsChange?: (config: TempoEffectsConfig) => void;
}) {
  const [selectedSource, setSelectedSource] = useState<"link" | "midi_clock">(
    "link"
  );
  const [selectedMidiDevice, setSelectedMidiDevice] = useState<string>("");
  const [bpmInput, setBpmInput] = useState("120");
  const [bpmInputFocused, setBpmInputFocused] = useState(false);
  const sourcesLoaded = useRef(false);

  useEffect(() => {
    if (tempoState.bpm !== null && !bpmInputFocused) {
      setBpmInput(String(Math.round(tempoState.bpm)));
    }
  }, [tempoState.bpm, bpmInputFocused]);

  useEffect(() => {
    if (sources && !sourcesLoaded.current) {
      sourcesLoaded.current = true;
      if (sources.sources.link?.available) {
        setSelectedSource("link");
      } else if (sources.sources.midi_clock?.available) {
        setSelectedSource("midi_clock");
        const devices = sources.sources.midi_clock.devices ?? [];
        if (devices.length > 0) {
          setSelectedMidiDevice(devices[0]);
        }
      }
    }
  }, [sources]);

  const handleToggle = (pressed: boolean) => {
    if (pressed) {
      const request: TempoEnableRequest = {
        source: selectedSource,
        bpm: parseFloat(bpmInput) || 120,
      };
      if (selectedSource === "midi_clock" && selectedMidiDevice) {
        request.midi_device = selectedMidiDevice;
      }
      onEnable(request);
    } else {
      onDisable();
    }
  };

  const linkAvailable = sources?.sources.link?.available ?? false;
  const midiAvailable = sources?.sources.midi_clock?.available ?? false;
  const midiDevices = sources?.sources.midi_clock?.devices ?? [];

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <LabelWithTooltip
          label="Tempo Sync"
          tooltip="Synchronize with external tempo sources (Ableton Link, MIDI Clock) for beat-locked visuals."
          className="text-sm font-medium"
        />
        <Toggle
          pressed={tempoState.enabled}
          onPressedChange={handleToggle}
          disabled={loading || (!linkAvailable && !midiAvailable)}
          variant="outline"
          size="sm"
          className="h-7"
        >
          {tempoState.enabled ? "ON" : "OFF"}
        </Toggle>
      </div>

      {!linkAvailable && !midiAvailable && (
        <p className="text-xs text-muted-foreground">
          No tempo sources available. Install{" "}
          <code className="text-[10px]">aalink</code> for Ableton Link or{" "}
          <code className="text-[10px]">mido python-rtmidi</code> for MIDI
          Clock.
        </p>
      )}

      {(linkAvailable || midiAvailable) && (
        <>
          <div className="space-y-2">
            <LabelWithTooltip
              label="Source"
              tooltip="Select the tempo source to synchronize with."
              className="text-xs text-muted-foreground"
            />
            <Select
              value={selectedSource}
              onValueChange={v => setSelectedSource(v as "link" | "midi_clock")}
              disabled={tempoState.enabled || loading}
            >
              <SelectTrigger className="h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {linkAvailable && (
                  <SelectItem value="link">Ableton Link</SelectItem>
                )}
                {midiAvailable && (
                  <SelectItem value="midi_clock">MIDI Clock</SelectItem>
                )}
              </SelectContent>
            </Select>
          </div>

          {selectedSource === "midi_clock" && midiDevices.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <LabelWithTooltip
                  label="MIDI Device"
                  tooltip="Select the MIDI input device to receive clock from."
                  className="text-xs text-muted-foreground"
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 text-xs px-2"
                  onClick={onRefreshSources}
                  disabled={loading}
                >
                  Refresh
                </Button>
              </div>
              <Select
                value={selectedMidiDevice}
                onValueChange={setSelectedMidiDevice}
                disabled={tempoState.enabled || loading}
              >
                <SelectTrigger className="h-8 text-sm">
                  <SelectValue placeholder="Select device..." />
                </SelectTrigger>
                <SelectContent>
                  {midiDevices.map(device => (
                    <SelectItem key={device} value={device}>
                      {device}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
        </>
      )}

      {tempoState.enabled && tempoState.bpm !== null && (
        <div className="rounded-lg border bg-card/50 p-3 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-2xl font-mono font-bold tabular-nums">
              {tempoState.bpm.toFixed(1)}
            </span>
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              BPM
            </span>
          </div>

          <div className="flex items-center justify-between">
            <BeatIndicator
              beatPhase={tempoState.beatPhase}
              barPosition={tempoState.barPosition}
              beatsPerBar={tempoState.beatsPerBar}
            />
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {tempoState.sourceType === "link" &&
                tempoState.numPeers !== null && (
                  <span>
                    {tempoState.numPeers} peer
                    {tempoState.numPeers !== 1 ? "s" : ""}
                  </span>
                )}
              <span className="w-2 h-2 rounded-full bg-green-500" />
            </div>
          </div>

          {onSetBpm && (
            <div className="flex items-center gap-2 pt-1">
              <input
                type="number"
                min={20}
                max={300}
                step={1}
                value={bpmInput}
                onChange={e => setBpmInput(e.target.value)}
                onFocus={() => setBpmInputFocused(true)}
                onBlur={() => setBpmInputFocused(false)}
                onKeyDown={e => {
                  if (e.key === "Enter") {
                    const val = parseFloat(bpmInput);
                    if (val >= 20 && val <= 300) onSetBpm(val);
                  }
                }}
                className="w-16 h-7 text-sm font-mono tabular-nums bg-transparent border border-border/50 focus:border-foreground outline-none text-center rounded"
              />
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs px-2"
                onClick={() => {
                  const val = parseFloat(bpmInput);
                  if (val >= 20 && val <= 300) onSetBpm(val);
                }}
              >
                Set BPM
              </Button>
            </div>
          )}
        </div>
      )}

      {tempoState.enabled && onTempoEffectsChange && (
        <TempoEffectsPanel onConfigChange={onTempoEffectsChange} />
      )}

      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

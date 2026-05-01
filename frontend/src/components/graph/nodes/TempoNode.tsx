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
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import type {
  TempoEnableRequest,
  TempoSourcesResponse,
} from "../../../lib/api";
import { COLOR_NUMBER as COLOR } from "../nodeColors";

type TempoNodeType = Node<FlowNodeData, "tempo">;

const BEAT_RATE_OPTIONS = [
  { value: "beat", label: "Beat" },
  { value: "bar", label: "Bar" },
  { value: "2_bar", label: "2 Bars" },
  { value: "4_bar", label: "4 Bars" },
] as const;

const OUTPUT_LABELS = [
  "bpm",
  "beat_phase",
  "beat_count",
  "bar_position",
  "is_playing",
];

function BeatDot({ phase }: { phase: number }) {
  const brightness = 1 - phase;
  return (
    <div
      className="w-2.5 h-2.5 rounded-full transition-all duration-75"
      style={{
        backgroundColor: `rgba(245, 158, 11, ${0.25 + brightness * 0.75})`,
        boxShadow: `0 0 ${3 + brightness * 5}px rgba(245, 158, 11, ${brightness * 0.6})`,
      }}
    />
  );
}

export function TempoNode({ id, data, selected }: NodeProps<TempoNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();

  const enabled = data.tempoEnabled ?? false;
  const bpm = data.tempoBpm ?? null;
  const beatPhase = data.tempoBeatPhase ?? 0;
  const beatCountRaw = (data.tempoBeatCount as number) ?? 0;
  const beatCountOffset = (data.tempoBeatCountOffset as number) ?? 0;
  const beatCount = beatCountRaw - beatCountOffset;
  const barPosition = data.tempoBarPosition ?? 0;
  const isPlaying = (data.tempoIsPlaying as boolean) ?? false;
  const sourceType = data.tempoSourceType as string | null;
  const numPeers = data.tempoNumPeers as number | null;
  const beatsPerBar = data.tempoBeatsPerBar ?? 4;
  const loading = data.tempoLoading ?? false;
  const tempoError = data.tempoError as string | null | undefined;
  const sources = data.tempoSources as TempoSourcesResponse | null | undefined;
  const streaming = data.isStreaming ?? false;
  const quantizeMode = (data.tempoQuantizeMode as string) ?? "none";
  const lookaheadMs = (data.tempoLookaheadMs as number) ?? 0;
  const beatResetRate = (data.tempoBeatResetRate as string) ?? "none";

  const onEnableTempo = data.onEnableTempo as
    | ((req: TempoEnableRequest) => void)
    | undefined;
  const onDisableTempo = data.onDisableTempo as (() => void) | undefined;
  const onSetTempo = data.onSetTempo as ((bpm: number) => void) | undefined;
  const onRefreshTempoSources = data.onRefreshTempoSources as
    | (() => void)
    | undefined;

  const [selectedSource, setSelectedSource] = useState<"link" | "midi_clock">(
    "link"
  );
  const [selectedMidiDevice, setSelectedMidiDevice] = useState("");
  const [bpmInput, setBpmInput] = useState("120");
  const bpmInputFocusedRef = useRef(false);
  const [bpb, setBpb] = useState(beatsPerBar);

  useEffect(() => {
    if (bpm !== null && !bpmInputFocusedRef.current) {
      setBpmInput(String(Math.round(bpm)));
    }
  }, [bpm]);

  useEffect(() => {
    if (sources) {
      const linkAvail = sources.sources?.link?.available ?? false;
      const midiAvail = sources.sources?.midi_clock?.available ?? false;
      if (linkAvail) setSelectedSource("link");
      else if (midiAvail) setSelectedSource("midi_clock");

      const devices =
        (sources.sources?.midi_clock?.devices as string[] | undefined) ?? [];
      if (devices.length > 0 && !selectedMidiDevice) {
        setSelectedMidiDevice(devices[0]);
      }
    }
  }, [sources, selectedMidiDevice]);

  const linkAvailable = sources?.sources?.link?.available ?? false;
  const midiAvailable = sources?.sources?.midi_clock?.available ?? false;
  const midiDevices =
    (sources?.sources?.midi_clock?.devices as string[] | undefined) ?? [];
  const anyAvailable = linkAvailable || midiAvailable;

  const handleToggle = useCallback(() => {
    if (enabled) {
      onDisableTempo?.();
    } else {
      const request: TempoEnableRequest = {
        source: selectedSource,
        bpm: parseFloat(bpmInput) || 120,
        beats_per_bar: bpb,
      };
      if (selectedSource === "midi_clock" && selectedMidiDevice) {
        request.midi_device = selectedMidiDevice;
      }
      onEnableTempo?.(request);
    }
  }, [
    enabled,
    selectedSource,
    selectedMidiDevice,
    bpmInput,
    bpb,
    onEnableTempo,
    onDisableTempo,
  ]);

  const handleSetBpm = useCallback(() => {
    const val = parseFloat(bpmInput);
    if (val >= 1 && val <= 999 && onSetTempo) {
      onSetTempo(val);
    }
  }, [bpmInput, onSetTempo]);

  const outputValues = streaming
    ? [
        bpm?.toFixed(1) ?? "—",
        beatPhase.toFixed(2),
        String(beatCount),
        barPosition.toFixed(2),
        isPlaying ? "1" : "0",
      ]
    : ["—", "—", "—", "—", "—"];

  const { setRowRef, rowPositions } = useHandlePositions([
    enabled,
    bpm !== null,
    selectedSource,
    midiDevices.length,
    sourceType,
    quantizeMode,
    lookaheadMs,
    beatResetRate,
  ]);

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "Tempo"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          <div className="flex flex-col gap-1.5">
            {/* Enable / source row */}
            <div className="flex items-center gap-1.5 flex-wrap">
              <button
                type="button"
                onClick={handleToggle}
                onPointerDown={e => e.stopPropagation()}
                disabled={loading || !anyAvailable}
                className={`${NODE_TOKENS.pill} text-[9px] cursor-pointer transition-colors shrink-0 ${
                  enabled
                    ? "!bg-amber-500/20 !border-amber-500/40 text-amber-400"
                    : "text-[#aaa] hover:text-[#fff]"
                } ${loading || !anyAvailable ? "opacity-50 cursor-not-allowed" : ""}`}
              >
                {enabled ? "ON" : "OFF"}
              </button>

              {anyAvailable && !enabled && (
                <>
                  <select
                    value={selectedSource}
                    onChange={e =>
                      setSelectedSource(e.target.value as "link" | "midi_clock")
                    }
                    onPointerDown={e => e.stopPropagation()}
                    className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1.5 py-0.5 focus:outline-none cursor-pointer`}
                  >
                    {linkAvailable && <option value="link">Link</option>}
                    {midiAvailable && <option value="midi_clock">MIDI</option>}
                  </select>
                  <input
                    className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[36px] !text-[9px] !px-1 !py-0`}
                    type="number"
                    value={bpmInput}
                    onChange={e => setBpmInput(e.target.value)}
                    onFocus={() => {
                      bpmInputFocusedRef.current = true;
                    }}
                    onBlur={() => {
                      bpmInputFocusedRef.current = false;
                    }}
                    onMouseDown={e => e.stopPropagation()}
                    placeholder="BPM"
                  />
                  <select
                    value={String(bpb)}
                    onChange={e => setBpb(Number(e.target.value))}
                    onPointerDown={e => e.stopPropagation()}
                    className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1 py-0.5 focus:outline-none cursor-pointer`}
                  >
                    {[2, 3, 4, 5, 6, 7, 8].map(n => (
                      <option key={n} value={String(n)}>
                        {n}/4
                      </option>
                    ))}
                  </select>
                </>
              )}

              {!anyAvailable && (
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  No sources
                </span>
              )}
            </div>

            {/* MIDI device selector */}
            {selectedSource === "midi_clock" &&
              !enabled &&
              midiDevices.length > 0 && (
                <div className="flex items-center gap-1">
                  <select
                    value={selectedMidiDevice}
                    onChange={e => setSelectedMidiDevice(e.target.value)}
                    onPointerDown={e => e.stopPropagation()}
                    className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1.5 py-0.5 focus:outline-none cursor-pointer flex-1 min-w-0 truncate`}
                  >
                    {midiDevices.map(device => (
                      <option key={device} value={device}>
                        {device}
                      </option>
                    ))}
                  </select>
                  {onRefreshTempoSources && (
                    <button
                      type="button"
                      onClick={onRefreshTempoSources}
                      onPointerDown={e => e.stopPropagation()}
                      className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors shrink-0 !px-1`}
                    >
                      ↻
                    </button>
                  )}
                </div>
              )}

            {/* Active BPM display */}
            {enabled && bpm !== null && (
              <div className="flex items-center gap-2">
                <BeatDot phase={streaming ? beatPhase : 0} />
                <span className="text-[14px] font-mono font-bold tabular-nums text-[#fafafa]">
                  {bpm.toFixed(1)}
                </span>
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  BPM
                </span>
                {sourceType === "link" && numPeers !== null && (
                  <span
                    className={`${NODE_TOKENS.labelText} text-[8px] ml-auto`}
                  >
                    {numPeers} peer{numPeers !== 1 ? "s" : ""}
                  </span>
                )}
                {onRefreshTempoSources && (
                  <button
                    type="button"
                    onClick={onRefreshTempoSources}
                    onPointerDown={e => e.stopPropagation()}
                    className={`${NODE_TOKENS.labelText} text-[8px] hover:text-[#fff] transition-colors cursor-pointer`}
                  >
                    ↻
                  </button>
                )}
              </div>
            )}

            {/* Set BPM (Link only, when enabled) */}
            {enabled && sourceType === "link" && onSetTempo && (
              <div className="flex items-center gap-1">
                <input
                  className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[40px] !text-[9px] !px-1 !py-0`}
                  type="number"
                  value={bpmInput}
                  onChange={e => setBpmInput(e.target.value)}
                  onFocus={() => {
                    bpmInputFocusedRef.current = true;
                  }}
                  onBlur={() => {
                    bpmInputFocusedRef.current = false;
                  }}
                  onKeyDown={e => {
                    if (e.key === "Enter") handleSetBpm();
                  }}
                  onMouseDown={e => e.stopPropagation()}
                  placeholder="BPM"
                />
                <button
                  type="button"
                  onClick={handleSetBpm}
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors`}
                >
                  Set
                </button>
              </div>
            )}

            {/* Beat Quantize (when enabled) */}
            {enabled && (
              <div className="flex items-center justify-between gap-1">
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  Quantize
                </span>
                <select
                  value={quantizeMode}
                  onChange={e =>
                    updateData({ tempoQuantizeMode: e.target.value })
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1 py-0 focus:outline-none cursor-pointer`}
                >
                  <option value="none">Off</option>
                  {BEAT_RATE_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Lookahead (when quantize active) */}
            {enabled && quantizeMode !== "none" && (
              <div className="flex items-center justify-between gap-1">
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  Lookahead
                </span>
                <div className="flex items-center gap-0.5">
                  <input
                    className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[32px] !text-[9px] !px-0.5 !py-0 text-center`}
                    type="number"
                    min={0}
                    max={1000}
                    step={10}
                    value={lookaheadMs}
                    onChange={e =>
                      updateData({
                        tempoLookaheadMs: Number(e.target.value) || 0,
                      })
                    }
                    onMouseDown={e => e.stopPropagation()}
                  />
                  <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                    ms
                  </span>
                </div>
              </div>
            )}

            {/* Beat Reset (when enabled) */}
            {enabled && (
              <div className="flex items-center justify-between gap-1">
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  Beat Reset
                </span>
                <select
                  value={beatResetRate}
                  onChange={e =>
                    updateData({ tempoBeatResetRate: e.target.value })
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] rounded-full px-1 py-0 focus:outline-none cursor-pointer`}
                >
                  <option value="none">Off</option>
                  {BEAT_RATE_OPTIONS.map(opt => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Reset beat count */}
            {enabled && (
              <div className="flex items-center justify-between gap-1">
                <span className={`${NODE_TOKENS.labelText} text-[8px]`}>
                  Beat Count
                </span>
                <button
                  type="button"
                  onClick={() =>
                    updateData({ tempoBeatCountOffset: beatCountRaw })
                  }
                  onPointerDown={e => e.stopPropagation()}
                  className={`${NODE_TOKENS.pill} text-[9px] text-[#aaa] hover:text-[#fff] cursor-pointer transition-colors`}
                >
                  Reset
                </button>
              </div>
            )}

            {/* Error display */}
            {tempoError && (
              <span
                className={`${NODE_TOKENS.labelText} text-[8px] text-red-400`}
              >
                {tempoError}
              </span>
            )}

            {/* Output labels */}
            <div className="flex flex-col gap-0.5">
              {OUTPUT_LABELS.map((label, i) => (
                <div
                  key={label}
                  ref={setRowRef(label)}
                  className="flex items-center justify-between h-[20px]"
                >
                  <span className={`${NODE_TOKENS.labelText} text-[9px]`}>
                    {label}
                  </span>
                  <span className="text-[9px] font-mono tabular-nums text-[#aaa]">
                    {outputValues[i]}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </NodeBody>
      )}

      {/* Output handles */}
      {OUTPUT_LABELS.map(label => (
        <Handle
          key={label}
          type="source"
          position={Position.Right}
          id={buildHandleId("param", label)}
          className={
            collapsed
              ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
              : "!w-2.5 !h-2.5 !border-0"
          }
          style={
            collapsed
              ? { ...collapsedHandleStyle("right"), opacity: 0 }
              : {
                  top: rowPositions[label] ?? 0,
                  right: 0,
                  backgroundColor: COLOR,
                }
          }
        />
      ))}
    </NodeCard>
  );
}

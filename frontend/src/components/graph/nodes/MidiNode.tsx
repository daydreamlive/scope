import { Handle, Position, useUpdateNodeInternals } from "@xyflow/react";
import type { NodeProps, Node } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";
import type { FlowNodeData } from "../../../lib/graphUtils";
import { buildHandleId } from "../../../lib/graphUtils";
import { useNodeData } from "../hooks/node/useNodeData";
import { useNodeCollapse } from "../hooks/node/useNodeCollapse";
import {
  NodeCard,
  NodeHeader,
  NodeBody,
  NodeParamRow,
  NodePillSelect,
  NODE_TOKENS,
  collapsedHandleStyle,
} from "../ui";
import { COLOR_NUMBER as COLOR } from "../nodeColors";

type MidiNodeType = Node<FlowNodeData, "midi">;

/** Height of a single channel row (used for handle positioning). */
const ROW_HEIGHT = 32;
const HEADER_HEIGHT = 28;
const BODY_PAD_TOP = 6;
const DEVICE_ROW_HEIGHT = 30;

export interface MidiChannelDef {
  label: string;
  type: "cc" | "note";
  channel: number;
  cc: number;
  value: number;
}

function defaultChannel(index: number): MidiChannelDef {
  return {
    label: `CC ${index + 1}`,
    type: "cc",
    channel: 0,
    cc: index + 1,
    value: 0,
  };
}

/** A single MIDI channel row with learn button, live value indicator, and settings */
function MidiChannelRow({
  ch,
  index,
  onFieldChange,
  onRemove,
  canRemove,
  onLearn,
  isLearning,
}: {
  ch: MidiChannelDef;
  index: number;
  onFieldChange: (
    index: number,
    field: keyof MidiChannelDef,
    value: string | number
  ) => void;
  onRemove: (index: number) => void;
  canRemove: boolean;
  onLearn: (index: number) => void;
  isLearning: boolean;
}) {
  const pct = Math.min(Math.max(ch.value, 0), 1) * 100;

  return (
    <div
      className="flex items-center gap-1 relative"
      style={{ height: ROW_HEIGHT }}
    >
      {/* Value bar background */}
      <div
        className="absolute left-0 top-0 h-full rounded pointer-events-none"
        style={{
          width: `${pct}%`,
          background: COLOR,
          opacity: 0.12,
          transition: "width 60ms linear",
        }}
      />

      {/* Learn button */}
      <button
        className={`w-[18px] h-[18px] rounded text-[8px] font-bold flex items-center justify-center shrink-0 transition-colors z-10 ${
          isLearning
            ? "bg-cyan-500 text-black animate-pulse"
            : "bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-cyan-400 hover:border-cyan-400/40"
        }`}
        onClick={() => onLearn(index)}
        title={isLearning ? "Listening..." : "Learn MIDI"}
      >
        L
      </button>

      {/* Type badge */}
      <span
        className="text-[8px] font-mono text-[#999] uppercase w-[16px] text-center shrink-0 z-10"
        title={ch.type === "cc" ? "Control Change" : "Note On/Off"}
      >
        {ch.type === "cc" ? "CC" : "N"}
      </span>

      {/* CC/Note number */}
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[24px] !text-[8px] !px-0.5 !py-0 shrink-0 z-10`}
        type="number"
        min={0}
        max={127}
        value={ch.cc}
        onChange={e =>
          onFieldChange(
            index,
            "cc",
            Math.min(127, Math.max(0, parseInt(e.target.value) || 0))
          )
        }
        onMouseDown={e => e.stopPropagation()}
        title={ch.type === "cc" ? "CC Number" : "Note Number"}
      />

      {/* Channel */}
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputNumber} !w-[20px] !text-[8px] !px-0.5 !py-0 shrink-0 z-10`}
        type="number"
        min={0}
        max={15}
        value={ch.channel}
        onChange={e =>
          onFieldChange(
            index,
            "channel",
            Math.min(15, Math.max(0, parseInt(e.target.value) || 0))
          )
        }
        onMouseDown={e => e.stopPropagation()}
        title="MIDI Channel (0-15)"
      />

      {/* Label */}
      <input
        className={`${NODE_TOKENS.pillInput} ${NODE_TOKENS.pillInputText} !w-auto flex-1 min-w-0 !text-[9px] !px-1 !py-0 z-10`}
        value={ch.label}
        onChange={e => onFieldChange(index, "label", e.target.value)}
        onMouseDown={e => e.stopPropagation()}
      />

      {/* Live value */}
      <span
        className={`${NODE_TOKENS.primaryText} w-[28px] text-center shrink-0 text-[9px] z-10`}
      >
        {ch.value.toFixed(2)}
      </span>

      {/* Remove button */}
      {canRemove && (
        <button
          className="w-4 h-4 rounded text-[#555] hover:text-red-400 text-[10px] flex items-center justify-center leading-none shrink-0 z-10"
          onClick={() => onRemove(index)}
        >
          ×
        </button>
      )}
    </div>
  );
}

export function MidiNode({ id, data, selected }: NodeProps<MidiNodeType>) {
  const { updateData } = useNodeData(id);
  const { collapsed, toggleCollapse } = useNodeCollapse();
  const updateNodeInternals = useUpdateNodeInternals();
  const channels: MidiChannelDef[] =
    data.midiChannels && data.midiChannels.length > 0
      ? data.midiChannels
      : [defaultChannel(0)];
  const prevChannelCount = useRef(channels.length);

  const [devices, setDevices] = useState<MIDIInput[]>([]);
  const [learningIndex, setLearningIndex] = useState<number | null>(null);
  const midiAccessRef = useRef<MIDIAccess | null>(null);
  const listenerRef = useRef<((e: MIDIMessageEvent) => void) | null>(null);
  const selectedInputRef = useRef<MIDIInput | null>(null);
  const channelsRef = useRef(channels);
  channelsRef.current = channels;
  const learningIndexRef = useRef<number | null>(null);

  // Keep learningIndex ref in sync
  useEffect(() => {
    learningIndexRef.current = learningIndex;
  }, [learningIndex]);

  // Request MIDI access on mount
  useEffect(() => {
    if (!navigator.requestMIDIAccess) return;
    navigator.requestMIDIAccess({ sysex: false }).then(
      access => {
        midiAccessRef.current = access;
        const inputs = Array.from(access.inputs.values());
        setDevices(inputs);

        access.onstatechange = () => {
          setDevices(Array.from(access.inputs.values()));
        };
      },
      () => {
        /* MIDI not available */
      }
    );
  }, []);

  // Connect to selected device & listen for messages
  useEffect(() => {
    const access = midiAccessRef.current;
    if (!access) return;

    // Disconnect old listener
    if (selectedInputRef.current && listenerRef.current) {
      selectedInputRef.current.removeEventListener(
        "midimessage",
        listenerRef.current as EventListener
      );
    }

    const deviceId = data.midiDeviceId;
    if (!deviceId) {
      selectedInputRef.current = null;
      return;
    }

    const input = access.inputs.get(deviceId);
    if (!input) {
      selectedInputRef.current = null;
      return;
    }

    selectedInputRef.current = input;

    const handler = (event: MIDIMessageEvent) => {
      const d = event.data;
      if (!d || d.length < 2) return;

      const status = d[0];
      const command = status & 0xf0;
      const midiChannel = status & 0x0f;
      const noteOrCC = d[1];
      const value = d.length > 2 ? d[2] : 0;
      const normalized = value / 127;

      // Learn mode — capture first CC or Note On
      if (learningIndexRef.current !== null) {
        const idx = learningIndexRef.current;
        const chs = channelsRef.current;
        if (idx < chs.length) {
          const updated = [...chs];
          if (command === 0xb0) {
            // CC
            updated[idx] = {
              ...updated[idx],
              type: "cc",
              channel: midiChannel,
              cc: noteOrCC,
              label: `CC ${noteOrCC}`,
              value: normalized,
            };
          } else if (command === 0x90 && value > 0) {
            // Note On
            updated[idx] = {
              ...updated[idx],
              type: "note",
              channel: midiChannel,
              cc: noteOrCC,
              label: `Note ${noteOrCC}`,
              value: normalized,
            };
          } else {
            return; // ignore other messages during learn
          }
          updateData({ midiChannels: updated });
          setLearningIndex(null);
        }
        return;
      }

      // Normal: match to channels and update values
      const chs = channelsRef.current;
      let changed = false;
      const updated = chs.map(ch => {
        if (ch.channel !== midiChannel) return ch;
        if (ch.type === "cc" && command === 0xb0 && ch.cc === noteOrCC) {
          changed = true;
          return { ...ch, value: normalized };
        }
        if (ch.type === "note" && ch.cc === noteOrCC) {
          if (command === 0x90 && value > 0) {
            changed = true;
            return { ...ch, value: normalized };
          }
          if (command === 0x80 || (command === 0x90 && value === 0)) {
            changed = true;
            return { ...ch, value: 0 };
          }
        }
        return ch;
      });

      if (changed) {
        updateData({ midiChannels: updated });
      }
    };

    listenerRef.current = handler;
    input.addEventListener("midimessage", handler as EventListener);

    return () => {
      if (input && handler) {
        input.removeEventListener("midimessage", handler as EventListener);
      }
    };
  }, [data.midiDeviceId, devices, updateData]);

  useEffect(() => {
    if (channels.length !== prevChannelCount.current) {
      prevChannelCount.current = channels.length;
      updateData({
        parameterOutputs: channels.map((_, i) => ({
          name: `midi_${i}`,
          type: "number" as const,
          defaultValue: 0,
        })),
      });
      updateNodeInternals(id);
    }
  }, [channels.length, id, updateData, updateNodeInternals]);

  const updateChannels = useCallback(
    (newChannels: MidiChannelDef[]) => {
      updateData({ midiChannels: newChannels });
    },
    [updateData]
  );

  const handleFieldChange = useCallback(
    (index: number, field: keyof MidiChannelDef, value: string | number) => {
      const updated = [...channels];
      updated[index] = { ...updated[index], [field]: value };
      updateChannels(updated);
    },
    [channels, updateChannels]
  );

  const handleRemove = useCallback(
    (index: number) => {
      if (channels.length <= 1) return;
      const updated = channels.filter((_, i) => i !== index);
      updateChannels(updated);
    },
    [channels, updateChannels]
  );

  const handleAdd = useCallback(() => {
    updateChannels([...channels, defaultChannel(channels.length)]);
  }, [channels, updateChannels]);

  const handleLearn = useCallback(
    (index: number) => {
      if (learningIndex === index) {
        setLearningIndex(null);
      } else {
        setLearningIndex(index);
      }
    },
    [learningIndex]
  );

  const deviceOptions = devices.map(d => ({
    value: d.id,
    label: d.name || d.id,
  }));

  return (
    <NodeCard
      selected={selected}
      autoMinHeight={!collapsed}
      collapsed={collapsed}
    >
      <NodeHeader
        title={data.customTitle || "MIDI"}
        onTitleChange={newTitle => updateData({ customTitle: newTitle })}
        collapsed={collapsed}
        onCollapseToggle={toggleCollapse}
      />
      {!collapsed && (
        <NodeBody>
          {/* Device selector */}
          <NodeParamRow label="Device">
            {deviceOptions.length > 0 ? (
              <NodePillSelect
                value={data.midiDeviceId || ""}
                options={[{ value: "", label: "None" }, ...deviceOptions]}
                onChange={v => updateData({ midiDeviceId: v || undefined })}
              />
            ) : (
              <span className="text-[9px] text-[#666] italic">
                No MIDI devices
              </span>
            )}
          </NodeParamRow>

          {/* Channel rows */}
          <div className="flex flex-col gap-0.5 py-0.5">
            {channels.map((ch, i) => (
              <MidiChannelRow
                key={i}
                ch={ch}
                index={i}
                onFieldChange={handleFieldChange}
                onRemove={handleRemove}
                canRemove={channels.length > 1}
                onLearn={handleLearn}
                isLearning={learningIndex === i}
              />
            ))}
          </div>

          {/* Add button */}
          <button
            className="w-full py-0.5 mt-0.5 rounded bg-[#1b1a1a] border border-[rgba(119,119,119,0.15)] text-[#888] hover:text-[#ccc] hover:border-[rgba(119,119,119,0.4)] text-[10px] transition-colors"
            onClick={handleAdd}
          >
            + Add Channel
          </button>
        </NodeBody>
      )}

      {/* Output handles for each channel, vertically aligned to each row */}
      {channels.map((_, i) => {
        const yOffset =
          HEADER_HEIGHT +
          BODY_PAD_TOP +
          DEVICE_ROW_HEIGHT +
          ROW_HEIGHT * i +
          ROW_HEIGHT / 2;
        const isFirst = i === 0;
        return (
          <Handle
            key={`handle-${i}`}
            type="source"
            position={Position.Right}
            id={buildHandleId("param", `midi_${i}`)}
            className={
              collapsed && !isFirst
                ? "!w-0 !h-0 !border-0 !min-w-0 !min-h-0"
                : "!w-2.5 !h-2.5 !border-0"
            }
            style={
              collapsed
                ? isFirst
                  ? collapsedHandleStyle("right")
                  : { ...collapsedHandleStyle("right"), opacity: 0 }
                : { top: yOffset, right: 0, backgroundColor: COLOR }
            }
          />
        );
      })}
    </NodeCard>
  );
}

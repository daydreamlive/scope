import { useEffect, useRef, useCallback, useState } from "react";
import type { MIDIMappingProfile } from "../types/midi";

const CONTINUOUS_THROTTLE_MS = 16; // ~60Hz max for continuous CC sends

export interface MIDIControllerConfig {
  mappingProfile?: MIDIMappingProfile;
  enabled?: boolean;
  deviceId?: string;
  currentDenoisingSteps?: number[];
  onDenoisingStepsChange?: (steps: number[]) => void;
  onLearnComplete?: (
    mappingIndex: number,
    source: import("../types/midi").MIDISource
  ) => void;
  currentNoiseController?: boolean;
  onSwitchPrompt?: (index: number) => void;
  onPromptWeightChange?: (index: number, weight: number) => void;
  onParameterActivity?: (paramId: string) => void;
  onPlayPauseToggle?: () => void;
}

export function useMIDIController(
  sendParameterUpdate: (params: Record<string, unknown>) => void,
  config?: MIDIControllerConfig
) {
  const {
    mappingProfile,
    enabled = false,
    deviceId,
    currentDenoisingSteps,
    onDenoisingStepsChange,
    onLearnComplete,
    currentNoiseController,
    onSwitchPrompt,
    onPromptWeightChange,
    onParameterActivity,
    onPlayPauseToggle,
  } = config || {};

  const [midiAccess, setMidiAccess] = useState<MIDIAccess | null>(null);
  const [devices, setDevices] = useState<MIDIInput[]>([]);
  const [selectedInput, setSelectedInput] = useState<MIDIInput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const toggleStatesRef = useRef<Map<string, boolean>>(new Map());
  const enumStatesRef = useRef<Map<string, number>>(new Map());
  const lastCCStateRef = useRef<Map<string, number>>(new Map());
  const learningMappingIndexRef = useRef<number | null>(null);
  const throttleTimersRef = useRef<Map<string, number>>(new Map());
  const pendingUpdatesRef = useRef<Map<string, Record<string, unknown>>>(
    new Map()
  );

  // Sync toggle states from current settings
  useEffect(() => {
    if (currentNoiseController !== undefined) {
      toggleStatesRef.current.set("noise_controller", currentNoiseController);
    }
  }, [currentNoiseController]);

  // Refs to avoid stale closures in the stable MIDI handler
  const sendParameterUpdateRef = useRef(sendParameterUpdate);
  const enabledRef = useRef(enabled);
  const mappingProfileRef = useRef(mappingProfile);
  const currentDenoisingStepsRef = useRef(currentDenoisingSteps);
  const onDenoisingStepsChangeRef = useRef(onDenoisingStepsChange);
  const onLearnCompleteRef = useRef(onLearnComplete);
  const onSwitchPromptRef = useRef(onSwitchPrompt);
  const onPromptWeightChangeRef = useRef(onPromptWeightChange);
  const onParameterActivityRef = useRef(onParameterActivity);
  const onPlayPauseToggleRef = useRef(onPlayPauseToggle);

  sendParameterUpdateRef.current = sendParameterUpdate;
  enabledRef.current = enabled;
  mappingProfileRef.current = mappingProfile;
  currentDenoisingStepsRef.current = currentDenoisingSteps;
  onDenoisingStepsChangeRef.current = onDenoisingStepsChange;
  onLearnCompleteRef.current = onLearnComplete;
  onSwitchPromptRef.current = onSwitchPrompt;
  onPromptWeightChangeRef.current = onPromptWeightChange;
  onParameterActivityRef.current = onParameterActivity;
  onPlayPauseToggleRef.current = onPlayPauseToggle;

  // Throttled send for continuous CC — batches rapid updates per parameter key
  const throttledSendRef = useRef(
    (paramKey: string, params: Record<string, unknown>) => {
      pendingUpdatesRef.current.set(paramKey, params);
      if (throttleTimersRef.current.has(paramKey)) return;
      // Send immediately on first call, then throttle subsequent
      sendParameterUpdateRef.current(params);
      throttleTimersRef.current.set(
        paramKey,
        window.setTimeout(() => {
          throttleTimersRef.current.delete(paramKey);
          const pending = pendingUpdatesRef.current.get(paramKey);
          if (pending) {
            pendingUpdatesRef.current.delete(paramKey);
            sendParameterUpdateRef.current(pending);
          }
        }, CONTINUOUS_THROTTLE_MS)
      );
    }
  );

  const updateDeviceList = useCallback((access: MIDIAccess) => {
    const inputs: MIDIInput[] = [];
    access.inputs.forEach(input => inputs.push(input));
    setDevices(inputs);
  }, []);

  useEffect(() => {
    if (!enabled) {
      // Clear MIDI access and devices when disabled
      setMidiAccess(null);
      setDevices([]);
      setSelectedInput(null);
      setError(null);
      return;
    }

    if (typeof navigator === "undefined" || !navigator.requestMIDIAccess) {
      setError("Web MIDI API not available in this browser");
      return;
    }

    let pollTimer: number | undefined;
    navigator
      .requestMIDIAccess({ sysex: false })
      .then(access => {
        setMidiAccess(access);
        updateDeviceList(access);
        access.addEventListener("statechange", () => updateDeviceList(access));

        // Poll briefly to catch already-connected devices that enumerate late
        let polls = 0;
        const poll = () => {
          polls++;
          updateDeviceList(access);
          if (polls < 5) pollTimer = window.setTimeout(poll, 200);
        };
        pollTimer = window.setTimeout(poll, 100);
      })
      .catch(err => {
        setError(`Failed to access MIDI: ${err.message}`);
      });

    return () => {
      if (pollTimer !== undefined) clearTimeout(pollTimer);
    };
  }, [enabled, updateDeviceList]);

  useEffect(() => {
    if (!midiAccess || !deviceId) {
      setSelectedInput(null);
      return;
    }
    setSelectedInput(midiAccess.inputs.get(deviceId) || null);
  }, [midiAccess, deviceId]);

  // Stable MIDI message handler — reads everything from refs
  const handleMIDIMessage = useCallback((event: MIDIMessageEvent) => {
    const data = event.data;
    if (!data || data.length < 2) return;

    const status = data[0];
    const command = status & 0xf0;
    const channel = status & 0x0f;
    const noteOrCC = data[1];
    const value = data.length > 2 ? data[2] : 0;

    // Learn mode — capture first CC or Note On
    if (learningMappingIndexRef.current !== null) {
      const mappingIndex = learningMappingIndexRef.current;
      const source: import("../types/midi").MIDISource = { channel };

      if (command === 0xb0) {
        source.midi_cc = noteOrCC;
        learningMappingIndexRef.current = null;
        onLearnCompleteRef.current?.(mappingIndex, source);
        return;
      }
      if (command === 0x90 && value > 0) {
        source.midi_note = noteOrCC;
        learningMappingIndexRef.current = null;
        onLearnCompleteRef.current?.(mappingIndex, source);
        return;
      }
      return;
    }

    // Normal processing
    const enabled = enabledRef.current;
    const mappingProfile = mappingProfileRef.current;
    const sendParameterUpdate = sendParameterUpdateRef.current;
    const currentDenoisingSteps = currentDenoisingStepsRef.current;
    const onDenoisingStepsChange = onDenoisingStepsChangeRef.current;
    const onSwitchPrompt = onSwitchPromptRef.current;

    if (!enabled || !mappingProfile?.mappings.length) return;

    const BOOLEAN_PARAMS = ["noise_controller"];

    for (const mapping of mappingProfile.mappings) {
      const source = mapping.source;
      if (source.channel !== channel) continue;

      // --- Continuous (CC) mappings ---
      if (mapping.type === "continuous" && source.midi_cc !== undefined) {
        if (command !== 0xb0 || noteOrCC !== source.midi_cc) continue;
        if (!mapping.target.parameter) continue;

        const notifyActivity = onParameterActivityRef.current;

        // Boolean params sent via CC: treat as toggle on press, ignore release
        if (BOOLEAN_PARAMS.includes(mapping.target.parameter)) {
          const ccKey = `cc_${source.midi_cc}_ch${source.channel}`;
          if (value > 0) {
            const last = lastCCStateRef.current.get(ccKey) ?? 0;
            if (last <= 0) {
              const key = mapping.target.parameter;
              const cur = toggleStatesRef.current.get(key) ?? false;
              toggleStatesRef.current.set(key, !cur);
              sendParameterUpdate({ [key]: !cur });
              notifyActivity?.(key);
            }
            lastCCStateRef.current.set(ccKey, value);
          } else {
            lastCCStateRef.current.set(ccKey, value);
          }
          continue;
        }

        // Array parameter (denoising steps)
        if (
          mapping.target.parameter === "denoising_step_list" &&
          mapping.target.arrayIndex !== undefined
        ) {
          if (!currentDenoisingSteps || !onDenoisingStepsChange) continue;
          const idx = mapping.target.arrayIndex;
          if (idx < 0 || idx >= currentDenoisingSteps.length) continue;

          const normalized = value / 127.0;
          const min = mapping.range?.min ?? 0;
          const max = mapping.range?.max ?? 1000;
          const paramValue = Math.round(min + normalized * (max - min));

          const steps = [...currentDenoisingSteps];
          steps[idx] = paramValue;

          // Push outward from changed index to maintain descending order
          for (let i = idx + 1; i < steps.length; i++) {
            if (steps[i] >= steps[i - 1])
              steps[i] = Math.max(0, steps[i - 1] - 1);
          }
          for (let i = idx - 1; i >= 0; i--) {
            if (steps[i] <= steps[i + 1])
              steps[i] = Math.min(1000, steps[i + 1] + 1);
          }

          onDenoisingStepsChange(steps);
          throttledSendRef.current(`denoising_step_list[${idx}]`, {
            denoising_step_list: steps,
          });
          notifyActivity?.(`denoising_step_list[${idx}]`);
          continue;
        }

        if (
          mapping.target.parameter === "prompt_weight" &&
          mapping.target.arrayIndex !== undefined
        ) {
          const idx = mapping.target.arrayIndex;
          const min = mapping.range?.min ?? 0;
          const max = mapping.range?.max ?? 100;
          const normalized = value / 127.0;
          const mappedWeight = Math.round(min + normalized * (max - min));

          onPromptWeightChangeRef.current?.(idx, mappedWeight);
          notifyActivity?.(`prompt_weight[${idx}]`);
          continue;
        }

        // Scalar parameter — throttle to avoid flooding the data channel
        const normalized = value / 127.0;
        const min = mapping.range?.min ?? 0;
        const max = mapping.range?.max ?? 1;
        throttledSendRef.current(mapping.target.parameter, {
          [mapping.target.parameter]: min + normalized * (max - min),
        });
        notifyActivity?.(mapping.target.parameter);
        continue;
      }

      // --- Toggle / Trigger / Enum cycle (Note On or CC press) ---
      if (
        mapping.type === "toggle" ||
        mapping.type === "trigger" ||
        mapping.type === "enum_cycle"
      ) {
        const isNoteOn =
          command === 0x90 &&
          source.midi_note !== undefined &&
          noteOrCC === source.midi_note &&
          value > 0;
        const isCCMatch =
          command === 0xb0 &&
          source.midi_cc !== undefined &&
          noteOrCC === source.midi_cc;
        const ccKey =
          source.midi_cc !== undefined
            ? `cc_${source.midi_cc}_ch${source.channel}`
            : null;

        // CC release — update state, skip
        if (isCCMatch && value <= 0 && ccKey) {
          lastCCStateRef.current.set(ccKey, value);
          continue;
        }

        // CC press — debounce repeated presses
        const isCCPress = isCCMatch && value > 0;
        if (isCCPress && ccKey) {
          if ((lastCCStateRef.current.get(ccKey) ?? 0) > 0) continue;
          lastCCStateRef.current.set(ccKey, value);
        }

        if (!(isNoteOn || isCCPress)) continue;

        const notifyAct = onParameterActivityRef.current;

        if (mapping.type === "toggle") {
          if (!mapping.target.parameter) continue;
          const key = mapping.target.parameter;
          const cur = toggleStatesRef.current.get(key) ?? false;
          toggleStatesRef.current.set(key, !cur);
          sendParameterUpdate({ [key]: !cur });
          notifyAct?.(key);
        } else if (mapping.type === "trigger") {
          const target = mapping.target;
          const actionId = target.action ?? "";

          if (
            target.action === "switch_prompt" &&
            target.prompt_index !== undefined
          ) {
            onSwitchPrompt?.(target.prompt_index);
          } else if (target.action?.startsWith("switch_prompt_")) {
            const idx = parseInt(
              target.action.replace("switch_prompt_", ""),
              10
            );
            if (!isNaN(idx) && idx >= 0 && idx < 4) onSwitchPrompt?.(idx);
          } else if (target.action === "reset_cache") {
            sendParameterUpdate({ reset_cache: true });
          } else if (target.action === "toggle_pause") {
            onPlayPauseToggleRef.current?.();
          } else if (target.action === "add_denoising_step") {
            if (!currentDenoisingSteps || !onDenoisingStepsChange) continue;
            if (currentDenoisingSteps.length >= 10) continue;
            const last =
              currentDenoisingSteps[currentDenoisingSteps.length - 1];
            const steps = [...currentDenoisingSteps, Math.max(0, last - 100)];
            onDenoisingStepsChange(steps);
            sendParameterUpdate({ denoising_step_list: steps });
          } else if (target.action === "remove_denoising_step") {
            if (!currentDenoisingSteps || !onDenoisingStepsChange) continue;
            if (currentDenoisingSteps.length <= 1) continue;
            const steps = currentDenoisingSteps.slice(0, -1);
            onDenoisingStepsChange(steps);
            sendParameterUpdate({ denoising_step_list: steps });
          }

          if (actionId) notifyAct?.(actionId);
        } else if (mapping.type === "enum_cycle") {
          if (!mapping.target.parameter || !mapping.target.values) continue;
          const key = mapping.target.parameter;
          const values = mapping.target.values;
          const nextIdx =
            ((enumStatesRef.current.get(key) ?? 0) + 1) % values.length;
          enumStatesRef.current.set(key, nextIdx);
          sendParameterUpdate({ [key]: values[nextIdx] });
          notifyAct?.(key);
        }
      }
    }
  }, []);

  useEffect(() => {
    if (!selectedInput) return;
    const handler = (event: MIDIMessageEvent) => {
      if (event.data) handleMIDIMessage(event);
    };
    selectedInput.addEventListener("midimessage", handler);
    return () => selectedInput.removeEventListener("midimessage", handler);
  }, [selectedInput, handleMIDIMessage]);

  // Cleanup throttle timers on unmount
  useEffect(() => {
    const timers = throttleTimersRef.current;
    return () => {
      timers.forEach(timer => clearTimeout(timer));
      timers.clear();
    };
  }, []);

  const startLearn = useCallback((mappingIndex: number) => {
    learningMappingIndexRef.current = mappingIndex;
  }, []);

  const cancelLearn = useCallback(() => {
    learningMappingIndexRef.current = null;
  }, []);

  return {
    devices,
    selectedInput,
    error,
    isEnabled: enabled && selectedInput !== null,
    startLearn,
    cancelLearn,
  };
}

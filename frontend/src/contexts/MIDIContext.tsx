import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef } from "react";
import { useMIDIController } from "../hooks/useMIDIController";
import type { MIDIMappingProfile, MIDISource, MIDIMapping, MIDIMappingType } from "../types/midi";
import { toast } from "sonner";

const MAPPING_PROFILES_KEY = "scope_midi_mapping_profiles";
const MIDI_ENABLED_KEY = "scope_midi_enabled";
const MIDI_DEVICE_KEY = "scope_midi_device_id";

const EMPTY_PROFILE: MIDIMappingProfile = { name: "Default", mappings: [] };

function loadMappingProfiles(): MIDIMappingProfile[] {
  try {
    const stored = localStorage.getItem(MAPPING_PROFILES_KEY);
    if (!stored) return [EMPTY_PROFILE];
    return JSON.parse(stored) as MIDIMappingProfile[];
  } catch {
    return [EMPTY_PROFILE];
  }
}

function loadMIDIEnabled(): boolean {
  // Always start with MIDI disabled to prevent permission requests on page load
  // User must explicitly enable it via the settings toggle
  return false;
}

function loadMIDIDeviceId(): string {
  try { return localStorage.getItem(MIDI_DEVICE_KEY) || ""; }
  catch { return ""; }
}

function saveMappingProfiles(profiles: MIDIMappingProfile[]) {
  try {
    localStorage.setItem(MAPPING_PROFILES_KEY, JSON.stringify(profiles));
  } catch {
    toast.error("Failed to save MIDI mapping");
  }
}

interface MIDIContextValue {
  midiEnabled: boolean;
  setMidiEnabled: (enabled: boolean) => void;
  selectedDeviceId: string;
  setSelectedDeviceId: (deviceId: string) => void;
  devices: MIDIInput[];
  mappingProfile: MIDIMappingProfile;
  setMappingProfile: (profile: MIDIMappingProfile) => void;
  isMappingMode: boolean;
  setMappingMode: (enabled: boolean) => void;
  learningParameter: string | null;
  error: string | null;
  startLearning: (parameterId: string, arrayIndex?: number, actionId?: string, mappingType?: "continuous" | "toggle" | "trigger" | "enum_cycle", range?: { min: number; max: number }, enumValues?: string[]) => void;
  cancelLearning: () => void;
  getMappedSource: (parameterId: string, arrayIndex?: number, actionId?: string) => string | null;
  deleteMapping: (index: number) => void;
  clearAllMappings: () => void;
}

const MIDIContext = createContext<MIDIContextValue | null>(null);

interface MIDIProviderProps {
  children: React.ReactNode;
  sendParameterUpdate: (params: Record<string, unknown>) => void;
  currentDenoisingSteps?: number[];
  onDenoisingStepsChange?: (steps: number[]) => void;
  currentNoiseController?: boolean;
  currentManageCache?: boolean;
  onSwitchPrompt?: (index: number) => void;
}

export function MIDIProvider({
  children,
  sendParameterUpdate,
  currentDenoisingSteps,
  onDenoisingStepsChange,
  currentNoiseController,
  currentManageCache,
  onSwitchPrompt,
}: MIDIProviderProps) {
  const [midiEnabled, setMidiEnabledState] = useState(loadMIDIEnabled);
  const [selectedDeviceId, setSelectedDeviceIdState] = useState(loadMIDIDeviceId);
  const [mappingProfiles, setMappingProfiles] = useState<MIDIMappingProfile[]>(loadMappingProfiles);
  const [selectedProfileIndex] = useState(0);
  const [isMappingMode, setMappingMode] = useState(false);
  const [learningParameter, setLearningParameter] = useState<string | null>(null);
  const learningTimeoutRef = useRef<number | null>(null);
  const learningMappingIndexRef = useRef<number | null>(null);

  useEffect(() => { localStorage.setItem(MIDI_ENABLED_KEY, midiEnabled.toString()); }, [midiEnabled]);
  useEffect(() => { if (selectedDeviceId) localStorage.setItem(MIDI_DEVICE_KEY, selectedDeviceId); }, [selectedDeviceId]);

  const mappingProfile = useMemo(
    () => mappingProfiles[selectedProfileIndex] || mappingProfiles[0] || EMPTY_PROFILE,
    [mappingProfiles, selectedProfileIndex]
  );

  const findOrCreateMapping = useCallback(
    (parameterId: string, arrayIndex?: number, actionId?: string, mappingType?: "continuous" | "toggle" | "trigger" | "enum_cycle", range?: { min: number; max: number }, enumValues?: string[]): { mapping: MIDIMapping; index: number } => {
      let mappingIndex = mappingProfile.mappings.findIndex((m) => {
        if (actionId) return m.target.action === actionId;
        if (arrayIndex !== undefined) return m.target.parameter === parameterId && m.target.arrayIndex === arrayIndex;
        return m.target.parameter === parameterId && m.target.arrayIndex === undefined;
      });

      if (mappingIndex === -1) {
        let inferredType: MIDIMappingType = mappingType || (actionId ? "trigger" : "continuous");

        const baseTarget = actionId
          ? { action: actionId as any }
          : arrayIndex !== undefined
            ? { parameter: parameterId, arrayIndex }
            : { parameter: parameterId };

        const newMapping: MIDIMapping = {
          type: inferredType,
          source: { channel: 0 },
          target: enumValues ? { ...baseTarget, values: enumValues } : baseTarget,
          ...(range && { range }),
        };
        const updatedProfile = { ...mappingProfile, mappings: [...mappingProfile.mappings, newMapping] };
        const updatedProfiles = [...mappingProfiles];
        updatedProfiles[selectedProfileIndex] = updatedProfile;
        setMappingProfiles(updatedProfiles);
        saveMappingProfiles(updatedProfiles);
        return { mapping: newMapping, index: updatedProfile.mappings.length - 1 };
      }

      const existing = mappingProfile.mappings[mappingIndex];
      let updated = false;
      const updatedMapping = { ...existing };

      if (mappingType && existing.type !== mappingType) {
        updatedMapping.type = mappingType;
        updated = true;
      }
      if (range && (!existing.range || existing.range.min !== range.min || existing.range.max !== range.max)) {
        updatedMapping.range = range;
        updated = true;
      }
      if (enumValues && (!existing.target.values || JSON.stringify(existing.target.values) !== JSON.stringify(enumValues))) {
        updatedMapping.target = { ...existing.target, values: enumValues };
        updated = true;
      }

      if (updated) {
        const updatedProfile = {
          ...mappingProfile,
          mappings: mappingProfile.mappings.map((m, i) => i === mappingIndex ? updatedMapping : m),
        };
        const updatedProfiles = [...mappingProfiles];
        updatedProfiles[selectedProfileIndex] = updatedProfile;
        setMappingProfiles(updatedProfiles);
        saveMappingProfiles(updatedProfiles);
        return { mapping: updatedMapping, index: mappingIndex };
      }

      return { mapping: existing, index: mappingIndex };
    },
    [mappingProfile, mappingProfiles, selectedProfileIndex]
  );

  const handleLearnComplete = useCallback(
    (mappingIndex: number, source: MIDISource) => {
      setLearningParameter(null);
      if (learningTimeoutRef.current) {
        clearTimeout(learningTimeoutRef.current);
        learningTimeoutRef.current = null;
      }
      const updatedProfiles = [...mappingProfiles];
      const profile = updatedProfiles[selectedProfileIndex];
      if (profile?.mappings[mappingIndex]) {
        profile.mappings[mappingIndex].source = source;
        setMappingProfiles(updatedProfiles);
        saveMappingProfiles(updatedProfiles);
        toast.success("MIDI mapping learned!");
      }
    },
    [mappingProfiles, selectedProfileIndex]
  );

  const startLearning = useCallback(
    (parameterId: string, arrayIndex?: number, actionId?: string, mappingType?: "continuous" | "toggle" | "trigger" | "enum_cycle", range?: { min: number; max: number }, enumValues?: string[]) => {
      if (!midiEnabled || !selectedDeviceId) {
        toast.error("Please enable MIDI and select a device first");
        return;
      }
      const { index } = findOrCreateMapping(parameterId, arrayIndex, actionId, mappingType, range, enumValues);
      setLearningParameter(`${parameterId}${arrayIndex !== undefined ? `[${arrayIndex}]` : ""}${actionId || ""}`);
      learningMappingIndexRef.current = index;

      if (learningTimeoutRef.current) clearTimeout(learningTimeoutRef.current);
      learningTimeoutRef.current = window.setTimeout(() => {
        setLearningParameter(null);
        learningMappingIndexRef.current = null;
        toast.warning("No MIDI input detected. Try moving a knob or pressing a pad.");
      }, 10000);
    },
    [midiEnabled, selectedDeviceId, findOrCreateMapping]
  );

  const cancelLearning = useCallback(() => {
    setLearningParameter(null);
    learningMappingIndexRef.current = null;
    if (learningTimeoutRef.current) {
      clearTimeout(learningTimeoutRef.current);
      learningTimeoutRef.current = null;
    }
  }, []);

  const getMappedSource = useCallback(
    (parameterId: string, arrayIndex?: number, actionId?: string): string | null => {
      const mapping = mappingProfile.mappings.find((m) => {
        if (actionId) return m.target.action === actionId;
        if (arrayIndex !== undefined) return m.target.parameter === parameterId && m.target.arrayIndex === arrayIndex;
        return m.target.parameter === parameterId && m.target.arrayIndex === undefined;
      });

      if (!mapping || (!mapping.source.midi_cc && !mapping.source.midi_note)) return null;
      if (mapping.source.midi_cc !== undefined) return `CC ${mapping.source.midi_cc} (Ch ${mapping.source.channel})`;
      if (mapping.source.midi_note !== undefined) return `Note ${mapping.source.midi_note} (Ch ${mapping.source.channel})`;
      return null;
    },
    [mappingProfile]
  );

  const setMappingProfile = useCallback(
    (profile: MIDIMappingProfile) => {
      const updatedProfiles = [...mappingProfiles];
      updatedProfiles[selectedProfileIndex] = profile;
      setMappingProfiles(updatedProfiles);
      saveMappingProfiles(updatedProfiles);
    },
    [mappingProfiles, selectedProfileIndex]
  );

  const deleteMapping = useCallback(
    (index: number) => {
      const updatedProfile = {
        ...mappingProfile,
        mappings: mappingProfile.mappings.filter((_, i) => i !== index),
      };
      const updatedProfiles = [...mappingProfiles];
      updatedProfiles[selectedProfileIndex] = updatedProfile;
      setMappingProfiles(updatedProfiles);
      saveMappingProfiles(updatedProfiles);
      toast.success("Mapping deleted");
    },
    [mappingProfile, mappingProfiles, selectedProfileIndex]
  );

  const clearAllMappings = useCallback(() => {
    const updatedProfile = {
      ...mappingProfile,
      mappings: [],
    };
    const updatedProfiles = [...mappingProfiles];
    updatedProfiles[selectedProfileIndex] = updatedProfile;
    setMappingProfiles(updatedProfiles);
    saveMappingProfiles(updatedProfiles);
    toast.success("All mappings cleared");
  }, [mappingProfile, mappingProfiles, selectedProfileIndex]);

  const { devices, error, startLearn, cancelLearn: cancelLearnHook } = useMIDIController(
    sendParameterUpdate,
    {
      enabled: midiEnabled,
      deviceId: selectedDeviceId,
      mappingProfile,
      currentDenoisingSteps,
      onDenoisingStepsChange,
      onLearnComplete: handleLearnComplete,
      currentNoiseController,
      currentManageCache,
      onSwitchPrompt,
    }
  );

  useEffect(() => {
    if (devices.length > 0 && !selectedDeviceId) setSelectedDeviceIdState(devices[0].id);
  }, [devices, selectedDeviceId]);

  useEffect(() => {
    if (learningMappingIndexRef.current !== null && learningParameter !== null) {
      startLearn(learningMappingIndexRef.current);
    } else if (learningParameter === null && learningMappingIndexRef.current === null) {
      cancelLearnHook();
    }
  }, [learningParameter, startLearn, cancelLearnHook]);

  useEffect(() => {
    return () => { if (learningTimeoutRef.current) clearTimeout(learningTimeoutRef.current); };
  }, []);

  const value: MIDIContextValue = {
    midiEnabled,
    setMidiEnabled: setMidiEnabledState,
    selectedDeviceId,
    setSelectedDeviceId: setSelectedDeviceIdState,
    devices,
    mappingProfile,
    setMappingProfile,
    isMappingMode,
    setMappingMode,
    learningParameter,
    error,
    startLearning,
    cancelLearning,
    getMappedSource,
    deleteMapping,
    clearAllMappings,
  };

  return <MIDIContext.Provider value={value}>{children}</MIDIContext.Provider>;
}

export function useMIDI() {
  const context = useContext(MIDIContext);
  if (!context) throw new Error("useMIDI must be used within MIDIProvider");
  return context;
}

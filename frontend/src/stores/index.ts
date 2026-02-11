import { create } from "zustand";
import type { SettingsState, DownloadProgress } from "../types";
import type { PromptItem } from "../lib/api";

export interface TimelinePrompt {
  id: string;
  text: string;
  startTime: number;
  endTime: number;
  prompts?: Array<{ text: string; weight: number }>;
}

// ---- Slice types ----

interface SettingsSlice {
  settings: SettingsState;
  updateSettings: (patch: Partial<SettingsState>) => void;
}

interface PromptSlice {
  promptItems: PromptItem[];
  interpolationMethod: "linear" | "slerp";
  temporalInterpolationMethod: "linear" | "slerp";
  transitionSteps: number;
  selectedTimelinePrompt: TimelinePrompt | null;
  externalSelectedPromptId: string | null;
  setPromptItems: (items: PromptItem[]) => void;
  setInterpolationMethod: (method: "linear" | "slerp") => void;
  setTemporalInterpolationMethod: (method: "linear" | "slerp") => void;
  setTransitionSteps: (steps: number) => void;
  setSelectedTimelinePrompt: (prompt: TimelinePrompt | null) => void;
  setExternalSelectedPromptId: (id: string | null) => void;
}

interface TimelineSlice {
  timelinePrompts: TimelinePrompt[];
  timelineCurrentTime: number;
  isTimelinePlaying: boolean;
  isTimelineCollapsed: boolean;
  isLive: boolean;
  setTimelinePrompts: (prompts: TimelinePrompt[]) => void;
  setTimelineCurrentTime: (time: number) => void;
  setIsTimelinePlaying: (playing: boolean) => void;
  setIsTimelineCollapsed: (collapsed: boolean) => void;
  setIsLive: (live: boolean) => void;
}

interface VideoSlice {
  shouldReinitializeVideo: boolean;
  customVideoResolution: { width: number; height: number } | null;
  videoScaleMode: "fit" | "native";
  isRecording: boolean;
  setShouldReinitializeVideo: (value: boolean) => void;
  setCustomVideoResolution: (
    res: { width: number; height: number } | null
  ) => void;
  setVideoScaleMode: (mode: "fit" | "native") => void;
  setIsRecording: (recording: boolean) => void;
}

interface DownloadSlice {
  showDownloadDialog: boolean;
  isDownloading: boolean;
  downloadProgress: DownloadProgress | null;
  pipelinesNeedingModels: string[];
  currentDownloadPipeline: string | null;
  downloadError: string | null;
  setShowDownloadDialog: (show: boolean) => void;
  setIsDownloading: (downloading: boolean) => void;
  setDownloadProgress: (progress: DownloadProgress | null) => void;
  setPipelinesNeedingModels: (pipelines: string[]) => void;
  setCurrentDownloadPipeline: (pipeline: string | null) => void;
  setDownloadError: (error: string | null) => void;
}

interface UiSlice {
  openSettingsTab: string | null;
  isCloudConnecting: boolean;
  isBackendCloudConnected: boolean;
  setOpenSettingsTab: (tab: string | null) => void;
  setIsCloudConnecting: (connecting: boolean) => void;
  setIsBackendCloudConnected: (connected: boolean) => void;
}

// ---- Combined store type ----

export type AppState = SettingsSlice &
  PromptSlice &
  TimelineSlice &
  VideoSlice &
  DownloadSlice &
  UiSlice;

// ---- Default values ----

const DEFAULT_SETTINGS: SettingsState = {
  pipelineId: "longlive",
  resolution: { height: 320, width: 576 },
  denoisingSteps: [1000, 750, 500, 250],
  noiseScale: undefined,
  noiseController: undefined,
  manageCache: true,
  quantization: null,
  kvCacheAttentionBias: 0.3,
  paused: false,
  loraMergeStrategy: "permanent_merge",
  inputMode: "text",
};

// ---- Store ----

export const useAppStore = create<AppState>()(set => ({
  // Settings slice
  settings: DEFAULT_SETTINGS,
  updateSettings: patch =>
    set(state => ({ settings: { ...state.settings, ...patch } })),

  // Prompt slice
  promptItems: [{ text: "", weight: 100 }],
  interpolationMethod: "linear",
  temporalInterpolationMethod: "slerp",
  transitionSteps: 4,
  selectedTimelinePrompt: null,
  externalSelectedPromptId: null,
  setPromptItems: items => set({ promptItems: items }),
  setInterpolationMethod: method => set({ interpolationMethod: method }),
  setTemporalInterpolationMethod: method =>
    set({ temporalInterpolationMethod: method }),
  setTransitionSteps: steps => set({ transitionSteps: steps }),
  setSelectedTimelinePrompt: prompt => set({ selectedTimelinePrompt: prompt }),
  setExternalSelectedPromptId: id => set({ externalSelectedPromptId: id }),

  // Timeline slice
  timelinePrompts: [],
  timelineCurrentTime: 0,
  isTimelinePlaying: false,
  isTimelineCollapsed: false,
  isLive: false,
  setTimelinePrompts: prompts => set({ timelinePrompts: prompts }),
  setTimelineCurrentTime: time => set({ timelineCurrentTime: time }),
  setIsTimelinePlaying: playing => set({ isTimelinePlaying: playing }),
  setIsTimelineCollapsed: collapsed => set({ isTimelineCollapsed: collapsed }),
  setIsLive: live => set({ isLive: live }),

  // Video slice
  shouldReinitializeVideo: false,
  customVideoResolution: null,
  videoScaleMode: "fit",
  isRecording: false,
  setShouldReinitializeVideo: value => set({ shouldReinitializeVideo: value }),
  setCustomVideoResolution: res => set({ customVideoResolution: res }),
  setVideoScaleMode: mode => set({ videoScaleMode: mode }),
  setIsRecording: recording => set({ isRecording: recording }),

  // Download slice
  showDownloadDialog: false,
  isDownloading: false,
  downloadProgress: null,
  pipelinesNeedingModels: [],
  currentDownloadPipeline: null,
  downloadError: null,
  setShowDownloadDialog: show => set({ showDownloadDialog: show }),
  setIsDownloading: downloading => set({ isDownloading: downloading }),
  setDownloadProgress: progress => set({ downloadProgress: progress }),
  setPipelinesNeedingModels: pipelines =>
    set({ pipelinesNeedingModels: pipelines }),
  setCurrentDownloadPipeline: pipeline =>
    set({ currentDownloadPipeline: pipeline }),
  setDownloadError: error => set({ downloadError: error }),

  // UI slice
  openSettingsTab: null,
  isCloudConnecting: false,
  isBackendCloudConnected: false,
  setOpenSettingsTab: tab => set({ openSettingsTab: tab }),
  setIsCloudConnecting: connecting => set({ isCloudConnecting: connecting }),
  setIsBackendCloudConnected: connected =>
    set({ isBackendCloudConnected: connected }),
}));

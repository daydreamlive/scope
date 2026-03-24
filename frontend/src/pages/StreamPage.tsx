import { useState, useEffect, useRef, useCallback } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { OutputsPanel } from "../components/OutputsPanel";
import { TempoSyncSection } from "../components/settings/TempoSyncSection";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { PromptInputWithTimeline } from "../components/PromptInputWithTimeline";
import { DownloadDialog } from "../components/DownloadDialog";
import { WorkflowExportDialog } from "../components/WorkflowExportDialog";
import { WorkflowImportDialog } from "../components/WorkflowImportDialog";
import {
  buildScopeWorkflow,
  type WorkflowPromptState,
} from "../lib/workflowSettings";
import { GraphEditor } from "../components/graph/GraphEditor";
import type { GraphEditorHandle } from "../components/graph/GraphEditor";
import type { TimelinePrompt } from "../components/PromptTimeline";
import { StatusBar } from "../components/StatusBar";
import { LogPanel } from "../components/LogPanel";
import { useUnifiedWebRTC } from "../hooks/useUnifiedWebRTC";
import { useTempoSync } from "../hooks/useTempoSync";
import { MIDIProvider } from "../contexts/MIDIContext";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { useControllerInput } from "../hooks/useControllerInput";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useApi } from "../hooks/useApi";
import { useCloudContext } from "../lib/cloudContext";
import { useCloudStatus } from "../hooks/useCloudStatus";
import { useLogStream } from "../hooks/useLogStream";
import { getDefaultPromptForMode } from "../data/pipelines";
import {
  adjustResolutionForPipeline,
  fitResolutionToPixelBudget,
  getResolutionScaleFactor,
} from "../lib/utils";
import type {
  ExtensionMode,
  InputMode,
  PipelineId,
  LoRAConfig,
  LoraMergeStrategy,
  DownloadProgress,
  SettingsState,
} from "../types";
import type {
  PromptItem,
  PromptTransition,
  GraphConfig,
  PipelineLoadItem,
  PluginInfo,
} from "../lib/api";
import {
  getInputSourceResolution,
  fetchDaydreamWorkflow,
  getDmxStatus,
} from "../lib/api";
import type { ScopeWorkflow } from "../lib/workflowApi";
import { linearGraphFromSettings, stripUIFields } from "../lib/graphUtils";
import { resolveLoRAPath } from "../lib/workflowSettings";
import { useLoRAsContext } from "../contexts/LoRAsContext";
import { usePluginsContext } from "../contexts/PluginsContext";
import { useServerInfoContext } from "../contexts/ServerInfoContext";
import { sendLoRAScaleUpdates } from "../utils/loraHelpers";
import { toast } from "sonner";
import {
  isAuthenticated as checkIsAuthenticated,
  getDaydreamAPIKey,
  redirectToSignIn,
} from "../lib/auth";
import { createDaydreamImportSession } from "../lib/daydreamExport";
import { openExternalUrl } from "../lib/openExternal";

interface OscCommand {
  key: string;
  value: unknown;
}

// Delay before resetting video reinitialization flag (ms)
// This allows useVideoSource to detect the flag change and trigger reinitialization
const VIDEO_REINITIALIZE_DELAY_MS = 100;

function buildLoRAParams(
  loras?: LoRAConfig[],
  strategy?: LoraMergeStrategy
): {
  loras?: { path: string; scale: number; merge_mode?: string }[];
  lora_merge_mode: string;
} {
  return {
    loras: loras?.map(({ path, scale, mergeMode }) => ({
      path,
      scale,
      ...(mergeMode && { merge_mode: mergeMode }),
    })),
    lora_merge_mode: strategy ?? "permanent_merge",
  };
}

function getVaceParams(
  refImages?: string[],
  vaceContextScale?: number
):
  | { vace_ref_images: string[]; vace_context_scale: number }
  | Record<string, never> {
  if (refImages && refImages.length > 0) {
    return {
      vace_ref_images: refImages,
      vace_context_scale: vaceContextScale ?? 1.0,
    };
  }
  return {};
}

export function StreamPage() {
  // Get API functions that work in both local and cloud modes
  const api = useApi();
  const { isCloudMode: isDirectCloudMode, isReady: isCloudReady } =
    useCloudContext();

  // Track backend cloud relay mode (local backend connected to cloud or connecting)
  const {
    isConnected: isBackendCloudConnected,
    isConnecting: isBackendCloudConnecting,
    connectStage: cloudConnectStage,
  } = useCloudStatus();

  const { loraFiles } = useLoRAsContext();
  const { plugins } = usePluginsContext();
  const { version: scopeVersion } = useServerInfoContext();

  // Combined cloud mode: either frontend direct-to-cloud or backend relay to cloud
  const isCloudMode = isDirectCloudMode || isBackendCloudConnected;

  // Log stream for the log panel
  const {
    logs,
    isOpen: isLogPanelOpen,
    toggle: toggleLogPanel,
    clearLogs,
    unreadCount: logUnreadCount,
  } = useLogStream();

  // Show loading state while connecting to cloud
  useEffect(() => {
    if (isDirectCloudMode) {
      console.log("[StreamPage] Cloud mode enabled, ready:", isCloudReady);
    }
  }, [isDirectCloudMode, isCloudReady]);

  // Fetch available pipelines dynamically
  const { pipelines, refreshPipelines } = usePipelinesContext();

  // Helper to get default mode for a pipeline
  const getPipelineDefaultMode = (pipelineId: string): InputMode => {
    return pipelines?.[pipelineId]?.defaultMode ?? "text";
  };

  // Use the stream state hook for settings management
  const {
    settings,
    updateSettings,
    getDefaults,
    supportsNoiseControls,
    spoutAvailable,
    ndiOutputAvailable,
    syphonOutputAvailable,
    availableInputSources,
    refreshPipelineSchemas,
    refreshHardwareInfo,
    hardwareInfo,
    skipNextModeReset,
  } = useStreamState();

  // Derive NDI and Syphon input availability from dynamic input sources list
  const ndiAvailable = availableInputSources.some(
    s => s.source_id === "ndi" && s.available
  );
  const syphonAvailable = availableInputSources.some(
    s => s.source_id === "syphon" && s.available
  );
  // Output availability flags are passed to GraphEditor for output nodes
  const hasAvailableOutputs =
    spoutAvailable || ndiOutputAvailable || syphonOutputAvailable;

  // Combined refresh function for pipeline schemas, pipelines list, and hardware info
  const handlePipelinesRefresh = useCallback(async () => {
    // Refresh all hooks to keep them in sync when cloud mode toggles
    await Promise.all([
      refreshPipelineSchemas(),
      refreshPipelines(),
      refreshHardwareInfo(),
    ]);
  }, [refreshPipelineSchemas, refreshPipelines, refreshHardwareInfo]);

  // Prompt state - use unified default prompts based on mode
  const initialMode =
    settings.inputMode || getPipelineDefaultMode(settings.pipelineId);

  // Ref to access latest settings without re-creating sendParameterUpdate on every change
  const settingsRef = useRef(settings);
  settingsRef.current = settings;

  const [promptItems, setPromptItems] = useState<PromptItem[]>([
    { text: getDefaultPromptForMode(initialMode), weight: 100 },
  ]);
  const [interpolationMethod, setInterpolationMethod] = useState<
    "linear" | "slerp"
  >("linear");
  const [temporalInterpolationMethod, setTemporalInterpolationMethod] =
    useState<"linear" | "slerp">("slerp");
  const [transitionSteps, setTransitionSteps] = useState(4);

  // Track when we need to reinitialize video source
  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

  // Store custom video resolution from user uploads - persists across mode/pipeline changes
  const [customVideoResolution, setCustomVideoResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);

  // Cap an input resolution to fit within a pipeline's default pixel budget
  const capResolution = useCallback(
    (
      input: { width: number; height: number },
      pipelineId: PipelineId,
      mode?: InputMode
    ) => {
      const defaults = getDefaults(pipelineId, mode);
      const scaleFactor = getResolutionScaleFactor(pipelineId) ?? 1;
      const maxPixels = defaults.width * defaults.height;
      return fitResolutionToPixelBudget(
        input.width,
        input.height,
        maxPixels,
        scaleFactor
      );
    },
    [getDefaults]
  );

  const [isLive, setIsLive] = useState(false);
  const [isTimelineCollapsed, setIsTimelineCollapsed] = useState(false);
  const [selectedTimelinePrompt, setSelectedTimelinePrompt] =
    useState<TimelinePrompt | null>(null);

  // Timeline state for left panel
  const [timelinePrompts, setTimelinePrompts] = useState<TimelinePrompt[]>([]);
  const [timelineCurrentTime, setTimelineCurrentTime] = useState(0);
  const [isTimelinePlaying, setIsTimelinePlaying] = useState(false);

  // Recording toggle state
  const [isRecording, setIsRecording] = useState(false);

  // Track when waiting for cloud WebSocket to connect after clicking Play
  const [isCloudConnecting, setIsCloudConnecting] = useState(false);

  // Graph mode state
  const [graphMode, setGraphMode] = useState(false);
  const graphEditorRef = useRef<GraphEditorHandle>(null);

  // When true, pipeline controls are disabled in Perform Mode
  // (set when user edits anything in Graph Mode, cleared when user clicks Clear)
  const [nonLinearGraph, setNonLinearGraph] = useState(false);

  // Called by GraphEditor whenever user edits the graph
  const handleGraphChange = useCallback(() => {
    setNonLinearGraph(true);
  }, []);

  // Called by GraphEditor when user clicks Clear
  const handleGraphClear = useCallback(() => {
    setNonLinearGraph(false);
  }, []);

  // Clear graph from SettingsPanel (triggers the full graph clear via ref)
  const handleClearGraphFromSettings = useCallback(() => {
    graphEditorRef.current?.clearGraph();
  }, []);

  // Video display state
  const [videoScaleMode, setVideoScaleMode] = useState<"fit" | "native">("fit");

  // External control of timeline selection
  const [externalSelectedPromptId, setExternalSelectedPromptId] = useState<
    string | null
  >(null);

  // Settings dialog navigation state
  const [openSettingsTab, setOpenSettingsTab] = useState<string | null>(null);

  // Open account tab after sign-in (success or error)
  useEffect(() => {
    const handleAuthEvent = () => {
      setOpenSettingsTab("account");
    };
    window.addEventListener("daydream-auth-success", handleAuthEvent);
    window.addEventListener("daydream-auth-error", handleAuthEvent);
    return () => {
      window.removeEventListener("daydream-auth-success", handleAuthEvent);
      window.removeEventListener("daydream-auth-error", handleAuthEvent);
    };
  }, []);

  // Download dialog state
  const [showDownloadDialog, setShowDownloadDialog] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] =
    useState<DownloadProgress | null>(null);
  const [pipelinesNeedingModels, setPipelinesNeedingModels] = useState<
    string[]
  >([]);
  const [currentDownloadPipeline, setCurrentDownloadPipeline] = useState<
    string | null
  >(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  // Workflow export/import dialog state
  const [showWorkflowExport, setShowWorkflowExport] = useState(false);
  const [showWorkflowImport, setShowWorkflowImport] = useState(false);
  const [preloadedWorkflow, setPreloadedWorkflow] =
    useState<ScopeWorkflow | null>(null);

  // Daydream export state
  const [isExportingToDaydream, setIsExportingToDaydream] = useState(false);
  const [isDaydreamAuthenticated, setIsDaydreamAuthenticated] = useState(
    checkIsAuthenticated()
  );

  useEffect(() => {
    const handleAuthChange = () => {
      setIsDaydreamAuthenticated(checkIsAuthenticated());
    };
    window.addEventListener("daydream-auth-change", handleAuthChange);
    return () => {
      window.removeEventListener("daydream-auth-change", handleAuthChange);
    };
  }, []);

  const handleExportToDaydream = useCallback(async () => {
    if (!isDaydreamAuthenticated) {
      redirectToSignIn();
      return;
    }

    const apiKey = getDaydreamAPIKey();
    if (!apiKey) {
      toast.error("Not authenticated with Daydream");
      return;
    }

    const isElectron = Boolean(
      (window as unknown as { scope?: { openExternal?: unknown } }).scope
        ?.openExternal
    );
    // Open a blank tab synchronously while user-activation is still live,
    // so popup blockers don't interfere. Electron uses IPC and doesn't need this.
    const pendingTab = isElectron ? null : window.open("about:blank", "_blank");

    setIsExportingToDaydream(true);
    try {
      const pluginInfoMap = new Map<string, PluginInfo>(
        plugins.map(p => [p.name, p])
      );

      const workflow = buildScopeWorkflow({
        name: "Untitled Workflow",
        settings,
        timelinePrompts,
        promptState: {
          promptItems,
          interpolationMethod,
          transitionSteps,
          temporalInterpolationMethod,
        },
        pipelineInfoMap: pipelines ?? {},
        loraFiles,
        pluginInfoMap,
        scopeVersion: scopeVersion ?? "unknown",
      });

      const result = await createDaydreamImportSession(
        apiKey,
        workflow,
        workflow.metadata.name
      );

      if (pendingTab) {
        pendingTab.location.href = result.createUrl;
      } else {
        openExternalUrl(result.createUrl);
      }
      toast.success("Opening daydream.live...", {
        description:
          "Your workflow has been sent to daydream.live for publishing.",
      });
    } catch (err) {
      pendingTab?.close();
      console.error("Export to daydream.live failed:", err);
      toast.error("Export failed", {
        description: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setIsExportingToDaydream(false);
    }
  }, [
    isDaydreamAuthenticated,
    plugins,
    settings,
    timelinePrompts,
    promptItems,
    interpolationMethod,
    transitionSteps,
    temporalInterpolationMethod,
    pipelines,
    loraFiles,
    scopeVersion,
  ]);

  // Handle install-workflow deep links from Electron
  useEffect(() => {
    if (!window.scope?.onDeepLinkAction) return;
    return window.scope.onDeepLinkAction(data => {
      if (data.action !== "install-workflow") return;
      const workflowId = data.id;
      toast.info("Fetching workflow...");
      fetchDaydreamWorkflow(workflowId)
        .then((workflow: ScopeWorkflow) => {
          setPreloadedWorkflow(workflow);
          setShowWorkflowImport(true);
        })
        .catch((err: unknown) => {
          console.error("Failed to fetch workflow from deep link:", err);
          toast.error("Failed to import workflow", {
            description: err instanceof Error ? err.message : String(err),
          });
        });
    });
  }, []);

  // Stable ref for OSC command handler (avoids hook dependency cycles)
  const oscCommandHandlerRef = useRef<(cmd: OscCommand) => void>(() => {});

  // Ref to access timeline functions
  const timelineRef = useRef<{
    getCurrentTimelinePrompt: () => string;
    submitLivePrompt: (prompts: PromptItem[]) => void;
    updatePrompt: (prompt: TimelinePrompt) => void;
    clearTimeline: () => void;
    resetPlayhead: () => void;
    resetTimelineCompletely: () => void;
    loadPrompts: (prompts: TimelinePrompt[]) => void;
    getPrompts: () => TimelinePrompt[];
    getCurrentTime: () => number;
    getIsPlaying: () => boolean;
  }>(null);

  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
    pipelineInfo,
    pipelineInfoRef,
    loadingStage: pipelineLoadingStage,
  } = usePipeline();

  // Tempo sync
  const {
    tempoState,
    sources: tempoSources,
    loading: tempoLoading,
    error: tempoError,
    enable: enableTempoSync,
    disable: disableTempoSync,
    setSessionTempo: setTempoSessionBpm,
    fetchSources: refreshTempoSources,
    updateFromNotification: updateTempoFromNotification,
  } = useTempoSync();

  // Apply backend parameter values to frontend state (used for both local
  // sends and external updates pushed via the data channel).
  const applyBackendParamsToSettings = useCallback(
    (params: Record<string, unknown>) => {
      const settingsUpdate: Partial<SettingsState> = {};

      if (params.noise_scale !== undefined) {
        settingsUpdate.noiseScale = params.noise_scale as number;
      }
      if (params.noise_controller !== undefined) {
        settingsUpdate.noiseController = params.noise_controller as boolean;
      }
      if (params.manage_cache !== undefined) {
        settingsUpdate.manageCache = params.manage_cache as boolean;
      }
      if (params.kv_cache_attention_bias !== undefined) {
        settingsUpdate.kvCacheAttentionBias =
          params.kv_cache_attention_bias as number;
      }
      if (params.vace_context_scale !== undefined) {
        settingsUpdate.vaceContextScale = params.vace_context_scale as number;
      }

      // Sync any remaining params to schemaFieldOverrides (for plugin parameters)
      const knownKeys = new Set([
        "noise_scale",
        "noise_controller",
        "manage_cache",
        "kv_cache_attention_bias",
        "vace_context_scale",
        "denoising_step_list",
        "reset_cache",
        "paused",
        "prompts",
        "quantize_mode",
        "lookahead_ms",
        "_quantized",
      ]);
      const overrideUpdates: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(params)) {
        if (!knownKeys.has(k)) {
          overrideUpdates[k] = v;
        }
      }
      if (Object.keys(overrideUpdates).length > 0) {
        const current = settingsRef.current;
        settingsUpdate.schemaFieldOverrides = {
          ...(current.schemaFieldOverrides ?? {}),
          ...overrideUpdates,
        };
      }

      if (Object.keys(settingsUpdate).length > 0) {
        updateSettings(settingsUpdate);
      }

      if (params.prompts) {
        setPromptItems(
          params.prompts as Array<{ text: string; weight: number }>
        );
      }
    },
    [updateSettings]
  );

  // WebRTC for streaming (unified hook works in both local and cloud modes)
  const {
    remoteStream,
    remoteStreams,
    isStreaming,
    isConnecting,
    peerConnectionRef,
    sinkNodeIdsRef,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate: sendParameterUpdateWebRTC,
    sessionId,
  } = useUnifiedWebRTC({
    onParametersUpdated: applyBackendParamsToSettings,
    onTempoUpdate: updateTempoFromNotification,
  });

  // Whether beat-quantized output gating is active
  const isQuantizeActive =
    isStreaming &&
    tempoState.enabled &&
    (settings.quantizeMode || "none") !== "none";

  // Wrapper for sendParameterUpdate that also syncs frontend state.
  // Uses settingsRef to avoid depending on the full `settings` object,
  // which would cause this callback (and dependent useEffects) to
  // re-fire on every settings change, flooding the backend with
  // unnecessary parameter messages.
  const sendParameterUpdate = useCallback(
    (params: Record<string, unknown>) => {
      // Auto-flag discrete params for beat-quantized gating
      if (isQuantizeActive) {
        const NEVER_QUANTIZE = new Set([
          "paused",
          "quantize_mode",
          "lookahead_ms",
          "_quantized",
          "prompt_interpolation_method",
        ]);
        const DISCRETE_PARAM_KEYS = new Set([
          "prompts",
          "reset_cache",
          "transition",
          "denoising_step_list",
        ]);
        const hasDiscrete = Object.entries(params).some(([key, value]) => {
          if (NEVER_QUANTIZE.has(key)) return false;
          return (
            DISCRETE_PARAM_KEYS.has(key) ||
            typeof value === "boolean" ||
            typeof value === "string"
          );
        });
        if (hasDiscrete) {
          params = { ...params, _quantized: true };
        }
      }

      // Send to backend via WebRTC
      sendParameterUpdateWebRTC(params);

      // Also update frontend state for known parameters
      applyBackendParamsToSettings(params);
    },
    [sendParameterUpdateWebRTC, applyBackendParamsToSettings, isQuantizeActive]
  );

  // Computed loading state - true when downloading models, loading pipeline, connecting WebRTC, or waiting for cloud
  const isLoading =
    isDownloading || isPipelineLoading || isConnecting || isCloudConnecting;

  // Get per-sink WebRTC stats (FPS / bitrate)
  const { perSinkStats } = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
    sinkNodeIdsRef,
  });

  // Video container ref for controller input pointer lock
  const videoContainerRef = useRef<HTMLDivElement>(null);

  // Check if current pipeline supports controller input
  const currentPipelineSupportsController =
    pipelines?.[settings.pipelineId]?.supportsControllerInput ?? false;

  // Controller input hook - captures WASD/mouse and streams to backend
  const { isPointerLocked, requestPointerLock } = useControllerInput(
    sendParameterUpdate,
    isStreaming && currentPipelineSupportsController,
    videoContainerRef
  );

  // Video source for preview (camera or video)
  // Enable based on input mode, not pipeline category
  const {
    localStream,
    isInitializing,
    error: videoSourceError,
    mode,
    videoResolution,
    switchMode,
    handleVideoFileUpload,
  } = useVideoSource({
    onStreamUpdate: updateVideoTrack,
    onStopStream: stopStream,
    shouldReinitialize: shouldReinitializeVideo,
    enabled: settings.inputMode === "video" || graphMode,
    // Sync output resolution when user uploads a custom video
    // Store the custom resolution so it persists across mode/pipeline changes
    onCustomVideoResolution: resolution => {
      setCustomVideoResolution(resolution);
      updateSettings({
        resolution: capResolution(
          resolution,
          settings.pipelineId,
          settings.inputMode
        ),
      });
    },
  });

  // Per-node local streams for multi-source graph mode
  const [nodeLocalStreams, setNodeLocalStreams] = useState<
    Record<string, MediaStream>
  >({});
  const nodeLocalStreamsRef = useRef(nodeLocalStreams);
  nodeLocalStreamsRef.current = nodeLocalStreams;

  const createCameraStreamForNode = useCallback(async (nodeId: string) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      setNodeLocalStreams(prev => ({ ...prev, [nodeId]: stream }));
    } catch (e) {
      console.error(`Failed to get camera for node ${nodeId}:`, e);
    }
  }, []);

  const handlePerNodeSourceModeChange = useCallback(
    (newMode: string, nodeId?: string) => {
      if (!nodeId) {
        switchMode(newMode as "video" | "camera" | "spout" | "ndi" | "syphon");
        return;
      }
      const oldStream = nodeLocalStreamsRef.current[nodeId];
      if (oldStream) {
        oldStream.getTracks().forEach(t => t.stop());
        setNodeLocalStreams(prev => {
          const next = { ...prev };
          delete next[nodeId];
          return next;
        });
      }
      if (newMode === "camera") {
        createCameraStreamForNode(nodeId);
      }
    },
    [switchMode, createCameraStreamForNode]
  );

  const handlePerNodeVideoFileUpload = useCallback(
    async (file: File, nodeId?: string): Promise<boolean> => {
      if (!nodeId) {
        return handleVideoFileUpload(file);
      }
      try {
        const video = document.createElement("video");
        video.src = URL.createObjectURL(file);
        video.loop = true;
        video.muted = true;
        video.playsInline = true;
        await video.play();
        const stream = (
          video as HTMLVideoElement & { captureStream(): MediaStream }
        ).captureStream();
        const oldStream = nodeLocalStreamsRef.current[nodeId];
        if (oldStream) {
          oldStream.getTracks().forEach(t => t.stop());
        }
        setNodeLocalStreams(prev => ({ ...prev, [nodeId]: stream }));
        return true;
      } catch (e) {
        console.error(`Failed to create video stream for node ${nodeId}:`, e);
        return false;
      }
    },
    [handleVideoFileUpload]
  );

  useEffect(() => {
    return () => {
      Object.values(nodeLocalStreamsRef.current).forEach(stream => {
        stream.getTracks().forEach(t => t.stop());
      });
    };
  }, []);

  const handlePromptsSubmit = (prompts: PromptItem[]) => {
    setPromptItems(prompts);
  };

  const buildSoloPromptItems = useCallback(
    (index: number) =>
      promptItems.map((prompt, promptIndex) => ({
        ...prompt,
        weight: promptIndex === index ? 100 : 0,
      })),
    [promptItems]
  );

  const handleTransitionSubmit = (transition: PromptTransition) => {
    setPromptItems(transition.target_prompts);

    // Add to timeline if available
    if (timelineRef.current) {
      timelineRef.current.submitLivePrompt(transition.target_prompts);
    }

    // Send transition to backend
    sendParameterUpdate({
      transition,
    });
  };

  // Handler for input mode changes (text vs video)
  const handleInputModeChange = (newMode: InputMode) => {
    // Stop stream if currently streaming
    if (isStreaming) {
      stopStream();
    }

    // Get mode-specific defaults from backend schema
    const modeDefaults = getDefaults(settings.pipelineId, newMode);

    // Use custom video resolution (capped to pixel budget) if switching to video mode
    // This preserves the user's uploaded video aspect ratio across mode switches
    const resolution =
      newMode === "video" && customVideoResolution
        ? capResolution(customVideoResolution, settings.pipelineId, newMode)
        : { height: modeDefaults.height, width: modeDefaults.width };

    // Clear pre/postprocessors that don't support the new mode
    const preprocessorStillValid = settings.preprocessorIds?.every(id =>
      pipelines?.[id]?.supportedModes?.includes(newMode)
    );
    const postprocessorStillValid = settings.postprocessorIds?.every(id =>
      pipelines?.[id]?.supportedModes?.includes(newMode)
    );

    // Update settings with new mode and ALL mode-specific defaults including resolution
    updateSettings({
      inputMode: newMode,
      resolution,
      denoisingSteps: modeDefaults.denoisingSteps,
      noiseScale: modeDefaults.noiseScale,
      noiseController: modeDefaults.noiseController,
      ...(preprocessorStillValid
        ? {}
        : { preprocessorIds: [], preprocessorSchemaFieldOverrides: {} }),
      ...(postprocessorStillValid
        ? {}
        : { postprocessorIds: [], postprocessorSchemaFieldOverrides: {} }),
    });

    // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
    setPromptItems([{ text: getDefaultPromptForMode(newMode), weight: 100 }]);

    // Update temporal interpolation steps to mode-specific default
    const pipeline = pipelines?.[settings.pipelineId];
    const pipelineDefaultSteps =
      pipeline?.defaultTemporalInterpolationSteps ?? 4;
    setTransitionSteps(
      modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps
    );

    // Handle video source based on mode
    if (newMode === "video") {
      // Trigger video source reinitialization
      setShouldReinitializeVideo(true);
      setTimeout(
        () => setShouldReinitializeVideo(false),
        VIDEO_REINITIALIZE_DELAY_MS
      );
    }
    // Note: useVideoSource hook will automatically stop when enabled becomes false
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
    // User manually changed pipeline, clear non-linear flag
    setNonLinearGraph(false);

    // Stop the stream if it's currently running
    if (isStreaming) {
      stopStream();
    }

    const newPipeline = pipelines?.[pipelineId];
    const modeToUse = newPipeline?.defaultMode || "text";
    const currentMode = settings.inputMode || "text";

    // Trigger video reinitialization if switching to video mode
    if (modeToUse === "video" && currentMode !== "video") {
      setShouldReinitializeVideo(true);
      setTimeout(
        () => setShouldReinitializeVideo(false),
        VIDEO_REINITIALIZE_DELAY_MS
      );
    }

    // Reset timeline completely but preserve collapse state
    if (timelineRef.current) {
      timelineRef.current.resetTimelineCompletely();
    }

    // Reset selected timeline prompt to exit Edit mode and return to Append mode
    setSelectedTimelinePrompt(null);
    setExternalSelectedPromptId(null);

    // Get all defaults for the new pipeline + mode from backend schema
    const defaults = getDefaults(pipelineId, modeToUse);

    // Update prompts to mode-specific defaults (unified per mode, not per pipeline)
    setPromptItems([{ text: getDefaultPromptForMode(modeToUse), weight: 100 }]);

    // Use custom video resolution (capped to pixel budget) if mode is video
    // This preserves the user's uploaded video aspect ratio across pipeline switches
    const resolution =
      modeToUse === "video" && customVideoResolution
        ? capResolution(customVideoResolution, pipelineId, modeToUse)
        : { height: defaults.height, width: defaults.width };

    // Update the pipeline in settings with the appropriate mode and defaults
    updateSettings({
      pipelineId,
      inputMode: modeToUse,
      denoisingSteps: defaults.denoisingSteps,
      resolution,
      noiseScale: defaults.noiseScale,
      noiseController: defaults.noiseController,
      loras: [], // Clear LoRA controls when switching pipelines
      preprocessorSchemaFieldOverrides: {},
      postprocessorSchemaFieldOverrides: {},
    });
  };

  const downloadPipelineSequentially = async (
    pipelineId: string,
    remainingPipelines: string[]
  ) => {
    setCurrentDownloadPipeline(pipelineId);
    setDownloadProgress(null);

    try {
      await api.downloadPipelineModels(pipelineId);

      // Enhanced polling with progress updates
      const checkDownloadProgress = async () => {
        try {
          const status = await api.checkModelStatus(pipelineId);

          // Update progress state
          if (status.progress) {
            setDownloadProgress(status.progress);
          }

          // Check for download error
          if (status.progress?.error) {
            const errorMessage = status.progress.error;
            console.error("Download failed:", errorMessage);
            toast.error(errorMessage);
            setIsDownloading(false);
            setDownloadProgress(null);
            setDownloadError(errorMessage);
            setCurrentDownloadPipeline(null);
            return;
          }

          if (status.downloaded) {
            // Download complete for this pipeline
            // Remove it from the list
            const newRemaining = remainingPipelines;
            setPipelinesNeedingModels(newRemaining);

            // Check if this was a preprocessor or the main pipeline
            const pipelineInfo = pipelines?.[pipelineId];
            const isPreprocessor =
              pipelineInfo?.usage?.includes("preprocessor") ?? false;

            // Only update the main pipeline ID if this was NOT a preprocessor
            // and it matches the current pipeline ID
            if (!isPreprocessor && pipelineId === settings.pipelineId) {
              // This is the main pipeline, update settings
              if (timelineRef.current) {
                timelineRef.current.resetTimelineCompletely();
              }

              setSelectedTimelinePrompt(null);
              setExternalSelectedPromptId(null);

              // Preserve the current input mode that the user selected before download
              const newPipeline = pipelines?.[pipelineId];
              const currentMode =
                settings.inputMode || newPipeline?.defaultMode || "text";
              const defaults = getDefaults(
                pipelineId as PipelineId,
                currentMode
              );

              // Use custom video resolution (capped to pixel budget) if mode is video
              const resolution =
                currentMode === "video" && customVideoResolution
                  ? capResolution(
                      customVideoResolution,
                      pipelineId as PipelineId,
                      currentMode
                    )
                  : { height: defaults.height, width: defaults.width };

              // Only update pipeline-related settings, preserving current input mode and prompts
              updateSettings({
                pipelineId: pipelineId as PipelineId,
                inputMode: currentMode,
                denoisingSteps: defaults.denoisingSteps,
                resolution,
                noiseScale: defaults.noiseScale,
                noiseController: defaults.noiseController,
              });
            }

            // If there are more pipelines to download, continue with the next one
            if (newRemaining.length > 0) {
              // Continue with next pipeline
              setTimeout(() => {
                downloadPipelineSequentially(
                  newRemaining[0],
                  newRemaining.slice(1)
                );
              }, 1000);
            } else {
              // All downloads complete
              setIsDownloading(false);
              setDownloadProgress(null);
              setShowDownloadDialog(false);
              setCurrentDownloadPipeline(null);

              // Automatically start the stream after all downloads complete
              setTimeout(async () => {
                const started = await handleStartStream();
                // If stream started successfully, also start the timeline
                if (started && timelinePlayPauseRef.current) {
                  setTimeout(() => {
                    timelinePlayPauseRef.current?.();
                  }, 2000); // Give stream time to fully initialize
                }
              }, 100);
            }
          } else {
            setTimeout(checkDownloadProgress, 2000);
          }
        } catch (error) {
          console.error("Error checking download status:", error);
          setIsDownloading(false);
          setDownloadProgress(null);
          setShowDownloadDialog(false);
          setCurrentDownloadPipeline(null);
        }
      };

      // Start checking
      setTimeout(checkDownloadProgress, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      setIsDownloading(false);
      setDownloadProgress(null);
      setShowDownloadDialog(false);
      setCurrentDownloadPipeline(null);
    }
  };

  const handleDownloadModels = async () => {
    if (pipelinesNeedingModels.length === 0) return;

    setIsDownloading(true);
    setDownloadError(null);
    setShowDownloadDialog(true); // Keep dialog open to show progress

    // Start downloading the first pipeline in the list
    const firstPipeline = pipelinesNeedingModels[0];
    const remaining = pipelinesNeedingModels.slice(1);
    await downloadPipelineSequentially(firstPipeline, remaining);
  };

  const handleDialogClose = () => {
    setShowDownloadDialog(false);
    setPipelinesNeedingModels([]);
    setCurrentDownloadPipeline(null);
    setDownloadError(null);

    // When user cancels, no stream or timeline has started yet, so nothing to clean up
    // Just close the dialog and return early without any state changes
  };

  const handleResolutionChange = (resolution: {
    height: number;
    width: number;
  }) => {
    updateSettings({ resolution });
  };

  const handleDenoisingStepsChange = (denoisingSteps: number[]) => {
    updateSettings({ denoisingSteps });
    // Send denoising steps update to backend
    sendParameterUpdate({
      denoising_step_list: denoisingSteps,
    });
  };

  const handleNoiseScaleChange = (noiseScale: number) => {
    updateSettings({ noiseScale });
    // Send noise scale update to backend
    sendParameterUpdate({
      noise_scale: noiseScale,
    });
  };

  const handleNoiseControllerChange = (enabled: boolean) => {
    updateSettings({ noiseController: enabled });
    // Send noise controller update to backend
    sendParameterUpdate({
      noise_controller: enabled,
    });
  };

  const handleManageCacheChange = (enabled: boolean) => {
    updateSettings({ manageCache: enabled });
    // Send manage cache update to backend
    sendParameterUpdate({
      manage_cache: enabled,
    });
  };

  const handleQuantizationChange = (quantization: "fp8_e4m3fn" | null) => {
    updateSettings({ quantization });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleKvCacheAttentionBiasChange = (bias: number) => {
    updateSettings({ kvCacheAttentionBias: bias });
    // Send KV cache attention bias update to backend
    sendParameterUpdate({
      kv_cache_attention_bias: bias,
    });
  };

  const handleLorasChange = (loras: LoRAConfig[]) => {
    updateSettings({ loras });

    // If streaming, send scale updates to backend for runtime adjustment
    if (isStreaming) {
      sendLoRAScaleUpdates(
        loras,
        pipelineInfo?.loaded_lora_adapters,
        ({ lora_scales }) => {
          // Forward only the lora_scales field over the data channel.
          sendParameterUpdate({
            // TypeScript doesn't know about lora_scales on this payload yet.
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            ...({ lora_scales } as any),
          });
        }
      );
    }
    // Note: Adding/removing LoRAs requires pipeline reload
  };

  const handleVaceEnabledChange = (enabled: boolean) => {
    updateSettings({ vaceEnabled: enabled });
    // Note: This setting requires pipeline reload, so we don't send parameter update here
  };

  const handleVaceUseInputVideoChange = (enabled: boolean) => {
    updateSettings({ vaceUseInputVideo: enabled });
    // Send parameter update to backend if streaming
    if (isStreaming) {
      sendParameterUpdate({
        vace_use_input_video: enabled,
      });
    }
  };

  const handleRefImagesChange = (images: string[]) => {
    updateSettings({ refImages: images });
  };

  const handleSendHints = (imagePaths: string[]) => {
    const currentPipeline = pipelines?.[settings.pipelineId];

    if (currentPipeline?.supportsVACE) {
      // VACE pipeline - use vace_ref_images
      sendParameterUpdate({
        vace_ref_images: imagePaths,
      });
    } else if (currentPipeline?.supportsImages) {
      // Non-VACE pipeline with images support - use images
      sendParameterUpdate({
        images: imagePaths,
      });
    }
  };

  // Shared helpers for pre/postprocessor pipeline settings
  const pipelineSettingsKeys = {
    preprocessor: {
      ids: "preprocessorIds",
      overrides: "preprocessorSchemaFieldOverrides",
    },
    postprocessor: {
      ids: "postprocessorIds",
      overrides: "postprocessorSchemaFieldOverrides",
    },
  } as const;

  type PipelineKind = keyof typeof pipelineSettingsKeys;

  const makePipelineIdsHandler = (kind: PipelineKind) => (ids: string[]) => {
    // User manually changed pipeline chain, clear non-linear flag
    setNonLinearGraph(false);
    const k = pipelineSettingsKeys[kind];
    // Preserve overrides for processors that remain in the list
    const currentOverrides =
      (settings[k.overrides] as
        | Record<string, Record<string, unknown>>
        | undefined) ?? {};
    const kept: Record<string, Record<string, unknown>> = {};
    for (const id of ids) {
      if (currentOverrides[id]) {
        kept[id] = currentOverrides[id];
      }
    }
    updateSettings({ [k.ids]: ids, [k.overrides]: kept });
  };

  const makePipelineOverrideHandler =
    (kind: PipelineKind) =>
    (
      processorId: string,
      key: string,
      value: unknown,
      isRuntimeParam?: boolean
    ) => {
      const k = pipelineSettingsKeys[kind];
      const currentOverrides =
        (settings[k.overrides] as
          | Record<string, Record<string, unknown>>
          | undefined) ?? {};
      const processorOverrides = currentOverrides[processorId] ?? {};
      updateSettings({
        [k.overrides]: {
          ...currentOverrides,
          [processorId]: { ...processorOverrides, [key]: value },
        },
      });
      if (isRuntimeParam && isStreaming) {
        sendParameterUpdate({ [key]: value });
      }
    };

  const handlePreprocessorIdsChange = makePipelineIdsHandler("preprocessor");
  const handlePostprocessorIdsChange = makePipelineIdsHandler("postprocessor");
  const handlePreprocessorSchemaFieldOverrideChange =
    makePipelineOverrideHandler("preprocessor");
  const handlePostprocessorSchemaFieldOverrideChange =
    makePipelineOverrideHandler("postprocessor");

  const pipelineHasRuntimeField = useCallback(
    (pipelineId: string, key: string): boolean => {
      const props = (
        pipelines?.[pipelineId]?.configSchema as
          | {
              properties?: Record<string, { ui?: { is_load_param?: boolean } }>;
            }
          | undefined
      )?.properties;
      const field = props?.[key];
      if (!field) {
        return false;
      }
      return field.ui?.is_load_param === false;
    },
    [pipelines]
  );

  const handleVaceContextScaleChange = (scale: number) => {
    updateSettings({ vaceContextScale: scale });
    // Send VACE context scale update to backend if streaming
    if (isStreaming) {
      sendParameterUpdate({
        vace_context_scale: scale,
      });
    }
  };

  // Derive the appropriate extension mode based on which frame images are set
  const deriveExtensionMode = (
    first: string | undefined,
    last: string | undefined
  ): ExtensionMode | undefined => {
    if (first && last) return "firstlastframe";
    if (first) return "firstframe";
    if (last) return "lastframe";
    return undefined;
  };

  const handleFirstFrameImageChange = (imagePath: string | undefined) => {
    updateSettings({
      firstFrameImage: imagePath,
      extensionMode: deriveExtensionMode(imagePath, settings.lastFrameImage),
    });
  };

  const handleLastFrameImageChange = (imagePath: string | undefined) => {
    updateSettings({
      lastFrameImage: imagePath,
      extensionMode: deriveExtensionMode(settings.firstFrameImage, imagePath),
    });
  };

  const handleExtensionModeChange = (mode: ExtensionMode) => {
    updateSettings({ extensionMode: mode });
  };

  const handleSendExtensionFrames = () => {
    const mode = settings.extensionMode || "firstframe";
    const params: Record<string, string> = {};

    if (mode === "firstframe" && settings.firstFrameImage) {
      params.first_frame_image = settings.firstFrameImage;
    } else if (mode === "lastframe" && settings.lastFrameImage) {
      params.last_frame_image = settings.lastFrameImage;
    } else if (mode === "firstlastframe") {
      if (settings.firstFrameImage) {
        params.first_frame_image = settings.firstFrameImage;
      }
      if (settings.lastFrameImage) {
        params.last_frame_image = settings.lastFrameImage;
      }
    }

    if (Object.keys(params).length > 0) {
      sendParameterUpdate(params);
    }
  };

  const handleResetCache = () => {
    // Send reset cache command to backend
    sendParameterUpdate({
      reset_cache: true,
    });
  };

  const handleOutputSinkChange = (
    sinkType: string,
    config: { enabled: boolean; name: string }
  ) => {
    const updated = { ...settings.outputSinks, [sinkType]: config };
    updateSettings({ outputSinks: updated });
    if (isStreaming) {
      sendParameterUpdate({
        output_sinks: updated,
      });
    }
  };

  const handleOutputSinkBulkChange = (
    sinks: Record<string, { enabled: boolean; name: string }>
  ) => {
    updateSettings({ outputSinks: sinks });
    if (isStreaming) {
      sendParameterUpdate({
        output_sinks: sinks,
      });
    }
  };

  // Handle Spout input name change from InputAndControlsPanel
  const handleSpoutSourceChange = (name: string) => {
    updateSettings({
      inputSource: {
        enabled: mode === "spout",
        source_type: "spout",
        source_name: name,
      },
    });
  };

  // Sync input source settings with mode changes
  const handleModeChange = (newMode: typeof mode) => {
    if (newMode === "spout") {
      updateSettings({
        inputSource: {
          enabled: true,
          source_type: "spout",
          source_name: settings.inputSource?.source_name ?? "",
        },
      });
    } else if (newMode === "ndi") {
      updateSettings({
        inputSource: {
          enabled: true,
          source_type: "ndi",
          source_name: settings.inputSource?.source_name ?? "",
        },
      });
    } else if (newMode === "syphon") {
      updateSettings({
        inputSource: {
          enabled: true,
          source_type: "syphon",
          source_name: settings.inputSource?.source_name ?? "",
        },
      });
    } else {
      updateSettings({
        inputSource: { enabled: false, source_type: "", source_name: "" },
      });
    }
    switchMode(newMode);
  };

  // Handle NDI source selection — probe resolution and update pipeline dimensions
  const handleNdiSourceChange = async (identifier: string) => {
    updateSettings({
      inputSource: {
        enabled: true,
        source_type: "ndi",
        source_name: identifier,
      },
    });

    // Probe the source's native resolution and scale it to fit the pipeline's pixel budget
    try {
      const { width, height } = await getInputSourceResolution(
        "ndi",
        identifier
      );
      updateSettings({
        resolution: capResolution(
          { width, height },
          settings.pipelineId,
          settings.inputMode
        ),
      });
    } catch (e) {
      console.warn("Could not probe NDI source resolution:", e);
    }
  };

  // Handle Syphon source selection — probe resolution and update pipeline dimensions
  const handleSyphonSourceChange = async (identifier: string) => {
    updateSettings({
      inputSource: {
        enabled: true,
        source_type: "syphon",
        source_name: identifier,
      },
    });

    try {
      const { width, height } = await getInputSourceResolution(
        "syphon",
        identifier
      );
      updateSettings({
        resolution: capResolution(
          { width, height },
          settings.pipelineId,
          settings.inputMode
        ),
      });
    } catch (e) {
      console.warn("Could not probe Syphon source resolution:", e);
    }
  };

  const handleLivePromptSubmit = useCallback(
    (prompts: PromptItem[]) => {
      // Use the timeline ref to submit the prompt
      if (timelineRef.current) {
        timelineRef.current.submitLivePrompt(prompts);
      }

      // Also send the updated parameters to the backend immediately
      // Preserve the full blend while live
      sendParameterUpdate({
        prompts,
        prompt_interpolation_method: interpolationMethod,
        denoising_step_list: settings.denoisingSteps || [700, 500],
      });
    },
    [interpolationMethod, sendParameterUpdate, settings.denoisingSteps]
  );

  const handleMidiPromptSolo = useCallback(
    (index: number) => {
      if (index < 0 || index >= promptItems.length) return;

      const newPromptItems = buildSoloPromptItems(index);
      setPromptItems(newPromptItems);

      if (isStreaming) {
        handleLivePromptSubmit(newPromptItems);
      }
    },
    [
      buildSoloPromptItems,
      handleLivePromptSubmit,
      isStreaming,
      promptItems.length,
    ]
  );

  const handleMidiPromptWeightChange = useCallback(
    (index: number, weight: number) => {
      if (index < 0 || index >= promptItems.length) return;

      const nextPromptItems = [...promptItems];
      const remainingWeight = 100 - weight;
      const otherWeightsSum = promptItems.reduce(
        (sum, prompt, promptIndex) =>
          promptIndex === index ? sum : sum + prompt.weight,
        0
      );

      nextPromptItems[index] = { ...nextPromptItems[index], weight };

      if (promptItems.length > 1) {
        if (otherWeightsSum > 0) {
          nextPromptItems.forEach((prompt, promptIndex) => {
            if (promptIndex !== index) {
              const proportion =
                promptItems[promptIndex].weight / otherWeightsSum;
              nextPromptItems[promptIndex] = {
                ...prompt,
                weight: remainingWeight * proportion,
              };
            }
          });
        } else {
          const evenWeight = remainingWeight / (promptItems.length - 1);
          nextPromptItems.forEach((prompt, promptIndex) => {
            if (promptIndex !== index) {
              nextPromptItems[promptIndex] = {
                ...prompt,
                weight: evenWeight,
              };
            }
          });
        }
      }

      setPromptItems(nextPromptItems);

      if (isStreaming) {
        handleLivePromptSubmit(nextPromptItems);
      }
    },
    [handleLivePromptSubmit, isStreaming, promptItems]
  );

  const handleTimelinePromptEdit = (prompt: TimelinePrompt | null) => {
    setSelectedTimelinePrompt(prompt);
    // Sync external selection state
    setExternalSelectedPromptId(prompt?.id || null);
  };

  const handleTimelinePromptUpdate = (prompt: TimelinePrompt) => {
    setSelectedTimelinePrompt(prompt);

    // Update the prompt in the timeline
    if (timelineRef.current) {
      timelineRef.current.updatePrompt(prompt);
    }
  };

  // Event-driven timeline state updates for left panel
  const handleTimelinePromptsChange = (prompts: TimelinePrompt[]) => {
    setTimelinePrompts(prompts);
  };

  const handleTimelineCurrentTimeChange = (currentTime: number) => {
    setTimelineCurrentTime(currentTime);
  };

  const handleTimelinePlayingChange = (isPlaying: boolean) => {
    setIsTimelinePlaying(isPlaying);
  };

  // Keep the OSC command handler ref in sync with current state/handlers
  useEffect(() => {
    oscCommandHandlerRef.current = (cmd: OscCommand) => {
      const { key, value } = cmd;

      switch (key) {
        case "prompt": {
          const prompts: PromptItem[] = [{ text: String(value), weight: 1.0 }];
          setPromptItems(prompts);

          if (isStreaming && transitionSteps > 0) {
            handleTransitionSubmit({
              target_prompts: prompts,
              num_steps: transitionSteps,
              temporal_interpolation_method: temporalInterpolationMethod,
            });
          } else {
            handleLivePromptSubmit(prompts);
          }
          break;
        }
        case "transition_steps":
          setTransitionSteps(Number(value));
          break;
        case "interpolation_method":
          setInterpolationMethod(value as "linear" | "slerp");
          break;
        case "temporal_interpolation_method":
          setTemporalInterpolationMethod(value as "linear" | "slerp");
          break;
        case "noise_scale":
          handleNoiseScaleChange(Number(value));
          break;
        case "noise_controller":
          handleNoiseControllerChange(Boolean(value));
          break;
        case "kv_cache_attention_bias":
          handleKvCacheAttentionBiasChange(Number(value));
          break;
        case "manage_cache":
          handleManageCacheChange(Boolean(value));
          break;
        case "reset_cache":
          handleResetCache();
          break;
        case "vace_context_scale":
          handleVaceContextScaleChange(Number(value));
          break;
        case "paused":
          updateSettings({ paused: Boolean(value) });
          sendParameterUpdate({ paused: Boolean(value) });
          break;
        case "input_mode":
          handleInputModeChange(String(value) as InputMode);
          break;
        case "denoising_step_list": {
          const steps = (Array.isArray(value) ? value : [value])
            .map(v => Number(v))
            .filter(v => Number.isFinite(v))
            .map(v => Math.trunc(v));
          if (steps.length > 0) {
            handleDenoisingStepsChange(steps);
            // Some schema-driven UIs surface denoising as "denoising_steps".
            // Keep that override in sync so controls visibly update.
            updateSettings({
              schemaFieldOverrides: {
                ...(settings.schemaFieldOverrides ?? {}),
                denoising_steps: steps,
              },
            });
          }
          break;
        }
        default: {
          // Pipeline-specific runtime params:
          // update frontend override state first, then forward to backend.
          if (pipelineHasRuntimeField(settings.pipelineId, key)) {
            updateSettings({
              schemaFieldOverrides: {
                ...(settings.schemaFieldOverrides ?? {}),
                [key]: value,
              },
            });
          }

          const updateProcessorOverrides = (
            processorIds: string[] | undefined,
            currentOverrides:
              | Record<string, Record<string, unknown>>
              | undefined,
            overridesKey:
              | "preprocessorSchemaFieldOverrides"
              | "postprocessorSchemaFieldOverrides"
          ) => {
            if (!processorIds?.length) {
              return;
            }
            const nextOverrides = { ...(currentOverrides ?? {}) };
            let changed = false;

            for (const pid of processorIds) {
              if (!pipelineHasRuntimeField(pid, key)) {
                continue;
              }
              nextOverrides[pid] = {
                ...(nextOverrides[pid] ?? {}),
                [key]: value,
              };
              changed = true;
            }

            if (changed) {
              updateSettings({ [overridesKey]: nextOverrides });
            }
          };

          updateProcessorOverrides(
            settings.preprocessorIds,
            settings.preprocessorSchemaFieldOverrides,
            "preprocessorSchemaFieldOverrides"
          );
          updateProcessorOverrides(
            settings.postprocessorIds,
            settings.postprocessorSchemaFieldOverrides,
            "postprocessorSchemaFieldOverrides"
          );

          if (isStreaming) {
            sendParameterUpdate({ [key]: value } as Record<string, unknown>);
          }
          break;
        }
      }
    };
  });

  // Handle ESC key to exit Edit mode and return to Append mode
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && selectedTimelinePrompt) {
        setSelectedTimelinePrompt(null);
        setExternalSelectedPromptId(null);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [selectedTimelinePrompt]);

  // Subscribe to OSC commands via SSE so each update arrives individually
  // and triggers its own render tick, giving smooth slider movement.
  useEffect(() => {
    const es = new EventSource("/api/v1/osc/stream");

    es.onmessage = event => {
      try {
        const cmd = JSON.parse(event.data) as OscCommand;
        oscCommandHandlerRef.current(cmd);
      } catch (err) {
        console.debug("[StreamPage] Failed to parse OSC SSE event:", err);
      }
    };

    es.onerror = () => {
      console.debug(
        "[StreamPage] OSC SSE connection error, will auto-reconnect"
      );
    };

    return () => es.close();
  }, []);

  // Send quantize_mode and lookahead_ms to backend when settings change.
  // Uses sendParameterUpdateWebRTC directly to avoid depending on the
  // wrapper (which would re-fire this effect on unrelated changes).
  useEffect(() => {
    if (isStreaming) {
      sendParameterUpdateWebRTC({
        quantize_mode: settings.quantizeMode || "none",
        lookahead_ms: settings.lookaheadMs ?? 0,
      });
    }
  }, [
    settings.quantizeMode,
    settings.lookaheadMs,
    isStreaming,
    sendParameterUpdateWebRTC,
  ]);

  // Send modulation config to backend when it changes or stream starts.
  useEffect(() => {
    if (isStreaming && settings.modulations) {
      sendParameterUpdateWebRTC({ modulations: settings.modulations });
    }
  }, [settings.modulations, isStreaming, sendParameterUpdateWebRTC]);

  // Send beat cache reset rate to backend when it changes or stream starts.
  useEffect(() => {
    if (isStreaming) {
      sendParameterUpdateWebRTC({
        beat_cache_reset_rate: settings.beatCacheResetRate || "none",
      });
    }
  }, [settings.beatCacheResetRate, isStreaming, sendParameterUpdateWebRTC]);

  // Send input_source config to backend when it changes during streaming.
  // This enables live switching between source types (e.g. Syphon → File)
  // without restarting the stream.
  useEffect(() => {
    if (isStreaming && settings.inputSource) {
      sendParameterUpdateWebRTC({
        input_source: settings.inputSource,
      });
    }
  }, [settings.inputSource, isStreaming, sendParameterUpdateWebRTC]);

  // Beat-synced prompt cycling: rotate through the queued prompt items on beat
  // boundaries. Each prompt item is applied individually at full weight in sequence.
  // The promptItems list in the UI stays unchanged; only the backend receives the
  // currently-active single prompt.
  const promptCycleBoundaryRef = useRef(-1);
  const promptCycleIndexRef = useRef(0);
  const promptCycleItemsRef = useRef<PromptItem[]>([]);
  // Snapshot the prompt list when cycling is enabled so edits don't disrupt the cycle
  useEffect(() => {
    if (
      (settings.promptCycleRate || "none") !== "none" &&
      promptItems.length >= 2
    ) {
      promptCycleItemsRef.current = promptItems;
      promptCycleIndexRef.current = 0;
      promptCycleBoundaryRef.current = -1;
    }
  }, [settings.promptCycleRate, promptItems]);

  const tempoEnabled = tempoState.enabled;
  const tempoBeatCount = tempoState.beatCount;
  const tempoBeatsPerBar = tempoState.beatsPerBar;

  useEffect(() => {
    const rate = settings.promptCycleRate || "none";
    const items = promptCycleItemsRef.current;
    if (rate === "none" || !isStreaming || !tempoEnabled || items.length < 2) {
      promptCycleBoundaryRef.current = -1;
      return;
    }

    let boundary: number;
    if (rate === "beat") boundary = tempoBeatCount;
    else if (rate === "bar")
      boundary = Math.floor(tempoBeatCount / tempoBeatsPerBar);
    else if (rate === "2_bar")
      boundary = Math.floor(tempoBeatCount / (tempoBeatsPerBar * 2));
    else if (rate === "4_bar")
      boundary = Math.floor(tempoBeatCount / (tempoBeatsPerBar * 4));
    else return;

    if (
      boundary !== promptCycleBoundaryRef.current &&
      promptCycleBoundaryRef.current >= 0
    ) {
      promptCycleIndexRef.current =
        (promptCycleIndexRef.current + 1) % items.length;
      const active = items[promptCycleIndexRef.current];

      // Send directly via WebRTC to bypass the quantize logic in
      // sendParameterUpdate — this effect already fires on the boundary.
      const promptParams = {
        prompts: [{ text: active.text, weight: 100 }],
      };
      sendParameterUpdateWebRTC(promptParams);
      applyBackendParamsToSettings(promptParams);
    }
    promptCycleBoundaryRef.current = boundary;
  }, [
    settings.promptCycleRate,
    tempoBeatCount,
    tempoBeatsPerBar,
    tempoEnabled,
    isStreaming,
    sendParameterUpdateWebRTC,
    applyBackendParamsToSettings,
  ]);

  // Subscribe to DMX commands via SSE only when DMX is enabled
  const [dmxEnabled, setDmxEnabled] = useState(false);

  useEffect(() => {
    getDmxStatus()
      .then(s => setDmxEnabled(s.enabled))
      .catch(() => setDmxEnabled(false));
  }, []);

  useEffect(() => {
    if (!dmxEnabled) return;

    const es = new EventSource("/api/v1/dmx/stream");

    es.onmessage = event => {
      try {
        const cmd = JSON.parse(event.data) as OscCommand;
        oscCommandHandlerRef.current(cmd);
      } catch (err) {
        console.debug("[StreamPage] Failed to parse DMX SSE event:", err);
      }
    };

    es.onerror = () => {
      console.debug(
        "[StreamPage] DMX SSE connection error; browser will retry"
      );
    };

    return () => es.close();
  }, [dmxEnabled]);

  // Update temporal interpolation defaults and clear prompts when pipeline changes
  useEffect(() => {
    const pipeline = pipelines?.[settings.pipelineId];
    if (pipeline) {
      const defaultMethod =
        pipeline.defaultTemporalInterpolationMethod || "slerp";
      const pipelineDefaultSteps =
        pipeline.defaultTemporalInterpolationSteps ?? 4;
      // Get mode-specific default if available
      const modeDefaults = getDefaults(settings.pipelineId, settings.inputMode);
      const defaultSteps =
        modeDefaults.defaultTemporalInterpolationSteps ?? pipelineDefaultSteps;

      setTemporalInterpolationMethod(defaultMethod);
      setTransitionSteps(defaultSteps);

      // Clear prompts if pipeline doesn't support them
      if (pipeline.supportsPrompts === false) {
        setPromptItems([{ text: "", weight: 1.0 }]);
      }
    }
  }, [settings.pipelineId, pipelines, settings.inputMode, getDefaults]);

  const handlePlayPauseToggle = () => {
    const newPausedState = !settings.paused;
    updateSettings({ paused: newPausedState });
    sendParameterUpdate({
      paused: newPausedState,
    });

    // Deselect any selected prompt when video starts playing
    if (!newPausedState && selectedTimelinePrompt) {
      setSelectedTimelinePrompt(null);
      setExternalSelectedPromptId(null); // Also clear external selection
    }
  };

  // Ref to access the timeline's play/pause handler
  const timelinePlayPauseRef = useRef<(() => Promise<void>) | null>(null);

  // Ref to store callback that should execute when video starts playing
  const onVideoPlayingCallbackRef = useRef<(() => void) | null>(null);

  // Note: We intentionally do NOT auto-sync videoResolution to settings.resolution.
  // Mode defaults from the backend schema take precedence. Users can manually
  // adjust resolution if needed. This prevents the video source resolution from
  // overriding the carefully tuned per-mode defaults.

  // Wait for an in-progress cloud connection to complete before starting WebRTC
  const waitForCloudConnection = async (): Promise<boolean> => {
    const maxWaitMs = 180_000; // 3 minutes
    const pollIntervalMs = 2000;
    const start = Date.now();

    while (Date.now() - start < maxWaitMs) {
      try {
        const response = await fetch("/api/v1/cloud/status");
        if (response.ok) {
          const data = await response.json();
          if (data.connected) return true;
          if (!data.connecting) {
            // Not connecting and not connected — connection failed
            console.error("Cloud connection failed:", data.error);
            return false;
          }
        }
      } catch (e) {
        console.error("Error polling cloud status:", e);
      }
      await new Promise(resolve => setTimeout(resolve, pollIntervalMs));
    }
    console.error("Timed out waiting for cloud connection");
    return false;
  };

  const handleStartStream = async (
    overridePipelineId?: PipelineId
  ): Promise<boolean> => {
    if (isStreaming) {
      stopStream();
      return true;
    }

    // Use override pipeline ID if provided, otherwise use current settings
    let pipelineIdToUse = overridePipelineId || settings.pipelineId;

    try {
      // Build pipeline chain: preprocessors + main pipeline + postprocessors
      const pipelineIds: string[] = [];
      if (settings.preprocessorIds && settings.preprocessorIds.length > 0) {
        pipelineIds.push(...settings.preprocessorIds);
      }
      pipelineIds.push(pipelineIdToUse);
      if (settings.postprocessorIds && settings.postprocessorIds.length > 0) {
        pipelineIds.push(...settings.postprocessorIds);
      }

      // In graph mode (or when a custom graph exists from graph mode),
      // extract pipeline IDs and source mode from the graph so the
      // workflow builder settings dominate over perform-mode defaults.
      let graphSourceMode: string | null = null;
      let graphInputSource: {
        enabled: boolean;
        source_type: string;
        source_name: string;
      } | null = null;
      // Sink node IDs for multi-track WebRTC
      const graphSinkNodeIds: string[] = [];
      // The graph config to pass via initialParameters (sent over WebRTC)
      let graphConfigForStream: ReturnType<
        NonNullable<typeof graphEditorRef.current>["getCurrentGraphConfig"]
      > | null = null;

      if (graphMode || nonLinearGraph) {
        try {
          // Read graph from frontend React state (always up-to-date)
          const frontendGraph = graphEditorRef.current?.getCurrentGraphConfig();
          if (frontendGraph) {
            graphConfigForStream = frontendGraph;
          }

          const graphNodes = frontendGraph?.nodes ?? null;

          if (graphNodes) {
            const graphPipelineIds = graphNodes
              .filter(n => n.type === "pipeline" && n.pipeline_id)
              .map(n => n.pipeline_id as string);
            if (graphPipelineIds.length > 0) {
              pipelineIds.length = 0;
              pipelineIds.push(...graphPipelineIds);
              // Find the main pipeline (first non-preprocessor, non-postprocessor)
              const mainPid = graphPipelineIds.find(pid => {
                const schema = pipelines?.[pid];
                const usage = schema?.usage ?? [];
                return (
                  !usage.includes("preprocessor") &&
                  !usage.includes("postprocessor")
                );
              });
              pipelineIdToUse = mainPid ?? graphPipelineIds[0];
            }

            // Extract sink node IDs for multi-track WebRTC
            graphSinkNodeIds.push(
              ...graphNodes.filter(n => n.type === "sink").map(n => n.id)
            );

            // Extract source mode from all source nodes and normalize
            // to a valid InputMode. All source_mode values (video, camera,
            // spout, ndi, syphon) need video input, so graphSourceMode is
            // always "video". For server-side sources we also capture the
            // input source config so the backend receives it.
            const sourceNodes = graphNodes.filter(n => n.type === "source");
            if (sourceNodes.length > 0) {
              graphSourceMode = "video";

              // Use first server-side source for backward compat input_source param
              for (const sourceNode of sourceNodes) {
                const sm = sourceNode.source_mode || "video";
                if (sm === "spout" || sm === "ndi" || sm === "syphon") {
                  graphInputSource = {
                    enabled: true,
                    source_type: sm,
                    source_name: sourceNode.source_name ?? "",
                  };
                  break;
                }
              }
            }
          }

          // Adjust resolution in graph node params BEFORE anything reads them.
          // This ensures all downstream code (loadItems, initialParameters, etc.)
          // sees the corrected values and the UI reflects them immediately.
          if (graphConfigForStream?.ui_state) {
            const nParams = graphConfigForStream.ui_state.node_params as
              | Record<string, Record<string, unknown>>
              | undefined;
            if (nParams && graphNodes) {
              for (const node of graphNodes) {
                if (node.type !== "pipeline" || !node.pipeline_id) continue;
                const bag = nParams[node.id];
                if (!bag) continue;
                const h =
                  typeof bag.height === "number"
                    ? Math.round(bag.height)
                    : undefined;
                const w =
                  typeof bag.width === "number"
                    ? Math.round(bag.width)
                    : undefined;
                if (h != null && w != null) {
                  const { resolution: adj, wasAdjusted } =
                    adjustResolutionForPipeline(
                      node.pipeline_id as PipelineId,
                      { height: h, width: w }
                    );
                  // Always write back rounded integers
                  bag.height = wasAdjusted ? adj.height : h;
                  bag.width = wasAdjusted ? adj.width : w;
                  if (wasAdjusted) {
                    console.log(
                      `[GraphMode] Adjusted ${node.pipeline_id} resolution: ${w}×${h} → ${bag.width}×${bag.height}`
                    );
                  }
                  // Update the graph editor UI so the user sees corrected values
                  graphEditorRef.current?.updateNodeParam(
                    node.id,
                    "height",
                    bag.height
                  );
                  graphEditorRef.current?.updateNodeParam(
                    node.id,
                    "width",
                    bag.width
                  );
                }
              }
            }
          }
        } catch (err) {
          console.warn("Failed to extract pipeline IDs from graph:", err);
        }
      }

      // Check if models are needed but not downloaded for all pipelines in the chain
      // Skip this check if cloud is connecting - we'll wait for connection and then
      // the model check will happen on the cloud side
      if (!isBackendCloudConnecting) {
        const missingPipelines: string[] = [];
        for (const pipelineId of pipelineIds) {
          const pipelineInfo = pipelines?.[pipelineId];
          if (pipelineInfo?.requiresModels) {
            try {
              const status = await api.checkModelStatus(pipelineId);
              if (!status.downloaded) {
                missingPipelines.push(pipelineId);
              }
            } catch (error) {
              console.error(
                `Error checking model status for ${pipelineId}:`,
                error
              );
              // Continue anyway if check fails
            }
          }
        }

        // If any pipelines are missing models, show download dialog
        if (missingPipelines.length > 0) {
          setPipelinesNeedingModels(missingPipelines);
          setShowDownloadDialog(true);
          return false; // Stream did not start
        }
      }

      // If cloud connection is in progress, wait for it before loading pipeline
      // (pipeline load is proxied to cloud only when connected)
      // Check API directly rather than React state to avoid stale values
      try {
        const cloudRes = await fetch("/api/v1/cloud/status");
        if (cloudRes.ok) {
          const cloudData = await cloudRes.json();
          if (cloudData.connecting && !cloudData.connected) {
            console.log(
              "[StreamPage] Cloud connecting, waiting before pipeline load..."
            );
            setIsCloudConnecting(true);
            try {
              const cloudReady = await waitForCloudConnection();
              if (!cloudReady) {
                console.error("Cloud connection failed, cannot start stream");
                return false;
              }
            } finally {
              setIsCloudConnecting(false);
            }
          }
        }
      } catch (e) {
        console.error("Error checking cloud status before stream:", e);
      }

      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${pipelineIdToUse} pipeline...`);

      // Determine current input mode – in graph mode, prefer the source node's
      // mode so the backend receives the correct input_mode (e.g. "video")
      let currentMode =
        settings.inputMode || getPipelineDefaultMode(pipelineIdToUse) || "text";
      if ((graphMode || nonLinearGraph) && graphSourceMode) {
        currentMode = graphSourceMode as InputMode;
      }

      // Use settings.resolution if available, otherwise fall back to videoResolution
      let resolution = settings.resolution || videoResolution;

      // Adjust resolution to be divisible by required scale factor for the pipeline
      if (resolution) {
        const { resolution: adjustedResolution, wasAdjusted } =
          adjustResolutionForPipeline(pipelineIdToUse, resolution);

        if (wasAdjusted) {
          // Update settings with adjusted resolution
          updateSettings({ resolution: adjustedResolution });
          resolution = adjustedResolution;
        }
      }

      // Build load parameters dynamically based on pipeline capabilities and settings
      // The backend will use only the parameters it needs based on the pipeline schema
      const currentPipeline = pipelines?.[pipelineIdToUse];
      // Compute VACE enabled state - needed for both loadParams and initialParameters
      const vaceEnabled = currentPipeline?.supportsVACE
        ? (settings.vaceEnabled ?? currentMode !== "video")
        : false;

      let loadParams: Record<string, unknown> | null = null;

      if (resolution) {
        // Start with common parameters
        loadParams = {
          height: resolution.height,
          width: resolution.width,
        };

        // Add quantization when pipeline supports it
        if (currentPipeline?.supportsQuantization) {
          loadParams.quantization = settings.quantization ?? null;
        }

        // Add LoRA parameters if pipeline supports LoRA
        if (currentPipeline?.supportsLoRA && settings.loras) {
          const loraParams = buildLoRAParams(
            settings.loras,
            settings.loraMergeStrategy
          );
          loadParams = { ...loadParams, ...loraParams };
        }

        // Add VACE parameters if pipeline supports VACE
        if (currentPipeline?.supportsVACE) {
          loadParams.vace_enabled = vaceEnabled;
          loadParams.vace_context_scale = settings.vaceContextScale ?? 1.0;

          // Add VACE reference images if provided
          const vaceParams = getVaceParams(
            settings.refImages,
            settings.vaceContextScale
          );
          loadParams = { ...loadParams, ...vaceParams };
        }

        // Merge schema-driven primitive fields (e.g. new_param) so backend receives them.
        // Exclude height/width — they are handled by the dedicated resolution logic
        // above and must not be overridden by stale schemaFieldOverrides values.
        if (
          settings.schemaFieldOverrides &&
          Object.keys(settings.schemaFieldOverrides).length > 0
        ) {
          const schemaRest = Object.fromEntries(
            Object.entries(settings.schemaFieldOverrides).filter(
              ([k]) => k !== "height" && k !== "width"
            )
          );
          if (Object.keys(schemaRest).length > 0) {
            loadParams = { ...loadParams, ...schemaRest };
          }
        }

        // Include per-processor schema overrides as flat params
        for (const kind of ["preprocessor", "postprocessor"] as const) {
          const k = pipelineSettingsKeys[kind];
          const ids = settings[k.ids] as string[] | undefined;
          const overrides = settings[k.overrides] as
            | Record<string, Record<string, unknown>>
            | undefined;
          if (ids?.length && overrides) {
            for (const id of ids) {
              const processorOverrides = overrides[id];
              if (
                processorOverrides &&
                Object.keys(processorOverrides).length > 0
              ) {
                Object.assign(loadParams, processorOverrides);
              }
            }
          }
        }

        console.log(
          `Loading ${pipelineIds.length} pipeline(s) (${pipelineIds.join(", ")}) with resolution ${resolution.width}x${resolution.height}`,
          loadParams
        );
      }

      // Build a linear graph for perform mode so backend uses the unified graph path.
      // In graph/nonLinearGraph mode, graphConfigForStream is already set above.
      if (!graphMode && !nonLinearGraph) {
        // Resolve which pipelines should receive input as vace_input_frames
        // (mirrors the backend's _setup_pipelines_sync fallback logic)
        const vaceInputVideoIds = new Set<string>();
        if (
          vaceEnabled &&
          (settings.vaceUseInputVideo ?? false) &&
          currentMode === "video"
        ) {
          const allIds = [
            ...(settings.preprocessorIds ?? []),
            pipelineIdToUse,
            ...(settings.postprocessorIds ?? []),
          ];
          for (const pid of allIds) {
            if (pipelines?.[pid]?.supportsVACE) {
              vaceInputVideoIds.add(pid);
            }
          }
        }

        graphConfigForStream = linearGraphFromSettings(
          pipelineIdToUse,
          settings.preprocessorIds ?? [],
          settings.postprocessorIds ?? [],
          vaceInputVideoIds.size > 0 ? vaceInputVideoIds : undefined
        );
      }

      // Build PipelineLoadItem[] from graph nodes (always available at this
      // point — perform mode builds a linear graph above).
      // Per-node load_params ensures each pipeline gets vace_enabled when it supports VACE
      // and LoRA adapters when configured via LoRA nodes.
      const loraSettings = graphEditorRef.current?.getGraphLoRASettings() ?? [];
      const loraByNode = new Map(loraSettings.map(s => [s.pipelineNodeId, s]));

      const loadItems: PipelineLoadItem[] = graphConfigForStream
        ? graphConfigForStream.nodes
            .filter(n => n.type === "pipeline" && n.pipeline_id)
            .map(n => {
              const pid = n.pipeline_id as string;
              const pipeSchema = pipelines?.[pid];
              const nodeLoadParams = { ...(loadParams ?? {}) };
              const nParams = graphConfigForStream?.ui_state?.node_params as
                | Record<string, Record<string, unknown>>
                | undefined;
              const nodeBag = nParams?.[n.id];
              if (typeof nodeBag?.height === "number")
                nodeLoadParams.height = Math.round(nodeBag.height);
              if (typeof nodeBag?.width === "number")
                nodeLoadParams.width = Math.round(nodeBag.width);
              // Ensure resolution is divisible by pipeline's required scale factor
              if (
                typeof nodeLoadParams.height === "number" &&
                typeof nodeLoadParams.width === "number"
              ) {
                const { resolution: adjRes, wasAdjusted } =
                  adjustResolutionForPipeline(pid as PipelineId, {
                    height: nodeLoadParams.height as number,
                    width: nodeLoadParams.width as number,
                  });
                if (wasAdjusted) {
                  nodeLoadParams.height = adjRes.height;
                  nodeLoadParams.width = adjRes.width;
                }
              }
              if (pipeSchema?.supportsQuantization) {
                const nodeQuant = nodeBag?.quantization;
                if (typeof nodeQuant === "string") {
                  nodeLoadParams.quantization = nodeQuant;
                } else {
                  // Compute VRAM-based default for this pipeline
                  const vramThreshold =
                    pipeSchema.recommendedQuantizationVramThreshold;
                  if (vramThreshold != null && hardwareInfo?.vram_gb != null) {
                    nodeLoadParams.quantization =
                      hardwareInfo.vram_gb > vramThreshold
                        ? null
                        : "fp8_e4m3fn";
                  } else {
                    nodeLoadParams.quantization = settings.quantization ?? null;
                  }
                }
              }
              if (pipeSchema?.supportsVACE) {
                const hasVaceEdge = graphConfigForStream!.edges.some(
                  e =>
                    e.to_node === n.id &&
                    (e.to_port === "vace_input_frames" ||
                      e.to_port === "vace_input_masks")
                );
                nodeLoadParams.vace_enabled = hasVaceEdge;
                const nodeVaceScale = nodeBag?.vace_context_scale;
                nodeLoadParams.vace_context_scale =
                  typeof nodeVaceScale === "number"
                    ? nodeVaceScale
                    : (settings.vaceContextScale ?? 1.0);
              }
              const loraConfig = loraByNode.get(n.id);
              if (loraConfig && loraConfig.loras.length > 0) {
                nodeLoadParams.loras = loraConfig.loras.map(l => ({
                  ...l,
                  path: resolveLoRAPath(l.path, loraFiles),
                }));
                nodeLoadParams.lora_merge_mode = loraConfig.lora_merge_mode;
              }
              return {
                node_id: n.id,
                pipeline_id: pid,
                load_params: nodeLoadParams,
              };
            })
        : pipelineIds.map(pid => ({
            node_id: pid,
            pipeline_id: pid,
            load_params: loadParams,
          }));

      // Log the resolution values being sent to loadPipeline for debugging
      for (const item of loadItems) {
        const lp = item.load_params as Record<string, unknown> | undefined;
        console.log(
          `[GraphMode] Loading pipeline ${item.pipeline_id} (node ${item.node_id}) with resolution: ${lp?.width}×${lp?.height}`
        );
      }

      const loadSuccess = await loadPipeline(loadItems);
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        return false;
      }

      // Check video requirements based on input mode
      const needsVideoInput = currentMode === "video";
      const isSpoutMode =
        mode === "spout" && settings.inputSource?.source_type === "spout";
      const isNdiMode =
        mode === "ndi" && settings.inputSource?.source_type === "ndi";
      const isSyphonMode =
        mode === "syphon" && settings.inputSource?.source_type === "syphon";
      const isServerSideInput = isSpoutMode || isNdiMode || isSyphonMode;

      // Only send video stream for pipelines that need video input (not in Spout/NDI mode)
      const streamToSend =
        needsVideoInput && !isServerSideInput
          ? localStream || undefined
          : undefined;

      if (needsVideoInput && !isServerSideInput && !localStream) {
        console.error("Video input required but no local stream available");
        return false;
      }

      // Build initial parameters based on pipeline type
      const initialParameters: {
        input_mode?: "text" | "video";
        prompts?: PromptItem[];
        prompt_interpolation_method?: "linear" | "slerp";
        denoising_step_list?: number[];
        noise_scale?: number;
        noise_controller?: boolean;
        manage_cache?: boolean;
        kv_cache_attention_bias?: number;
        output_sinks?: Record<string, { enabled: boolean; name: string }>;
        vace_ref_images?: string[];
        vace_use_input_video?: boolean;
        vace_context_scale?: number;
        vace_enabled?: boolean;
        pipeline_ids?: string[];
        produces_video?: boolean;
        produces_audio?: boolean;
        first_frame_image?: string;
        last_frame_image?: string;
        images?: string[];
        recording?: boolean;
        input_source?: {
          enabled: boolean;
          source_type: string;
          source_name: string;
        };
        graph?: GraphConfig;
      } = {
        // Signal the intended input mode to the backend so it doesn't
        // briefly fall back to text mode before video frames arrive
        input_mode: currentMode,
      };

      // Common parameters for pipelines that support prompts
      if (currentPipeline?.supportsPrompts !== false) {
        initialParameters.prompts = promptItems;
        initialParameters.prompt_interpolation_method = interpolationMethod;
        initialParameters.denoising_step_list = settings.denoisingSteps || [
          700, 500,
        ];
      }

      // In graph mode, use the graph node's prompt instead of perform mode defaults
      if (graphMode || nonLinearGraph) {
        const graphPrompts = graphEditorRef.current?.getGraphNodePrompts();
        if (graphPrompts && graphPrompts.length > 0) {
          initialParameters.prompts = [
            { text: graphPrompts[0].text, weight: 100 },
          ];
        }
      }

      // Cache management for pipelines that support it
      if (currentPipeline?.supportsCacheManagement) {
        initialParameters.manage_cache = settings.manageCache ?? true;
      }

      // KV cache bias for pipelines that support it
      if (currentPipeline?.supportsKvCacheBias) {
        initialParameters.kv_cache_attention_bias =
          settings.kvCacheAttentionBias ?? 1.0;
      }

      // Pipeline chain: preprocessors + main pipeline (already built above)
      initialParameters.pipeline_ids = pipelineIds;

      // Media modalities from pipeline status — used by the backend to decide
      // which tracks to create (avoids unnecessary audio processing for
      // video-only pipelines, and vice versa). Read from the ref (not React
      // state) to guarantee fresh values in the same tick loadPipeline resolved.
      const latestInfo = pipelineInfoRef.current;
      initialParameters.produces_video = latestInfo?.produces_video ?? true;
      initialParameters.produces_audio = latestInfo?.produces_audio ?? false;

      // VACE-specific parameters
      if (graphMode || nonLinearGraph) {
        // In graph mode, extract VACE settings from VaceNode connections.
        // The VaceNode's compound output (__vace) carries context_scale,
        // ref_images, and frame references to the connected pipeline node.
        const vaceSettings =
          graphEditorRef.current?.getGraphVaceSettings() ?? [];
        if (vaceSettings.length > 0) {
          // Use the first VaceNode's settings for global initialParameters
          // (backend broadcasts initial_parameters to all processors)
          const first = vaceSettings[0];
          initialParameters.vace_enabled = true;
          initialParameters.vace_context_scale = first.vace_context_scale;
          initialParameters.vace_use_input_video = first.vace_use_input_video;
          if (first.vace_ref_images?.length) {
            initialParameters.vace_ref_images = first.vace_ref_images;
          }
          if (first.first_frame_image) {
            initialParameters.first_frame_image = first.first_frame_image;
          }
          if (first.last_frame_image) {
            initialParameters.last_frame_image = first.last_frame_image;
          }
        } else {
          // No VaceNode in graph; check if any pipeline supports VACE and
          // use perform-mode settings as fallback
          const anyVace = pipelineIds.some(
            pid => pipelines?.[pid]?.supportsVACE
          );
          if (anyVace) {
            initialParameters.vace_enabled = vaceEnabled;
            initialParameters.vace_context_scale =
              settings.vaceContextScale ?? 1.0;
            if (currentMode === "video") {
              initialParameters.vace_use_input_video =
                settings.vaceUseInputVideo ?? false;
            }
            const vaceParams = getVaceParams(
              settings.refImages,
              settings.vaceContextScale
            );
            if ("vace_ref_images" in vaceParams) {
              initialParameters.vace_ref_images = vaceParams.vace_ref_images;
            }
          }
        }
      } else if (currentPipeline?.supportsVACE) {
        // Perform mode: use settings panel values
        const vaceParams = getVaceParams(
          settings.refImages,
          settings.vaceContextScale
        );
        if ("vace_ref_images" in vaceParams) {
          initialParameters.vace_ref_images = vaceParams.vace_ref_images;
        }
        initialParameters.vace_context_scale = settings.vaceContextScale ?? 1.0;
        if (currentMode === "video") {
          initialParameters.vace_use_input_video =
            settings.vaceUseInputVideo ?? false;
        }
        initialParameters.vace_enabled = vaceEnabled;
      } else if (
        currentPipeline?.supportsImages &&
        settings.refImages?.length
      ) {
        initialParameters.images = settings.refImages;
      }

      // Add FFLF (first-frame-last-frame) parameters if set
      if (settings.firstFrameImage) {
        initialParameters.first_frame_image = settings.firstFrameImage;
      }
      if (settings.lastFrameImage) {
        initialParameters.last_frame_image = settings.lastFrameImage;
      }

      // Video mode parameters - applies to all pipelines in video mode
      if (currentMode === "video") {
        initialParameters.noise_scale = settings.noiseScale ?? 0.7;
        initialParameters.noise_controller = settings.noiseController ?? true;
      }

      // Output sinks - send if any are enabled
      if (settings.outputSinks) {
        const enabledSinks = Object.fromEntries(
          Object.entries(settings.outputSinks).filter(([, v]) => v.enabled)
        );
        if (Object.keys(enabledSinks).length > 0) {
          initialParameters.output_sinks = enabledSinks;
        }
      }

      // Generic input source (NDI, Spout, etc.) - send if enabled.
      // In graph/workflow mode, prefer the graph's source config over perform-mode settings.
      if ((graphMode || nonLinearGraph) && graphInputSource) {
        initialParameters.input_source = graphInputSource;
      } else if (settings.inputSource?.enabled) {
        initialParameters.input_source = settings.inputSource;
      }

      // Pass graph config via initialParameters for the backend to use
      // Strip frontend-only fields (position, size, ui_state) before sending.
      if (graphConfigForStream) {
        initialParameters.graph = stripUIFields(graphConfigForStream);
      }

      // Include recording toggle state
      initialParameters.recording = isRecording;

      // Include runtime schema field overrides so they reach __call__ on first frame.
      // Exclude height/width — they are load-time params baked into the pipeline
      // during initialization and must not leak into runtime parameters (which
      // would override the adjusted resolution on every __call__).
      if (
        settings.schemaFieldOverrides &&
        Object.keys(settings.schemaFieldOverrides).length > 0
      ) {
        const schemaRuntime = Object.fromEntries(
          Object.entries(settings.schemaFieldOverrides).filter(
            ([k]) => k !== "height" && k !== "width"
          )
        );
        Object.assign(initialParameters, schemaRuntime);
      }

      // Override initialParameters with graph node params for the main pipeline.
      // When a workflow is imported into graph mode, per-node params (denoising
      // steps, noise scale, etc.) are stored in ui_state.node_params but not
      // reflected in settings.*. Apply them so the backend receives the correct
      // values at stream start.
      if ((graphMode || nonLinearGraph) && graphConfigForStream?.ui_state) {
        const nodeParams = graphConfigForStream.ui_state.node_params as
          | Record<string, Record<string, unknown>>
          | undefined;
        if (nodeParams) {
          const mainNode = graphConfigForStream.nodes.find(
            n => n.type === "pipeline" && n.pipeline_id === pipelineIdToUse
          );
          const mainNodeParams = mainNode ? nodeParams[mainNode.id] : undefined;
          if (mainNodeParams) {
            const RUNTIME_PARAMS = [
              "denoising_step_list",
              "noise_scale",
              "noise_controller",
              "manage_cache",
              "kv_cache_attention_bias",
              "vace_enabled",
              "vace_context_scale",
              "vace_use_input_video",
            ] as const;
            for (const key of RUNTIME_PARAMS) {
              if (mainNodeParams[key] !== undefined) {
                (initialParameters as Record<string, unknown>)[key] =
                  mainNodeParams[key];
              }
            }
          }
        }
      }

      // Reset paused state when starting a fresh stream
      updateSettings({ paused: false });

      // Build per-source-node streams for multi-source WebRTC
      let sourceNodeStreamsForWebRTC: Record<string, MediaStream> | undefined;
      if (graphConfigForStream) {
        const webrtcSourceNodes = (graphConfigForStream.nodes ?? []).filter(
          n =>
            n.type === "source" &&
            (n.source_mode || "video") !== "spout" &&
            (n.source_mode || "video") !== "ndi" &&
            (n.source_mode || "video") !== "syphon"
        );
        if (webrtcSourceNodes.length > 1) {
          const streams: Record<string, MediaStream> = {};
          for (const node of webrtcSourceNodes) {
            const nodeStream = nodeLocalStreams[node.id];
            if (nodeStream) {
              streams[node.id] = nodeStream;
            } else if (localStream) {
              streams[node.id] = localStream;
            }
          }
          if (Object.keys(streams).length > 0) {
            sourceNodeStreamsForWebRTC = streams;
          }
        }
      }

      // Pipeline is loaded, now start WebRTC stream
      startStream(
        initialParameters,
        sourceNodeStreamsForWebRTC ? undefined : streamToSend,
        graphSinkNodeIds.length > 0 ? graphSinkNodeIds : undefined,
        sourceNodeStreamsForWebRTC
      );

      return true; // Stream started successfully
    } catch (error) {
      console.error("Error during stream start:", error);
      return false;
    }
  };

  const handleSaveGeneration = async () => {
    try {
      if (!sessionId) {
        toast.error("No active session", {
          description: "Please start a stream before downloading the recording",
          duration: 5000,
        });
        return;
      }
      await api.downloadRecording(sessionId);
    } catch (error) {
      console.error("Error downloading recording:", error);
      toast.error("Error downloading recording", {
        description:
          error instanceof Error
            ? error.message
            : "An error occurred while downloading the recording",
        duration: 5000,
      });
    }
  };

  const handleStartRecording = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.startRecording(sessionId);
    } catch (error) {
      console.error("Error starting recording:", error);
      toast.error("Failed to start recording");
    }
  }, [sessionId, api]);

  const handleStopRecording = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.downloadRecording(sessionId);
    } catch (error) {
      console.error("Error downloading recording:", error);
      toast.error("Failed to save recording");
    }
    try {
      await api.stopRecording(sessionId);
    } catch (error) {
      console.error("Error stopping recording:", error);
      toast.error("Failed to stop recording");
    }
  }, [sessionId, api]);

  // Handle workflow import: load settings, timeline, and prompt state
  const handleWorkflowLoad = useCallback(
    (
      importedSettings: Partial<typeof settings>,
      importedTimeline: TimelinePrompt[],
      promptState: WorkflowPromptState | null
    ) => {
      // Prevent the auto-mode-reset effect from overriding the workflow's inputMode
      if (importedSettings.pipelineId) {
        skipNextModeReset(importedSettings.pipelineId);
      }

      updateSettings(importedSettings);

      // Trigger video source reinitialization if the workflow uses video mode
      if (
        importedSettings.inputMode === "video" &&
        settings.inputMode !== "video"
      ) {
        setShouldReinitializeVideo(true);
        setTimeout(
          () => setShouldReinitializeVideo(false),
          VIDEO_REINITIALIZE_DELAY_MS
        );
      }

      if (timelineRef.current) {
        timelineRef.current.loadPrompts(importedTimeline);
      }

      // Restore active prompt state
      if (promptState) {
        setPromptItems(promptState.promptItems);
        setInterpolationMethod(promptState.interpolationMethod);
        setTransitionSteps(promptState.transitionSteps);
        setTemporalInterpolationMethod(promptState.temporalInterpolationMethod);
      }
    },
    [updateSettings, skipNextModeReset, settings.inputMode]
  );

  const handleWorkflowLoadToGraph = useCallback((workflow: ScopeWorkflow) => {
    graphEditorRef.current?.loadWorkflow(workflow);
  }, []);

  return (
    <MIDIProvider
      sendParameterUpdate={sendParameterUpdate}
      currentDenoisingSteps={settings.denoisingSteps}
      onDenoisingStepsChange={handleDenoisingStepsChange}
      currentNoiseController={settings.noiseController}
      currentManageCache={settings.manageCache}
      onSwitchPrompt={handleMidiPromptSolo}
      onPromptWeightChange={handleMidiPromptWeightChange}
      onPlayPauseToggle={handlePlayPauseToggle}
    >
      <div className="h-screen flex flex-col bg-background">
        {/* Header */}
        <Header
          onPipelinesRefresh={handlePipelinesRefresh}
          cloudDisabled={isStreaming}
          openSettingsTab={openSettingsTab}
          onSettingsTabOpened={() => setOpenSettingsTab(null)}
          graphMode={graphMode}
          onGraphModeToggle={() => {
            if (!graphMode) {
              // Switching Perform → Graph: seed a linear graph if none exists in localStorage
              const currentGraph =
                graphEditorRef.current?.getCurrentGraphConfig();
              if (
                !currentGraph ||
                (currentGraph.nodes.length === 0 &&
                  currentGraph.edges.length === 0)
              ) {
                // No graph yet — create a linear one from current settings
                // and save to localStorage so the graph editor picks it up
                const graph = linearGraphFromSettings(
                  settings.pipelineId,
                  settings.preprocessorIds ?? [],
                  settings.postprocessorIds ?? []
                );
                try {
                  localStorage.setItem(
                    "scope:graph:backup",
                    JSON.stringify(graph)
                  );
                } catch {
                  /* ignore */
                }
              }
              // Refresh the graph editor so it picks up the current graph
              graphEditorRef.current?.refreshGraph();
            } else {
              // Switching Graph → Perform: sync pipeline ID and source mode
              // from the graph so perform mode reflects the workflow builder.
              try {
                const frontendGraph =
                  graphEditorRef.current?.getCurrentGraphConfig();
                const graphNodes = frontendGraph?.nodes ?? null;

                if (graphNodes) {
                  // Sync pipeline ID from the graph's first pipeline node
                  const firstPipeline = graphNodes.find(
                    n => n.type === "pipeline" && n.pipeline_id
                  );
                  if (firstPipeline?.pipeline_id) {
                    skipNextModeReset(firstPipeline.pipeline_id);
                    updateSettings({ pipelineId: firstPipeline.pipeline_id });
                  }

                  const sourceNode = graphNodes.find(n => n.type === "source");
                  // Default to "video" if source node has no explicit mode
                  const sourceMode = (sourceNode?.source_mode || "video") as
                    | "video"
                    | "camera"
                    | "spout"
                    | "ndi"
                    | "syphon";

                  // Sync inputMode setting so perform mode reflects the graph's choice
                  const inputMode: InputMode =
                    sourceMode === "video" || sourceMode === "camera"
                      ? "video"
                      : "video"; // server-side sources still use "video" inputMode

                  // Also sync resolution to match the new input mode so
                  // perform mode shows the correct video-mode defaults.
                  const pid = (firstPipeline?.pipeline_id ??
                    settings.pipelineId) as PipelineId;
                  const modeDefaults = getDefaults(pid, inputMode);
                  const resolution = customVideoResolution
                    ? capResolution(customVideoResolution, pid, inputMode)
                    : {
                        height: modeDefaults.height,
                        width: modeDefaults.width,
                      };
                  updateSettings({ inputMode, resolution });

                  // Sync to useVideoSource
                  if (
                    sourceMode === "spout" ||
                    sourceMode === "ndi" ||
                    sourceMode === "syphon"
                  ) {
                    // For server-side sources, update settings.inputSource with graph's source_name
                    updateSettings({
                      inputSource: {
                        enabled: true,
                        source_type: sourceMode,
                        source_name:
                          sourceNode?.source_name ??
                          settings.inputSource?.source_name ??
                          "",
                      },
                    });
                    switchMode(sourceMode);
                  } else {
                    switchMode(sourceMode);
                  }
                }
              } catch {
                /* ignore */
              }
            }
            // Graph → Perform: just switch mode (modes are independent)
            setGraphMode(prev => !prev);
          }}
        />

        {/* Graph Editor - always mounted so control/value node animations and
            value-forwarding effects keep running even in perform mode */}
        <div
          className={
            graphMode
              ? "flex-1 min-h-0 overflow-hidden"
              : "fixed inset-0 -z-50 invisible pointer-events-none"
          }
        >
          <GraphEditor
            ref={graphEditorRef}
            visible={graphMode}
            isStreaming={isStreaming}
            isConnecting={isConnecting || isCloudConnecting}
            isLoading={isPipelineLoading || isDownloading}
            onNodeParameterChange={(nodeId, key, value) => {
              sendParameterUpdate({ node_id: nodeId, [key]: value });
            }}
            onGraphChange={handleGraphChange}
            onGraphClear={handleGraphClear}
            localStream={localStream}
            localStreams={nodeLocalStreams}
            remoteStream={remoteStream}
            remoteStreams={remoteStreams}
            sinkStats={perSinkStats}
            onVideoFileUpload={handlePerNodeVideoFileUpload}
            onStartStream={() => handleStartStream()}
            onStopStream={stopStream}
            onSourceModeChange={handlePerNodeSourceModeChange}
            spoutAvailable={spoutAvailable}
            ndiAvailable={ndiAvailable}
            syphonAvailable={syphonAvailable}
            onSpoutSourceChange={handleSpoutSourceChange}
            onNdiSourceChange={handleNdiSourceChange}
            onSyphonSourceChange={handleSyphonSourceChange}
            onOutputSinkChange={handleOutputSinkChange}
            onOutputSinkBulkChange={handleOutputSinkBulkChange}
            spoutOutputAvailable={spoutAvailable}
            ndiOutputAvailable={ndiOutputAvailable}
            syphonOutputAvailable={syphonOutputAvailable}
            onStartRecording={handleStartRecording}
            onStopRecording={handleStopRecording}
          />
        </div>

        {/* Main Content Area - Perform Mode */}
        {!graphMode && (
          <div className="flex-1 flex gap-4 px-4 pb-4 min-h-0 overflow-hidden">
            {/* Left Panel - Input & Controls */}
            <div className="w-1/5 flex flex-col gap-3 min-h-0 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:transition-colors [&::-webkit-scrollbar-thumb:hover]:bg-gray-400">
              <InputAndControlsPanel
                className=""
                pipelines={pipelines}
                localStream={localStream}
                isInitializing={isInitializing}
                error={videoSourceError}
                mode={mode}
                onModeChange={handleModeChange}
                isStreaming={isStreaming}
                isConnecting={isConnecting || isCloudConnecting}
                isPipelineLoading={isPipelineLoading}
                canStartStream={
                  settings.inputMode === "text"
                    ? !isInitializing
                    : mode === "spout" || mode === "ndi" || mode === "syphon"
                      ? !isInitializing
                      : !!localStream && !isInitializing
                }
                onStartStream={handleStartStream}
                onStopStream={stopStream}
                onVideoFileUpload={handleVideoFileUpload}
                pipelineId={settings.pipelineId}
                prompts={promptItems}
                onPromptsChange={setPromptItems}
                onPromptsSubmit={handlePromptsSubmit}
                onTransitionSubmit={handleTransitionSubmit}
                interpolationMethod={interpolationMethod}
                onInterpolationMethodChange={setInterpolationMethod}
                temporalInterpolationMethod={temporalInterpolationMethod}
                onTemporalInterpolationMethodChange={
                  setTemporalInterpolationMethod
                }
                isLive={isLive}
                onLivePromptSubmit={handleLivePromptSubmit}
                selectedTimelinePrompt={selectedTimelinePrompt}
                onTimelinePromptUpdate={handleTimelinePromptUpdate}
                isVideoPaused={settings.paused}
                isTimelinePlaying={isTimelinePlaying}
                currentTime={timelineCurrentTime}
                timelinePrompts={timelinePrompts}
                transitionSteps={transitionSteps}
                onTransitionStepsChange={setTransitionSteps}
                spoutReceiverName={
                  settings.inputSource?.source_type === "spout"
                    ? (settings.inputSource?.source_name ?? "")
                    : ""
                }
                onSpoutReceiverChange={handleSpoutSourceChange}
                inputMode={
                  settings.inputMode ||
                  getPipelineDefaultMode(settings.pipelineId)
                }
                onInputModeChange={handleInputModeChange}
                spoutAvailable={spoutAvailable}
                ndiAvailable={ndiAvailable}
                syphonAvailable={syphonAvailable}
                selectedNdiSource={settings.inputSource?.source_name ?? ""}
                onNdiSourceChange={handleNdiSourceChange}
                selectedSyphonSource={
                  settings.inputSource?.source_type === "syphon"
                    ? (settings.inputSource?.source_name ?? "")
                    : ""
                }
                onSyphonSourceChange={handleSyphonSourceChange}
                vaceEnabled={
                  settings.vaceEnabled ??
                  (pipelines?.[settings.pipelineId]?.supportsVACE &&
                    settings.inputMode !== "video")
                }
                refImages={settings.refImages || []}
                onRefImagesChange={handleRefImagesChange}
                onSendHints={handleSendHints}
                isDownloading={isDownloading}
                supportsImages={
                  pipelines?.[settings.pipelineId]?.supportsImages
                }
                firstFrameImage={settings.firstFrameImage}
                onFirstFrameImageChange={handleFirstFrameImageChange}
                lastFrameImage={settings.lastFrameImage}
                onLastFrameImageChange={handleLastFrameImageChange}
                extensionMode={settings.extensionMode || "firstframe"}
                onExtensionModeChange={handleExtensionModeChange}
                onSendExtensionFrames={handleSendExtensionFrames}
                configSchema={
                  pipelines?.[settings.pipelineId]?.configSchema as
                    | import("../lib/schemaSettings").ConfigSchemaLike
                    | undefined
                }
                schemaFieldOverrides={settings.schemaFieldOverrides ?? {}}
                onSchemaFieldOverrideChange={(
                  key: string,
                  value: unknown,
                  isRuntimeParam?: boolean
                ) => {
                  updateSettings({
                    schemaFieldOverrides: {
                      ...(settings.schemaFieldOverrides ?? {}),
                      [key]: value,
                    },
                  });
                  if (isRuntimeParam && isStreaming) {
                    sendParameterUpdate({ [key]: value });
                  }
                }}
              />
              <Card>
                <CardHeader className="px-4 py-3">
                  <CardTitle className="text-base font-medium">
                    Tempo Sync
                  </CardTitle>
                </CardHeader>
                <CardContent className="px-4 pb-4 pt-0">
                  <TempoSyncSection
                    tempoState={tempoState}
                    sources={tempoSources ?? null}
                    loading={tempoLoading}
                    error={tempoError}
                    onEnable={enableTempoSync}
                    onDisable={disableTempoSync}
                    onSetBpm={setTempoSessionBpm}
                    onRefreshSources={refreshTempoSources}
                    quantizeMode={settings.quantizeMode || "none"}
                    onQuantizeModeChange={mode =>
                      updateSettings({
                        quantizeMode: mode as SettingsState["quantizeMode"],
                      })
                    }
                    lookaheadMs={settings.lookaheadMs ?? 0}
                    onLookaheadMsChange={ms =>
                      updateSettings({ lookaheadMs: ms })
                    }
                    modulations={settings.modulations}
                    onModulationsChange={modulations =>
                      updateSettings({ modulations })
                    }
                    configSchema={
                      pipelines?.[settings.pipelineId]?.configSchema
                    }
                    beatCacheResetRate={settings.beatCacheResetRate || "none"}
                    onBeatCacheResetRateChange={rate =>
                      updateSettings({
                        beatCacheResetRate:
                          rate as SettingsState["beatCacheResetRate"],
                      })
                    }
                    promptCycleRate={settings.promptCycleRate || "none"}
                    onPromptCycleRateChange={rate =>
                      updateSettings({
                        promptCycleRate:
                          rate as SettingsState["promptCycleRate"],
                      })
                    }
                  />
                </CardContent>
              </Card>
              {hasAvailableOutputs && (
                <OutputsPanel
                  className=""
                  outputSinks={settings.outputSinks}
                  onOutputSinkChange={handleOutputSinkChange}
                  spoutAvailable={spoutAvailable}
                  ndiAvailable={ndiOutputAvailable}
                  syphonAvailable={syphonOutputAvailable}
                  isStreaming={isStreaming}
                />
              )}
            </div>

            {/* Center Panel - Video Output + Timeline */}
            <div className="flex-1 flex flex-col min-h-0">
              {/* Video area - takes remaining space but can shrink */}
              <div className="flex-1 min-h-0">
                <VideoOutput
                  className="h-full"
                  remoteStream={remoteStream}
                  isPipelineLoading={isPipelineLoading}
                  isCloudConnecting={isCloudConnecting}
                  isConnecting={isConnecting}
                  pipelineError={pipelineError}
                  pipelineLoadingStage={pipelineLoadingStage}
                  cloudConnectStage={cloudConnectStage}
                  isPlaying={!settings.paused}
                  isDownloading={isDownloading}
                  onPlayPauseToggle={() => {
                    // Use timeline's play/pause handler instead of direct video toggle
                    if (timelinePlayPauseRef.current) {
                      timelinePlayPauseRef.current();
                    }
                  }}
                  onStartStream={() => {
                    // Use timeline's play/pause handler to start stream
                    if (timelinePlayPauseRef.current) {
                      timelinePlayPauseRef.current();
                    }
                  }}
                  onVideoPlaying={() => {
                    // Execute callback when video starts playing
                    if (onVideoPlayingCallbackRef.current) {
                      onVideoPlayingCallbackRef.current();
                      onVideoPlayingCallbackRef.current = null; // Clear after execution
                    }
                  }}
                  // Controller input props
                  supportsControllerInput={currentPipelineSupportsController}
                  isPointerLocked={isPointerLocked}
                  onRequestPointerLock={requestPointerLock}
                  videoContainerRef={videoContainerRef}
                  // Video scale mode
                  videoScaleMode={videoScaleMode}
                />
              </div>
              {/* Timeline area - compact, always visible */}
              <div className="flex-shrink-0 mt-2">
                <PromptInputWithTimeline
                  currentPrompt={promptItems[0]?.text || ""}
                  currentPromptItems={promptItems}
                  transitionSteps={transitionSteps}
                  temporalInterpolationMethod={temporalInterpolationMethod}
                  onPromptSubmit={text => {
                    // Update the left panel's prompt state to reflect current timeline prompt
                    const prompts = [{ text, weight: 100 }];
                    setPromptItems(prompts);

                    // Send to backend - use transition if streaming and transition steps > 0
                    if (isStreaming && transitionSteps > 0) {
                      sendParameterUpdate({
                        transition: {
                          target_prompts: prompts,
                          num_steps: transitionSteps,
                          temporal_interpolation_method:
                            temporalInterpolationMethod,
                        },
                      });
                    } else {
                      // Send direct prompts without transition
                      sendParameterUpdate({
                        prompts,
                        prompt_interpolation_method: interpolationMethod,
                        denoising_step_list: settings.denoisingSteps || [
                          700, 500,
                        ],
                      });
                    }
                  }}
                  onPromptItemsSubmit={(
                    prompts,
                    blockTransitionSteps,
                    blockTemporalInterpolationMethod
                  ) => {
                    // Update the left panel's prompt state to reflect current timeline prompt blend
                    setPromptItems(prompts);

                    // Use transition params from block if provided, otherwise use global settings
                    const effectiveTransitionSteps =
                      blockTransitionSteps ?? transitionSteps;
                    const effectiveTemporalInterpolationMethod =
                      blockTemporalInterpolationMethod ??
                      temporalInterpolationMethod;

                    // Update the left panel's transition settings to reflect current block's values
                    if (blockTransitionSteps !== undefined) {
                      setTransitionSteps(blockTransitionSteps);
                    }
                    if (blockTemporalInterpolationMethod !== undefined) {
                      setTemporalInterpolationMethod(
                        blockTemporalInterpolationMethod
                      );
                    }

                    // Send to backend - use transition if streaming and transition steps > 0
                    if (isStreaming && effectiveTransitionSteps > 0) {
                      sendParameterUpdate({
                        transition: {
                          target_prompts: prompts,
                          num_steps: effectiveTransitionSteps,
                          temporal_interpolation_method:
                            effectiveTemporalInterpolationMethod,
                        },
                      });
                    } else {
                      // Send direct prompts without transition
                      sendParameterUpdate({
                        prompts,
                        prompt_interpolation_method: interpolationMethod,
                        denoising_step_list: settings.denoisingSteps || [
                          700, 500,
                        ],
                      });
                    }
                  }}
                  disabled={
                    isPipelineLoading ||
                    isConnecting ||
                    isCloudConnecting ||
                    showDownloadDialog
                  }
                  isStreaming={isStreaming}
                  isVideoPaused={settings.paused}
                  timelineRef={timelineRef}
                  onLiveStateChange={setIsLive}
                  onLivePromptSubmit={handleLivePromptSubmit}
                  onDisconnect={stopStream}
                  onStartStream={handleStartStream}
                  onVideoPlayPauseToggle={handlePlayPauseToggle}
                  onPromptEdit={handleTimelinePromptEdit}
                  isCollapsed={isTimelineCollapsed}
                  onCollapseToggle={setIsTimelineCollapsed}
                  externalSelectedPromptId={externalSelectedPromptId}
                  onPlayPauseRef={timelinePlayPauseRef}
                  onVideoPlayingCallbackRef={onVideoPlayingCallbackRef}
                  onTimelinePromptsChange={handleTimelinePromptsChange}
                  onTimelineCurrentTimeChange={handleTimelineCurrentTimeChange}
                  onTimelinePlayingChange={handleTimelinePlayingChange}
                  isLoading={isLoading}
                  videoScaleMode={videoScaleMode}
                  onVideoScaleModeToggle={() =>
                    setVideoScaleMode(prev =>
                      prev === "fit" ? "native" : "fit"
                    )
                  }
                  isDownloading={isDownloading}
                  onSaveGeneration={handleSaveGeneration}
                  isRecording={isRecording}
                  onRecordingToggle={() => setIsRecording(prev => !prev)}
                  onWorkflowExport={() => setShowWorkflowExport(true)}
                  onWorkflowImport={() => setShowWorkflowImport(true)}
                  onExportToDaydream={handleExportToDaydream}
                  isAuthenticated={isDaydreamAuthenticated}
                  isExportingToDaydream={isExportingToDaydream}
                />
              </div>
            </div>

            {/* Right Panel - Settings */}
            <div className="w-1/5 flex flex-col gap-3 min-h-0">
              <SettingsPanel
                className="flex-1 min-h-0"
                pipelines={pipelines}
                pipelineId={settings.pipelineId}
                onPipelineIdChange={handlePipelineIdChange}
                isStreaming={isStreaming}
                isLoading={isLoading}
                resolution={
                  settings.resolution || {
                    height: getDefaults(settings.pipelineId, settings.inputMode)
                      .height,
                    width: getDefaults(settings.pipelineId, settings.inputMode)
                      .width,
                  }
                }
                onResolutionChange={handleResolutionChange}
                denoisingSteps={
                  settings.denoisingSteps ||
                  getDefaults(settings.pipelineId, settings.inputMode)
                    .denoisingSteps || [750, 250]
                }
                onDenoisingStepsChange={handleDenoisingStepsChange}
                defaultDenoisingSteps={
                  getDefaults(settings.pipelineId, settings.inputMode)
                    .denoisingSteps || [750, 250]
                }
                noiseScale={settings.noiseScale ?? 0.7}
                onNoiseScaleChange={handleNoiseScaleChange}
                noiseController={settings.noiseController ?? true}
                onNoiseControllerChange={handleNoiseControllerChange}
                manageCache={settings.manageCache ?? true}
                onManageCacheChange={handleManageCacheChange}
                quantization={
                  settings.quantization !== undefined
                    ? settings.quantization
                    : "fp8_e4m3fn"
                }
                onQuantizationChange={handleQuantizationChange}
                kvCacheAttentionBias={settings.kvCacheAttentionBias ?? 0.3}
                onKvCacheAttentionBiasChange={handleKvCacheAttentionBiasChange}
                onResetCache={handleResetCache}
                loras={settings.loras || []}
                onLorasChange={handleLorasChange}
                loraMergeStrategy={
                  settings.loraMergeStrategy ?? "permanent_merge"
                }
                inputMode={settings.inputMode}
                supportsNoiseControls={supportsNoiseControls(
                  settings.pipelineId
                )}
                vaceEnabled={
                  settings.vaceEnabled ??
                  (pipelines?.[settings.pipelineId]?.supportsVACE &&
                    settings.inputMode !== "video")
                }
                onVaceEnabledChange={handleVaceEnabledChange}
                vaceUseInputVideo={settings.vaceUseInputVideo ?? false}
                onVaceUseInputVideoChange={handleVaceUseInputVideoChange}
                vaceContextScale={settings.vaceContextScale ?? 1.0}
                onVaceContextScaleChange={handleVaceContextScaleChange}
                preprocessorIds={settings.preprocessorIds ?? []}
                onPreprocessorIdsChange={handlePreprocessorIdsChange}
                postprocessorIds={settings.postprocessorIds ?? []}
                onPostprocessorIdsChange={handlePostprocessorIdsChange}
                preprocessorSchemaFieldOverrides={
                  settings.preprocessorSchemaFieldOverrides ?? {}
                }
                postprocessorSchemaFieldOverrides={
                  settings.postprocessorSchemaFieldOverrides ?? {}
                }
                onPreprocessorSchemaFieldOverrideChange={
                  handlePreprocessorSchemaFieldOverrideChange
                }
                onPostprocessorSchemaFieldOverrideChange={
                  handlePostprocessorSchemaFieldOverrideChange
                }
                schemaFieldOverrides={settings.schemaFieldOverrides ?? {}}
                onSchemaFieldOverrideChange={(key, value, isRuntimeParam) => {
                  updateSettings({
                    schemaFieldOverrides: {
                      ...(settings.schemaFieldOverrides ?? {}),
                      [key]: value,
                    },
                  });
                  if (isRuntimeParam && isStreaming) {
                    sendParameterUpdate({ [key]: value });
                  }
                }}
                isCloudMode={isCloudMode}
                nonLinearGraph={nonLinearGraph}
                onClearGraph={handleClearGraphFromSettings}
                onOpenLoRAsSettings={() => setOpenSettingsTab("loras")}
              />
            </div>
          </div>
        )}

        {/* Log Panel */}
        <LogPanel
          logs={logs}
          isOpen={isLogPanelOpen}
          onClose={toggleLogPanel}
          onClear={clearLogs}
        />

        {/* Status Bar */}
        <StatusBar
          fps={
            Object.values(perSinkStats).length > 0
              ? Object.values(perSinkStats)[0].fps
              : 0
          }
          bitrate={
            Object.values(perSinkStats).length > 0
              ? Object.values(perSinkStats).reduce((s, v) => s + v.bitrate, 0)
              : 0
          }
          onLogToggle={toggleLogPanel}
          isLogOpen={isLogPanelOpen}
          logUnreadCount={logUnreadCount}
        />

        {/* Download Dialog */}
        {pipelinesNeedingModels.length > 0 && (
          <DownloadDialog
            open={showDownloadDialog}
            pipelines={pipelines}
            pipelineIds={pipelinesNeedingModels}
            currentDownloadPipeline={currentDownloadPipeline}
            onClose={handleDialogClose}
            onDownload={handleDownloadModels}
            isDownloading={isDownloading}
            progress={downloadProgress}
            error={downloadError}
            onOpenSettings={tab => {
              setShowDownloadDialog(false);
              setOpenSettingsTab(tab);
            }}
          />
        )}

        {/* Workflow Export Dialog */}
        <WorkflowExportDialog
          open={showWorkflowExport}
          onClose={() => setShowWorkflowExport(false)}
          settings={settings}
          timelinePrompts={timelinePrompts}
          promptState={{
            promptItems,
            interpolationMethod,
            transitionSteps,
            temporalInterpolationMethod,
          }}
        />

        {/* Workflow Import Dialog */}
        <WorkflowImportDialog
          open={showWorkflowImport}
          onClose={() => {
            setShowWorkflowImport(false);
            setPreloadedWorkflow(null);
          }}
          onLoad={handleWorkflowLoad}
          onLoadToGraph={graphMode ? handleWorkflowLoadToGraph : undefined}
          initialWorkflow={preloadedWorkflow}
        />
      </div>
    </MIDIProvider>
  );
}

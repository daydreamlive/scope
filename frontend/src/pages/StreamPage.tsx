import { useState } from "react";
import { Header } from "../components/Header";
import { InputAndControlsPanel } from "../components/InputAndControlsPanel";
import { VideoOutput } from "../components/VideoOutput";
import { SettingsPanel } from "../components/SettingsPanel";
import { PromptInput } from "../components/PromptInput";
import { StatusBar } from "../components/StatusBar";
import { useWebRTC } from "../hooks/useWebRTC";
import { useVideoSource } from "../hooks/useVideoSource";
import { useWebRTCStats } from "../hooks/useWebRTCStats";
import { usePipeline } from "../hooks/usePipeline";
import { useStreamState } from "../hooks/useStreamState";
import { PIPELINES } from "../data/pipelines";
import { getDefaultDenoisingSteps } from "../lib/utils";
import type { PipelineId } from "../types";

export function StreamPage() {
  // Use the stream state hook for settings management
  const { settings, updateSettings } = useStreamState();

  // Track current parameter state
  const [currentPrompts, setCurrentPrompts] = useState<string[]>([
    PIPELINES[settings.pipelineId]?.defaultPrompt || "",
  ]);

  // Track when we need to reinitialize video source
  const [shouldReinitializeVideo, setShouldReinitializeVideo] = useState(false);

  // Pipeline management
  const {
    isLoading: isPipelineLoading,
    error: pipelineError,
    loadPipeline,
  } = usePipeline();

  // WebRTC for streaming
  const {
    remoteStream,
    isStreaming,
    isConnecting,
    peerConnectionRef,
    startStream,
    stopStream,
    updateVideoTrack,
    sendParameterUpdate,
  } = useWebRTC();

  // Get WebRTC stats for FPS
  const webrtcStats = useWebRTCStats({
    peerConnectionRef,
    isStreaming,
  });

  // Video source for preview (camera or video)
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
    enabled: PIPELINES[settings.pipelineId]?.category === "video-input",
  });

  const handlePromptChange = (prompt: string) => {
    setCurrentPrompts([prompt]);
  };

  const handlePromptSubmit = (prompt: string) => {
    const prompts = [prompt];
    setCurrentPrompts(prompts);
    sendParameterUpdate({
      prompts,
      denoising_step_list: settings.denoisingSteps || [700, 500],
    });
  };

  const handlePipelineIdChange = (pipelineId: PipelineId) => {
    // Stop the stream if it's currently running
    if (isStreaming) {
      stopStream();
    }

    // Check if we're switching from no-video-input to video-input pipeline
    const currentPipelineCategory = PIPELINES[settings.pipelineId]?.category;
    const newPipelineCategory = PIPELINES[pipelineId]?.category;

    if (
      currentPipelineCategory === "no-video-input" &&
      newPipelineCategory === "video-input"
    ) {
      // Trigger video source reinitialization
      // Otherwise the camera or video file is not visible while switching the pipeline types
      setShouldReinitializeVideo(true);
      // Reset the flag after a short delay to allow the effect to trigger
      setTimeout(() => setShouldReinitializeVideo(false), 100);
    }

    // Update the prompt to the new pipeline's default
    const newDefaultPrompt = PIPELINES[pipelineId]?.defaultPrompt || "";
    setCurrentPrompts([newDefaultPrompt]);

    // Update denoising steps based on pipeline
    const newDenoisingSteps = getDefaultDenoisingSteps(pipelineId);

    // Update the pipeline in settings
    updateSettings({ pipelineId, denoisingSteps: newDenoisingSteps });
  };

  const handleResolutionChange = (resolution: {
    height: number;
    width: number;
  }) => {
    updateSettings({ resolution });
  };

  const handleSeedChange = (seed: number) => {
    updateSettings({ seed });
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

  const handleStartStream = async () => {
    if (isStreaming) {
      stopStream();
      return;
    }

    try {
      // Always load pipeline with current parameters - backend will handle the rest
      console.log(`Loading ${settings.pipelineId} pipeline...`);

      // Prepare load parameters based on pipeline type and video resolution
      let loadParams = null;
      if (settings.pipelineId === "streamdiffusionv2" && videoResolution) {
        loadParams = {
          height: videoResolution.height,
          width: videoResolution.width,
          seed: settings.seed ?? 42,
        };
        console.log(
          `Loading with resolution: ${videoResolution.width}x${videoResolution.height}, seed: ${loadParams.seed}`
        );
      } else if (settings.pipelineId === "passthrough" && videoResolution) {
        loadParams = {
          height: videoResolution.height,
          width: videoResolution.width,
        };
        console.log(
          `Loading with resolution: ${videoResolution.width}x${videoResolution.height}`
        );
      } else if (settings.pipelineId === "longlive") {
        loadParams = {
          height: settings.resolution?.height ?? 320,
          width: settings.resolution?.width ?? 576,
          seed: settings.seed ?? 42,
        };
        console.log(
          `Loading with resolution: ${loadParams.width}x${loadParams.height}, seed: ${loadParams.seed}`
        );
      }

      const loadSuccess = await loadPipeline(
        settings.pipelineId,
        loadParams || undefined
      );
      if (!loadSuccess) {
        console.error("Failed to load pipeline, cannot start stream");
        return;
      }

      // Check if this pipeline needs video input
      const pipelineCategory = PIPELINES[settings.pipelineId]?.category;
      const needsVideoInput = pipelineCategory === "video-input";

      // Only send video stream for pipelines that need video input
      const streamToSend = needsVideoInput
        ? localStream || undefined
        : undefined;

      if (needsVideoInput && !localStream) {
        console.error("Video input required but no local stream available");
        return;
      }

      // Pipeline is loaded, now start WebRTC stream
      startStream(
        {
          prompts: currentPrompts,
          denoising_step_list: settings.denoisingSteps || [700, 500],
          noise_scale: settings.noiseScale ?? 0.7,
          noise_controller: settings.noiseController ?? true,
        },
        streamToSend
      );
    } catch (error) {
      console.error("Error during stream start:", error);
    }
  };

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Header */}
      <Header />

      {/* Main Content Area */}
      <div className="flex-1 flex gap-4 px-4 pb-4 pt-2 min-h-0 overflow-hidden">
        {/* Left Panel - Input & Controls */}
        <div className="w-1/5">
          <InputAndControlsPanel
            className="h-full"
            localStream={localStream}
            isInitializing={isInitializing}
            error={videoSourceError}
            mode={mode}
            onModeChange={switchMode}
            isStreaming={isStreaming}
            isConnecting={isConnecting}
            isPipelineLoading={isPipelineLoading}
            canStartStream={
              PIPELINES[settings.pipelineId]?.category === "no-video-input"
                ? !isInitializing
                : !!localStream && !isInitializing
            }
            onStartStream={handleStartStream}
            onStopStream={stopStream}
            onVideoFileUpload={handleVideoFileUpload}
            pipelineId={settings.pipelineId}
          />
        </div>

        {/* Center Panel - Video Output + Prompt */}
        <div className="flex-1 flex flex-col">
          <div className="flex-1">
            <VideoOutput
              className="h-full"
              remoteStream={remoteStream}
              isPipelineLoading={isPipelineLoading}
              isConnecting={isConnecting}
              pipelineError={pipelineError}
            />
          </div>
          <div className="mx-24 mt-4">
            <PromptInput
              currentPrompt={currentPrompts[0] || ""}
              onPromptChange={handlePromptChange}
              onPromptSubmit={handlePromptSubmit}
              disabled={
                settings.pipelineId === "passthrough" ||
                settings.pipelineId === "vod"
              }
            />
          </div>
        </div>

        {/* Right Panel - Settings */}
        <div className="w-1/5">
          <SettingsPanel
            className="h-full"
            pipelineId={settings.pipelineId}
            onPipelineIdChange={handlePipelineIdChange}
            isStreaming={isStreaming}
            resolution={settings.resolution || { height: 320, width: 576 }}
            onResolutionChange={handleResolutionChange}
            seed={settings.seed ?? 42}
            onSeedChange={handleSeedChange}
            denoisingSteps={settings.denoisingSteps || [700, 500]}
            onDenoisingStepsChange={handleDenoisingStepsChange}
            noiseScale={settings.noiseScale ?? 0.7}
            onNoiseScaleChange={handleNoiseScaleChange}
            noiseController={settings.noiseController ?? true}
            onNoiseControllerChange={handleNoiseControllerChange}
          />
        </div>
      </div>

      {/* Status Bar */}
      <StatusBar fps={webrtcStats.fps} bitrate={webrtcStats.bitrate} />
    </div>
  );
}

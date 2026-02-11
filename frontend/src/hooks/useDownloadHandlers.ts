/**
 * Download orchestration handlers.
 *
 * Manages sequential model downloads with progress polling,
 * and auto-starts the stream after all downloads complete.
 */

import { useCallback, useRef } from "react";
import { useAppStore } from "../stores";
import { usePipelinesContext } from "../contexts/PipelinesContext";
import { useStreamState } from "./useStreamState";
import { useApi } from "./useApi";
import { toast } from "sonner";
import type { PipelineId } from "../types";
import type { PromptItem } from "../lib/api";
import type { TimelinePrompt } from "../components/PromptTimeline";

interface UseDownloadHandlersParams {
  handleStartStreamRef: React.RefObject<
    ((overridePipelineId?: PipelineId) => Promise<boolean>) | null
  >;
  timelinePlayPauseRef: React.RefObject<(() => Promise<void>) | null>;
  timelineRef: React.RefObject<{
    getCurrentTimelinePrompt: () => string;
    submitLivePrompt: (prompts: PromptItem[]) => void;
    updatePrompt: (prompt: TimelinePrompt) => void;
    clearTimeline: () => void;
    resetPlayhead: () => void;
    resetTimelineCompletely: () => void;
    getPrompts: () => TimelinePrompt[];
    getCurrentTime: () => number;
    getIsPlaying: () => boolean;
  } | null>;
}

export function useDownloadHandlers(params: UseDownloadHandlersParams) {
  const { handleStartStreamRef, timelinePlayPauseRef, timelineRef } = params;

  const api = useApi();
  const { pipelines } = usePipelinesContext();
  const { getDefaults } = useStreamState();

  const store = useAppStore;

  const downloadPipelineSequentiallyRef = useRef<
    ((pipelineId: string, remainingPipelines: string[]) => Promise<void>) | null
  >(null);

  downloadPipelineSequentiallyRef.current = async (
    pipelineId: string,
    remainingPipelines: string[]
  ) => {
    store.getState().setCurrentDownloadPipeline(pipelineId);
    store.getState().setDownloadProgress(null);

    try {
      await api.downloadPipelineModels(pipelineId);

      const checkDownloadProgress = async () => {
        try {
          const status = await api.checkModelStatus(pipelineId);

          if (status.progress) {
            store.getState().setDownloadProgress(status.progress);
          }

          if (status.progress?.error) {
            const errorMessage = status.progress.error;
            console.error("Download failed:", errorMessage);
            toast.error(errorMessage);
            store.getState().setIsDownloading(false);
            store.getState().setDownloadProgress(null);
            store.getState().setDownloadError(errorMessage);
            store.getState().setCurrentDownloadPipeline(null);
            return;
          }

          if (status.downloaded) {
            const newRemaining = remainingPipelines;
            store.getState().setPipelinesNeedingModels(newRemaining);

            const { settings } = store.getState();
            const pipelineInfoLocal = pipelines?.[pipelineId];
            const isPreprocessor =
              pipelineInfoLocal?.usage?.includes("preprocessor") ?? false;

            if (!isPreprocessor && pipelineId === settings.pipelineId) {
              if (timelineRef.current) {
                timelineRef.current.resetTimelineCompletely();
              }
              store.getState().setSelectedTimelinePrompt(null);
              store.getState().setExternalSelectedPromptId(null);

              const newPipeline = pipelines?.[pipelineId];
              const currentMode =
                settings.inputMode || newPipeline?.defaultMode || "text";
              const defaults = getDefaults(
                pipelineId as PipelineId,
                currentMode
              );
              const { customVideoResolution } = store.getState();

              const resolution =
                currentMode === "video" && customVideoResolution
                  ? customVideoResolution
                  : { height: defaults.height, width: defaults.width };

              store.getState().updateSettings({
                pipelineId: pipelineId as PipelineId,
                inputMode: currentMode,
                denoisingSteps: defaults.denoisingSteps,
                resolution,
                noiseScale: defaults.noiseScale,
                noiseController: defaults.noiseController,
              });
            }

            if (newRemaining.length > 0) {
              setTimeout(() => {
                downloadPipelineSequentiallyRef.current?.(
                  newRemaining[0],
                  newRemaining.slice(1)
                );
              }, 1000);
            } else {
              store.getState().setIsDownloading(false);
              store.getState().setDownloadProgress(null);
              store.getState().setShowDownloadDialog(false);
              store.getState().setCurrentDownloadPipeline(null);

              setTimeout(async () => {
                const started = await handleStartStreamRef.current?.();
                if (started && timelinePlayPauseRef.current) {
                  setTimeout(() => {
                    timelinePlayPauseRef.current?.();
                  }, 2000);
                }
              }, 100);
            }
          } else {
            setTimeout(checkDownloadProgress, 2000);
          }
        } catch (error) {
          console.error("Error checking download status:", error);
          store.getState().setIsDownloading(false);
          store.getState().setDownloadProgress(null);
          store.getState().setShowDownloadDialog(false);
          store.getState().setCurrentDownloadPipeline(null);
        }
      };

      setTimeout(checkDownloadProgress, 5000);
    } catch (error) {
      console.error("Error downloading models:", error);
      store.getState().setIsDownloading(false);
      store.getState().setDownloadProgress(null);
      store.getState().setShowDownloadDialog(false);
      store.getState().setCurrentDownloadPipeline(null);
    }
  };

  const handleDownloadModels = useCallback(async () => {
    const { pipelinesNeedingModels } = store.getState();
    if (pipelinesNeedingModels.length === 0) return;

    store.getState().setIsDownloading(true);
    store.getState().setDownloadError(null);
    store.getState().setShowDownloadDialog(true);

    const firstPipeline = pipelinesNeedingModels[0];
    const remaining = pipelinesNeedingModels.slice(1);
    await downloadPipelineSequentiallyRef.current?.(firstPipeline, remaining);
  }, []);

  const handleDialogClose = useCallback(() => {
    store.getState().setShowDownloadDialog(false);
    store.getState().setPipelinesNeedingModels([]);
    store.getState().setCurrentDownloadPipeline(null);
    store.getState().setDownloadError(null);
  }, []);

  return {
    handleDownloadModels,
    handleDialogClose,
  };
}

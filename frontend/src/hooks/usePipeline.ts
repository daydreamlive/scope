import { useState, useEffect, useCallback, useRef } from "react";
import { useApi } from "./useApi";
import type { PipelineStatusResponse, PipelineLoadParams } from "../lib/api";
import { toast } from "sonner";

interface UsePipelineOptions {
  pollInterval?: number; // milliseconds
  maxTimeout?: number; // milliseconds
}

export function usePipeline(options: UsePipelineOptions = {}) {
  const { pollInterval = 2000, maxTimeout = 600000 } = options;

  // Use the unified API hook - handles cloud/local routing automatically
  const { getPipelineStatus, loadPipeline: loadPipelineRequest } = useApi();

  const [status, setStatus] =
    useState<PipelineStatusResponse["status"]>("not_loaded");
  const [pipelineInfo, setPipelineInfo] =
    useState<PipelineStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimeoutRef = useRef<number | null>(null);
  const loadTimeoutRef = useRef<number | null>(null);
  const isPollingRef = useRef(false);
  const shownErrorRef = useRef<string | null>(null); // Track which error we've shown

  // Check initial pipeline status
  const checkStatus = useCallback(async () => {
    try {
      const statusResponse = await getPipelineStatus();
      setStatus(statusResponse.status);
      setPipelineInfo(statusResponse);

      if (statusResponse.status === "error") {
        const errorMessage = statusResponse.error || "Unknown pipeline error";
        // Show toast if we haven't shown this error yet
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 8000,
          });
          shownErrorRef.current = errorMessage;
        }
        // Don't set error in state - it's shown as toast and cleared on backend
        setError(null);
      } else {
        setError(null);
        shownErrorRef.current = null; // Reset when status is not error
      }
    } catch (err) {
      console.error("Failed to get pipeline status:", err);
      const errorMessage =
        err instanceof Error ? err.message : "Failed to get pipeline status";
      // Show toast for API errors
      if (shownErrorRef.current !== errorMessage) {
        toast.error("Pipeline Error", {
          description: errorMessage,
          duration: 5000,
        });
        shownErrorRef.current = errorMessage;
      }
      setError(null); // Don't persist in state
    }
  }, []);

  // Stop polling
  const stopPolling = useCallback(() => {
    isPollingRef.current = false;
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  // Load pipeline
  const triggerLoad = useCallback(
    async (
      pipelineIds?: string[],
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      if (isLoading) {
        console.log("Pipeline already loading");
        return false;
      }

      if (!pipelineIds || pipelineIds.length === 0) {
        console.error("No pipeline IDs provided");
        return false;
      }

      try {
        setIsLoading(true);
        setError(null);
        shownErrorRef.current = null; // Reset error tracking when starting new load

        // Start the load request
        await loadPipelineRequest({
          pipeline_ids: pipelineIds,
          load_params: loadParams,
        });

        // Extend timeout when torch.compile is enabled (first-time compilation is slow)
        const effectiveTimeout = loadParams?.compile ? maxTimeout * 3 : maxTimeout;

        // Set up timeout for the load operation
        const timeoutPromise = new Promise<boolean>((_, reject) => {
          loadTimeoutRef.current = setTimeout(() => {
            reject(
              new Error(
                `Pipeline load timeout after ${effectiveTimeout / 1000} seconds`
              )
            );
          }, effectiveTimeout);
        });

        // Wait for pipeline to be loaded or error
        const loadPromise = new Promise<boolean>((resolve, reject) => {
          const checkComplete = async () => {
            try {
              const currentStatus = await getPipelineStatus();
              // Keep hook state synchronized while polling so callers can
              // reliably read fields like loaded_lora_adapters after load.
              setStatus(currentStatus.status);
              setPipelineInfo(currentStatus);
              if (currentStatus.status === "loaded") {
                resolve(true);
              } else if (currentStatus.status === "error") {
                const errorMsg = currentStatus.error || "Pipeline load failed";
                // Show toast for load completion errors
                if (shownErrorRef.current !== errorMsg) {
                  toast.error("Pipeline Error", {
                    description: errorMsg,
                    duration: 8000,
                  });
                  shownErrorRef.current = errorMsg;
                }
                reject(new Error(errorMsg));
              } else {
                // Continue polling
                setTimeout(checkComplete, pollInterval);
              }
            } catch (err) {
              reject(err);
            }
          };
          checkComplete();
        });

        // Race between load completion and timeout
        const result = await Promise.race([loadPromise, timeoutPromise]);

        // Clear timeout if load completed
        if (loadTimeoutRef.current) {
          clearTimeout(loadTimeoutRef.current);
          loadTimeoutRef.current = null;
        }

        stopPolling();
        return result;
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load pipeline";
        console.error("Pipeline load error:", errorMessage);
        // Show toast for load errors
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 8000,
          });
          shownErrorRef.current = errorMessage;
        }
        setError(null); // Don't persist in state

        stopPolling();

        // Clear timeout on error
        if (loadTimeoutRef.current) {
          clearTimeout(loadTimeoutRef.current);
          loadTimeoutRef.current = null;
        }

        return false;
      } finally {
        setIsLoading(false);
      }
    },
    [
      isLoading,
      maxTimeout,
      pollInterval,
      stopPolling,
      getPipelineStatus,
      loadPipelineRequest,
    ]
  );

  // Load pipeline with proper state management
  const loadPipelineAsync = useCallback(
    async (
      pipelineIds?: string[],
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      // Always trigger load - let the backend decide if reload is needed
      return await triggerLoad(pipelineIds, loadParams);
    },
    [triggerLoad]
  );

  // Initial status check on mount
  useEffect(() => {
    checkStatus();
  }, [checkStatus]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
      if (loadTimeoutRef.current) {
        clearTimeout(loadTimeoutRef.current);
      }
    };
  }, [stopPolling]);

  return {
    status,
    pipelineInfo,
    isLoading,
    error,
    loadPipeline: loadPipelineAsync,
    checkStatus,
    isLoaded: status === "loaded",
    isError: status === "error",
    loadingStage: pipelineInfo?.loading_stage ?? null,
  };
}

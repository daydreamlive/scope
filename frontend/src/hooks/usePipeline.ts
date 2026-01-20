import { useState, useEffect, useCallback, useRef } from "react";
import { loadPipeline, getPipelineStatus } from "../lib/api";
import type { PipelineStatusResponse, PipelineLoadParams } from "../lib/api";
import { toast } from "sonner";

interface UsePipelineOptions {
  pollInterval?: number; // milliseconds
  maxTimeout?: number; // milliseconds
}

export function usePipeline(options: UsePipelineOptions = {}) {
  const { pollInterval = 2000, maxTimeout = 600000 } = options;

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

  // Start polling for status updates
  const startPolling = useCallback(() => {
    if (isPollingRef.current) return;

    isPollingRef.current = true;

    const poll = async () => {
      if (!isPollingRef.current) return;

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

        // Stop polling if loaded or error
        if (
          statusResponse.status === "loaded" ||
          statusResponse.status === "error"
        ) {
          stopPolling();
          return;
        }
      } catch (err) {
        console.error("Polling error:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to get pipeline status";
        // Show toast for polling errors
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 5000,
          });
          shownErrorRef.current = errorMessage;
        }
        setError(null); // Don't persist in state
      }

      if (isPollingRef.current) {
        pollTimeoutRef.current = setTimeout(poll, pollInterval);
      }
    };

    poll();
  }, [pollInterval, stopPolling]);

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
        await loadPipeline({
          pipeline_ids: pipelineIds,
          load_params: loadParams,
        });

        // Start polling for updates
        startPolling();

        // Set up timeout for the load operation
        const timeoutPromise = new Promise<boolean>((_, reject) => {
          loadTimeoutRef.current = setTimeout(() => {
            reject(
              new Error(
                `Pipeline load timeout after ${maxTimeout / 1000} seconds`
              )
            );
          }, maxTimeout);
        });

        // Wait for pipeline to be loaded or error
        const loadPromise = new Promise<boolean>((resolve, reject) => {
          const checkComplete = async () => {
            try {
              const currentStatus = await getPipelineStatus();
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
    [isLoading, maxTimeout, pollInterval, startPolling, stopPolling]
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
  };
}

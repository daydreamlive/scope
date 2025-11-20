import { useState, useEffect, useCallback, useRef } from "react";
import { loadPipeline, getPipelineStatus } from "../lib/api";
import type { PipelineStatusResponse, PipelineLoadParams } from "../lib/api";
import { toast } from "sonner";

interface UsePipelineOptions {
  pollInterval?: number; // milliseconds
}

export function usePipeline(options: UsePipelineOptions = {}) {
  const { pollInterval = 2000 } = options;

  const [status, setStatus] =
    useState<PipelineStatusResponse["status"]>("not_loaded");
  const [pipelineInfo, setPipelineInfo] =
    useState<PipelineStatusResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollTimeoutRef = useRef<number | null>(null);
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
            duration: 5000,
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

      // Don't show error toast for timeout errors on initial check
      // They might be temporary and the pipeline could still be loading
      const isTimeoutError = errorMessage.includes("524") ||
                            errorMessage.includes("timeout") ||
                            errorMessage.includes("timed out");

      if (!isTimeoutError) {
        // Show toast for non-timeout API errors
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 5000,
          });
          shownErrorRef.current = errorMessage;
        }
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
              duration: 5000,
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
          setIsLoading(false);
          stopPolling();
          return;
        }
      } catch (err) {
        console.error("Polling error:", err);
        const errorMessage =
          err instanceof Error ? err.message : "Failed to get pipeline status";

        // Check if it's a Cloudflare timeout error (524) or request timeout
        const isTimeoutError = errorMessage.includes("524") ||
                              errorMessage.includes("timeout") ||
                              errorMessage.includes("timed out");

        // For timeout errors during loading, don't show error toast - just continue polling
        // The pipeline might still be loading, so we don't want to alarm the user
        if (isTimeoutError && status === "loading") {
          console.log("Pipeline status request timed out, but pipeline is still loading. Continuing to poll...");
          // Continue polling without showing error
          if (isPollingRef.current) {
            pollTimeoutRef.current = setTimeout(poll, pollInterval);
          }
          return;
        }

        // Show toast for other polling errors
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 5000,
          });
          shownErrorRef.current = errorMessage;
        }
        setError(null); // Don't persist in state

        // Continue polling even on error (unless it's a fatal error)
        if (isPollingRef.current) {
          pollTimeoutRef.current = setTimeout(poll, pollInterval);
        }
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
      pipelineId?: string,
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      if (isLoading) {
        console.log("Pipeline already loading");
        return false;
      }

      try {
        setIsLoading(true);
        setError(null);
        shownErrorRef.current = null; // Reset error tracking when starting new load

        // Start the load request (backend handles this asynchronously)
        await loadPipeline({
          pipeline_id: pipelineId,
          load_params: loadParams,
        });

        // Start polling for updates - this will handle status updates asynchronously
        startPolling();

        // Return immediately - don't wait for pipeline to load
        // The polling mechanism will handle status updates and show errors via toasts
        return true;
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Failed to load pipeline";
        console.error("Pipeline load error:", errorMessage);
        // Show toast for load errors
        if (shownErrorRef.current !== errorMessage) {
          toast.error("Pipeline Error", {
            description: errorMessage,
            duration: 5000,
          });
          shownErrorRef.current = errorMessage;
        }
        setError(null); // Don't persist in state

        setIsLoading(false);
        return false;
      }
    },
    [isLoading, startPolling]
  );

  // Load pipeline with proper state management
  const loadPipelineAsync = useCallback(
    async (
      pipelineId?: string,
      loadParams?: PipelineLoadParams
    ): Promise<boolean> => {
      // Always trigger load - let the backend decide if reload is needed
      return await triggerLoad(pipelineId, loadParams);
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

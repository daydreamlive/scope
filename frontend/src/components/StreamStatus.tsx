import { useEffect, useRef } from "react";
import {
  getStreamStatus,
  type StreamStatus as StreamStatusType,
} from "../lib/daydream";

interface StreamStatusProps {
  streamId: string | null;
  isActive: boolean;
  onStatusChange?: (status: StreamStatusType) => void;
  pollInterval?: number;
}

/**
 * Stream status monitor for cloud mode.
 * Polls the Daydream API for stream status when active.
 * Does not render any visible UI.
 */
export function StreamStatus({
  streamId,
  isActive,
  onStatusChange,
  pollInterval = 5000,
}: StreamStatusProps) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    // Clear any existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Only poll when active and we have a stream ID
    if (!isActive || !streamId) {
      return;
    }

    const pollStatus = async () => {
      try {
        const status = await getStreamStatus(streamId);
        onStatusChange?.(status);
      } catch (error) {
        console.error("Failed to fetch stream status:", error);
      }
    };

    // Poll immediately, then at interval
    pollStatus();
    intervalRef.current = setInterval(pollStatus, pollInterval);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [streamId, isActive, onStatusChange, pollInterval]);

  // This component is a monitor and renders nothing
  return null;
}

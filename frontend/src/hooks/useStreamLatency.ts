import { useState, useEffect, useRef } from "react";
import { getStreamStats } from "../lib/api";

interface UseStreamLatencyProps {
  isStreaming: boolean;
  inputMode: "text" | "video";
}

export function useStreamLatency({
  isStreaming,
  inputMode,
}: UseStreamLatencyProps) {
  const [latency, setLatency] = useState<number | null>(null);
  const statsIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    // Only fetch latency for V2V mode (video input)
    if (isStreaming && inputMode === "video") {
      const fetchLatency = async () => {
        try {
          const stats = await getStreamStats();
          if (stats.latency !== undefined && stats.latency !== null) {
            setLatency(stats.latency);
          }
        } catch (error) {
          // Silently fail - latency is optional
          console.debug("Failed to fetch latency:", error);
        }
      };

      // Fetch immediately
      fetchLatency();

      // Then fetch every 1s for updates
      statsIntervalRef.current = setInterval(fetchLatency, 1000);
    } else {
      // Clear interval and reset latency when not streaming or not V2V
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }
      setLatency(null);
    }

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }
    };
  }, [isStreaming, inputMode]);

  return latency;
}

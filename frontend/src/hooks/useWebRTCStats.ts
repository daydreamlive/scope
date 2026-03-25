import { useState, useEffect, useRef, useCallback } from "react";

interface WebRTCStats {
  fps: number;
  bitrate: number;
}

interface UseWebRTCStatsProps {
  peerConnectionRef: React.MutableRefObject<RTCPeerConnection | null>;
  isStreaming: boolean;
}

export function useWebRTCStats({
  peerConnectionRef,
  isStreaming,
}: UseWebRTCStatsProps) {
  const [stats, setStats] = useState<WebRTCStats>({
    fps: 0,
    bitrate: 0,
  });

  const statsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const previousStatsRef = useRef<{
    framesReceived?: number;
    bytesReceived?: number;
    timestamp?: number;
  }>({});

  const fpsHistoryRef = useRef<number[]>([]);
  const bitrateHistoryRef = useRef<number[]>([]);

  const calculateStats = useCallback(async () => {
    const peerConnection = peerConnectionRef.current;

    if (!peerConnection || !isStreaming) {
      setStats(prev => ({ ...prev, fps: 0, bitrate: 0 }));
      return;
    }

    try {
      const statsReport = await peerConnection.getStats();
      let incomingFPS = 0;
      let incomingBitrate = 0;

      // Process stats for incoming stream (from server)
      statsReport.forEach(report => {
        if (report.type === "inbound-rtp" && report.mediaType === "video") {
          const currentFramesReceived = report.framesReceived || 0;
          const currentBytesReceived = report.bytesReceived || 0;
          const currentTimestamp = report.timestamp;

          if (
            previousStatsRef.current.framesReceived !== undefined &&
            previousStatsRef.current.bytesReceived !== undefined &&
            previousStatsRef.current.timestamp !== undefined
          ) {
            const timeDiff =
              (currentTimestamp - previousStatsRef.current.timestamp) / 1000; // Convert to seconds
            const framesDiff =
              currentFramesReceived - previousStatsRef.current.framesReceived;
            const bytesDiff =
              currentBytesReceived - previousStatsRef.current.bytesReceived;

            if (timeDiff > 0 && framesDiff >= 0) {
              incomingFPS = Math.round((framesDiff / timeDiff) * 10) / 10; // Round to 1 decimal place
              // Clamp FPS to reasonable range
              incomingFPS = Math.max(0, Math.min(incomingFPS, 60));
            }

            if (timeDiff > 0 && bytesDiff >= 0) {
              // Calculate bitrate: (bytes * 8) / time = bits per second
              incomingBitrate = (bytesDiff * 8) / timeDiff;
            }
          }

          // Store current values for next calculation
          previousStatsRef.current.framesReceived = currentFramesReceived;
          previousStatsRef.current.bytesReceived = currentBytesReceived;
          previousStatsRef.current.timestamp = currentTimestamp;
        }
      });

      // Use rolling average for smoother FPS display
      if (incomingFPS > 0) {
        fpsHistoryRef.current.push(incomingFPS);
        if (fpsHistoryRef.current.length > 5) {
          fpsHistoryRef.current.shift(); // Keep only last 5 measurements
        }
      }

      // Use rolling average for smoother bitrate display
      if (incomingBitrate > 0) {
        bitrateHistoryRef.current.push(incomingBitrate);
        if (bitrateHistoryRef.current.length > 5) {
          bitrateHistoryRef.current.shift(); // Keep only last 5 measurements
        }
      }

      // Calculate averages
      const avgFps =
        fpsHistoryRef.current.length > 0
          ? fpsHistoryRef.current.reduce((sum, val) => sum + val, 0) /
            fpsHistoryRef.current.length
          : incomingFPS;

      const avgBitrate =
        bitrateHistoryRef.current.length > 0
          ? bitrateHistoryRef.current.reduce((sum, val) => sum + val, 0) /
            bitrateHistoryRef.current.length
          : incomingBitrate;

      setStats(prev => ({
        fps: avgFps > 0 ? avgFps : prev.fps, // Keep previous FPS if current is 0
        bitrate: avgBitrate > 0 ? avgBitrate : prev.bitrate, // Keep previous bitrate if current is 0
      }));
    } catch (error) {
      console.error("Error getting WebRTC stats:", error);
    }
  }, [peerConnectionRef, isStreaming]);

  // Start/stop stats collection based on streaming state
  useEffect(() => {
    const peerConnection = peerConnectionRef.current;
    if (isStreaming && peerConnection) {
      // Start collecting stats immediately
      calculateStats();

      // Then collect every 1s for more responsive updates
      statsIntervalRef.current = setInterval(calculateStats, 1000);
    } else {
      // Clear interval and reset stats
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }

      // Reset previous stats when not streaming
      previousStatsRef.current = {};
      fpsHistoryRef.current = [];
      bitrateHistoryRef.current = [];
      setStats({
        fps: 0,
        bitrate: 0,
      });
    }

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }
    };
  }, [isStreaming, peerConnectionRef, calculateStats]);

  return stats;
}

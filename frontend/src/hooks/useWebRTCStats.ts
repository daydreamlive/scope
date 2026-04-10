import { useState, useEffect, useRef, useCallback } from "react";

export interface WebRTCStats {
  fps: number;
  bitrate: number;
}

interface PerTrackPrev {
  framesReceived: number;
  bytesReceived: number;
  timestamp: number;
}

interface UseWebRTCStatsProps {
  peerConnectionRef: React.MutableRefObject<RTCPeerConnection | null>;
  isStreaming: boolean;
  sinkNodeIdsRef?: React.RefObject<string[]>;
  sinkMidMapRef?: React.RefObject<Record<string, string>>;
}

interface InboundVideoStats extends RTCStats {
  mediaType?: string;
  kind?: string;
  mid?: string;
  framesReceived?: number;
  framesDecoded?: number;
  framesPerSecond?: number;
  bytesReceived?: number;
  timestamp: number;
}

function isInboundVideoStats(report: RTCStats): report is InboundVideoStats {
  if (report.type !== "inbound-rtp") return false;

  const candidate = report as InboundVideoStats;
  return (
    // Safari exposes a slightly different RTCInboundRtpStreamStats shape than
    // Chromium, so accept either browser's video markers/counters here.
    candidate.mediaType === "video" ||
    candidate.kind === "video" ||
    typeof candidate.framesReceived === "number" ||
    typeof candidate.framesDecoded === "number" ||
    typeof candidate.framesPerSecond === "number"
  );
}

export function useWebRTCStats({
  peerConnectionRef,
  isStreaming,
  sinkNodeIdsRef,
  sinkMidMapRef,
}: UseWebRTCStatsProps) {
  const [perSinkStats, setPerSinkStats] = useState<Record<string, WebRTCStats>>(
    {}
  );

  const statsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const previousPerTrackRef = useRef<Record<string, PerTrackPrev>>({});
  const fpsHistoryPerTrackRef = useRef<Record<string, number[]>>({});
  const bitrateHistoryPerTrackRef = useRef<Record<string, number[]>>({});

  const calculateStats = useCallback(async () => {
    const pc = peerConnectionRef.current;
    const sinkIds = sinkNodeIdsRef?.current ?? [];

    if (!pc || !isStreaming) {
      setPerSinkStats({});
      return;
    }

    try {
      const statsReport = await pc.getStats();
      const newPerSink: Record<string, WebRTCStats> = {};

      statsReport.forEach(report => {
        if (isInboundVideoStats(report)) {
          const mid = report.mid;
          const trackKey = mid ?? "default";

          // Safari may omit framesReceived on inbound-rtp video stats while
          // still exposing framesDecoded, so fall back to that counter.
          const currentFrames =
            report.framesReceived ?? report.framesDecoded ?? 0;
          const currentBytes = report.bytesReceived || 0;
          const currentTimestamp = report.timestamp;

          let fps = 0;
          let bitrate = 0;

          const prev = previousPerTrackRef.current[trackKey];
          if (prev) {
            const timeDiff = (currentTimestamp - prev.timestamp) / 1000;
            const framesDiff = currentFrames - prev.framesReceived;
            const bytesDiff = currentBytes - prev.bytesReceived;

            if (timeDiff > 0 && framesDiff >= 0) {
              fps = Math.max(
                0,
                Math.min(Math.round((framesDiff / timeDiff) * 10) / 10, 60)
              );
            }
            if (timeDiff > 0 && bytesDiff >= 0) {
              bitrate = (bytesDiff * 8) / timeDiff;
            }
          }

          // Some Safari builds report instantaneous FPS directly instead of
          // enough counters to derive it from deltas.
          if (fps <= 0 && typeof report.framesPerSecond === "number") {
            fps = Math.max(0, Math.min(report.framesPerSecond, 60));
          }

          previousPerTrackRef.current[trackKey] = {
            framesReceived: currentFrames,
            bytesReceived: currentBytes,
            timestamp: currentTimestamp,
          };

          // Rolling averages
          if (fps > 0) {
            const hist = (fpsHistoryPerTrackRef.current[trackKey] ??= []);
            hist.push(fps);
            if (hist.length > 5) hist.shift();
          }
          if (bitrate > 0) {
            const hist = (bitrateHistoryPerTrackRef.current[trackKey] ??= []);
            hist.push(bitrate);
            if (hist.length > 5) hist.shift();
          }

          const fpsHist = fpsHistoryPerTrackRef.current[trackKey] ?? [];
          const bitrateHist = bitrateHistoryPerTrackRef.current[trackKey] ?? [];

          const avgFps =
            fpsHist.length > 0
              ? fpsHist.reduce((s, v) => s + v, 0) / fpsHist.length
              : fps;
          const avgBitrate =
            bitrateHist.length > 0
              ? bitrateHist.reduce((s, v) => s + v, 0) / bitrateHist.length
              : bitrate;

          // Map MID to sink node ID using the mapping built during ontrack
          const midMap = sinkMidMapRef?.current ?? {};
          let sinkId: string | undefined;
          if (mid != null && midMap[mid]) {
            sinkId = midMap[mid];
          } else if (sinkIds.length === 1) {
            sinkId = sinkIds[0];
          } else if (sinkIds.length === 0) {
            sinkId = "_default";
          }

          if (sinkId) {
            newPerSink[sinkId] = {
              fps: avgFps > 0 ? avgFps : 0,
              bitrate: avgBitrate > 0 ? avgBitrate : 0,
            };
          }
        }
      });

      setPerSinkStats(prev => {
        // Keep previous values for sinks that didn't report this cycle
        const merged = { ...prev };
        for (const [id, s] of Object.entries(newPerSink)) {
          merged[id] = {
            fps: s.fps > 0 ? s.fps : (prev[id]?.fps ?? 0),
            bitrate: s.bitrate > 0 ? s.bitrate : (prev[id]?.bitrate ?? 0),
          };
        }
        return merged;
      });
    } catch (error) {
      console.error("Error getting WebRTC stats:", error);
    }
  }, [peerConnectionRef, isStreaming, sinkNodeIdsRef, sinkMidMapRef]);

  useEffect(() => {
    const pc = peerConnectionRef.current;
    if (isStreaming && pc) {
      calculateStats();
      statsIntervalRef.current = setInterval(calculateStats, 1000);
    } else {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }
      previousPerTrackRef.current = {};
      fpsHistoryPerTrackRef.current = {};
      bitrateHistoryPerTrackRef.current = {};
      setPerSinkStats({});
    }

    return () => {
      if (statsIntervalRef.current) {
        clearInterval(statsIntervalRef.current);
        statsIntervalRef.current = null;
      }
    };
  }, [isStreaming, peerConnectionRef, calculateStats]);

  return { perSinkStats };
}

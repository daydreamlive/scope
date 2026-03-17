import { useState, useCallback, useEffect, useRef } from "react";
import {
  getTempoStatus,
  getTempoSources,
  enableTempo,
  disableTempo,
  setTempo,
  type TempoStatusResponse,
  type TempoSourcesResponse,
  type TempoEnableRequest,
} from "../lib/api";

export interface TempoState {
  enabled: boolean;
  bpm: number | null;
  beatPhase: number;
  barPosition: number;
  beatCount: number;
  isPlaying: boolean;
  sourceType: string | null;
  numPeers: number | null;
  beatsPerBar: number;
}

const INITIAL_STATE: TempoState = {
  enabled: false,
  bpm: null,
  beatPhase: 0,
  barPosition: 0,
  beatCount: 0,
  isPlaying: false,
  sourceType: null,
  numPeers: null,
  beatsPerBar: 4,
};

export function useTempoSync() {
  const [tempoState, setTempoState] = useState<TempoState>(INITIAL_STATE);
  const [sources, setSources] = useState<TempoSourcesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const applyStatus = useCallback((status: TempoStatusResponse) => {
    setTempoState({
      enabled: status.enabled,
      bpm: status.beat_state?.bpm ?? null,
      beatPhase: status.beat_state?.beat_phase ?? 0,
      barPosition: status.beat_state?.bar_position ?? 0,
      beatCount: status.beat_state?.beat_count ?? 0,
      isPlaying: status.beat_state?.is_playing ?? false,
      sourceType: status.source?.type ?? null,
      numPeers: status.source?.num_peers ?? null,
      beatsPerBar: status.beats_per_bar,
    });
  }, []);

  const updateFromNotification = useCallback(
    (data: {
      bpm: number;
      beat_phase: number;
      bar_position: number;
      beat_count: number;
      is_playing: boolean;
    }) => {
      setTempoState(prev => ({
        ...prev,
        bpm: data.bpm,
        beatPhase: data.beat_phase,
        barPosition: data.bar_position,
        beatCount: data.beat_count,
        isPlaying: data.is_playing,
      }));
    },
    []
  );

  const fetchStatus = useCallback(async () => {
    try {
      const status = await getTempoStatus();
      applyStatus(status);
    } catch {
      // Silently ignore polling errors
    }
  }, [applyStatus]);

  const fetchSources = useCallback(async () => {
    try {
      const result = await getTempoSources();
      setSources(result);
    } catch {
      // Ignore
    }
  }, []);

  const enable = useCallback(
    async (request: TempoEnableRequest) => {
      setLoading(true);
      setError(null);
      try {
        const status = await enableTempo(request);
        applyStatus(status);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Failed to enable tempo";
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [applyStatus]
  );

  const disable = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const status = await disableTempo();
      applyStatus(status);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to disable tempo";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [applyStatus]);

  const setSessionTempo = useCallback(
    async (bpm: number) => {
      setError(null);
      try {
        const status = await setTempo(bpm);
        applyStatus(status);
      } catch (e) {
        const msg = e instanceof Error ? e.message : "Failed to set tempo";
        setError(msg);
      }
    },
    [applyStatus]
  );

  useEffect(() => {
    fetchStatus();
    fetchSources();
  }, [fetchStatus, fetchSources]);

  // Poll status when enabled (fallback for when data channel notifications aren't available)
  useEffect(() => {
    if (tempoState.enabled) {
      pollRef.current = setInterval(fetchStatus, 2000);
    } else if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
    };
  }, [tempoState.enabled, fetchStatus]);

  return {
    tempoState,
    sources,
    loading,
    error,
    enable,
    disable,
    setSessionTempo,
    fetchSources,
    updateFromNotification,
  };
}

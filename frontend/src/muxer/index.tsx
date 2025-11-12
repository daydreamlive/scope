"use client";

import {
  createContext,
  PropsWithChildren,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Muxer } from "./core";
import type { MuxerApi, MuxerSource } from "./types";

const MuxerCtx = createContext<MuxerApi | null>(null);

export function useMuxer(): MuxerApi {
  const ctx = useContext(MuxerCtx);
  if (!ctx) throw new Error("useMuxer must be used within <MuxerProvider>");
  return ctx;
}

export const MuxerProvider = ({
  children,
  width = 512,
  height = 512,
  fps: initialFps = 30,
  crossfadeMs = 500,
  sendFps = initialFps,
  dpr,
}: PropsWithChildren<{
  width?: number;
  height?: number;
  fps?: number;
  crossfadeMs?: number;
  sendFps?: number;
  dpr?: number;
}>) => {
  const engineRef = useRef<Muxer | null>(null);
  const pendingSourceRef = useRef<MuxerSource | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [size, setSize] = useState({
    width,
    height,
    dpr: Math.min(
      2,
      typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1,
    ),
  });
  const [fps, setFps] = useState(initialFps);
  const [currentSendFps, setCurrentSendFps] = useState(sendFps);

  useLayoutEffect(() => {
    const engine = new Muxer({
      width,
      height,
      fps: initialFps,
      crossfadeMs,
      sendFps,
      dpr,
      onSendFpsChange: (nfps: number) => setCurrentSendFps(nfps),
      disableSilentAudio: true,
    });
    engineRef.current = engine;
    setStream(engine.stream);
    setSize(engine.canvasSize);
    if (pendingSourceRef.current) {
      try {
        engine.setSource(pendingSourceRef.current);
      } finally {
        pendingSourceRef.current = null;
      }
    }
    return () => {
      engine.destroy();
      engineRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!engineRef.current) return;
    engineRef.current.setCanvasSize(size.width, size.height, size.dpr);
    setStream(engineRef.current.stream);
  }, [size.width, size.height, size.dpr]);

  useEffect(() => {
    if (!engineRef.current) return;
    engineRef.current.setFps(fps);
    setStream(engineRef.current.stream);
  }, [fps]);

  useEffect(() => {
    if (!engineRef.current) return;
    engineRef.current.setSendFps(currentSendFps);
  }, [currentSendFps]);

  const api = useMemo<MuxerApi>(
    () => ({
      getSource: () => engineRef.current?.getSource() ?? null,
      setSource: (source: MuxerSource) => {
        const eng = engineRef.current;
        if (eng) eng.setSource(source);
        else pendingSourceRef.current = source;
      },
      clearSource: () => engineRef.current?.clearSource(),
      stream,
      canvasSize: size,
      setCanvasSize: (w: number, h: number, dpr?: number) =>
        setSize({ width: w, height: h, dpr: dpr ?? size.dpr }),
      fps,
      setFps,
      addAudioTrack: (track: MediaStreamTrack) =>
        engineRef.current?.addAudioTrack(track),
      removeAudioTrack: (trackId: string) =>
        engineRef.current?.removeAudioTrack(trackId),
      sendFps: currentSendFps,
      setSendFps: (nfps: number) => setCurrentSendFps(nfps),
      unlockAudio: () =>
        engineRef.current?.unlockAudio() ?? Promise.resolve(false),
    }),
    [stream, size, fps, currentSendFps],
  );

  return <MuxerCtx.Provider value={api}>{children}</MuxerCtx.Provider>;
};

export type { FitMode, MuxerApi, MuxerSource } from "./types";

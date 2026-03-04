import { useCallback, useEffect, useRef, useState } from "react";

import {
  negotiateMimeType,
  downloadBlob,
  getSourceDimensions,
  buildFilename,
} from "../lib/recordingUtils";

export interface UseRecordingOptions {
  /** Canvas width in px. Defaults to the source element's intrinsic width. */
  width?: number;
  /** Canvas height in px. Defaults to the source element's intrinsic height. */
  height?: number;
  /** Frames-per-second passed to `canvas.captureStream()`. Default 30. */
  fps?: number;
  /** MediaRecorder timeslice in ms. Default 200. */
  timeslice?: number;
  /** Mirror the source horizontally before recording. Default false. */
  mirror?: boolean;
  /** Called when an error occurs during recording. */
  onError?: (err: Error) => void;
}

export interface RecordingResult {
  blob: Blob;
  /** Object URL suitable for `<video src>` preview. Revoked by `cleanup()`. */
  url: string;
  mimeType: string;
  fileExtension: "mp4" | "webm";
  durationMs: number;
}

export interface UseRecordingReturn {
  isRecording: boolean;
  isInitializing: boolean;
  result: RecordingResult | null;
  startRecording: () => void;
  stopRecording: () => void;
  /** Download the current result. Optionally override the filename. */
  download: (filename?: string) => void;
  /** Revoke any object URLs held by the hook and clear the result. */
  cleanup: () => void;
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  source: HTMLVideoElement | HTMLCanvasElement,
  canvasW: number,
  canvasH: number,
  mirror: boolean
): void {
  ctx.fillStyle = "#000000";
  ctx.fillRect(0, 0, canvasW, canvasH);

  if (source instanceof HTMLVideoElement && (source.paused || source.ended)) {
    return;
  }

  const { width: srcW, height: srcH } = getSourceDimensions(source);
  const scale = Math.min(canvasW / srcW, canvasH / srcH);
  const drawW = srcW * scale;
  const drawH = srcH * scale;
  const offsetX = (canvasW - drawW) / 2;
  const offsetY = (canvasH - drawH) / 2;

  if (mirror) {
    ctx.save();
    ctx.translate(canvasW - offsetX, offsetY);
    ctx.scale(-1, 1);
    ctx.drawImage(source, 0, 0, drawW, drawH);
    ctx.restore();
  } else {
    ctx.drawImage(source, offsetX, offsetY, drawW, drawH);
  }
}

/**
 * Record any `<video>` or `<canvas>` element to a downloadable Blob.
 *
 * Internally creates an offscreen canvas, draws frames via
 * `requestAnimationFrame`, captures a `MediaStream`, and records with
 * `MediaRecorder`.
 */
export function useRecording(
  sourceRef: React.RefObject<HTMLVideoElement | HTMLCanvasElement | null>,
  options: UseRecordingOptions = {}
): UseRecordingReturn {
  const { fps = 30, timeslice = 200, mirror = false, onError } = options;

  const [isRecording, setIsRecording] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [result, setResult] = useState<RecordingResult | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef<number | null>(null);
  const recordingRef = useRef(false);
  const startTimeRef = useRef(0);

  const emitError = useCallback(
    (err: unknown) => {
      const error = err instanceof Error ? err : new Error(String(err));
      console.error("[useRecording]", error);
      onError?.(error);
    },
    [onError]
  );

  const startRecording = useCallback(() => {
    try {
      setIsInitializing(true);

      const source = sourceRef.current;
      if (!source) {
        emitError(new Error("Source element ref is null"));
        setIsInitializing(false);
        return;
      }

      const { width: intrinsicW, height: intrinsicH } =
        getSourceDimensions(source);
      const canvasW = options.width ?? intrinsicW;
      const canvasH = options.height ?? intrinsicH;

      if (!canvasRef.current) {
        canvasRef.current = document.createElement("canvas");
      }
      const canvas = canvasRef.current;
      canvas.width = canvasW;
      canvas.height = canvasH;

      const ctx = canvas.getContext("2d");
      if (!ctx) {
        emitError(new Error("Could not get 2d context from offscreen canvas"));
        setIsInitializing(false);
        return;
      }

      const draw = () => {
        if (!recordingRef.current) return;
        try {
          drawFrame(ctx, source, canvasW, canvasH, mirror);
        } catch (err) {
          emitError(err);
        }
        if (recordingRef.current) {
          rafRef.current = requestAnimationFrame(draw);
        }
      };

      const stream = canvas.captureStream(fps);
      const { mimeType } = negotiateMimeType();

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream, { mimeType });
      recorderRef.current = recorder;

      recorder.ondataavailable = e => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data);
        }
      };

      recorder.start(timeslice);
      recordingRef.current = true;
      startTimeRef.current = performance.now();
      setIsRecording(true);

      draw();
    } catch (err) {
      emitError(err);
    } finally {
      setIsInitializing(false);
    }
  }, [
    sourceRef,
    options.width,
    options.height,
    fps,
    timeslice,
    mirror,
    emitError,
  ]);

  const stopRecording = useCallback(() => {
    recordingRef.current = false;

    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }

    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== "recording") {
      setIsRecording(false);
      return;
    }

    recorder.onstop = () => {
      const mimeType = recorder.mimeType || "video/webm";
      const blob = new Blob(chunksRef.current, { type: mimeType });
      const url = URL.createObjectURL(blob);
      const { fileExtension } = negotiateMimeType();
      const durationMs = performance.now() - startTimeRef.current;

      setResult({ blob, url, mimeType, fileExtension, durationMs });
      setIsRecording(false);
      chunksRef.current = [];
    };

    recorder.stop();
  }, []);

  const download = useCallback(
    (filename?: string) => {
      if (!result) return;
      const name = filename ?? buildFilename("recording", result.fileExtension);
      downloadBlob(result.blob, name);
    },
    [result]
  );

  const cleanup = useCallback(() => {
    if (result?.url) {
      URL.revokeObjectURL(result.url);
    }
    setResult(null);
  }, [result]);

  useEffect(() => {
    return () => {
      if (result?.url) {
        URL.revokeObjectURL(result.url);
      }
    };
  }, [result]);

  return {
    isRecording,
    isInitializing,
    result,
    startRecording,
    stopRecording,
    download,
    cleanup,
  };
}

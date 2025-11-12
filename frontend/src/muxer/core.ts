import type {
  CanvasSize,
  Ctx2D,
  CustomSource,
  FitMode,
  MuxerSource,
} from "./types";

export type MuxerOptions = {
  width: number;
  height: number;
  fps: number;
  dpr?: number;
  crossfadeMs?: number;
  keepalive?: boolean;
  sendFps?: number;
  onSendFpsChange?: (fps: number) => void;
  autoUnlockAudio?: boolean;
  unlockEvents?: string[];
  disableSilentAudio?: boolean;
};

export class Muxer {
  private captureCanvas: HTMLCanvasElement | null = null;
  private captureCtx: Ctx2D | null = null;
  private offscreen: OffscreenCanvas | HTMLCanvasElement | null = null;
  private offscreenCtx: Ctx2D | null = null;
  private outputStream: MediaStream | null = null;

  private currentSource: MuxerSource | null = null;
  private pendingSource: MuxerSource | null = null;
  private cleanupRef: (() => void) | void | undefined;

  private lastFrameAt = 0;
  private fps: number;
  private sendFps: number;
  private size: CanvasSize;
  private crossfadeMs: number;
  private crossfadeStart: number | null = null;
  private keepalive: boolean;
  private onSendFpsChange?: (fps: number) => void;

  private rafId: number | null = null;
  private videoFrameRequestId: number | null = null;
  private videoFrameSource: HTMLVideoElement | null = null;
  private rafFallbackActive = false;

  private backgroundIntervalId: number | null = null;
  private visibilityListener: (() => void) | null = null;
  private lastVisibleSendFps: number | null = null;
  private externalAudioTrackIds: Set<string> = new Set();
  private frameIndex = 0;
  private audioCtx: AudioContext | null = null;
  private silentOsc: OscillatorNode | null = null;
  private silentGain: GainNode | null = null;
  private audioDst: MediaStreamAudioDestinationNode | null = null;
  private silentAudioTrack: MediaStreamTrack | null = null;
  private externalAudioEndHandlers: Map<string, (ev: Event) => void> =
    new Map();

  private audioAutoUnlock: boolean = true;
  private audioUnlockEvents: string[] = [];
  private audioUnlockHandler: ((ev: Event) => void) | null = null;
  private audioUnlockAttached = false;
  private audioStateListenerAttached = false;
  private disableSilentAudio: boolean = false;

  private rectCache: WeakMap<
    HTMLCanvasElement | HTMLVideoElement,
    {
      canvasW: number;
      canvasH: number;
      sourceW: number;
      sourceH: number;
      dx: number;
      dy: number;
      dw: number;
      dh: number;
      fit: FitMode;
    }
  > = new WeakMap();

  constructor(options: MuxerOptions) {
    const dpr = Math.min(
      2,
      options.dpr ??
        (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1),
    );
    this.size = { width: options.width, height: options.height, dpr };
    this.fps = Math.max(1, options.fps);
    this.sendFps = Math.max(1, options.sendFps ?? options.fps);
    this.crossfadeMs = options.crossfadeMs ?? 500;
    this.keepalive = options.keepalive ?? true;
    this.onSendFpsChange = options.onSendFpsChange;
    this.disableSilentAudio = options.disableSilentAudio ?? false;
    this.initCanvas();
    this.setupVisibilityHandling();
    this.audioAutoUnlock = options.autoUnlockAudio ?? true;
    this.audioUnlockEvents =
      options.unlockEvents && options.unlockEvents.length > 0
        ? options.unlockEvents
        : ["pointerdown", "click", "touchstart", "keydown"];
    this.setupAudioAutoUnlock();
  }

  private initCanvas() {
    const canvas = document.createElement("canvas");
    canvas.style.display = "none";
    const pxW = Math.round(this.size.width * this.size.dpr);
    const pxH = Math.round(this.size.height * this.size.dpr);
    const outW = Math.round(this.size.width);
    const outH = Math.round(this.size.height);
    canvas.width = outW;
    canvas.height = outH;
    const ctx = canvas.getContext("2d", {
      alpha: false,
      desynchronized: true,
    }) as Ctx2D | null;
    if (!ctx) throw new Error("2D context not available");

    this.captureCanvas = canvas;
    this.captureCtx = ctx;

    try {
      const off = new OffscreenCanvas(pxW, pxH);
      this.offscreen = off;
      const offCtx = off.getContext("2d", { alpha: false }) as Ctx2D | null;
      if (!offCtx) throw new Error("2D context not available for Offscreen");
      offCtx.imageSmoothingEnabled = true;
      this.offscreenCtx = offCtx;
    } catch (_) {
      const off = document.createElement("canvas");
      off.width = pxW;
      off.height = pxH;
      const offCtx = off.getContext("2d", { alpha: false }) as Ctx2D | null;
      if (!offCtx)
        throw new Error("2D context not available for Offscreen fallback");
      offCtx.imageSmoothingEnabled = true;
      this.offscreen = off;
      this.offscreenCtx = offCtx;
    }

    const stream = canvas.captureStream(this.fps);
    this.outputStream = stream;

    this.ensureSilentAudioTrack();

    try {
      const vtrack = stream.getVideoTracks()[0];
      if (vtrack) {
        try {
          if (vtrack.contentHint !== undefined) {
            vtrack.contentHint = "detail";
          }
        } catch (_) {}
        this.applyVideoTrackConstraints();
      }
    } catch (_) {}

    const offCtx = this.offscreenCtx!;
    offCtx.fillStyle = "#111";
    offCtx.fillRect(0, 0, pxW, pxH);
    this.captureCtx!.drawImage(
      this.offscreen as any,
      0,
      0,
      pxW,
      pxH,
      0,
      0,
      outW,
      outH,
    );
  }

  private recreateStream() {
    if (!this.captureCanvas) return;
    const newStream = this.captureCanvas.captureStream(this.fps);
    const prev = this.outputStream;
    if (prev && prev !== newStream) {
      try {
        prev.getAudioTracks().forEach(t => {
          try {
            newStream.addTrack(t);
          } catch (_) {}
        });
      } catch (_) {}
    }
    this.outputStream = newStream;

    if (this.outputStream && this.outputStream.getAudioTracks().length === 0) {
      this.ensureSilentAudioTrack();
    }

    try {
      const w = this.captureCanvas.width;
      const h = this.captureCanvas.height;
      const vtrack = newStream.getVideoTracks()[0];
      if (vtrack) {
        try {
          if (vtrack.contentHint !== undefined) {
            vtrack.contentHint = "detail";
          }
        } catch (_) {}
        this.applyVideoTrackConstraints();
      }
    } catch (_) {}

    if (prev && prev !== newStream) {
      try {
        prev.getVideoTracks().forEach(t => {
          try {
            t.stop();
          } catch (_) {}
        });
      } catch (_) {}
    }
  }

  get stream(): MediaStream | null {
    return this.outputStream;
  }

  get canvasSize(): CanvasSize {
    return this.size;
  }

  get currentFps(): number {
    return this.fps;
  }

  get currentSendFps(): number {
    return this.sendFps;
  }

  setFps(fps: number) {
    const next = Math.max(1, fps);
    if (this.fps === next) return;
    this.fps = next;
    this.recreateStream();
  }

  setSendFps(fps: number) {
    const next = Math.max(1, fps);
    if (this.sendFps === next) return;
    this.sendFps = next;
    try {
      if (typeof this.onSendFpsChange === "function") {
        this.onSendFpsChange(this.sendFps);
      }
    } catch (_) {}
    this.applyVideoTrackConstraints();
  }

  setCanvasSize(w: number, h: number, dpr?: number) {
    const nextDpr = Math.min(
      2,
      dpr ?? (typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1),
    );
    if (
      this.size.width === w &&
      this.size.height === h &&
      this.size.dpr === nextDpr
    )
      return;

    this.size = { width: w, height: h, dpr: nextDpr };

    const pxW = Math.round(w * nextDpr);
    const pxH = Math.round(h * nextDpr);

    if (this.captureCanvas) {
      const outW = Math.round(w);
      const outH = Math.round(h);
      this.captureCanvas.width = outW;
      this.captureCanvas.height = outH;
    }

    if (this.offscreen instanceof HTMLCanvasElement) {
      this.offscreen.width = pxW;
      this.offscreen.height = pxH;
    } else if (this.offscreen instanceof OffscreenCanvas) {
      this.offscreen.width = pxW;
      this.offscreen.height = pxH;
    }

    this.recreateStream();
    this.rectCache = new WeakMap();
  }

  setCrossfade(ms: number) {
    this.crossfadeMs = Math.max(0, ms);
  }

  setBackgroundKeepalive(enabled: boolean) {
    this.keepalive = !!enabled;
  }

  getSource(): MuxerSource | null {
    return this.currentSource;
  }

  setSource(source: MuxerSource) {
    this.cancelSchedulers();
    this.cleanupRef?.();
    this.cleanupRef = undefined;
    this.pendingSource = source;
    this.crossfadeStart = null;

    const track = this.outputStream?.getVideoTracks()[0];

    if (source.kind === "custom") {
      const off = this.offscreenCtx;
      if (off) {
        const cleanup = (source as CustomSource).onStart(off);
        this.cleanupRef = cleanup || undefined;
      }
    }

    this.lastFrameAt = 0;
    this.scheduleForCurrentSource();
  }

  clearSource() {
    this.cancelSchedulers();
    this.cleanupRef?.();
    this.cleanupRef = undefined;
    this.currentSource = null;
    this.pendingSource = null;
    this.crossfadeStart = null;
  }

  addAudioTrack(track: MediaStreamTrack) {
    if (!this.outputStream) return;
    try {
      if (this.silentAudioTrack) {
        try {
          this.outputStream.removeTrack(this.silentAudioTrack);
        } catch (_) {}
      }
      const exists = this.outputStream
        .getAudioTracks()
        .some(t => t.id === track.id);
      if (!exists) {
        this.outputStream.addTrack(track);
      }
      this.externalAudioTrackIds.add(track.id);
      try {
        const onEnded = (_ev: Event) => {
          try {
            if (!this.outputStream) return;
            this.outputStream.getAudioTracks().forEach(t => {
              if (t.id === track.id) {
                try {
                  this.outputStream!.removeTrack(t);
                } catch (_) {}
              }
            });
            this.externalAudioTrackIds.delete(track.id);
            this.externalAudioEndHandlers.delete(track.id);
            if (this.outputStream.getAudioTracks().length === 0) {
              this.ensureSilentAudioTrack();
            }
          } catch (_) {}
          try {
            track.removeEventListener("ended", onEnded);
          } catch (_) {}
        };
        track.addEventListener("ended", onEnded);
        this.externalAudioEndHandlers.set(track.id, onEnded);
      } catch (_) {}
    } catch (_) {}
  }

  removeAudioTrack(trackId: string) {
    if (!this.outputStream) return;
    this.outputStream.getAudioTracks().forEach(t => {
      if (t.id === trackId) this.outputStream!.removeTrack(t);
    });
    this.externalAudioTrackIds.delete(trackId);
    try {
      const handler = this.externalAudioEndHandlers.get(trackId);
      const tracks = this.outputStream.getAudioTracks();
      const tr = tracks.find(t => t.id === trackId);
      if (tr && handler) {
        try {
          tr.removeEventListener("ended", handler);
        } catch (_) {}
      }
    } catch (_) {}
    this.externalAudioEndHandlers.delete(trackId);
    if (this.outputStream.getAudioTracks().length === 0) {
      this.ensureSilentAudioTrack();
    }
  }

  destroy() {
    this.cancelSchedulers();
    this.cleanupRef?.();
    this.cleanupRef = undefined;
    this.cleanupVisibilityHandling();
    this.cleanupAudioAutoUnlock();
    try {
      if (this.audioCtx && (this.audioCtx as any).onstatechange) {
        (this.audioCtx as any).onstatechange = null;
      }
    } catch (_) {}
    this.audioStateListenerAttached = false;
    this.onSendFpsChange = undefined;
    const s = this.outputStream;
    if (s) {
      try {
        s.getVideoTracks().forEach(t => {
          try {
            t.stop();
          } catch (_) {}
        });
      } catch (_) {}
    }
    this.removeSilentAudioTrack();
    try {
      this.externalAudioEndHandlers.forEach((handler, id) => {
        try {
          const tr = s?.getAudioTracks().find(t => t.id === id);
          if (tr) tr.removeEventListener("ended", handler);
        } catch (_) {}
      });
    } catch (_) {}
    this.externalAudioEndHandlers.clear();
    this.outputStream = null;
    this.captureCanvas = null;
    this.captureCtx = null;
    this.offscreen = null;
    this.offscreenCtx = null;
  }

  async unlockAudio(): Promise<boolean> {
    try {
      if (typeof window === "undefined") return false;
      if (!this.audioCtx || this.audioCtx.state === "closed") {
        this.audioCtx = new (window.AudioContext ||
          (window as any).webkitAudioContext)({ sampleRate: 48000 });
      }
      const ac = this.audioCtx;
      if (!ac) return false;
      try {
        await ac.resume();
      } catch (_) {}
      this.attachAudioCtxStateListener();
      if (ac.state === "running") {
        this.rebuildSilentAudioTrack();
        this.cleanupAudioAutoUnlock();
        return true;
      }
      return false;
    } catch (_) {
      return false;
    }
  }

  private scheduleForCurrentSource() {
    const videoEl = this.chooseVideoElement();
    if (videoEl && typeof videoEl.requestVideoFrameCallback === "function") {
      this.videoFrameSource = videoEl;
      const cb = () => {
        this.renderFrame();
        if (this.videoFrameSource === videoEl) {
          try {
            this.videoFrameRequestId = videoEl.requestVideoFrameCallback(cb);
          } catch (_) {}
        }
      };
      try {
        this.videoFrameRequestId = videoEl.requestVideoFrameCallback(cb);
      } catch (_) {}
      this.scheduleWithRaf(true);
      return;
    }
    this.scheduleWithRaf(false);
  }

  private ensureSilentAudioTrack() {
    if (this.disableSilentAudio) return;
    try {
      if (!this.outputStream) return;
      const alreadyHasAudio = this.outputStream.getAudioTracks().length > 0;
      if (alreadyHasAudio) return;
      if (
        this.silentAudioTrack &&
        this.silentAudioTrack.readyState === "live"
      ) {
        try {
          this.outputStream.addTrack(this.silentAudioTrack);
        } catch (_) {}
        return;
      }
      if (!this.audioCtx) {
        this.audioCtx = new (window.AudioContext ||
          (window as any).webkitAudioContext)({ sampleRate: 48000 });
        try {
          this.audioCtx.resume().catch(() => {});
        } catch (_) {}
        this.attachAudioCtxStateListener();
      }
      const ac = this.audioCtx;
      if (!ac) return;
      this.silentOsc = ac.createOscillator();
      this.silentGain = ac.createGain();
      this.audioDst = ac.createMediaStreamDestination();
      this.silentGain.gain.setValueAtTime(0.0001, ac.currentTime);
      this.silentOsc.frequency.setValueAtTime(440, ac.currentTime);
      this.silentOsc.type = "sine";
      this.silentOsc.connect(this.silentGain);
      this.silentGain.connect(this.audioDst);
      this.silentOsc.start();
      const track = this.audioDst.stream.getAudioTracks()[0];
      if (track) {
        this.silentAudioTrack = track;
        try {
          this.outputStream.addTrack(track);
        } catch (_) {}
      }
    } catch (_) {}
  }

  private removeSilentAudioTrack() {
    try {
      if (this.outputStream && this.silentAudioTrack) {
        try {
          this.outputStream.removeTrack(this.silentAudioTrack);
        } catch (_) {}
      }
      try {
        if (this.silentOsc) this.silentOsc.stop();
      } catch (_) {}
      try {
        if (this.silentOsc) this.silentOsc.disconnect();
      } catch (_) {}
      try {
        if (this.silentGain) this.silentGain.disconnect();
      } catch (_) {}
      this.silentOsc = null;
      this.silentGain = null;
      this.audioDst = null;
      this.silentAudioTrack = null;
      if (this.audioCtx) {
        try {
          this.audioCtx.close();
        } catch (_) {}
      }
      this.audioCtx = null;
    } catch (_) {}
  }

  private scheduleWithRaf(isFallback: boolean) {
    if (isFallback) {
      if (this.rafFallbackActive) return;
      this.rafFallbackActive = true;
    }
    const loop = () => {
      this.renderFrame();
      this.rafId = requestAnimationFrame(loop);
    };
    this.rafId = requestAnimationFrame(loop);
  }

  private cancelSchedulers() {
    if (this.rafId != null) {
      cancelAnimationFrame(this.rafId);
      this.rafId = null;
    }
    this.rafFallbackActive = false;
    if (this.videoFrameRequestId && this.videoFrameSource) {
      try {
        if (
          typeof this.videoFrameSource.cancelVideoFrameCallback === "function"
        ) {
          this.videoFrameSource.cancelVideoFrameCallback(
            this.videoFrameRequestId,
          );
        }
      } catch (_) {}
    }
    this.videoFrameRequestId = null;
    this.videoFrameSource = null;
  }

  private chooseVideoElement(): HTMLVideoElement | null {
    const pending = this.pendingSource;
    if (pending && pending.kind === "video" && this.isSourceReady(pending))
      return pending.element as HTMLVideoElement;
    const current = this.currentSource;
    if (current && current.kind === "video" && this.isSourceReady(current))
      return current.element as HTMLVideoElement;
    return null;
  }

  private isSourceReady(source: MuxerSource): boolean {
    if (source.kind === "video") {
      const v = source.element as HTMLVideoElement;
      return (
        typeof v.readyState === "number" &&
        v.readyState >= 2 &&
        (v.videoWidth || 0) > 0 &&
        (v.videoHeight || 0) > 0
      );
    }
    if (source.kind === "canvas") {
      const c = source.element as HTMLCanvasElement;
      return (c.width || 0) > 0 && (c.height || 0) > 0;
    }
    return true;
  }

  private renderFrame() {
    const off = this.offscreenCtx;
    const cap = this.captureCtx;
    const capCanvas = this.captureCanvas;
    if (!off || !cap || !capCanvas) return;

    const now =
      typeof performance !== "undefined" ? performance.now() : Date.now();

    if (this.pendingSource && this.isSourceReady(this.pendingSource)) {
      if (this.crossfadeStart === null) this.crossfadeStart = now;
    }

    const minIntervalMs = 1000 / Math.max(1, this.sendFps);
    if (this.lastFrameAt !== 0 && now - this.lastFrameAt < minIntervalMs)
      return;

    off.globalCompositeOperation = "source-over";
    const willDraw = !!(
      (this.pendingSource && this.isSourceReady(this.pendingSource)) ||
      this.currentSource
    );
    if (willDraw) {
      off.fillStyle = "#000";
      off.fillRect(0, 0, off.canvas.width, off.canvas.height);
    }

    const fading =
      this.pendingSource && this.crossfadeStart !== null && this.currentSource;
    if (fading) {
      const t = Math.min(
        1,
        (now - (this.crossfadeStart as number)) / this.crossfadeMs,
      );
      this.blitSource(this.currentSource as MuxerSource, 1 - t);
      this.blitSource(this.pendingSource as MuxerSource, t);
      if (t >= 1) {
        this.currentSource = this.pendingSource;
        this.pendingSource = null;
        this.crossfadeStart = null;
      }
    } else if (this.pendingSource && !this.currentSource) {
      if (this.isSourceReady(this.pendingSource)) {
        this.blitSource(this.pendingSource, 1);
        this.currentSource = this.pendingSource;
        this.pendingSource = null;
        this.crossfadeStart = null;
      }
    } else if (this.currentSource) {
      this.blitSource(this.currentSource, 1);
    }

    if (this.keepalive) {
      const w = off.canvas.width,
        h = off.canvas.height;
      const prevAlpha = off.globalAlpha;
      const prevFill = off.fillStyle;
      try {
        off.globalAlpha = 0.08;
        off.fillStyle = this.frameIndex % 2 ? "#101010" : "#0e0e0e";
        off.fillRect(w - 16, h - 16, 16, 16);
      } catch (_) {
      } finally {
        off.globalAlpha = prevAlpha;
        off.fillStyle = prevFill;
      }
    }

    this.frameIndex++;

    cap.drawImage(
      off.canvas,
      0,
      0,
      off.canvas.width,
      off.canvas.height,
      0,
      0,
      capCanvas.width,
      capCanvas.height,
    );

    const track = this.outputStream?.getVideoTracks()[0];
    if (track && typeof (track as any).requestFrame === "function") {
      try {
        (track as any).requestFrame();
      } catch (_) {}
    }
    this.lastFrameAt = now;
  }

  private applyVideoTrackConstraints() {
    try {
      const track = this.outputStream?.getVideoTracks()[0];
      const canvas = this.captureCanvas;
      if (!track || !canvas) return;
      const constraints: MediaTrackConstraints = {
        width: canvas.width,
        height: canvas.height,
        frameRate: Math.max(1, this.sendFps || this.fps),
      };
      try {
        (track as any).contentHint !== undefined &&
          ((track as any).contentHint = "detail");
      } catch (_) {}
      track.applyConstraints(constraints).catch(() => {});
    } catch (_) {}
  }

  private requestVideoTrackFrameOnce() {
    const track = this.outputStream?.getVideoTracks()[0];
    if (track && typeof (track as any).requestFrame === "function") {
      try {
        (track as any).requestFrame();
      } catch (_) {}
    }
  }

  private setupVisibilityHandling() {
    if (typeof document === "undefined") return;
    this.visibilityListener = () => {
      this.onVisibilityChange();
    };
    document.addEventListener("visibilitychange", this.visibilityListener);
    this.onVisibilityChange();
  }

  private cleanupVisibilityHandling() {
    if (typeof document !== "undefined" && this.visibilityListener) {
      try {
        document.removeEventListener(
          "visibilitychange",
          this.visibilityListener,
        );
      } catch (_) {}
      this.visibilityListener = null;
    }
    if (this.backgroundIntervalId != null) {
      clearInterval(this.backgroundIntervalId);
      this.backgroundIntervalId = null;
    }
    this.lastVisibleSendFps = null;
  }

  private setupAudioAutoUnlock() {
    if (!this.audioAutoUnlock) return;
    if (typeof document === "undefined") return;
    if (this.audioUnlockAttached) return;
    const handler = (_ev: Event) => {
      this.unlockAudio();
    };
    this.audioUnlockHandler = handler;
    this.audioUnlockEvents.forEach(evt => {
      try {
        document.addEventListener(
          evt as any,
          handler as any,
          { capture: true } as any,
        );
      } catch (_) {}
    });
    this.audioUnlockAttached = true;
  }

  private cleanupAudioAutoUnlock() {
    if (!this.audioUnlockAttached) return;
    if (typeof document !== "undefined" && this.audioUnlockHandler) {
      this.audioUnlockEvents.forEach(evt => {
        try {
          document.removeEventListener(
            evt as any,
            this.audioUnlockHandler as any,
            { capture: true } as any,
          );
        } catch (_) {}
      });
    }
    this.audioUnlockAttached = false;
    this.audioUnlockHandler = null;
  }

  private attachAudioCtxStateListener() {
    try {
      const ac = this.audioCtx;
      if (!ac || this.audioStateListenerAttached) return;
      const onStateChange = () => {
        try {
          if (this.audioCtx && this.audioCtx.state === "running") {
            this.rebuildSilentAudioTrack();
            this.cleanupAudioAutoUnlock();
          }
        } catch (_) {}
      };
      try {
        (ac as any).onstatechange = onStateChange as any;
        this.audioStateListenerAttached = true;
      } catch (_) {}
    } catch (_) {}
  }

  private onVisibilityChange() {
    if (typeof document === "undefined") return;
    const hidden = document.visibilityState === "hidden";
    if (hidden) {
      if (this.lastVisibleSendFps == null)
        this.lastVisibleSendFps = this.sendFps;
      if (this.sendFps !== 5) this.setSendFps(5);
      if (this.backgroundIntervalId == null) {
        this.backgroundIntervalId = setInterval(() => {
          this.renderFrame();
          this.requestVideoTrackFrameOnce();
        }, 1000) as unknown as number;
      }
    } else {
      if (this.backgroundIntervalId != null) {
        clearInterval(this.backgroundIntervalId);
        this.backgroundIntervalId = null;
      }
      if (
        this.lastVisibleSendFps != null &&
        this.sendFps !== this.lastVisibleSendFps
      ) {
        this.setSendFps(this.lastVisibleSendFps);
      }
      this.lastVisibleSendFps = null;
    }
  }

  private blitSource(source: MuxerSource, alpha: number) {
    if (!this.offscreenCtx) return;
    const ctx = this.offscreenCtx;
    if (source.kind === "custom") {
      if (source.onFrame) source.onFrame(ctx, this.lastFrameAt);
      return;
    }
    const el = source.element as HTMLCanvasElement | HTMLVideoElement;
    const rect = this.getDrawRect(el, (source.fit as FitMode) ?? "contain");
    if (!rect) return;
    const prev = ctx.globalAlpha;
    try {
      ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
      ctx.drawImage(el as any, rect.dx, rect.dy, rect.dw, rect.dh);
    } catch (_) {
    } finally {
      ctx.globalAlpha = prev;
    }
  }

  private getDrawRect(
    el: HTMLCanvasElement | HTMLVideoElement,
    fit: FitMode,
  ): { dx: number; dy: number; dw: number; dh: number } | null {
    const canvas = this.offscreenCtx?.canvas;
    if (!canvas) return null;
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    const sourceW = (el as any).videoWidth ?? el.width;
    const sourceH = (el as any).videoHeight ?? el.height;
    if (!sourceW || !sourceH) return null;
    const cached = this.rectCache.get(el);
    if (
      cached &&
      cached.canvasW === canvasW &&
      cached.canvasH === canvasH &&
      cached.sourceW === sourceW &&
      cached.sourceH === sourceH &&
      cached.fit === fit
    ) {
      const { dx, dy, dw, dh } = cached;
      return { dx, dy, dw, dh };
    }
    const scale =
      fit === "cover"
        ? Math.max(canvasW / sourceW, canvasH / sourceH)
        : Math.min(canvasW / sourceW, canvasH / sourceH);
    const dw = Math.floor(sourceW * scale);
    const dh = Math.floor(sourceH * scale);
    const dx = Math.floor((canvasW - dw) / 2);
    const dy = Math.floor((canvasH - dh) / 2);
    const computed = {
      canvasW,
      canvasH,
      sourceW,
      sourceH,
      dx,
      dy,
      dw,
      dh,
      fit,
    };
    this.rectCache.set(el, computed);
    return { dx, dy, dw, dh };
  }

  private rebuildSilentAudioTrack() {
    if (this.disableSilentAudio) return;
    try {
      if (!this.outputStream) return;
      if (this.externalAudioTrackIds.size > 0) return;
      if (this.silentAudioTrack) {
        try {
          this.outputStream.removeTrack(this.silentAudioTrack);
        } catch (_) {}
      }
      try {
        if (this.silentOsc) this.silentOsc.stop();
      } catch (_) {}
      try {
        if (this.silentOsc) this.silentOsc.disconnect();
      } catch (_) {}
      try {
        if (this.silentGain) this.silentGain.disconnect();
      } catch (_) {}
      this.silentOsc = null;
      this.silentGain = null;
      this.audioDst = null;
      this.silentAudioTrack = null;
      const ac = this.audioCtx;
      if (!ac || ac.state !== "running") return;
      this.attachAudioCtxStateListener();
      this.silentOsc = ac.createOscillator();
      this.silentGain = ac.createGain();
      this.audioDst = ac.createMediaStreamDestination();
      this.silentGain.gain.setValueAtTime(0.0001, ac.currentTime);
      this.silentOsc.frequency.setValueAtTime(440, ac.currentTime);
      this.silentOsc.type = "sine";
      this.silentOsc.connect(this.silentGain);
      this.silentGain.connect(this.audioDst);
      this.silentOsc.start();
      const track = this.audioDst.stream.getAudioTracks()[0];
      if (track) {
        this.silentAudioTrack = track;
        try {
          this.outputStream.addTrack(track);
        } catch (_) {}
      }
    } catch (_) {}
  }
}

export type Ctx2D = CanvasRenderingContext2D;

export type FitMode = "contain" | "cover";

export type ElementSource =
  | {
      kind: "video";
      element: HTMLVideoElement;
      fit?: FitMode;
      contentHint?: "detail" | "motion" | "";
    }
  | {
      kind: "canvas";
      element: HTMLCanvasElement;
      fit?: FitMode;
      contentHint?: "detail" | "motion" | "";
    };

export type StartFn = (ctx: Ctx2D) => void | (() => void);
export type FrameFn = (ctx: Ctx2D, t: number) => void;

export type CustomSource = {
  kind: "custom";
  onStart: StartFn;
  onFrame?: FrameFn;
};

export type MuxerSource = ElementSource | CustomSource;

export type MuxerSourceSpec = {
  id: string;
  onStart: StartFn;
  onFrame?: FrameFn;
};

export type CanvasSize = { width: number; height: number; dpr: number };

export type MuxerApi = {
  getSource: () => MuxerSource | null;
  setSource: (source: MuxerSource) => void;
  clearSource: () => void;
  stream: MediaStream | null;
  canvasSize: CanvasSize;
  setCanvasSize: (w: number, h: number, dpr?: number) => void;
  fps: number;
  setFps: (fps: number) => void;
  addAudioTrack: (track: MediaStreamTrack) => void;
  removeAudioTrack: (trackId: string) => void;
  sendFps: number;
  setSendFps: (fps: number) => void;
  unlockAudio: () => Promise<boolean>;
};

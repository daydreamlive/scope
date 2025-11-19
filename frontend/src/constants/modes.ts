export const GENERATION_MODE = {
  VIDEO: "video",
  TEXT: "text",
} as const;

export type GenerationMode =
  (typeof GENERATION_MODE)[keyof typeof GENERATION_MODE];

export const VIDEO_SOURCE_MODE = {
  VIDEO: "video",
  CAMERA: "camera",
} as const;

export type VideoSourceMode =
  (typeof VIDEO_SOURCE_MODE)[keyof typeof VIDEO_SOURCE_MODE];

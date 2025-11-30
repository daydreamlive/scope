export const INPUT_MODE = {
  VIDEO: "video",
  TEXT: "text",
} as const;

export type InputMode = (typeof INPUT_MODE)[keyof typeof INPUT_MODE];

export const VIDEO_SOURCE_MODE = {
  VIDEO: "video",
  CAMERA: "camera",
} as const;

export type VideoSourceMode =
  (typeof VIDEO_SOURCE_MODE)[keyof typeof VIDEO_SOURCE_MODE];

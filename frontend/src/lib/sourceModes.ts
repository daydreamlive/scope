import type { InputSourceType } from "./api";

export interface BrowserSourceMode {
  id: string;
  label: string;
  description: string;
  keywords: string[];
}

// Browser-driven (WebRTC) source modes. Not registered by the backend, so
// they're prepended to the dynamic list from /api/v1/input-sources.
export const BROWSER_SOURCE_MODES: BrowserSourceMode[] = [
  {
    id: "video",
    label: "File",
    description: "Upload or cycle sample video files",
    keywords: ["file", "video", "upload"],
  },
  {
    id: "camera",
    label: "Camera",
    description: "Use the browser's webcam",
    keywords: ["webcam", "camera", "capture"],
  },
];

// Headless-only alias for the browser "video" mode — never surface in UI.
export const HIDDEN_BACKEND_SOURCES = new Set(["video_file"]);

export function getVisibleBackendSources(
  inputSources: InputSourceType[]
): InputSourceType[] {
  return inputSources.filter(
    s => s.available && !HIDDEN_BACKEND_SOURCES.has(s.source_id)
  );
}

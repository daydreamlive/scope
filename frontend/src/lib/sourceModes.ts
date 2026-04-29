import type { InputSourceType } from "./api";

// Re-exported so source-related UI code can import everything from one place.
// The canonical definition lives in graphUtils.ts because `isBrowserSourceMode`
// (and its callers in StreamPage / SourceNode / useVideoSource) need the id
// list as shared logic, not just menu metadata.
export type { BrowserSourceMode } from "./graphUtils";
export { BROWSER_SOURCE_MODES } from "./graphUtils";

// Headless-only alias for the browser "video" mode — never surface in UI.
export const HIDDEN_BACKEND_SOURCES = new Set(["video_file"]);

export function getVisibleBackendSources(
  inputSources: InputSourceType[]
): InputSourceType[] {
  return inputSources.filter(
    s => s.available && !HIDDEN_BACKEND_SOURCES.has(s.source_id)
  );
}

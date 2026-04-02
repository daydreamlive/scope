/**
 * Centralized color palette for graph node types.
 *
 * Each semantic type maps to a single accent color used for handles,
 * slider fills, and other UI chrome. Import from here instead of
 * defining ad-hoc constants in individual node files.
 */

// Semantic type colors
export const COLOR_STREAM = "#eeeeee";
export const COLOR_STRING = "#fbbf24"; // amber-400
export const COLOR_NUMBER = "#38bdf8"; // sky-400
export const COLOR_BOOLEAN = "#34d399"; // emerald-400
export const COLOR_TRIGGER = "#f97316"; // orange-500
export const COLOR_VACE = "#a78bfa"; // violet-400
export const COLOR_LORA = "#f472b6"; // pink-400
export const COLOR_IMAGE = COLOR_STRING; // images are string-typed paths
export const COLOR_DOT = "#fafafa";
export const COLOR_DEFAULT = "#9ca3af"; // gray-400

/** Accent color lookup keyed by param type. */
export const PARAM_TYPE_COLORS: Record<string, string> = {
  stream: COLOR_STREAM,
  string: COLOR_STRING,
  number: COLOR_NUMBER,
  boolean: COLOR_BOOLEAN,
  trigger: COLOR_TRIGGER,
  list_number: COLOR_NUMBER,
  float: COLOR_NUMBER,
  int: COLOR_NUMBER,
  video_path: COLOR_STREAM,
};

/** Port-level colors for stream handles (video, VACE, source, sink, etc.). */
export const HANDLE_COLORS: Record<string, string> = {
  video: COLOR_STREAM,
  video2: COLOR_STREAM,
  vace_input_frames: "#ffffff",
  vace_input_masks: COLOR_LORA, // pink-400
  source: "#4ade80",
  sink: "#fb923c",
  record: "#ef4444",
};

/** Low-opacity background variant for reroute nodes. */
export const TYPE_BG: Record<string, string> = {
  string: "rgba(251,191,36,0.12)",
  number: "rgba(56,189,248,0.12)",
  boolean: "rgba(52,211,153,0.12)",
  trigger: "rgba(249,115,22,0.12)",
};

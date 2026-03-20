import type { CSSProperties } from "react";

const COLLAPSED_COLOR = "#9ca3af"; // gray-400

/**
 * Returns the inline style to apply to a React Flow Handle when
 * the parent node is collapsed.  All handles on the same side
 * overlap at the vertical centre of the pill so they appear as
 * a single grouped connector dot.
 */
export function collapsedHandleStyle(side: "left" | "right"): CSSProperties {
  return {
    top: "50%",
    ...(side === "left" ? { left: 0 } : { right: 0 }),
    backgroundColor: COLLAPSED_COLOR,
  };
}

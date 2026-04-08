/**
 * Tour step definitions for the workspace popover tour.
 *
 * Each step anchors to a `data-tour="<anchor>"` attribute on a DOM element.
 * The `position` determines which side of the anchor the popover appears on.
 */

export interface TourStepDef {
  /** data-tour attribute value to anchor to. `null` = centered on screen. */
  anchor: string | null;
  /** Fallback anchor if the primary isn't found in DOM. */
  fallbackAnchor?: string;
  title: string;
  description: string;
  /** Preferred popover position relative to anchor. */
  position: "top" | "bottom" | "left" | "right" | "center";
  /** Show "Skip tour" link. Spec: not shown on step 0. */
  showSkip: boolean;
  /** Show "Done" instead of "Next" on the last step. */
  showDone?: boolean;
  /** Optional URL to render as a clickable link in the description. */
  linkUrl?: string;
  /** Link display text (defaults to linkUrl). */
  linkText?: string;
}

/** Tour steps shown after simple-mode onboarding. */
export const SIMPLE_TOUR_STEPS: TourStepDef[] = [
  {
    anchor: "play-button",
    title: "Click Play to start generation",
    description: "",
    position: "bottom",
    showSkip: false,
  },
  {
    anchor: "workflows-button",
    title: "Explore Workflows",
    description:
      "When you're ready, try the other starter workflows or browse community creations.",
    position: "bottom",
    showSkip: false,
    showDone: true,
  },
];

/** Tour steps shown after teaching-mode onboarding. */
export const TEACHING_TOUR_STEPS: TourStepDef[] = [
  {
    anchor: "graph-canvas",
    title: "This is your workflow",
    description:
      "Each node does one thing — video flows left to right from Source to Output.",
    position: "right",
    showSkip: false,
  },
  {
    anchor: "input-controls-panel",
    title: "Change what goes in",
    description: "Set your text prompt or connect a live video source here.",
    position: "right",
    showSkip: false,
  },
  {
    anchor: "play-button",
    title: "Click Play to start",
    description: "Scope generates in real time — every frame is live.",
    position: "bottom",
    showSkip: false,
  },
  {
    anchor: "perform-mode-toggle",
    title: "Switch to Perform Mode",
    description:
      "Hide the graph and focus on your live output — great for performance.",
    position: "bottom",
    showSkip: false,
  },
  {
    anchor: "workflows-button",
    title: "Explore more workflows",
    description:
      "When you're ready, try the other starters or import your own.",
    position: "bottom",
    showSkip: false,
    showDone: true,
  },
];

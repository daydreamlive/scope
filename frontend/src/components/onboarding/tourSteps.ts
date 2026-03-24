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
      "When you're ready to explore, browse our library of community-built Workflows or submit your own.",
    position: "bottom",
    showSkip: false,
    showDone: true,
  },
];

/** Tour steps shown after teaching-mode onboarding. */
export const TEACHING_TOUR_STEPS: TourStepDef[] = [
  {
    anchor: "play-button",
    title: "Follow the Note Cards",
    description:
      "Before you click Play to start generation, please follow the instructions on the note cards below, starting from the left.",
    position: "bottom",
    showSkip: false,
  },
  {
    anchor: "workflows-button",
    title: "Explore Workflows",
    description:
      "When you're ready to explore, browse our library of community-built Workflows or submit your own.",
    position: "bottom",
    showSkip: false,
    showDone: true,
  },
];

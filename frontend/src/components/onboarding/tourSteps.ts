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

export const TOUR_STEPS: TourStepDef[] = [
  {
    anchor: "play-button",
    title: "Start Generating",
    description: "Hit play to start real-time AI video generation.",
    position: "bottom",
    showSkip: false,
  },
  {
    anchor: "add-node",
    fallbackAnchor: "play-button",
    title: "Inputs & Outputs",
    description:
      "Source and Output nodes send video between Scope and other apps via Syphon, NDI, or Spout.",
    position: "bottom",
    showSkip: true,
  },
  {
    anchor: "add-node",
    title: "Node Registry",
    description:
      "Browse and add nodes to build your own workflows. You'll find MIDI controllers, math operations, image inputs, and more.",
    position: "bottom",
    showSkip: true,
  },
  {
    anchor: "settings-button",
    title: "OSC & MIDI Control",
    description:
      "Map MIDI controllers or OSC messages to any parameter for live control. MIDI nodes are also available in the node registry.",
    position: "bottom",
    showSkip: true,
  },
  {
    anchor: "settings-button",
    title: "LoRA Styles",
    description:
      "Install LoRAs to change the visual style of your generations. Find them in Settings \u2192 LoRAs.",
    position: "bottom",
    showSkip: true,
  },
  {
    anchor: null,
    title: "Explore More",
    description: "Find more workflows, plugins, and LoRAs at",
    position: "center",
    showSkip: false,
    showDone: true,
    linkUrl: "https://daydream.live",
    linkText: "daydream.live",
  },
];

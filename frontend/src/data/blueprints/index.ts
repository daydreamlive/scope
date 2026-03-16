import type { Blueprint } from "./types";
import xyToMath from "./xy-to-math.json";
import intControlMath from "./int-control-math.json";
import lfoBoolGate from "./lfo-bool-gate.json";
import sliderDenoise from "./slider-denoise.json";
import midiPromptSwitcher from "./midi-prompt-switcher.json";
import manualPromptSwitcher from "./manual-prompt-switcher.json";

export type { Blueprint };

export const BLUEPRINTS: Blueprint[] = [
  sliderDenoise as Blueprint,
  midiPromptSwitcher as Blueprint,
  manualPromptSwitcher as Blueprint,
  xyToMath as Blueprint,
  intControlMath as Blueprint,
  lfoBoolGate as Blueprint,
];

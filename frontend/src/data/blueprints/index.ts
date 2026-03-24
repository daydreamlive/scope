import type { Blueprint } from "./types";
import xyToMath from "./xy-to-math.json";
import intControlMath from "./int-control-math.json";
import lfoBoolGate from "./lfo-bool-gate.json";
import sliderDenoise from "./slider-denoise.json";
import midiPromptSwitcher from "./midi-prompt-switcher.json";
import manualPromptSwitcher from "./manual-prompt-switcher.json";
import midiCcScaler from "./midi-cc-scaler.json";
import lfoRangeOscillator from "./lfo-range-oscillator.json";
import knobsPanel from "./knobs-panel.json";
import timedPromptCycler from "./timed-prompt-cycler.json";

export type { Blueprint };

export const BLUEPRINTS: Blueprint[] = [
  sliderDenoise as Blueprint,
  midiPromptSwitcher as Blueprint,
  manualPromptSwitcher as Blueprint,
  midiCcScaler as Blueprint,
  timedPromptCycler as Blueprint,
  lfoRangeOscillator as Blueprint,
  knobsPanel as Blueprint,
  xyToMath as Blueprint,
  intControlMath as Blueprint,
  lfoBoolGate as Blueprint,
];

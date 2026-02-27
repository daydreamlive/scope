export type MIDIMappingType =
  | "continuous"
  | "toggle"
  | "trigger"
  | "enum_cycle";

export interface MIDISource {
  midi_cc?: number;
  midi_note?: number;
  channel: number;
}

export interface MIDITarget {
  parameter?: string;
  action?:
    | "load_preset"
    | "switch_prompt"
    | "switch_prompt_0"
    | "switch_prompt_1"
    | "switch_prompt_2"
    | "switch_prompt_3"
    | "reset_cache"
    | "toggle_pause"
    | "add_denoising_step"
    | "remove_denoising_step";
  preset_index?: number;
  prompt_index?: number;
  values?: string[];
  arrayIndex?: number;
}

export interface MIDIMapping {
  type: MIDIMappingType;
  source: MIDISource;
  target: MIDITarget;
  range?: { min: number; max: number };
}

export interface MIDIMappingProfile {
  name: string;
  device?: string;
  mappings: MIDIMapping[];
}

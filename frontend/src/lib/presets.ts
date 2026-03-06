/**
 * Preset system for saving and loading parameter snapshots.
 */

import type { SettingsState } from "../types";

/**
 * A preset is a snapshot of runtime parameters
 */
export interface Preset {
  id: string;
  name: string;
  parameters: Record<string, unknown>;
  createdAt: number;
  updatedAt: number;
}

const PRESETS_STORAGE_KEY = "scope_midi_presets";
const MAX_PRESETS = 20;

/**
 * Load all presets from localStorage
 */
export function loadPresets(): Preset[] {
  try {
    const stored = localStorage.getItem(PRESETS_STORAGE_KEY);
    if (!stored) return [];
    return JSON.parse(stored) as Preset[];
  } catch (error) {
    console.error("Failed to load presets:", error);
    return [];
  }
}

/**
 * Save presets to localStorage
 */
export function savePresets(presets: Preset[]): void {
  try {
    // Limit to max presets
    const limited = presets.slice(0, MAX_PRESETS);
    localStorage.setItem(PRESETS_STORAGE_KEY, JSON.stringify(limited));
  } catch (error) {
    console.error("Failed to save presets:", error);
  }
}

/**
 * Create a preset from current settings state
 */
export function createPresetFromState(
  name: string,
  state: SettingsState
): Preset {
  // Extract runtime parameters (exclude pipelineId, resolution, etc.)
  const parameters: Record<string, unknown> = {};

  // Runtime parameters that can be saved
  if (state.noiseScale !== undefined) parameters.noise_scale = state.noiseScale;
  if (state.noiseController !== undefined)
    parameters.noise_controller = state.noiseController;
  if (state.manageCache !== undefined)
    parameters.manage_cache = state.manageCache;
  if (state.kvCacheAttentionBias !== undefined)
    parameters.kv_cache_attention_bias = state.kvCacheAttentionBias;
  if (state.vaceContextScale !== undefined)
    parameters.vace_context_scale = state.vaceContextScale;
  if (state.vaceUseInputVideo !== undefined)
    parameters.vace_use_input_video = state.vaceUseInputVideo;
  if (state.schemaFieldOverrides) {
    // Include schema-driven runtime parameters
    Object.assign(parameters, state.schemaFieldOverrides);
  }

  const now = Date.now();
  return {
    id: `preset_${now}`,
    name,
    parameters,
    createdAt: now,
    updatedAt: now,
  };
}

/**
 * Apply a preset to settings state
 */
export function applyPresetToState(
  preset: Preset,
  currentState: SettingsState
): Partial<SettingsState> {
  const updates: Partial<SettingsState> = {};

  // Map preset parameters back to state
  if (preset.parameters.noise_scale !== undefined) {
    updates.noiseScale = preset.parameters.noise_scale as number;
  }
  if (preset.parameters.noise_controller !== undefined) {
    updates.noiseController = preset.parameters.noise_controller as boolean;
  }
  if (preset.parameters.manage_cache !== undefined) {
    updates.manageCache = preset.parameters.manage_cache as boolean;
  }
  if (preset.parameters.kv_cache_attention_bias !== undefined) {
    updates.kvCacheAttentionBias = preset.parameters
      .kv_cache_attention_bias as number;
  }
  if (preset.parameters.vace_context_scale !== undefined) {
    updates.vaceContextScale = preset.parameters.vace_context_scale as number;
  }
  if (preset.parameters.vace_use_input_video !== undefined) {
    updates.vaceUseInputVideo = preset.parameters
      .vace_use_input_video as boolean;
  }

  // Handle schema field overrides
  const schemaOverrides: Record<string, unknown> = {
    ...currentState.schemaFieldOverrides,
  };
  for (const [key, value] of Object.entries(preset.parameters)) {
    // Skip known state fields
    if (
      ![
        "noise_scale",
        "noise_controller",
        "manage_cache",
        "kv_cache_attention_bias",
        "vace_context_scale",
        "vace_use_input_video",
      ].includes(key)
    ) {
      schemaOverrides[key] = value;
    }
  }
  if (Object.keys(schemaOverrides).length > 0) {
    updates.schemaFieldOverrides = schemaOverrides;
  }

  return updates;
}

/**
 * Save a preset
 */
export function savePreset(preset: Preset): void {
  const presets = loadPresets();
  const index = presets.findIndex((p) => p.id === preset.id);
  if (index >= 0) {
    presets[index] = { ...preset, updatedAt: Date.now() };
  } else {
    presets.push(preset);
  }
  savePresets(presets);
}

/**
 * Delete a preset
 */
export function deletePreset(presetId: string): void {
  const presets = loadPresets();
  const filtered = presets.filter((p) => p.id !== presetId);
  savePresets(filtered);
}

/**
 * Get a preset by ID
 */
export function getPreset(presetId: string): Preset | undefined {
  const presets = loadPresets();
  return presets.find((p) => p.id === presetId);
}

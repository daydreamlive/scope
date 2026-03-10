/**
 * Utility functions for LoRA adapter management
 */

export interface LoRAScaleData {
  adapter_name?: string;
  path: string;
  scale: number;
}

export interface LoadedAdapter {
  adapter_name?: string;
  path: string;
  scale: number;
}

/**
 * Send LoRA scale updates to backend, filtering for only loaded adapters.
 *
 * Matching strategy:
 * - Positional: user lora at index i maps to loadedAdapters[i] when the
 *   loaded adapter carries an adapter_name. This safely handles duplicate
 *   LoRA files that would be ambiguous when matched by path alone.
 * - Fallback: for legacy backends without adapter_name, matching falls
 *   back to path-based lookup.
 *
 * @param loras - Array of LoRA configurations with path and scale
 * @param loadedAdapters - Array of currently loaded adapters from pipeline
 * @param sendUpdate - Callback to send parameter update to backend
 */
export function sendLoRAScaleUpdates(
  loras: LoRAScaleData[] | undefined,
  loadedAdapters: LoadedAdapter[] | undefined,
  sendUpdate: (params: { lora_scales: LoRAScaleData[] }) => void
): void {
  if (!loras || !loadedAdapters) {
    return;
  }

  const hasAdapterNames = loadedAdapters.some(a => !!a.adapter_name);

  const lora_scales: LoRAScaleData[] = [];

  if (hasAdapterNames) {
    // Positional match: user lora[i] corresponds to loadedAdapters[i].
    const count = Math.min(loras.length, loadedAdapters.length);
    for (let i = 0; i < count; i++) {
      const loaded = loadedAdapters[i];
      const entry: LoRAScaleData = {
        path: loaded.path,
        scale: loras[i].scale,
      };
      if (loaded.adapter_name) {
        entry.adapter_name = loaded.adapter_name;
      }
      lora_scales.push(entry);
    }
  } else {
    // Legacy fallback: match by path.
    const loadedPaths = new Set(loadedAdapters.map(a => a.path));
    for (const lora of loras) {
      if (loadedPaths.has(lora.path)) {
        lora_scales.push({ path: lora.path, scale: lora.scale });
      }
    }
  }

  if (lora_scales.length > 0) {
    sendUpdate({ lora_scales });
  }
}

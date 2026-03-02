/**
 * Utility functions for LoRA adapter management
 */

export interface LoRAScaleData {
  path: string;
  scale: number;
}

export interface LoadedAdapter {
  path: string;
  scale: number;
}

/**
 * Send LoRA scale updates to backend, filtering for only loaded adapters.
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
  if (!loras) {
    return;
  }

  // Build path->scale updates from configured LoRAs (ignore incomplete entries).
  const configuredScales = loras
    .filter(lora => Boolean(lora.path))
    .map(lora => ({ path: lora.path, scale: lora.scale }));

  if (configuredScales.length === 0) {
    return;
  }

  // Prefer filtering by backend-reported loaded adapters when available.
  // If this metadata is missing/stale (common immediately after load),
  // fall back to sending configured scales so updates are not silently dropped.
  let lora_scales = configuredScales;
  if (loadedAdapters && loadedAdapters.length > 0) {
    const loadedPaths = new Set(loadedAdapters.map(adapter => adapter.path));
    const filtered = configuredScales.filter(lora => loadedPaths.has(lora.path));
    if (filtered.length > 0) {
      lora_scales = filtered;
    }
  }

  if (lora_scales.length > 0) {
    sendUpdate({ lora_scales });
  }
}

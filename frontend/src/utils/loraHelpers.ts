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
  if (!loras || !loadedAdapters) {
    return;
  }

  // Build set of loaded paths for efficient lookup
  const loadedPaths = new Set(loadedAdapters.map(adapter => adapter.path));

  // Filter to only include LoRAs that are actually loaded
  const lora_scales = loras
    .filter(lora => loadedPaths.has(lora.path))
    .map(lora => ({ path: lora.path, scale: lora.scale }));

  if (lora_scales.length > 0) {
    sendUpdate({ lora_scales });
  }
}

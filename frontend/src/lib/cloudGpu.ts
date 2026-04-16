/**
 * Cloud GPU catalog and local-storage helpers.
 *
 * The selected GPU is persisted per-browser so the settings selector comes
 * back pre-populated with the user's last choice on return visits.
 */

export type CloudGpu = "h100" | "rtx4090" | "rtx5090";

export interface CloudGpuOption {
  id: CloudGpu;
  label: string;
  creditsPerMin: number;
}

export const CLOUD_GPUS: readonly CloudGpuOption[] = [
  { id: "h100", label: "H100", creditsPerMin: 2.5 },
  { id: "rtx4090", label: "RTX 4090", creditsPerMin: 1.25 },
  { id: "rtx5090", label: "RTX 5090", creditsPerMin: 1.25 },
];

export const DEFAULT_CLOUD_GPU: CloudGpu = "h100";
export const CLOUD_GPU_STORAGE_KEY = "daydream-cloud-gpu";

const VALID_IDS: ReadonlySet<string> = new Set(CLOUD_GPUS.map(g => g.id));

function isCloudGpu(value: string | null): value is CloudGpu {
  return value !== null && VALID_IDS.has(value);
}

/** Read last-used GPU from localStorage, falling back to H100. */
export function getStoredCloudGpu(): CloudGpu {
  try {
    const raw = window.localStorage.getItem(CLOUD_GPU_STORAGE_KEY);
    if (isCloudGpu(raw)) return raw;
  } catch {
    // localStorage unavailable (private mode, SSR) — fall through
  }
  return DEFAULT_CLOUD_GPU;
}

/** Persist the chosen GPU to localStorage. */
export function setStoredCloudGpu(gpu: CloudGpu): void {
  try {
    window.localStorage.setItem(CLOUD_GPU_STORAGE_KEY, gpu);
  } catch {
    // noop
  }
}

export function cloudGpuLabel(gpu: CloudGpu): string {
  return CLOUD_GPUS.find(g => g.id === gpu)?.label ?? gpu;
}

export function formatCreditsPerMin(creditsPerMin: number): string {
  return `${creditsPerMin} credits / min`;
}

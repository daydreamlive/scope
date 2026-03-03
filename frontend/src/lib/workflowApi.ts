/**
 * TypeScript types and API wrappers for the workflow resolve endpoint,
 * plus the LoRA download endpoint.
 *
 * Types mirror the backend Pydantic models in:
 *   - scope.core.workflows.resolve
 *   - scope.server.lora_downloader
 *
 * Export is handled entirely client-side (see workflowSettings.ts).
 */

// ---------------------------------------------------------------------------
// Schema types (frontend-owned — the backend ignores extra fields)
// ---------------------------------------------------------------------------

export interface WorkflowLoRAProvenance {
  source: "huggingface" | "civitai" | "url" | "local";
  repo_id?: string | null;
  hf_filename?: string | null;
  model_id?: string | null;
  version_id?: string | null;
  url?: string | null;
}

export interface WorkflowLoRA {
  id?: string | null;
  filename: string;
  weight: number;
  merge_mode: string;
  provenance?: WorkflowLoRAProvenance | null;
  sha256?: string | null;
}

export interface WorkflowPipelineSource {
  type: "builtin" | "pypi" | "git" | "local";
  plugin_name?: string | null;
  plugin_version?: string | null;
  package_spec?: string | null;
}

export interface WorkflowPipeline {
  pipeline_id: string;
  pipeline_version?: string | null;
  source: WorkflowPipelineSource;
  loras: WorkflowLoRA[];
  params: Record<string, unknown>;
  role?: "preprocessor" | "main" | "postprocessor" | null;
}

export interface WorkflowPrompt {
  text: string;
  weight: number;
}

export interface WorkflowTimelineEntry {
  start_time: number;
  end_time: number;
  prompts: WorkflowPrompt[];
  transition_steps?: number | null;
  temporal_interpolation_method?: "linear" | "slerp" | null;
}

export interface WorkflowTimeline {
  entries: WorkflowTimelineEntry[];
}

export interface WorkflowMetadata {
  name: string;
  created_at: string;
  scope_version: string;
}

export interface ScopeWorkflow {
  format: "scope-workflow";
  format_version: string;
  metadata: WorkflowMetadata;
  pipelines: WorkflowPipeline[];
  timeline?: WorkflowTimeline | null;
  min_scope_version?: string | null;
  // Frontend-only fields (annotated post-backend, dropped by backend on validation)
  prompts?: WorkflowPrompt[];
  interpolation_method?: "linear" | "slerp" | null;
  transition_steps?: number | null;
  temporal_interpolation_method?: "linear" | "slerp" | null;
}

// ---------------------------------------------------------------------------
// Resolution types (scope.core.workflows.resolve)
// ---------------------------------------------------------------------------

export interface ResolutionItem {
  kind: "pipeline" | "plugin" | "lora";
  name: string;
  status: "ok" | "missing" | "version_mismatch";
  detail?: string | null;
  action?: string | null;
  can_auto_resolve: boolean;
}

export interface WorkflowResolutionPlan {
  can_apply: boolean;
  items: ResolutionItem[];
  warnings: string[];
}

// ---------------------------------------------------------------------------
// LoRA download types (scope.server.lora_downloader)
// ---------------------------------------------------------------------------

export interface LoRADownloadRequest {
  source: "huggingface" | "civitai" | "url";
  repo_id?: string | null;
  hf_filename?: string | null;
  model_id?: string | null;
  version_id?: string | null;
  url?: string | null;
  subfolder?: string | null;
  expected_sha256?: string | null;
}

export interface LoRADownloadResult {
  filename: string;
  path: string;
  sha256: string;
  size_bytes: number;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------

async function postJson<T>(
  url: string,
  body: unknown,
  label: string
): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`${label} failed: ${response.status} ${text}`);
  }
  return response.json();
}

// ---------------------------------------------------------------------------
// API functions
// ---------------------------------------------------------------------------

export function resolveWorkflow(
  workflow: ScopeWorkflow
): Promise<WorkflowResolutionPlan> {
  return postJson("/api/v1/workflow/resolve", workflow, "Resolution");
}

export function downloadLoRA(
  request: LoRADownloadRequest
): Promise<LoRADownloadResult> {
  return postJson("/api/v1/lora/download", request, "LoRA download");
}

/**
 * Bidirectional mapping between the frontend SettingsState and the
 * workflow export/import API types.
 *
 * Export direction: SettingsState -> ExportPipelineInput[] + WorkflowTimeline
 * Import direction: ScopeWorkflow -> partial SettingsState + TimelinePrompt[]
 */

import type { SettingsState, LoRAConfig, LoraMergeStrategy } from "../types";
import type { TimelinePrompt } from "../components/PromptTimeline";
import type { PromptItem } from "./api";
import type {
  ExportPipelineInput,
  ExportLoRAInput,
  WorkflowTimeline,
  WorkflowTimelineEntry,
  WorkflowPipeline,
  ScopeWorkflow,
} from "./workflowApi";

// ---------------------------------------------------------------------------
// Prompt state that lives outside SettingsState (separate React state vars)
// ---------------------------------------------------------------------------

export interface WorkflowPromptState {
  promptItems: PromptItem[];
  interpolationMethod: "linear" | "slerp";
  transitionSteps: number;
  temporalInterpolationMethod: "linear" | "slerp";
}

// ---------------------------------------------------------------------------
// Centralized param mapping: single source of truth for camelCase <-> snake_case
// ---------------------------------------------------------------------------

interface ParamMapping {
  /** SettingsState key (camelCase) */
  setting: keyof SettingsState;
  /** Backend param key (snake_case) */
  param: string;
  /** Expected typeof for import type-checking (skip if absent) */
  type?: "number" | "boolean" | "string";
  /** For import: restrict to specific allowed values */
  allowedValues?: readonly unknown[];
}

const PARAM_MAPPINGS: readonly ParamMapping[] = [
  { setting: "quantization", param: "quantization" },
  { setting: "denoisingSteps", param: "denoising_step_list" },
  { setting: "noiseScale", param: "noise_scale", type: "number" },
  { setting: "noiseController", param: "noise_controller", type: "boolean" },
  { setting: "manageCache", param: "manage_cache", type: "boolean" },
  {
    setting: "kvCacheAttentionBias",
    param: "kv_cache_attention_bias",
    type: "number",
  },
  { setting: "vaceEnabled", param: "vace_enabled", type: "boolean" },
  { setting: "vaceContextScale", param: "vace_context_scale", type: "number" },
  {
    setting: "vaceUseInputVideo",
    param: "vace_use_input_video",
    type: "boolean",
  },
  {
    setting: "inputMode",
    param: "input_mode",
    type: "string",
    allowedValues: ["text", "video"],
  },
] as const;

/** All snake_case param names that are explicitly mapped (plus resolution fields). */
const KNOWN_PARAMS = new Set([
  "height",
  "width",
  ...PARAM_MAPPINGS.map(m => m.param),
]);

// ---------------------------------------------------------------------------
// Export: SettingsState -> ExportPipelineInput[]
// ---------------------------------------------------------------------------

/**
 * Build the list of ExportPipelineInput objects from the current settings.
 *
 * Produces the full pipeline chain: preprocessors, then the main pipeline,
 * then postprocessors. The backend's `build_workflow` will filter params
 * against each pipeline's config schema.
 */
export function buildExportPipelines(
  settings: SettingsState
): ExportPipelineInput[] {
  const result: ExportPipelineInput[] = [];

  // --- Preprocessors ---
  for (const id of settings.preprocessorIds ?? []) {
    result.push({
      pipeline_id: id,
      params: { ...(settings.preprocessorSchemaFieldOverrides?.[id] ?? {}) },
      loras: [],
    });
  }

  // --- Main pipeline ---
  const params: Record<string, unknown> = {};

  // Resolution (compound mapping, handled separately)
  if (settings.resolution) {
    params.height = settings.resolution.height;
    params.width = settings.resolution.width;
  }

  // Simple 1:1 mappings
  for (const { setting, param } of PARAM_MAPPINGS) {
    const value = settings[setting];
    if (value !== undefined) {
      params[param] = value;
    }
  }

  // Schema-driven field overrides (already snake_case)
  if (settings.schemaFieldOverrides) {
    Object.assign(params, settings.schemaFieldOverrides);
  }

  // LoRAs
  const loras: ExportLoRAInput[] = (settings.loras ?? []).map(l => ({
    path: l.path,
    scale: l.scale,
    ...(l.mergeMode ? { merge_mode: l.mergeMode } : {}),
  }));

  result.push({
    pipeline_id: settings.pipelineId,
    params,
    loras,
  });

  // --- Postprocessors ---
  for (const id of settings.postprocessorIds ?? []) {
    result.push({
      pipeline_id: id,
      params: { ...(settings.postprocessorSchemaFieldOverrides?.[id] ?? {}) },
      loras: [],
    });
  }

  return result;
}

/**
 * Get the LoRA merge mode string for the export request.
 */
export function getExportMergeMode(settings: SettingsState): string {
  return settings.loraMergeStrategy ?? "permanent_merge";
}

// ---------------------------------------------------------------------------
// Export: Annotate a ScopeWorkflow with frontend-only metadata
// ---------------------------------------------------------------------------

/**
 * Annotate each pipeline in the workflow with its role (preprocessor/main/postprocessor)
 * based on the settings that produced the export request.
 */
export function annotateWorkflowRoles(
  workflow: ScopeWorkflow,
  settings: SettingsState
): void {
  const preCount = settings.preprocessorIds?.length ?? 0;
  const postCount = settings.postprocessorIds?.length ?? 0;
  for (let i = 0; i < workflow.pipelines.length; i++) {
    if (i < preCount) workflow.pipelines[i].role = "preprocessor";
    else if (i < preCount + 1) workflow.pipelines[i].role = "main";
    else if (i < preCount + 1 + postCount)
      workflow.pipelines[i].role = "postprocessor";
  }
}

/**
 * Annotate a workflow with the active prompt state so imports can restore
 * the current prompt blend independent of the timeline.
 */
export function annotateWorkflowPromptState(
  workflow: ScopeWorkflow,
  promptState: WorkflowPromptState
): void {
  if (promptState.promptItems.length > 0) {
    workflow.prompts = promptState.promptItems.map(p => ({
      text: p.text,
      weight: p.weight,
    }));
  }
  workflow.interpolation_method = promptState.interpolationMethod;
  workflow.transition_steps = promptState.transitionSteps;
  workflow.temporal_interpolation_method =
    promptState.temporalInterpolationMethod;
}

// ---------------------------------------------------------------------------
// Export: TimelinePrompt[] -> WorkflowTimeline
// ---------------------------------------------------------------------------

/**
 * Convert frontend TimelinePrompt[] into the WorkflowTimeline schema.
 * Returns null if there are no meaningful timeline entries.
 */
export function buildWorkflowTimeline(
  prompts: TimelinePrompt[]
): WorkflowTimeline | null {
  const entries: WorkflowTimelineEntry[] = prompts
    .filter(p => p.startTime !== p.endTime) // skip zero-length
    .map(p => {
      // Build the prompts array; fall back to the single `text` field
      const wPrompts = p.prompts?.length
        ? p.prompts.map(pp => ({ text: pp.text, weight: pp.weight }))
        : p.text
          ? [{ text: p.text, weight: 1.0 }]
          : [];

      const entry: WorkflowTimelineEntry = {
        start_time: p.startTime,
        end_time: p.endTime,
        prompts: wPrompts,
      };
      if (p.transitionSteps != null) {
        entry.transition_steps = p.transitionSteps;
      }
      if (p.temporalInterpolationMethod) {
        entry.temporal_interpolation_method = p.temporalInterpolationMethod;
      }
      return entry;
    });

  if (entries.length === 0) return null;
  return { entries };
}

// ---------------------------------------------------------------------------
// Import: ScopeWorkflow -> partial SettingsState
// ---------------------------------------------------------------------------

/**
 * Extract pipeline IDs and schema field overrides from a list of
 * processor pipelines (preprocessors or postprocessors).
 */
function extractProcessorSettings(pipelines: WorkflowPipeline[]): {
  ids: string[];
  overrides: Record<string, Record<string, unknown>> | undefined;
} {
  if (pipelines.length === 0) return { ids: [], overrides: undefined };

  const ids = pipelines.map(pp => pp.pipeline_id);
  const overrides: Record<string, Record<string, unknown>> = {};
  for (const pp of pipelines) {
    if (Object.keys(pp.params).length > 0) {
      overrides[pp.pipeline_id] = { ...pp.params };
    }
  }
  return {
    ids,
    overrides: Object.keys(overrides).length > 0 ? overrides : undefined,
  };
}

/**
 * Map a workflow's pipelines back to a partial SettingsState that can be
 * merged via `updateSettings()`.
 *
 * Pipelines are split by their `role` annotation: "preprocessor", "main",
 * and "postprocessor". For backward compatibility with older workflow files
 * that lack roles, the first pipeline is treated as the main pipeline.
 *
 * NOTE: LoRA paths are set to their filename only. The caller should
 * resolve full paths after the workflow is applied and LoRAs are present.
 */
export function workflowToSettings(
  workflow: ScopeWorkflow
): Partial<SettingsState> {
  if (workflow.pipelines.length === 0) return {};

  // Split pipelines by role, with backward-compat fallback
  const hasRoles = workflow.pipelines.some(p => p.role);

  let mainPipeline: WorkflowPipeline;
  let preprocessors: WorkflowPipeline[];
  let postprocessors: WorkflowPipeline[];

  if (hasRoles) {
    mainPipeline =
      workflow.pipelines.find(p => p.role === "main") ?? workflow.pipelines[0];
    preprocessors = workflow.pipelines.filter(p => p.role === "preprocessor");
    postprocessors = workflow.pipelines.filter(p => p.role === "postprocessor");
  } else {
    // Legacy: first pipeline is main, no pre/post processors
    mainPipeline = workflow.pipelines[0];
    preprocessors = [];
    postprocessors = [];
  }

  const p = mainPipeline.params;
  const partial: Partial<SettingsState> = {
    pipelineId: mainPipeline.pipeline_id,
  };

  // Resolution (compound mapping, handled separately)
  if (typeof p.height === "number" && typeof p.width === "number") {
    partial.resolution = { height: p.height, width: p.width };
  }

  // Simple 1:1 mappings
  for (const mapping of PARAM_MAPPINGS) {
    const value = p[mapping.param];
    if (value === undefined) continue;
    if (mapping.type && typeof value !== mapping.type) continue;
    if (mapping.allowedValues && !mapping.allowedValues.includes(value))
      continue;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (partial as any)[mapping.setting] = value;
  }

  // LoRAs
  if (mainPipeline.loras.length > 0) {
    partial.loras = mainPipeline.loras.map(
      (l): LoRAConfig => ({
        id: l.id ?? l.filename,
        path: l.filename, // filename-only; resolved after apply
        scale: l.weight,
        mergeMode: l.merge_mode as LoraMergeStrategy | undefined,
      })
    );
    // Use the first LoRA's merge_mode as the global strategy
    const mode = mainPipeline.loras[0].merge_mode;
    if (mode === "permanent_merge" || mode === "runtime_peft") {
      partial.loraMergeStrategy = mode;
    }
  }

  // Collect remaining unknown params into schemaFieldOverrides
  const overrides: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(p)) {
    if (!KNOWN_PARAMS.has(key)) {
      overrides[key] = value;
    }
  }
  if (Object.keys(overrides).length > 0) {
    partial.schemaFieldOverrides = overrides;
  }

  // Preprocessors
  const pre = extractProcessorSettings(preprocessors);
  if (pre.ids.length > 0) {
    partial.preprocessorIds = pre.ids;
    if (pre.overrides) partial.preprocessorSchemaFieldOverrides = pre.overrides;
  }

  // Postprocessors
  const post = extractProcessorSettings(postprocessors);
  if (post.ids.length > 0) {
    partial.postprocessorIds = post.ids;
    if (post.overrides)
      partial.postprocessorSchemaFieldOverrides = post.overrides;
  }

  return partial;
}

// ---------------------------------------------------------------------------
// Import: WorkflowTimeline -> TimelinePrompt[]
// ---------------------------------------------------------------------------

/**
 * Convert a WorkflowTimeline back to frontend TimelinePrompt[].
 */
export function workflowTimelineToPrompts(
  timeline: WorkflowTimeline | null | undefined
): TimelinePrompt[] {
  if (!timeline?.entries.length) return [];

  return timeline.entries.map((entry): TimelinePrompt => {
    const id = crypto.randomUUID();
    const mainText = entry.prompts.length > 0 ? entry.prompts[0].text : "";

    return {
      id,
      text: mainText,
      startTime: entry.start_time,
      endTime: entry.end_time,
      prompts:
        entry.prompts.length > 0
          ? entry.prompts.map(p => ({ text: p.text, weight: p.weight }))
          : undefined,
      transitionSteps: entry.transition_steps ?? undefined,
      temporalInterpolationMethod:
        entry.temporal_interpolation_method ?? undefined,
    };
  });
}

// ---------------------------------------------------------------------------
// Import: ScopeWorkflow -> WorkflowPromptState
// ---------------------------------------------------------------------------

/**
 * Extract the active prompt state from a workflow.
 *
 * Uses the top-level `prompts` / `interpolation_method` fields if present.
 * Falls back to the first timeline entry's prompts for older workflow files.
 * Returns null if no prompt state can be determined.
 */
export function workflowToPromptState(
  workflow: ScopeWorkflow
): WorkflowPromptState | null {
  const base = {
    interpolationMethod: workflow.interpolation_method ?? "linear",
    transitionSteps:
      typeof workflow.transition_steps === "number"
        ? workflow.transition_steps
        : 4,
    temporalInterpolationMethod:
      workflow.temporal_interpolation_method ?? "slerp",
  } as const;

  // Prefer explicit top-level prompt state
  if (workflow.prompts && workflow.prompts.length > 0) {
    return {
      ...base,
      promptItems: workflow.prompts.map(p => ({
        text: p.text,
        weight: p.weight,
      })),
    };
  }

  // Fallback: derive from first timeline entry
  const firstEntry = workflow.timeline?.entries?.[0];
  if (firstEntry && firstEntry.prompts.length > 0) {
    return {
      ...base,
      promptItems: firstEntry.prompts.map(p => ({
        text: p.text,
        weight: p.weight,
      })),
    };
  }

  return null;
}

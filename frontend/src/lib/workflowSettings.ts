/**
 * Bidirectional mapping between the frontend SettingsState and the
 * ScopeWorkflow schema.
 *
 * Export direction: SettingsState -> ScopeWorkflow (built entirely client-side)
 * Import direction: ScopeWorkflow -> partial SettingsState + TimelinePrompt[]
 */

import type {
  SettingsState,
  LoRAConfig,
  LoraMergeStrategy,
  PipelineInfo,
} from "../types";
import type { TimelinePrompt } from "../components/PromptTimeline";
import type { PromptItem, LoRAFileInfo, PluginInfo } from "./api";
import type {
  WorkflowPipeline,
  WorkflowPipelineSource,
  WorkflowLoRA,
  WorkflowLoRAProvenance,
  WorkflowTimeline,
  WorkflowTimelineEntry,
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
// Export helpers (private)
// ---------------------------------------------------------------------------

/** Extract just the filename from a full file path. */
function extractFilename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

/** Determine the pipeline source (builtin vs plugin). */
function buildPipelineSource(
  pipelineId: string,
  pipelineInfoMap: Record<string, PipelineInfo>,
  pluginInfoMap: Map<string, PluginInfo>
): WorkflowPipelineSource {
  const info = pipelineInfoMap[pipelineId];
  if (!info?.pluginName) {
    return { type: "builtin" };
  }

  const plugin = pluginInfoMap.get(info.pluginName);
  if (!plugin) {
    return { type: "builtin" };
  }

  return {
    type: plugin.source,
    plugin_name: plugin.name,
    plugin_version: plugin.version ?? null,
    package_spec: plugin.package_spec ?? null,
  };
}

/** Convert LoRAConfig[] to WorkflowLoRA[] with sha256/provenance enrichment. */
function buildWorkflowLoRAs(
  loraConfigs: LoRAConfig[],
  loraFiles: LoRAFileInfo[],
  mergeStrategy: string
): WorkflowLoRA[] {
  return loraConfigs.map(lora => {
    // Try to find the matching LoRA file for enrichment.
    // Match on full path first, then fall back to stem name comparison.
    const filename = extractFilename(lora.path);
    const matched =
      loraFiles.find(f => f.path === lora.path) ??
      loraFiles.find(
        f => extractFilename(f.path).toLowerCase() === filename.toLowerCase()
      );

    const result: WorkflowLoRA = {
      id: lora.id,
      filename,
      weight: lora.scale,
      merge_mode: lora.mergeMode ?? mergeStrategy,
    };

    if (matched?.sha256) {
      result.sha256 = matched.sha256;
    }
    if (matched?.provenance) {
      const p = matched.provenance;
      const prov: WorkflowLoRAProvenance = { source: p.source };
      if (p.repo_id != null) prov.repo_id = p.repo_id;
      if (p.hf_filename != null) prov.hf_filename = p.hf_filename;
      if (p.model_id != null) prov.model_id = p.model_id;
      if (p.version_id != null) prov.version_id = p.version_id;
      if (p.url != null) prov.url = p.url;
      result.provenance = prov;
    }

    return result;
  });
}

/** Extract main pipeline params from SettingsState using PARAM_MAPPINGS. */
function buildMainPipelineParams(
  settings: SettingsState
): Record<string, unknown> {
  const params: Record<string, unknown> = {};

  // Resolution (compound mapping)
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

  return params;
}

// ---------------------------------------------------------------------------
// Export: Build the complete ScopeWorkflow client-side
// ---------------------------------------------------------------------------

export interface BuildScopeWorkflowInput {
  name: string;
  settings: SettingsState;
  timelinePrompts: TimelinePrompt[];
  promptState: WorkflowPromptState;
  pipelineInfoMap: Record<string, PipelineInfo>;
  loraFiles: LoRAFileInfo[];
  pluginInfoMap: Map<string, PluginInfo>;
  scopeVersion: string;
}

/**
 * Assemble the full ScopeWorkflow entirely client-side.
 *
 * Enrichment data comes from:
 * - pipelineInfoMap: cached PipelinesContext (version, pluginName)
 * - loraFiles: cached LoRAsContext (sha256, provenance)
 * - pluginInfoMap: fetched on-demand via listPlugins()
 * - scopeVersion: fetched on-demand via getServerInfo()
 */
export function buildScopeWorkflow(
  input: BuildScopeWorkflowInput
): ScopeWorkflow {
  const {
    name,
    settings,
    timelinePrompts,
    promptState,
    pipelineInfoMap,
    loraFiles,
    pluginInfoMap,
    scopeVersion,
  } = input;

  const mergeStrategy = settings.loraMergeStrategy ?? "permanent_merge";

  // --- Build pipeline list ---
  const pipelines: WorkflowPipeline[] = [];

  // Preprocessors
  for (const id of settings.preprocessorIds ?? []) {
    pipelines.push({
      pipeline_id: id,
      pipeline_version: pipelineInfoMap[id]?.version ?? null,
      source: buildPipelineSource(id, pipelineInfoMap, pluginInfoMap),
      loras: [],
      params: { ...(settings.preprocessorSchemaFieldOverrides?.[id] ?? {}) },
      role: "preprocessor",
    });
  }

  // Main pipeline
  pipelines.push({
    pipeline_id: settings.pipelineId,
    pipeline_version: pipelineInfoMap[settings.pipelineId]?.version ?? null,
    source: buildPipelineSource(
      settings.pipelineId,
      pipelineInfoMap,
      pluginInfoMap
    ),
    loras: buildWorkflowLoRAs(settings.loras ?? [], loraFiles, mergeStrategy),
    params: buildMainPipelineParams(settings),
    role: "main",
  });

  // Postprocessors
  for (const id of settings.postprocessorIds ?? []) {
    pipelines.push({
      pipeline_id: id,
      pipeline_version: pipelineInfoMap[id]?.version ?? null,
      source: buildPipelineSource(id, pipelineInfoMap, pluginInfoMap),
      loras: [],
      params: { ...(settings.postprocessorSchemaFieldOverrides?.[id] ?? {}) },
      role: "postprocessor",
    });
  }

  // --- Build timeline ---
  const timeline = buildWorkflowTimeline(timelinePrompts);

  // --- Assemble workflow ---
  const workflow: ScopeWorkflow = {
    format: "scope-workflow",
    format_version: "1.0",
    metadata: {
      name,
      created_at: new Date().toISOString(),
      scope_version: scopeVersion,
    },
    pipelines,
    timeline,
  };

  // Annotate prompt state
  annotateWorkflowPromptState(workflow, promptState);

  return workflow;
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

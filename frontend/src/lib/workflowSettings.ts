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
import type { GraphConfig } from "./api";
import type {
  WorkflowPipeline,
  WorkflowPipelineSource,
  WorkflowLoRA,
  WorkflowTimeline,
  WorkflowTimelineEntry,
  WorkflowPrompt,
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

/** Strip prompt arrays down to just {text, weight}. */
const toPromptItems = (prompts: { text: string; weight: number }[]) =>
  prompts.map(p => ({ text: p.text, weight: p.weight }));

// ---------------------------------------------------------------------------
// Centralized param mapping: single source of truth for camelCase <-> snake_case
//
// These exist because some pipeline params have dedicated SettingsState fields
// (camelCase) rather than flowing through the generic schemaFieldOverrides
// bucket (already snake_case). Params NOT listed here are handled automatically
// via schemaFieldOverrides on both export and import — no mapping needed.
//
// If the dedicated SettingsState fields are ever migrated to
// schemaFieldOverrides, this mapping can be removed entirely.
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
export function extractFilename(path: string): string {
  const parts = path.split(/[/\\]/);
  return parts[parts.length - 1] || path;
}

/** Resolve a LoRA filename to the full path from the available files list. */
export function resolveLoRAPath(
  filename: string,
  availableLoRAs: LoRAFileInfo[]
): string {
  // Already a known full path
  const exact = availableLoRAs.find(f => f.path === filename);
  if (exact) return exact.path;

  // Match by basename (with extension)
  const basename = extractFilename(filename).toLowerCase();
  const byBasename = availableLoRAs.find(
    f => extractFilename(f.path).toLowerCase() === basename
  );
  if (byBasename) return byBasename.path;

  // Match by stem (without extension) — handles f.name which is the stem
  const byStem = availableLoRAs.find(
    f =>
      f.name.toLowerCase() === basename ||
      f.name.toLowerCase() === filename.toLowerCase()
  );
  if (byStem) return byStem.path;

  return filename;
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
      result.provenance = {
        source: p.source,
        ...(p.repo_id != null && { repo_id: p.repo_id }),
        ...(p.hf_filename != null && { hf_filename: p.hf_filename }),
        ...(p.model_id != null && { model_id: p.model_id }),
        ...(p.version_id != null && { version_id: p.version_id }),
        ...(p.url != null && { url: p.url }),
      };
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

/**
 * Extract LoRA configurations from ui_state LoRA nodes and their edges
 * to pipeline nodes. Returns a map of pipelineNodeId -> LoRA entries.
 */
function extractLoRAsFromUiState(
  uiState: Record<string, unknown> | undefined
): Map<string, Array<{ path: string; scale: number; mergeMode?: string }>> {
  const result = new Map<
    string,
    Array<{ path: string; scale: number; mergeMode?: string }>
  >();
  if (!uiState) return result;

  const uiNodes = (uiState.nodes ?? []) as Array<{
    id: string;
    data?: { nodeType?: string; loras?: unknown; loraMergeMode?: string };
  }>;
  const uiEdges = (uiState.edges ?? []) as Array<{
    source: string;
    target: string;
    sourceHandle?: string;
    targetHandle?: string;
  }>;

  const loraNodes = new Map(
    uiNodes.filter(n => n.data?.nodeType === "lora").map(n => [n.id, n])
  );

  for (const edge of uiEdges) {
    if (
      !edge.sourceHandle?.includes("__loras") ||
      !edge.targetHandle?.includes("__loras")
    )
      continue;

    const loraNode = loraNodes.get(edge.source);
    if (!loraNode?.data?.loras) continue;

    const entries = loraNode.data.loras as Array<{
      path: string;
      scale: number;
      mergeMode?: string;
    }>;
    const validEntries = entries.filter(l => l.path);
    if (validEntries.length > 0) {
      result.set(edge.target, validEntries);
    }
  }

  return result;
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
  const buildProcessorPipelines = (
    ids: string[],
    role: "preprocessor" | "postprocessor",
    overridesMap?: Record<string, Record<string, unknown>>
  ): WorkflowPipeline[] =>
    ids.map(id => ({
      pipeline_id: id,
      pipeline_version: pipelineInfoMap[id]?.version ?? null,
      source: buildPipelineSource(id, pipelineInfoMap, pluginInfoMap),
      loras: [],
      params: { ...(overridesMap?.[id] ?? {}) },
      role,
    }));

  const pipelines: WorkflowPipeline[] = [
    ...buildProcessorPipelines(
      settings.preprocessorIds ?? [],
      "preprocessor",
      settings.preprocessorSchemaFieldOverrides
    ),
    {
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
    },
    ...buildProcessorPipelines(
      settings.postprocessorIds ?? [],
      "postprocessor",
      settings.postprocessorSchemaFieldOverrides
    ),
  ];

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
    timeline: buildWorkflowTimeline(timelinePrompts),
    prompts:
      promptState.promptItems.length > 0
        ? toPromptItems(promptState.promptItems)
        : undefined,
    interpolation_method: promptState.interpolationMethod,
    transition_steps: promptState.transitionSteps,
    temporal_interpolation_method: promptState.temporalInterpolationMethod,
  };

  return workflow;
}

// ---------------------------------------------------------------------------
// Export: GraphConfig -> ScopeWorkflow (with embedded graph)
// ---------------------------------------------------------------------------

export interface BuildGraphWorkflowInput {
  name: string;
  graphConfig: GraphConfig;
  pipelineInfoMap: Record<string, PipelineInfo>;
  pluginInfoMap: Map<string, PluginInfo>;
  scopeVersion: string;
  loraFiles?: LoRAFileInfo[];
}

/**
 * Build a ScopeWorkflow from a graph-mode GraphConfig.
 *
 * The resulting workflow embeds the full graph topology in the `graph` field
 * so graph mode can fully restore it. It also populates the `pipelines`
 * array so perform mode (and the backend resolution API) can consume the
 * same file without needing the graph.
 */
export function buildGraphWorkflow(
  input: BuildGraphWorkflowInput
): ScopeWorkflow {
  const {
    name,
    graphConfig,
    pipelineInfoMap,
    pluginInfoMap,
    scopeVersion,
    loraFiles = [],
  } = input;

  const uiState = graphConfig.ui_state as Record<string, unknown> | undefined;
  const nodeParams = uiState?.node_params as
    | Record<string, Record<string, unknown>>
    | undefined;

  // Build a map: pipelineNodeId -> LoRA entries from LoRA nodes in ui_state
  const lorasByPipeline = extractLoRAsFromUiState(uiState);

  const pipelineNodes = graphConfig.nodes.filter(n => n.type === "pipeline");

  const pipelines: WorkflowPipeline[] = pipelineNodes.map(node => {
    const pipelineId = node.pipeline_id ?? "";
    const bag = nodeParams?.[node.id] ?? {};

    const params: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(bag)) {
      if (key === "__prompt") continue;
      params[key] = value;
    }

    const loraEntries = lorasByPipeline.get(node.id);
    const loras: WorkflowLoRA[] = loraEntries
      ? buildWorkflowLoRAs(
          loraEntries.map(l => ({
            id: `lora-${loraEntries.indexOf(l)}`,
            path: l.path,
            scale: l.scale,
            mergeMode: l.mergeMode as LoraMergeStrategy | undefined,
          })),
          loraFiles,
          "permanent_merge"
        )
      : [];

    return {
      pipeline_id: pipelineId,
      pipeline_version: pipelineInfoMap[pipelineId]?.version ?? null,
      source: buildPipelineSource(pipelineId, pipelineInfoMap, pluginInfoMap),
      loras,
      params,
    };
  });

  // Extract prompts: use __prompt from the first pipeline node that has one
  const prompts: WorkflowPrompt[] = [];
  for (const node of pipelineNodes) {
    const text = nodeParams?.[node.id]?.__prompt as string | undefined;
    if (text) {
      prompts.push({ text, weight: 1.0 });
      break;
    }
  }

  return {
    format: "scope-workflow",
    format_version: "1.0",
    metadata: {
      name,
      created_at: new Date().toISOString(),
      scope_version: scopeVersion,
    },
    pipelines,
    prompts: prompts.length > 0 ? prompts : undefined,
    graph: graphConfig,
  };
}

// ---------------------------------------------------------------------------
// Export: TimelinePrompt[] -> WorkflowTimeline
// ---------------------------------------------------------------------------

/**
 * Convert frontend TimelinePrompt[] into the WorkflowTimeline schema.
 * Returns null if there are no meaningful timeline entries.
 */
function buildWorkflowTimeline(
  prompts: TimelinePrompt[]
): WorkflowTimeline | null {
  const entries: WorkflowTimelineEntry[] = prompts
    .filter(p => p.startTime !== p.endTime) // skip zero-length
    .map(p => {
      // Build the prompts array; fall back to the single `text` field
      const wPrompts = p.prompts?.length
        ? toPromptItems(p.prompts)
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
  workflow: ScopeWorkflow,
  availableLoRAs: LoRAFileInfo[] = []
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
    const VALID_MERGE_MODES: LoraMergeStrategy[] = [
      "permanent_merge",
      "runtime_peft",
    ];
    partial.loras = mainPipeline.loras.map(
      (l): LoRAConfig => ({
        id: l.id ?? l.filename,
        path: resolveLoRAPath(l.filename, availableLoRAs),
        scale: l.weight,
        mergeMode: VALID_MERGE_MODES.includes(l.merge_mode as LoraMergeStrategy)
          ? (l.merge_mode as LoraMergeStrategy)
          : undefined,
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
        entry.prompts.length > 0 ? toPromptItems(entry.prompts) : undefined,
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

  // Prefer first timeline entry since that's what plays first
  const firstEntry = workflow.timeline?.entries?.[0];
  if (firstEntry && firstEntry.prompts.length > 0) {
    return {
      ...base,
      promptItems: toPromptItems(firstEntry.prompts),
    };
  }

  // Fallback: use top-level prompt state
  if (workflow.prompts && workflow.prompts.length > 0) {
    return {
      ...base,
      promptItems: toPromptItems(workflow.prompts),
    };
  }

  return null;
}

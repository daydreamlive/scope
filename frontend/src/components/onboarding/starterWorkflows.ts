// ---------------------------------------------------------------------------
// Starter workflow definitions with embedded .scope-workflow JSON
// ---------------------------------------------------------------------------

// Images live in public/assets/onboarding/ — reference by URL, not module import
const kubakubThumb = "/assets/onboarding/kubakub-butterfly.webp";
const dissolvingCatThumb = "/assets/onboarding/dissolving-cat.webp";
const pixelArtThumb = "/assets/onboarding/pixel-art.png";

/**
 * Build a graph field from a pipelines array so the graph editor can render
 * the workflow. Layout: input → pipeline1 → pipeline2 → ... → output
 */
function buildGraph(
  pipelines: Array<{ pipeline_id: string }>
) {
  const Y = 200;
  const GAP = 300;
  let x = 50;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const nodes: any[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const edges: any[] = [];

  // Source
  nodes.push({ id: "input", type: "source", x, y: Y });
  x += GAP;

  let prev = "input";
  for (const p of pipelines) {
    nodes.push({ id: p.pipeline_id, type: "pipeline", pipeline_id: p.pipeline_id, x, y: Y });
    edges.push({ from: prev, from_port: "video", to_node: p.pipeline_id, to_port: "video", kind: "stream" });
    prev = p.pipeline_id;
    x += GAP;
  }

  // Sink
  nodes.push({ id: "output", type: "sink", x, y: Y });
  edges.push({ from: prev, from_port: "video", to_node: "output", to_port: "video", kind: "stream" });

  return { nodes, edges };
}

export interface StarterWorkflow {
  id: string;
  title: string;
  category: string;
  description: string;
  color: string;
  thumbnail: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  workflow: Record<string, any>;
}

export const STARTER_WORKFLOWS: StarterWorkflow[] = [
  {
    id: "starter-style-lora",
    title: "Kubakub Butterfly Abstract",
    category: "Style LoRA",
    description:
      "Apply the Kubakub style LoRA for abstract, dissolving butterfly visuals. Pure text-to-video — no input source required.",
    color: "#a78bfa",
    thumbnail: kubakubThumb,
    workflow: {
      format: "scope-workflow",
      prompts: [
        {
          text: "abstract butterfly made of Kubakub dissolve",
          weight: 100,
        },
      ],
      metadata: {
        name: "Kubakub Butterfly Abstract",
        created_at: "2026-03-09T13:27:43.977Z",
        scope_version: "0.1.6",
      },
      timeline: {
        entries: [
          {
            prompts: [
              {
                text: "abstract butterfly made of Kubakub dissolve",
                weight: 100,
              },
            ],
            end_time: 71.47109999999404,
            start_time: 0,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
        ],
      },
      pipelines: [
        {
          role: "main",
          loras: [
            {
              id: "a28b8bd2-81f0-4b99-92fe-2bda347ad5f2",
              sha256:
                "bc5f39b3a6e55fcbf4f3c84806cb37b324996adb0d2a6ee9e9b9789e23948515",
              weight: 0.5,
              filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              merge_mode: "permanent_merge",
              provenance: {
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "ff72452c-b344-4004-bd69-4237687ad5bc",
              sha256:
                "c68897f7f50f6ab3e4370810b6c7cdfb1bf47eeb9e69aaac1998e76a57b58ea2",
              weight: 1,
              filename:
                "Kubakub_v1_Wan2.1_1-3B_t2v_torchoptadam80epochs.safetensors",
              merge_mode: "permanent_merge",
              provenance: {
                url: "https://civitai.com/api/download/models/1787596",
                source: "civitai",
                version_id: "1787596",
              },
            },
            {
              id: "2ecf282d-4007-45fc-9dbc-c8583e6f6c91",
              sha256:
                "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
              weight: 0.3,
              filename: "daydream-scope-dissolve.safetensors",
              merge_mode: "permanent_merge",
              provenance: {
                source: "civitai",
                version_id: "2680702",
              },
            },
          ],
          params: {
            width: 512,
            height: 512,
            input_mode: "text",
            noise_scale: 0.7,
            manage_cache: false,
            quantization: null,
            vace_enabled: true,
            noise_controller: true,
            vace_context_scale: 0.8,
            denoising_step_list: [1000, 750, 500, 250],
            vace_use_input_video: false,
            kv_cache_attention_bias: 0.3,
          },
          source: { type: "builtin" },
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
        },
        {
          role: "postprocessor",
          loras: [],
          params: {},
          source: { type: "builtin" },
          pipeline_id: "rife",
          pipeline_version: "1.0.0",
        },
      ],
      format_version: "1.0",
      transition_steps: 0,
      interpolation_method: "linear",
      temporal_interpolation_method: "slerp",
      graph: buildGraph([{ pipeline_id: "longlive" }, { pipeline_id: "rife" }]),
    },
  },
  {
    id: "starter-ref-image",
    title: "Dissolving Cat Flowers",
    category: "Reference Image Extension",
    description:
      "Use a depth-mapped reference video to guide generation with the dissolve style. Great for extending existing footage.",
    color: "#4ade80",
    thumbnail: dissolvingCatThumb,
    workflow: {
      format: "scope-workflow",
      prompts: [
        {
          text: "abstract dissolving flowers made of ral-dissolve swaying in the wind\n",
          weight: 100,
        },
      ],
      metadata: {
        name: "Dissolving Cat Flowers",
        created_at: "2026-03-05T19:48:25.876Z",
        scope_version: "0.1.5",
      },
      timeline: {
        entries: [
          {
            prompts: [
              {
                text: "A high resolution ral-dissolve scene. A **cat** sitting in the ral-dissolve, looking around in abstract dissolving particles",
                weight: 100,
              },
            ],
            end_time: 6.508999999999999,
            start_time: 0,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
          {
            prompts: [
              {
                text: "paint pour of ral-dissolve feline multicolored paint everywhere",
                weight: 1,
              },
            ],
            end_time: 12.344099999994034,
            start_time: 6.508999999999999,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
          {
            prompts: [
              {
                text: "abstract dissolving flowers made of ral-dissolve\n",
                weight: 1,
              },
            ],
            end_time: 34.0945,
            start_time: 12.344099999994034,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
          {
            prompts: [
              {
                text: "abstract dissolving flowers made of ral-dissolve swaying in the wind\n",
                weight: 100,
              },
            ],
            end_time: 35.27520000001788,
            start_time: 34.0945,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
        ],
      },
      pipelines: [
        {
          role: "preprocessor",
          loras: [],
          params: {},
          source: { type: "builtin" },
          pipeline_id: "video-depth-anything",
          pipeline_version: "1.0.0",
        },
        {
          role: "main",
          loras: [
            {
              id: "bda90241-8b74-40f7-9233-925a4328ad95",
              sha256:
                "bc5f39b3a6e55fcbf4f3c84806cb37b324996adb0d2a6ee9e9b9789e23948515",
              weight: 0.7,
              filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              merge_mode: "permanent_merge",
              provenance: {
                url: "https://huggingface.co/daydreamlive/Wan2.1-1.3b-lora-highresfix/resolve/main/Wan2.1-1.3b-lora-highresfix-v1_new.safetensors?download=true",
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "22717753-df3b-4de0-bd5f-614926a21a6f",
              sha256:
                "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
              weight: 1.5,
              filename: "daydream-scope-dissolve.safetensors",
              merge_mode: "permanent_merge",
              provenance: {
                source: "civitai",
                version_id: "2680702",
              },
            },
          ],
          params: {
            width: 512,
            height: 512,
            input_mode: "video",
            noise_scale: 0.7,
            manage_cache: false,
            quantization: null,
            vace_enabled: true,
            noise_controller: true,
            vace_context_scale: 0.5,
            denoising_step_list: [1000, 750],
            vace_use_input_video: true,
            kv_cache_attention_bias: 0.3,
          },
          source: { type: "builtin" },
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
        },
        {
          role: "postprocessor",
          loras: [],
          params: {},
          source: { type: "builtin" },
          pipeline_id: "rife",
          pipeline_version: "1.0.0",
        },
      ],
      format_version: "1.0",
      transition_steps: 0,
      interpolation_method: "linear",
      temporal_interpolation_method: "slerp",
      graph: buildGraph([
        { pipeline_id: "video-depth-anything" },
        { pipeline_id: "longlive" },
        { pipeline_id: "rife" },
      ]),
    },
  },
  {
    id: "starter-inpainting",
    title: "Pixel Art, Preserved Background",
    category: "Inpainting",
    description:
      "Use YOLO masking to preserve your background while transforming subjects into pixel art style.",
    color: "#60a5fa",
    thumbnail: pixelArtThumb,
    workflow: {
      format: "scope-workflow",
      prompts: [
        {
          text: "pixel art scene, 2d",
          weight: 100,
        },
      ],
      metadata: {
        name: "Pixel Art, Preserved Background",
        created_at: "2026-03-06T18:06:54.484Z",
        scope_version: "0.1.6",
      },
      timeline: {
        entries: [
          {
            prompts: [
              {
                text: "pixel art scene, 2d",
                weight: 100,
              },
            ],
            end_time: 75.49359999990463,
            start_time: 0,
            transition_steps: 0,
            temporal_interpolation_method: "slerp",
          },
        ],
      },
      pipelines: [
        {
          role: "preprocessor",
          loras: [],
          params: {},
          source: {
            type: "git",
            plugin_name: "scope-yolo-mask",
            package_spec: "git+https://github.com/daydreamlive/scope_yolo_mask",
            plugin_version: "0.1.0",
          },
          pipeline_id: "yolo_mask",
          pipeline_version: "1.0.0",
        },
        {
          role: "main",
          loras: [
            {
              id: "1889086a-9554-4549-8b53-cc53e58ec547",
              sha256:
                "2074e3fc23d7039bfa78d140337f720f5417ad1857b95326ac757243fd6f0607",
              weight: 1.6,
              filename: "[flux.2.klein]pixelart_redmond-000032.safetensors",
              merge_mode: "runtime_peft",
              provenance: {
                url: "https://civitai.com/api/download/models/2724902",
                source: "civitai",
                version_id: "2724902",
              },
            },
          ],
          params: {
            width: 512,
            height: 512,
            input_mode: "video",
            noise_scale: 0.7,
            manage_cache: true,
            quantization: null,
            vace_enabled: true,
            noise_controller: true,
            denoising_step_list: [1000, 858, 748, 550],
            vace_use_input_video: true,
            kv_cache_attention_bias: 0.3,
          },
          source: { type: "builtin" },
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
        },
      ],
      format_version: "1.0",
      transition_steps: 0,
      interpolation_method: "linear",
      temporal_interpolation_method: "slerp",
      graph: buildGraph([
        { pipeline_id: "yolo_mask" },
        { pipeline_id: "longlive" },
      ]),
    },
  },
];

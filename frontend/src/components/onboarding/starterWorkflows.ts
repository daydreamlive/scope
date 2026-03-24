// ---------------------------------------------------------------------------
// Starter workflow definitions with embedded .scope-workflow JSON
// ---------------------------------------------------------------------------

// Images live in public/assets/onboarding/ — reference by URL, not module import
const mythicalCreatureThumb = "/assets/onboarding/mythical-creature.png";
const dissolvingCatThumb = "/assets/onboarding/dissolving-cat.webp";
const pixelArtThumb = "/assets/onboarding/pixel-art.png";

export interface StarterWorkflow {
  id: string;
  title: string;
  category: string;
  description: string;
  color: string;
  thumbnail: string;
  /** Which onboarding style this workflow belongs to. */
  onboardingStyle: "teaching" | "simple" | "both";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  workflow: Record<string, any>;
}

/** Return workflows filtered for the given onboarding style. */
export function getWorkflowsForStyle(
  style: "teaching" | "simple" | null,
): StarterWorkflow[] {
  if (!style) return STARTER_WORKFLOWS;
  return STARTER_WORKFLOWS.filter(
    (wf) => wf.onboardingStyle === style || wf.onboardingStyle === "both",
  );
}

export const STARTER_WORKFLOWS: StarterWorkflow[] = [
  {
    id: "starter-mythical-creature",
    title: "Mythical Creature",
    category: "Style LoRA",
    description:
      "Morphing slime portal with acid-lime and dissolve LoRAs — creates vivid, otherworldly creature visuals from a video input.",
    color: "#a78bfa",
    thumbnail: mythicalCreatureThumb,
    onboardingStyle: "teaching",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Mythical Creature",
        created_at: "2026-03-20T22:12:35.393Z",
        scope_version: "0.1.8",
      },
      pipelines: [
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "lora-0",
              filename: "diffslime_acidzlime-000016.safetensors",
              weight: 1.5,
              merge_mode: "permanent_merge",
              provenance: {
                source: "civitai",
                version_id: "2704300",
              },
            },
            {
              id: "lora-1",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 1.5,
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
            vace_context_scale: 0.4,
            denoising_step_list: [1000, 750],
            vace_use_input_video: true,
            kv_cache_attention_bias: 0.3,
          },
        },
        {
          pipeline_id: "passthrough",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
      ],
      prompts: [
        {
          text: "god morphing outward infinitely into ral-acidzlime, morphing slime portal of of ral-dissolve abstract dissolving particles of multicolored slime",
          weight: 1,
        },
      ],
      graph: {
        nodes: [
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 342.27,
            y: 173.73,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "passthrough",
            x: 650,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 971.58,
            y: 187.50,
            w: 299,
            h: 238,
          },
        ],
        edges: [
          {
            from: "longlive",
            from_port: "video",
            to_node: "rife",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "rife",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119.22, y: 528.92 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/diffslime_acidzlime-000016.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "/tmp/.daydream-scope/assets/lora/daydream-scope-dissolve.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                ],
                loraMergeMode: "permanent_merge",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -117.71, y: 333.09 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "note",
              type: "note",
              position: { x: -131.51, y: -143.94 },
              width: 253,
              height: 203,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'WELCOME TO SCOPE\n\nThis educational workflow will walk you through some core concepts.\n\nFirst, let\'s add an input source. Right click on the canvas and search "Source".\n\nClick on the Source node to add it.',
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_1",
              type: "note",
              position: { x: 335.52, y: -144.70 },
              width: 200,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'Next,  click and hold the white dot on the Source , and connect it to the white dot next to "VACE Frames" on the LongLive  node. \n\nThis will supply the video source to the node where real-time generation takes place.',
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_2",
              type: "note",
              position: { x: 1002.67, y: -141.91 },
              width: 241,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Finally, click the Play button in the upper left corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_3",
              type: "note",
              position: { x: 1443.84, y: -143.21 },
              width: 320,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Extension Ideas\n\n- Add an Output node after Sink to connect your generation to Syphon\n- Add a MIDI node, and connect it to denoise steps on the LongLive node\n- Click the plug icon in the upper right, and experiment with installing community nodes",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_4",
              type: "note",
              position: { x: 666.73, y: -143.74 },
              width: 227,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'In the node below, click the dropdown and select "rife"\n\nRife is a node that interpolates between frames, creating smoother output video',
                locked: false,
                pinned: true,
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1006.75, y: 458.87 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_flags: {
            note: { locked: true, pinned: true },
            note_1: { locked: true, pinned: true },
            note_2: { locked: true, pinned: true },
            note_3: { locked: true, pinned: true },
            note_4: { pinned: true },
          },
          node_params: {
            longlive: {
              width: 512,
              height: 512,
              input_mode: "video",
              noise_scale: 0.7,
              manage_cache: false,
              quantization: null,
              vace_enabled: true,
              noise_controller: true,
              vace_context_scale: 0.4,
              denoising_step_list: [1000, 750],
              vace_use_input_video: true,
              kv_cache_attention_bias: 0.3,
              __prompt:
                "god morphing outward infinitely into ral-acidzlime, morphing slime portal of of ral-dissolve abstract dissolving particles of multicolored slime",
            },
          },
        },
      },
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
    onboardingStyle: "teaching",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Dissolving Cat Flowers",
        created_at: "2026-03-20T22:33:38.001Z",
        scope_version: "0.1.8",
      },
      prompts: [
        {
          text: "abstract dissolving flowers made of ral-dissolve swaying in the wind",
          weight: 1,
        },
      ],
      pipelines: [
        {
          pipeline_id: "video-depth-anything",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "lora-0",
              filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              weight: 0.7,
              merge_mode: "permanent_merge",
              provenance: {
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename:
                  "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "lora-1",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 1.5,
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
        },
        {
          pipeline_id: "passthrough",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
      ],
      graph: {
        nodes: [
          {
            id: "depth",
            type: "pipeline",
            pipeline_id: "video-depth-anything",
            x: 50,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 342,
            y: 174,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "passthrough",
            x: 650,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 972,
            y: 188,
            w: 299,
            h: 238,
          },
        ],
        edges: [
          {
            from: "longlive",
            from_port: "video",
            to_node: "rife",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "rife",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "depth",
            from_port: "video",
            to_node: "longlive",
            to_port: "vace_input_frames",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119, y: 529 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
                    scale: 0.7,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "/tmp/.daydream-scope/assets/lora/daydream-scope-dissolve.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                ],
                loraMergeMode: "permanent_merge",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -118, y: 333 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "note",
              type: "note",
              position: { x: -132, y: -144 },
              width: 253,
              height: 203,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "WELCOME TO SCOPE\n\nThis educational workflow will walk you through some core concepts.\n\nFirst, let's add an input source. Right click on the canvas and search \"Source\".\n\nClick on the Source node to add it.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_1",
              type: "note",
              position: { x: 336, y: -145 },
              width: 200,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Next, click and hold the white dot on the Source, and connect it to the Depth Anything node.\n\nThis preprocessor extracts depth information from your video, which guides the generation to follow the structure of your original footage.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_2",
              type: "note",
              position: { x: 1003, y: -142 },
              width: 241,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Finally, click the Play button in the upper left corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_3",
              type: "note",
              position: { x: 1444, y: -143 },
              width: 320,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Extension Ideas\n\n- Add an Output node after Sink to connect your generation to Syphon\n- Add a MIDI node, and connect it to denoise steps on the LongLive node\n- Click the plug icon in the upper right, and experiment with installing community nodes",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_4",
              type: "note",
              position: { x: 667, y: -144 },
              width: 227,
              height: 149,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'Next, click the dropdown and change it from "passthrough" to "rife" \n\nRife interpolates between frames, creating smoother output video.\n',
                locked: false,
                pinned: true,
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1007, y: 459 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_flags: {
            note: { locked: true, pinned: true },
            note_1: { locked: true, pinned: true },
            note_2: { locked: true, pinned: true },
            note_3: { locked: true, pinned: true },
            note_4: { pinned: true },
          },
          node_params: {
            longlive: {
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
              __prompt:
                "abstract dissolving flowers made of ral-dissolve swaying in the wind",
            },
          },
        },
      },
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
    onboardingStyle: "teaching",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Pixel Art, Preserved Background",
        created_at: "2026-03-20T22:31:18.153Z",
        scope_version: "0.1.8",
      },
      prompts: [
        {
          text: "pixel art scene, 2d",
          weight: 1,
        },
      ],
      pipelines: [
        {
          pipeline_id: "yolo_mask",
          pipeline_version: "1.0.0",
          source: {
            type: "git",
            plugin_name: "scope-yolo-mask",
            plugin_version: "0.1.0",
            package_spec: "git+https://github.com/daydreamlive/scope_yolo_mask",
          },
          loras: [],
          params: {},
        },
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "lora-0",
              filename: "[flux.2.klein]pixelart_redmond-000032.safetensors",
              weight: 1.6,
              merge_mode: "runtime_peft",
              provenance: {
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
        },
        {
          pipeline_id: "passthrough",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
      ],
      graph: {
        nodes: [
          {
            id: "yolo_mask",
            type: "pipeline",
            pipeline_id: "yolo_mask",
            x: 40.02,
            y: 158.22,
            w: 240,
            h: 114,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 379.12,
            y: 179.30,
            w: 240,
            h: 684,
          },
          {
            id: "output",
            type: "sink",
            x: 944.15,
            y: 170.29,
            w: 299,
            h: 238,
          },
          {
            id: "pipeline",
            type: "pipeline",
            pipeline_id: "passthrough",
            x: 659.47,
            y: 202.25,
            w: 240,
            h: 114,
          },
        ],
        edges: [
          {
            from: "yolo_mask",
            from_port: "video",
            to_node: "longlive",
            to_port: "vace_input_frames",
            kind: "stream",
          },
          {
            from: "longlive",
            from_port: "video",
            to_node: "pipeline",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "pipeline",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119, y: 529 },
              width: 240,
              height: 191,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/[flux.2.klein]pixelart_redmond-000032.safetensors",
                    scale: 1.6,
                    mergeMode: "runtime_peft",
                  },
                ],
                loraMergeMode: "runtime_peft",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -118, y: 333 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "note",
              type: "note",
              position: { x: -318.29, y: -150.75 },
              width: 253,
              height: 203,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "WELCOME TO SCOPE\n\nThis educational workflow will walk you through some core concepts.\n\nFirst, let's add an input source. Right click on the canvas and search \"Source\".\n\nClick on the Source node to add it.",
                locked: false,
                pinned: false,
              },
            },
            {
              id: "note_1",
              type: "note",
              position: { x: 44, y: -145 },
              width: 220,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Next, click the white dot on the Source node and drag it to connect the Source to the YOLO Mask node.\n\nThis preprocessor detects objects in your video and creates a mask, preserving your background while only transforming the detected subjects.",
                locked: false,
                pinned: true,
              },
            },
            {
              id: "note_2",
              type: "note",
              position: { x: 956.80, y: -128.58 },
              width: 241,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Finally, click the Play button in the upper left corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
                locked: false,
                pinned: false,
              },
            },
            {
              id: "note_3",
              type: "note",
              position: { x: 1319.52, y: -137.70 },
              width: 320,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Extension Ideas\n\n- Add an Output node after Sink to connect your generation to Syphon\n- Add a MIDI node, and connect it to denoise steps on the LongLive node\n- Click the plug icon in the upper right, and experiment with installing community nodes",
                locked: false,
                pinned: false,
              },
            },
            {
              id: "note_4",
              type: "note",
              position: { x: 640.87, y: -127.56 },
              width: 227,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'Next, click the dropdown and change "passthrough" to "rife".\n\nRife interpolates between frames for a smoother video',
                locked: false,
                pinned: false,
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1007.34, y: 453.06 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_flags: {
            note_1: { pinned: true },
          },
          node_params: {
            longlive: {
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
              __prompt: "pixel art scene, 2d",
            },
          },
        },
      },
    },
  },

  // -----------------------------------------------------------------------
  // Simple-mode workflows (with graph UI state, no teaching notes)
  // -----------------------------------------------------------------------
  {
    id: "simple-mythical-creature",
    title: "Mythical Creature",
    category: "Style LoRA",
    description:
      "Morphing slime portal with acid-lime and dissolve LoRAs — creates vivid, otherworldly creature visuals from a video input.",
    color: "#a78bfa",
    thumbnail: mythicalCreatureThumb,
    onboardingStyle: "simple",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Mythical Creature",
        created_at: "2026-03-09T11:39:11.961Z",
        scope_version: "0.1.6",
      },
      prompts: [
        {
          text: "god morphing outward infinitely into ral-acidzlime, morphing slime portal of of ral-dissolve abstract dissolving particles of multicolored slime",
          weight: 1,
        },
      ],
      pipelines: [
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "dd6e92b8-fdfd-4336-9d6e-2b9442772a20",
              sha256: "a4028744227d95ca03eb0db1a0906dc34d84356d44ab4778348ceb35661ec94a",
              filename: "diffslime_acidzlime-000016.safetensors",
              weight: 1.5,
              merge_mode: "permanent_merge",
              provenance: {
                url: "https://civitai.com/api/download/models/2704300",
                source: "civitai",
                version_id: "2704300",
              },
            },
            {
              id: "5e3c2671-81f3-4d14-a12b-8457258c48c4",
              sha256: "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 1.5,
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
            vace_context_scale: 0.4,
            denoising_step_list: [1000, 750],
            vace_use_input_video: true,
            kv_cache_attention_bias: 0.3,
          },
        },
        {
          pipeline_id: "rife",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
      ],
      graph: {
        nodes: [
          {
            id: "input",
            type: "source",
            source_mode: "video",
            x: 50,
            y: 200,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 342.27,
            y: 173.73,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "rife",
            x: 650,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 971.58,
            y: 187.5,
            w: 299,
            h: 238,
          },
        ],
        edges: [
          {
            from: "input",
            from_port: "video",
            to_node: "longlive",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "longlive",
            from_port: "video",
            to_node: "rife",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "rife",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119.22, y: 528.92 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/diffslime_acidzlime-000016.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "/tmp/.daydream-scope/assets/lora/daydream-scope-dissolve.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                ],
                loraMergeMode: "permanent_merge",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -117.71, y: 333.09 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1006.75, y: 458.87 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_params: {
            longlive: {
              width: 512,
              height: 512,
              input_mode: "video",
              noise_scale: 0.7,
              manage_cache: false,
              quantization: null,
              vace_enabled: true,
              noise_controller: true,
              vace_context_scale: 0.4,
              denoising_step_list: [1000, 750],
              vace_use_input_video: true,
              kv_cache_attention_bias: 0.3,
              __prompt:
                "god morphing outward infinitely into ral-acidzlime, morphing slime portal of of ral-dissolve abstract dissolving particles of multicolored slime",
            },
          },
        },
      },
    },
  },
  {
    id: "simple-ref-image",
    title: "Kubakub Butterfly Abstract",
    category: "Style LoRA",
    description:
      "Abstract butterfly visuals using Kubakub and dissolve LoRAs with high-res fix — text-to-video generation with rich organic style.",
    color: "#4ade80",
    thumbnail: dissolvingCatThumb,
    onboardingStyle: "simple",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Kubakub Butterfly Abstract",
        created_at: "2026-03-09T13:27:43.977Z",
        scope_version: "0.1.6",
      },
      prompts: [
        {
          text: "abstract butterfly made of Kubakub dissolve",
          weight: 1,
        },
      ],
      pipelines: [
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "a28b8bd2-81f0-4b99-92fe-2bda347ad5f2",
              sha256: "bc5f39b3a6e55fcbf4f3c84806cb37b324996adb0d2a6ee9e9b9789e23948515",
              filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              weight: 0.5,
              merge_mode: "permanent_merge",
              provenance: {
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "ff72452c-b344-4004-bd69-4237687ad5bc",
              sha256: "c68897f7f50f6ab3e4370810b6c7cdfb1bf47eeb9e69aaac1998e76a57b58ea2",
              filename: "Kubakub_v1_Wan2.1_1-3B_t2v_torchoptadam80epochs.safetensors",
              weight: 1,
              merge_mode: "permanent_merge",
              provenance: {
                url: "https://civitai.com/api/download/models/1787596",
                source: "civitai",
                version_id: "1787596",
              },
            },
            {
              id: "2ecf282d-4007-45fc-9dbc-c8583e6f6c91",
              sha256: "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 0.3,
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
        },
        {
          pipeline_id: "rife",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [],
          params: {},
        },
      ],
      graph: {
        nodes: [
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 342.27,
            y: 173.73,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "rife",
            x: 650,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 971.58,
            y: 187.5,
            w: 299,
            h: 238,
          },
        ],
        edges: [
          {
            from: "longlive",
            from_port: "video",
            to_node: "rife",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "rife",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119.22, y: 528.92 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
                    scale: 0.5,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "/tmp/.daydream-scope/assets/lora/Kubakub_v1_Wan2.1_1-3B_t2v_torchoptadam80epochs.safetensors",
                    scale: 1,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "/tmp/.daydream-scope/assets/lora/daydream-scope-dissolve.safetensors",
                    scale: 0.3,
                    mergeMode: "permanent_merge",
                  },
                ],
                loraMergeMode: "permanent_merge",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -117.71, y: 333.09 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1006.75, y: 458.87 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_params: {
            longlive: {
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
              __prompt:
                "abstract butterfly made of Kubakub dissolve",
            },
          },
        },
      },
    },
  },
  {
    id: "simple-inpainting",
    title: "Pixel Art, Preserved Background",
    category: "Inpainting",
    description:
      "Use YOLO masking to preserve your background while transforming subjects into pixel art style.",
    color: "#60a5fa",
    thumbnail: pixelArtThumb,
    onboardingStyle: "simple",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Pixel Art, Preserved Background",
        created_at: "2026-03-06T18:06:54.484Z",
        scope_version: "0.1.6",
      },
      prompts: [
        {
          text: "pixel art scene, 2d",
          weight: 1,
        },
      ],
      pipelines: [
        {
          pipeline_id: "yolo_mask",
          pipeline_version: "1.0.0",
          source: {
            type: "git",
            plugin_name: "scope-yolo-mask",
            plugin_version: "0.1.0",
            package_spec: "git+https://github.com/daydreamlive/scope_yolo_mask",
          },
          loras: [],
          params: {},
        },
        {
          pipeline_id: "longlive",
          pipeline_version: "1.0.0",
          source: { type: "builtin" },
          loras: [
            {
              id: "1889086a-9554-4549-8b53-cc53e58ec547",
              sha256: "2074e3fc23d7039bfa78d140337f720f5417ad1857b95326ac757243fd6f0607",
              filename: "[flux.2.klein]pixelart_redmond-000032.safetensors",
              weight: 1.6,
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
        },
      ],
      graph: {
        nodes: [
          {
            id: "input",
            type: "source",
            source_mode: "video",
            x: -200,
            y: 200,
          },
          {
            id: "yolo_mask",
            type: "pipeline",
            pipeline_id: "yolo_mask",
            x: 40.02,
            y: 158.22,
            w: 240,
            h: 114,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 379.12,
            y: 179.3,
            w: 240,
            h: 684,
          },
          {
            id: "output",
            type: "sink",
            x: 944.15,
            y: 170.29,
            w: 299,
            h: 238,
          },
        ],
        edges: [
          {
            from: "input",
            from_port: "video",
            to_node: "yolo_mask",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "input",
            from_port: "video",
            to_node: "longlive",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "yolo_mask",
            from_port: "video",
            to_node: "longlive",
            to_port: "vace_input_frames",
            kind: "stream",
          },
          {
            from: "longlive",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "lora-0",
              type: "lora",
              position: { x: -119, y: 529 },
              width: 240,
              height: 191,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "/tmp/.daydream-scope/assets/lora/[flux.2.klein]pixelart_redmond-000032.safetensors",
                    scale: 1.6,
                    mergeMode: "runtime_peft",
                  },
                ],
                loraMergeMode: "runtime_peft",
              },
            },
            {
              id: "vace",
              type: "vace",
              position: { x: -118, y: 333 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 1,
                vaceRefImage: "",
                vaceFirstFrame: "",
                vaceLastFrame: "",
                vaceVideo: "",
                parameterOutputs: [
                  { name: "__vace", type: "string", defaultValue: "" },
                ],
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1007.34, y: 453.06 },
              width: 180,
              height: 95,
              data: {
                label: "Record",
                nodeType: "record",
                parameterInputs: [
                  { name: "trigger", type: "boolean", defaultValue: false },
                ],
                isStreaming: false,
              },
            },
          ],
          edges: [
            {
              id: "e-lora-0-longlive",
              source: "lora-0",
              sourceHandle: "param:__loras",
              target: "longlive",
              targetHandle: "param:__loras",
            },
            {
              id: "xy-edge__vaceparam:__vace-longliveparam:__vace",
              source: "vace",
              sourceHandle: "param:__vace",
              target: "longlive",
              targetHandle: "param:__vace",
            },
          ],
          node_params: {
            longlive: {
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
              __prompt: "pixel art scene, 2d",
            },
          },
        },
      },
    },
  },
];

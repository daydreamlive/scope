// ---------------------------------------------------------------------------
// Starter workflow definitions with embedded .scope-workflow JSON
// ---------------------------------------------------------------------------

// Images live in public/assets/onboarding/ — reference by URL, not module import
const mythicalCreatureThumb = "/assets/onboarding/mythical-creature.png";
const dissolvingSunflowerThumb = "/assets/onboarding/dissolving-sunflower.png";
const ltxTextToVideoThumb = "/assets/onboarding/ltx-text-to-video.png";

export interface StarterWorkflow {
  id: string;
  title: string;
  category: string;
  description: string;
  color: string;
  thumbnail: string;
  /** Which onboarding style this workflow belongs to. */
  onboardingStyle: "teaching" | "simple" | "local" | "both";
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  workflow: Record<string, any>;
}

/** Return workflows filtered for the given onboarding style. */
export function getWorkflowsForStyle(
  style: "teaching" | "simple" | "local" | null
): StarterWorkflow[] {
  // Default to teaching workflows when no style is set
  const effectiveStyle = style ?? "teaching";
  return STARTER_WORKFLOWS.filter(
    wf => wf.onboardingStyle === effectiveStyle || wf.onboardingStyle === "both"
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
                    path: "diffslime_acidzlime-000016.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "daydream-scope-dissolve.safetensors",
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
              position: { x: 335.52, y: -144.7 },
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
                  "Finally, click the Run button in the upper right corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
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
            {
              id: "xy-edge__rifestream:video-recordstream:video",
              source: "rife",
              sourceHandle: "stream:video",
              target: "record",
              targetHandle: "stream:video",
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
    title: "Dissolving Sunflower",
    category: "Depth Map",
    description:
      "A dissolving sunflower in abstract particles using depth estimation and dissolve LoRA — transforms your camera feed into dreamy, particle-filled visuals.",
    color: "#4ade80",
    thumbnail: dissolvingSunflowerThumb,
    onboardingStyle: "teaching",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Dissolving Sunflower",
        created_at: "2026-03-25T21:25:43.546Z",
        scope_version: "0.1.9",
      },
      prompts: [
        {
          text: "A high resolution ral-dissolve scene. A **sunflower** sitting in the ral-dissolve, looking around in abstract dissolving particles",
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
              sha256:
                "bc5f39b3a6e55fcbf4f3c84806cb37b324996adb0d2a6ee9e9b9789e23948515",
              provenance: {
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "lora-1",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 1.5,
              merge_mode: "permanent_merge",
              sha256:
                "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
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
            id: "video-depth-anything",
            type: "pipeline",
            pipeline_id: "video-depth-anything",
            x: 350,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 650,
            y: 200,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "passthrough",
            x: 950,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 1250,
            y: 200,
            w: 240,
            h: 200,
          },
        ],
        edges: [
          {
            from: "video-depth-anything",
            from_port: "video",
            to_node: "longlive",
            to_port: "vace_input_frames",
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
              position: { x: 90.6, y: 823.8 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
                    scale: 0.7,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "daydream-scope-dissolve.safetensors",
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
              position: { x: 97.5, y: 533 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 0.5,
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
                  'WELCOME TO SCOPE\n\nThis educational workflow will walk you through some core concepts.\n\nFirst, let\'s add an input source. Right click on the canvas and search "Source".\n\nClick on the Source node to add it.',
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_1",
              type: "note",
              position: { x: 100, y: -145 },
              width: 220,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Next, click the white dot on the Source node and drag it to connect the Source to the Depth Anything node.\n\nThis preprocessor estimates depth in your video, preserving 3D structure while the AI stylizes the output.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_2",
              type: "note",
              position: { x: 1280, y: -128.58 },
              width: 241,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Finally, click the Run button in the upper right corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "note_3",
              type: "note",
              position: { x: 1600, y: -137.7 },
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
              position: { x: 950, y: -127.56 },
              width: 227,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  'Next, click the dropdown and change "passthrough" to "rife".\n\nRife interpolates between frames for a smoother video',
                locked: false,
                pinned: true,
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1330, y: 553.06 },
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
            {
              id: "xy-edge__rifestream:video-recordstream:video",
              source: "rife",
              sourceHandle: "stream:video",
              target: "record",
              targetHandle: "stream:video",
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
                "A high resolution ral-dissolve scene. A **sunflower** sitting in the ral-dissolve, looking around in abstract dissolving particles",
            },
          },
        },
      },
    },
  },
  {
    id: "starter-ltx-text-to-video",
    title: "LTX 2.3",
    category: "Text to Video",
    description:
      "Generate continuous video and audio from text prompts or reference media.",
    color: "#60a5fa",
    thumbnail: ltxTextToVideoThumb,
    onboardingStyle: "teaching",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "LTX-example",
        created_at: "2026-04-02T17:31:32.537Z",
        scope_version: "0.2.0",
      },
      pipelines: [
        {
          pipeline_id: "ltx2",
          pipeline_version: "0.3.0",
          source: {
            type: "git",
            plugin_name: "scope-ltx-2",
            plugin_version: "0.3.0",
            package_spec:
              "git+https://github.com/daydreamlive/scope-ltx-2",
          },
          loras: [],
          params: {
            height: 384,
            width: 320,
            quantization: null,
            manage_cache: true,
            kv_cache_attention_bias: 0.3,
            input_mode: "text",
            prompt_interpolation_method: "linear",
          },
        },
      ],
      prompts: [
        {
          text: "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
          weight: 1,
        },
      ],
      graph: {
        nodes: [
          {
            id: "ltx2",
            type: "pipeline",
            pipeline_id: "ltx2",
            x: 295.44,
            y: 178.88,
            w: 240,
            h: 650,
          },
          {
            id: "output",
            type: "sink",
            x: 658.8,
            y: 182.4,
            w: 500,
            h: 451,
          },
        ],
        edges: [
          {
            from: "ltx2",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "primitive",
              type: "primitive",
              position: { x: -198.95, y: 509.3 },
              width: 240,
              height: 135,
              data: {
                label: "Prompt Output",
                nodeType: "primitive",
                valueType: "string",
                value:
                  "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
                parameterOutputs: [
                  { name: "value", type: "string", defaultValue: "" },
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
                  "WELCOME TO SCOPE\n\nThis workflow generates video from a text prompt using LTX 2.3.\n\nThe Primitive node below contains the prompt text. Try changing it to generate different scenes.",
                locked: false,
                pinned: false,
              },
            },
            {
              id: "note_1",
              type: "note",
              position: { x: 658.8, y: -128.58 },
              width: 241,
              height: 131,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Click the Run button in the upper right corner.\n\nAfter 30-45 seconds, generation will begin playing in the Sink node.",
                locked: false,
                pinned: false,
              },
            },
            {
              id: "note_2",
              type: "note",
              position: { x: 1000, y: -137.7 },
              width: 320,
              height: 185,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "Extension Ideas\n\n- Add an Output node after Sink to connect your generation to Syphon\n- Click the plug icon in the upper right to install community nodes\n- Try adding a Source node and switching to video input mode for real-time style transfer",
                locked: false,
                pinned: false,
              },
            },
          ],
          edges: [
            {
              id: "xy-edge__primitiveparam:value-ltx2param:__prompt",
              source: "primitive",
              sourceHandle: "param:value",
              target: "ltx2",
              targetHandle: "param:__prompt",
            },
          ],
          node_params: {
            ltx2: {
              height: 384,
              width: 320,
              quantization: null,
              manage_cache: true,
              kv_cache_attention_bias: 0.3,
              input_mode: "text",
              prompt_interpolation_method: "linear",
              __prompt:
                "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
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
              sha256:
                "a4028744227d95ca03eb0db1a0906dc34d84356d44ab4778348ceb35661ec94a",
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
              sha256:
                "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
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
                    path: "diffslime_acidzlime-000016.safetensors",
                    scale: 1.5,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "daydream-scope-dissolve.safetensors",
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
            {
              id: "xy-edge__rifestream:video-recordstream:video",
              source: "rife",
              sourceHandle: "stream:video",
              target: "record",
              targetHandle: "stream:video",
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
    title: "Dissolving Sunflower",
    category: "Depth Map",
    description:
      "A dissolving sunflower in abstract particles using depth estimation and dissolve LoRA — transforms your camera feed into dreamy, particle-filled visuals.",
    color: "#4ade80",
    thumbnail: dissolvingSunflowerThumb,
    onboardingStyle: "simple",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Dissolving Sunflower",
        created_at: "2026-03-25T21:25:43.546Z",
        scope_version: "0.1.9",
      },
      prompts: [
        {
          text: "A high resolution ral-dissolve scene. A **sunflower** sitting in the ral-dissolve, looking around in abstract dissolving particles",
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
              sha256:
                "bc5f39b3a6e55fcbf4f3c84806cb37b324996adb0d2a6ee9e9b9789e23948515",
              provenance: {
                source: "huggingface",
                repo_id: "daydreamlive/Wan2.1-1.3b-lora-highresfix",
                hf_filename: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
              },
            },
            {
              id: "lora-1",
              filename: "daydream-scope-dissolve.safetensors",
              weight: 1.5,
              merge_mode: "permanent_merge",
              sha256:
                "fd373e0991a33df28f6d0d4a13d8553e2c9625483e309e8ec952a96a2570bec9",
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
            x: 34.1,
            y: 218.17,
          },
          {
            id: "video-depth-anything",
            type: "pipeline",
            pipeline_id: "video-depth-anything",
            x: 350,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "longlive",
            type: "pipeline",
            pipeline_id: "longlive",
            x: 650,
            y: 200,
            w: 240,
            h: 684,
          },
          {
            id: "rife",
            type: "pipeline",
            pipeline_id: "rife",
            x: 950,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 1250,
            y: 200,
            w: 240,
            h: 200,
          },
        ],
        edges: [
          {
            from: "input",
            from_port: "video",
            to_node: "video-depth-anything",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "video-depth-anything",
            from_port: "video",
            to_node: "longlive",
            to_port: "vace_input_frames",
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
              position: { x: 90.6, y: 823.8 },
              width: 240,
              height: 293,
              data: {
                label: "LoRA",
                nodeType: "lora",
                loras: [
                  {
                    path: "Wan2.1-1.3b-lora-highresfix-v1_new.safetensors",
                    scale: 0.7,
                    mergeMode: "permanent_merge",
                  },
                  {
                    path: "daydream-scope-dissolve.safetensors",
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
              position: { x: 97.5, y: 533 },
              width: 240,
              height: 178,
              data: {
                label: "VACE",
                nodeType: "vace",
                vaceContextScale: 0.5,
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
              position: { x: 1330, y: 553.06 },
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
            {
              id: "xy-edge__rifestream:video-recordstream:video",
              source: "rife",
              sourceHandle: "stream:video",
              target: "record",
              targetHandle: "stream:video",
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
              vace_context_scale: 0.5,
              denoising_step_list: [1000, 750],
              vace_use_input_video: true,
              kv_cache_attention_bias: 0.3,
              __prompt:
                "A high resolution ral-dissolve scene. A **sunflower** sitting in the ral-dissolve, looking around in abstract dissolving particles",
            },
          },
        },
      },
    },
  },
  {
    id: "simple-ltx-text-to-video",
    title: "LTX 2.3",
    category: "Text to Video",
    description:
      "Generate continuous video and audio from text prompts or reference media.",
    color: "#60a5fa",
    thumbnail: ltxTextToVideoThumb,
    onboardingStyle: "simple",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "LTX-example",
        created_at: "2026-04-02T17:31:32.537Z",
        scope_version: "0.2.0",
      },
      pipelines: [
        {
          pipeline_id: "ltx2",
          pipeline_version: "0.3.0",
          source: {
            type: "git",
            plugin_name: "scope-ltx-2",
            plugin_version: "0.3.0",
            package_spec:
              "git+https://github.com/daydreamlive/scope-ltx-2",
          },
          loras: [],
          params: {
            height: 384,
            width: 320,
            quantization: null,
            manage_cache: true,
            kv_cache_attention_bias: 0.3,
            input_mode: "text",
            prompt_interpolation_method: "linear",
          },
        },
      ],
      prompts: [
        {
          text: "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
          weight: 1,
        },
      ],
      graph: {
        nodes: [
          {
            id: "ltx2",
            type: "pipeline",
            pipeline_id: "ltx2",
            x: 295.44,
            y: 178.88,
            w: 240,
            h: 650,
          },
          {
            id: "output",
            type: "sink",
            x: 658.8,
            y: 182.4,
            w: 500,
            h: 451,
          },
        ],
        edges: [
          {
            from: "ltx2",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "primitive",
              type: "primitive",
              position: { x: -198.95, y: 509.3 },
              width: 240,
              height: 135,
              data: {
                label: "Prompt Output",
                nodeType: "primitive",
                valueType: "string",
                value:
                  "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
                parameterOutputs: [
                  { name: "value", type: "string", defaultValue: "" },
                ],
              },
            },
          ],
          edges: [
            {
              id: "xy-edge__primitiveparam:value-ltx2param:__prompt",
              source: "primitive",
              sourceHandle: "param:value",
              target: "ltx2",
              targetHandle: "param:__prompt",
            },
          ],
          node_params: {
            ltx2: {
              height: 384,
              width: 320,
              quantization: null,
              manage_cache: true,
              kv_cache_attention_bias: 0.3,
              input_mode: "text",
              prompt_interpolation_method: "linear",
              __prompt:
                "A majestic lion prowling through dense jungle undergrowth, slow deliberate steps forward, low tracking camera following at ground level, dappled sunlight filtering through canopy, warm golden tones, shallow depth of field, nature documentary cinematography.",
            },
          },
        },
      },
    },
  },

  // -----------------------------------------------------------------------
  // Local-mode workflow (no GPU required — runs on CPU)
  // -----------------------------------------------------------------------
  {
    id: "local-passthrough",
    title: "Camera Preview",
    category: "Getting Started",
    description:
      "A simple camera passthrough to verify your setup works. Most real-time video generation workflows require a GPU with at least 24GB of VRAM.",
    color: "#94a3b8",
    thumbnail: "",
    onboardingStyle: "local",
    workflow: {
      format: "scope-workflow",
      format_version: "1.0",
      metadata: {
        name: "Camera Check",
        created_at: "2026-03-26T16:25:35.191Z",
        scope_version: "0.1.9",
      },
      pipelines: [
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
            id: "input",
            type: "source",
            source_mode: "camera",
            x: 50,
            y: 200,
            w: 240,
            h: 200,
          },
          {
            id: "passthrough",
            type: "pipeline",
            pipeline_id: "passthrough",
            x: 350,
            y: 200,
            w: 240,
            h: 114,
          },
          {
            id: "output",
            type: "sink",
            x: 650,
            y: 200,
            w: 240,
            h: 200,
          },
        ],
        edges: [
          {
            from: "input",
            from_port: "video",
            to_node: "passthrough",
            to_port: "video",
            kind: "stream",
          },
          {
            from: "passthrough",
            from_port: "video",
            to_node: "output",
            to_port: "video",
            kind: "stream",
          },
        ],
        ui_state: {
          nodes: [
            {
              id: "note",
              type: "note",
              position: { x: 50, y: -150 },
              width: 350,
              height: 203,
              data: {
                label: "Note",
                nodeType: "note",
                noteText:
                  "WELCOME TO SCOPE\n\nThis is a simple camera passthrough to verify your setup.\n\nMost real-time AI video workflows require a GPU with at least 24GB of VRAM. If you don't have a compatible GPU, we recommend connecting to Daydream Cloud instead.\n\nYou can switch to Cloud mode anytime in Settings.",
                locked: true,
                pinned: true,
              },
            },
            {
              id: "record",
              type: "record",
              position: { x: 1023.65, y: 216.15 },
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
              id: "xy-edge__outputstream:out-recordstream:video",
              source: "output",
              sourceHandle: "stream:out",
              target: "record",
              targetHandle: "stream:video",
            },
          ],
          node_flags: {
            note: { locked: true, pinned: true },
          },
        },
      },
    },
  },
];

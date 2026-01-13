# RewardForcing

[RewardForcing](https://reward-forcing.github.io/) is a streaming pipeline and autoregressive video diffusion model from ZJU, Ant Group, SIAS-ZJU, HUST and SJTU.

The model is trained with Rewarded Distribution Matching Distillation using Wan2.1 1.3b as the base model.

## Examples

The following examples include timeline JSON files with the prompts used so you can try them as well.

https://github.com/user-attachments/assets/b47afba2-9689-48a5-81cc-3b7b944aca0f

[Timeline JSON File](./examples/timeline-reward-forcing-ink.json)

https://github.com/user-attachments/assets/c19a3d17-2cde-40f5-9166-a2cb077a09c3

[Timeline JSON File](./examples/timeline-reward-forcing-melt.json)

## Resolution

The generation will be faster for smaller resolutions resulting in smoother video. The visual quality will be better at 832x480 which is the resolution that the model was trained on, but you may need a more powerful GPU in order to achieve a higher FPS.

## Seed

The seed parameter in the UI can be used to reproduce generations. If you like the generation for a certain seed value and sequence of prompts you can re-use that value later with those same prompts to reproduce the generation.

## Prompting

**Subject and Background/Setting Anchors**

The model works better if you include a subject (who/what) and background/setting (where) in each prompt. If you want continuity in the next scene then you can continue referencing the same subject and/or background/setting.

For example:

"A 3D animated scene. A **panda** walks along a path towards the camera in a park on a spring day."

"A 3D animated scene. A **panda** halts along a path in a park on a spring day."

**Cinematic Long Takes**

The model works better for scene transitions that involve long cinematic long takes and works less well with rapid shot-by-shot transitions or fast cutscenes.

**Long, Detailed Prompts**

The model works better with long, detailed prompts. A helpful technique to extend prompts is to take a base prompt and then ask a LLM chatbot (eg. ChatGPT, Claude, Gemini, etc.) to write a more detailed version.

If your base prompt is:

"A cartoon dog jumping and then running."

Then, the extended prompt could be:

"A cartoon dog with big expressive eyes and floppy ears suddenly leaps into the frame, tail wagging, and then sprints joyfully toward the camera. Its oversized paws pound playfully on the ground, tongue hanging out in excitement. The animation style is colorful, smooth, and bouncy, with exaggerated motion to emphasize energy and fun. The background blurs slightly with speed lines, giving a lively, comic-style effect as if the dog is about to jump right into the viewer."

## Offline Generation

A test [script](../test.py) can be used for offline generation.

If the model weights are not downloaded yet:

```
# Run from scope directory
uv run download_models --pipeline reward_forcing
```

Then:

```
# Run from scope directory
uv run -m scope.core.pipelines.reward_forcing.test
```

This will create an `output.mp4` file in the `reward_forcing` directory.

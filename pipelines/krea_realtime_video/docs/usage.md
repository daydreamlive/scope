# Krea Realtime Video

[Krea Realtime Video](https://www.krea.ai/blog/krea-realtime-14b) is a streaming pipeline and autoregressive video diffusion model from Krea.

The model is trained using Self-Forcing on Wan2.1 14b.

## Examples

The following examples include timeline JSON files with the prompts used so you can try them as well.

A > 40 GB VRAM GPU (eg H100, RTX 6000 Pro) are recommended for these examples since they use a higher resolution.

https://github.com/user-attachments/assets/149d41ce-0392-4d9d-ae19-cd7896ff92b9

[Timeline JSON file](./examples/timeline-krea-flower-bloom.json)

https://github.com/user-attachments/assets/8b9086be-b59f-42f1-94fb-2523ddec41bd

[Timeline JSON file](./examples/timeline-abstract-shape.json)

## Resolution

The generation will be faster for smaller resolutions resulting in smoother video. The visual quality will be better at higher resolutsion (eg 480x832 and larger), but you may need a more powerful GPU in order to achieve a higher FPS.

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

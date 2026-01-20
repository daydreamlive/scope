# LongLive

[LongLive](https://nvlabs.github.io/LongLive) is a streaming pipeline and autoregressive video diffusion model from Nvidia, MIT, HKUST, HKU and THU.

The model is trained using [Self-Forcing](https://self-forcing.github.io/) on Wan2.1 1.3b with modifications to support smoother prompt switching and improved quality over longer time periods while maintaining fast generation.

## Examples

The following examples include timeline JSON files with the prompts used so you can try them as well.

https://github.com/user-attachments/assets/e9adaf9d-8639-42aa-911e-fc3574e7fdef

[Daydream Project](https://app.daydream.live/creators/yondonfu/creations/panda)

[Timeline JSON file](./examples/timeline-panda.json)

https://github.com/user-attachments/assets/63940a18-be6e-47fe-80bb-43d05162f2d6

[Daydream Project](https://app.daydream.live/creators/yondonfu/creations/factory)

[Timeline JSON file](./examples/timeline-factory.json)

## Resolution

The generation will be faster for smaller resolutions resulting in smoother video. The visual quality will be better at 832x480 which is the resolution that the model was trained on, but you may need a more powerful GPU in order to achieve a higher FPS (the ~20 FPS reported in the paper is on a H100).

## Seed

The seed parameter in the UI can be used to reproduce generations. If you like the generation for a certain seed value and sequence of prompts you can re-use that value later with those same prompts to reproduce the generation.

## Prompting

The [original project repo](https://github.com/NVlabs/LongLive) contains additional tips for prompting.

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
uv run download_models --pipeline longlive
```

Then:

```
# Run from scope directory
uv run -m score.core.pipelines.longlive.test
```

This will create an `output.mp4` file in the `longlive` directory.

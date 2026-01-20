# StreamDiffusionV2

[StreamDiffusionV2](https://streamdiffusionv2.github.io/) is a streaming inference pipeline and autoregressive video diffusion model from the creators of the original [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) project.

The model is trained using [Self-Forcing](https://self-forcing.github.io/) on Wan2.1 1.3b with modifications to support streaming.

## Examples

The following examples include timeline JSON files with the prompts used so you can try them as well.

https://github.com/user-attachments/assets/639dd5a2-1bb0-4916-94b6-7935fafe459a

[Timeline JSON file](./examples/timeline-evolution.json)

https://github.com/user-attachments/assets/26e570ab-f2f9-4b99-b82f-5bc36499bd89

[Timeline JSON file](./examples/timeline-feline.json)

https://github.com/user-attachments/assets/a78ec9c5-83aa-4f0d-826e-f32f6fd10366

[Timeline JSON file](./examples/timeline-prey.json)

## Resolution

The generation will be faster for smaller resolutions resulting in smoother video. Scope currently will use the input video's resolution as the output resolution. The visual quality will be better at 832x480 which is the resolution that the model was trained on, but you may need a more powerful GPU in order to achieve a higher FPS.

## Seed

The seed parameter in the UI can be used to reproduce generations. If you like the generation for a certain seed value, input video and sequence of prompts you can re-use that value later with those same input video and prompts to reproduce the generation.

## Prompting

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
uv run download_models --pipeline streamdiffusionv2
```

Then:

```
# Run from scope directory
uv run -m score.core.pipelines.streamdiffusionv2.test
```

This will create an `output.mp4` file in the `streamdiffusionv2` directory.

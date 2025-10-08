# StreamDiffusionV2

StreamDiffusionV2 is a streaming inference pipeline and autoregressive video diffusion model from the creators of the original [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion) project.

The model is trained using [Self-Forcing](https://self-forcing.github.io/) on Wan2.1 1.3b with modifications to support streaming.

## Examples

https://github.com/user-attachments/assets/60a15839-1e7c-480a-bc11-4f447ba6ec49

https://github.com/user-attachments/assets/dfcc04a3-f55c-4cc7-a063-7343aa584ea8

## Prompting

The model works better with long, detailed prompts. A helpful technique to extend prompts is to take a base prompt and then ask a LLM chatbot (eg. ChatGPT, Claude, Gemini, etc.) to write a more detailed version.

If your base prompt is:

"A cartoon dog jumping and then running."

Then, the extended prompt could be:

"A cartoon dog with big expressive eyes and floppy ears suddenly leaps into the frame, tail wagging, and then sprints joyfully toward the camera. Its oversized paws pound playfully on the ground, tongue hanging out in excitement. The animation style is colorful, smooth, and bouncy, with exaggerated motion to emphasize energy and fun. The background blurs slightly with speed lines, giving a lively, comic-style effect as if the dog is about to jump right into the viewer."

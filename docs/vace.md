# Using VACE (Video All-in-One Creation and Editing)

A subset of the pipeline in Scope support using a modified version of [VACE](https://ali-vilab.github.io/VACE-Page/) for additional video creation and editing tasks.

> [!IMPORTANT]
> VACE support is still experimental and the implementation is incomplete.

## Compatibility

At the moment, the following pipelines support VACE tasks:

### Wan2.1 1.3B based pipelines:
- `longlive`
- `reward-forcing`
- `memflow`

### Wan2.1 14B based pipelines:
- `krea_realtime_video`

`streamdiffusionv2` also supports VACE capabilities, but beware that the quality is poor right now.

> [!NOTE]
> KREA + VACE requires approximately 55GB of VRAM.
> FP8 Quantization is not currently supported.
> Continued subsequent prompting with KREA and VACE is limited in functionality due to the nature of cache recomputation, and therefore you may require resetting the cache.

## Supported Features

These features are currently supported:

- Reference-to-video (R2V) using reference images to guide generation.
- Video-to-video (VACE V2V) editing using control videos (eg. depth, pose, scribble, optical flow, etc.) to guide generation.
- Animate Anything (R2V + VACE V2V) where a reference image is used to define the character and style while the control video provides the structure.
- Real-time depth estimation using the built-in `video-depth-anything` preprocessor to automatically generate depth maps from source videos. Additional preprocessors will be available via plugins in the future.
- First Frame Last Frame (FFLF) extension mode for generating video that connects reference frames at the start and/or end.
- Inpainting using masks to selectively regenerate regions of video while preserving others.

## Unsupported Features

These features are not supported right now, but we're investigating them:

- Multiple reference images for R2V
- More complex tasks supported in the original VACE project such as Swap Anything, Reference Anything, Move Anything, Expand Anything

## Enabling VACE

Make sure that VACE is toggled to "On" in the Settings panel.

<img width="529" height="716" alt="Screenshot 2025-12-22 114746" src="https://github.com/user-attachments/assets/4a3831d4-a36e-429e-ad55-f34865f672d0" />

## R2V

Click "Add Image" under "Reference Images".

<img width="525" height="714" alt="Screenshot 2025-12-22 114718" src="https://github.com/user-attachments/assets/74901190-1ea2-4bbd-b574-caf7c50a2cbf" />

Use the media picker to either upload an image or select an image from your asset collection (previously uploaded images).

<img width="815" height="606" alt="Screenshot 2025-12-22 114729" src="https://github.com/user-attachments/assets/37091a2d-bc02-4ef5-b265-f3db566570e4" />

Then, you should see a preview of the selected reference image.

<img width="534" height="764" alt="Screenshot 2025-12-22 114738" src="https://github.com/user-attachments/assets/49807633-11e0-425e-a9d0-5b53ecba1302" />

> [!NOTE]
> Only a single reference image is supported right now.

## VACE V2V

Make sure that you have "Video" selected under "Input Mode" in the "Input & Controls Panel".

Upload a control video (eg. depth, pose, scribble, optical flow, etc.).

An example control video (pose) that can be used:

https://github.com/user-attachments/assets/9b2b1619-dbe9-4e46-9cfa-5bf304cc161f

<img width="516" height="1063" alt="Screenshot 2025-12-22 115520" src="https://github.com/user-attachments/assets/65210820-00af-4592-b314-5cb4aa991b88" />

## Animate Anything

R2V and VACE V2V can be combined for an "Animate Anything" task.

In this example, we're using this reference image with the `longlive` pipeline:

<img width="826" height="481" alt="Screenshot 2025-12-19 172128" src="https://github.com/user-attachments/assets/a08ca39c-ea15-43c5-9e49-10c5b8823872" />

https://github.com/user-attachments/assets/da126478-1f7f-4564-9fcb-c46a28977f3c

In this example, we also use the [Wan2.1 1.3B Arcane Jinx LoRA](https://civitai.com/models/1332383/wan-lora-arcane-jinx-v1-wan-13b) as described in the [LoRA guide](./lora.md) to improve the character and style consistency in the generation:

https://github.com/user-attachments/assets/ed65e627-3a48-4d54-9715-d25cb79655ed

## API Usage

*Coming soon*

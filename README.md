# Daydream Scope

[![Docs](https://img.shields.io/badge/Docs-blue?logo=gitbook&logoColor=white)](https://docs.daydream.live/scope/getting-started/quickstart)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

![timeline-panda](https://github.com/user-attachments/assets/21724fa1-d1c6-489e-bfb7-354b91e6f27b)

Scope is a tool for running and customizing real-time, interactive generative AI pipelines and models.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Runpod](#runpod)
- [Firewalls](#firewalls)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- Autoregressive video diffusion models with configurable [VAEs](./docs/vae.md)
  - [StreamDiffusionV2](./src/scope/core/pipelines/streamdiffusionv2/docs/usage.md) (text-to-video, video-to-video)
  - [LongLive](./src/scope/core/pipelines/longlive/docs/usage.md) (text-to-video, video-to-video)
  - [Krea Realtime Video](./src/scope/core/pipelines/krea_realtime_video/docs/usage.md) (text-to-video)
  - [RewardForcing](./src/scope/core/pipelines/reward_forcing/docs/usage.md) (text-to-video, video-to-video)
  - [MemFlow](./src/scope/core/pipelines/memflow/docs/usage.md) (text-to-video, video-to-video)
  - Additional models including [Waypoint-1](https://github.com/daydreamlive/scope-overworld) via plugins
- [Composable pipeline architecture](./docs/architecture/pipelines.md) enabling using additional video processing techniques such as real-time depth mapping and frame interpolation together with video diffusion
- [Plugins](./docs/plugins.md) to extend Scope's capabilities with new models, visual effects and more
- [LoRAs](./docs/lora.md) to customize concepts and styles used with autoregressive video diffusion models
- [VACE](./docs/vace.md) to use reference images and control videos to guide autoregressive video diffusion models
- [API](./docs/server.md) with WebRTC real-time streaming
- [Spout](./docs/spout.md) (Windows only) real-time video sharing with local applications
- Low latency async video processing pipelines
- Interactive UI with timeline editor, text prompting, model parameter controls and video/camera/text input modes

...and more to come!

## System Requirements

Check out the [Systems Requirements reference](https://docs.daydream.live/scope/reference/system-requirements).

## Quick Start

Check out the [Quick Start](https://docs.daydream.live/scope/getting-started/quickstart).

## Runpod

Check out the [instructions](https://docs.daydream.live/scope/getting-started/quickstart#cloud-runpod) for deploying Scope on Runpod using a template.

> [!IMPORTANT]
> The template will store model files under `/workspace/models` because RunPod mounts a volume disk at `/workspace` allowing any files there to be retained across pod restarts.

> [!NOTE]
> If you want to use the version from the main branch, you need to use the `daydreamlive/scope:main` docker image. You can configure this in the RunPod template by editing the Docker image setting.

## Firewalls

If you run Scope in a cloud environment with restrictive firewall settings (eg. Runpod), Scope supports using [TURN servers](https://webrtc.org/getting-started/turn-server) to establish a connection between your browser and the streaming server.

The easiest way to enable this feature is to follow the [HuggingFace Auth guide](https://docs.daydream.live/scope/guides/huggingface) which walks through using a HuggingFace account to access Cloudflare's TURN servers.

## Environment Variables

Check out the [Environment Variables reference](https://docs.daydream.live/scope/reference/environment-variables).

## Contributing

Check out the [contribution guide](./docs/contributing.md).

## Troubleshooting

Check out the [Troubleshooting page](https://docs.daydream.live/scope/getting-started/quickstart#troubleshooting).

## License

The alpha version of this project is licensed under [CC BY-NC-SA 4.0](./LICENSE).

You may use, modify, and share the code for non-commercial purposes only, provided that proper attribution is given.

We will consider re-licensing future versions under a more permissive license if/when non-commercial dependencies are refactored or replaced.

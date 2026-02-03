# Daydream Scope

[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/mnfGR4Fjhp)

![timeline-panda](https://github.com/user-attachments/assets/21724fa1-d1c6-489e-bfb7-354b91e6f27b)

Scope is a tool for running and customizing real-time, interactive generative AI pipelines and models.

🚧 This project is currently in **beta**. 🚧

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Install](#install)
  - [Manual Installation](#manual-installation)
    - [Clone](#clone)
    - [Build](#build)
    - [Run](#run)
  - [Runpod](#runpod)
- [First Generation](#first-generation)
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
  - Additional models including [Waypoint-1](https://github.com/daydreamlive/scope-overworld) via plugins (preview)
- [LoRAs](./docs/lora.md) to customize concepts and styles used with autoregressive video diffusion models
- [VACE (experimental)](./docs/vace.md) to use reference images and control videos to guide autoregressive video diffusion models
- [API](./docs/server.md) with WebRTC real-time streaming
- [Spout](./docs/spout.md) (Windows only) real-time video sharing with local applications
- Low latency async video processing pipelines
- Interactive UI with timeline editor, text prompting, model parameter controls and video/camera/text input modes

...and more to come!

## System Requirements

Scope currently supports the following operating systems:

- Linux
- Windows

Scope currently requires a Nvidia GPU with >= 24GB VRAM. As a baseline, we recommend a driver that supports CUDA >= 12.8 and a RTX 3090/4090/5090 (newer generations will support higher FPS throughput and lower latency).

The following models currently have more restrictive requirements:

**Krea Realtime Video**

- Requires a Nvidia GPU with >= 32 GB VRAM and we recommend > 40GB VRAM GPU in order to get better results
- At the default resolution of 576x320, a 32 GB VRAM GPU (eg RTX 5090) can run the model with fp8 quantization
- If you want to use a higher resolution like 832x480, we suggest using a > 40GB VRAM GPU (eg H100, RTX 6000 Pro)

If you do not have access to a GPU with these specs then we recommend installing on [Runpod](#runpod).

## Quick Start

The easiest way to get started with Scope is to download the latest version of the desktop app.

> [!IMPORTANT]
> The desktop app is only available on Windows right now. See the [Install](#install) section for manual and cloud based install options.

1. Download the `.exe` file under "Assets" from the [releases](https://github.com/daydreamlive/scope/releases) page.
2. Double click the `.exe` file which runs the installer that will walk you through the installation process.

## Install

### Manual Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) which is needed to run the server and [Node.js](https://nodejs.org/en/download) which is needed to build the frontend.

> [!IMPORTANT]
> If you are using Windows, install [Microsoft Visual C++ Redistributable (vcredist)](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

#### Clone

```
git clone git@github.com:daydreamlive/scope.git
cd scope
```

#### Build

This will build the frontend files which will be served by the Scope server.

```
uv run build
```

#### Run

> [!IMPORTANT]
> If you are running the server in a cloud environment, make sure to read the [Firewalls](#firewalls) section.

This will start the server.

```
uv run daydream-scope
```

After the server starts up, the frontend will be available at `http://localhost:8000`.

The frontend will present a dialog for downloading model weights for pipelines before running them (by pressing play with the pipeline selected) for the first time. The default directory where model weights are stored is `~/.daydream-scope/models`.

### Runpod

Use our RunPod template to quickly set up Scope in the cloud. This is the easiest way to get started if you don't have a compatible local GPU.

> [!IMPORTANT]
> Follow the instructions in [Firewalls](#firewalls) to get a HuggingFace access token.

**Deployment Steps:**

1. **Click the Runpod template link**: [Template](https://console.runpod.io/deploy?template=aca8mw9ivw&ref=5k8hxjq3)

2. **Select your GPU**: Choose a GPU that meets the [system requirements](#system-requirements). Please note that your driver must support CUDA >= 12.8.

> [!IMPORTANT]
> **H100 GPU Requirement**: When selecting an H100 GPU, you **must** select CUDA 12.8. CUDA 12.9 does not work with H100 GPUs.

3. **Configure environment variables**:
   - Click "Edit Template"
   - Add an environment variable:
     - Set name to `HF_TOKEN`
     - Set value to your HuggingFace access token
   - Click "Set Overrides"

4. **Deploy**: Click "Deploy On-Demand"

5. **Access the app**: Wait for deployment to complete, then open the app at port 8000

The template will configure everything needed and the frontend will present a dialog for downloading model weights for pipelines when running them (by pressing play with the pipeline selected) for the first time.

> [!IMPORTANT]
> The template will store model files under `/workspace/models` because RunPod mounts a volume disk at `/workspace` allowing any files there to be retained across pod restarts.

> [!NOTE]
> If you want to use the version from the main branch, you need to use the `daydreamlive/scope:main` docker image. You can configure this in the RunPod template by editing the Docker image setting.

## First Generation

The easiest way to get started is to replay an example generation and then modify prompts in the timeline to steer the generation in a different direction.

Examples with importable timeline files can be found here:

- [StreamDiffusionV2](./src/scope/core/pipelines/streamdiffusionv2/docs/usage.md)
- [LongLive](./src/scope/core/pipelines/longlive/docs/usage.md)
- [RewardForcing](./src/scope/core/pipelines/reward_forcing/docs/usage.md)
- [MemFlow](./src/scope/core/pipelines/memflow/docs/usage.md)
- [Krea Realtime Video](./src/scope/core/pipelines/krea_realtime_video/docs/usage.md)

After your first generation you can:

- Use [LoRAs](./docs/lora.md) to customize the concepts and styles used in your generations.
- Use [VACE (experimental)](./docs/vace.md) to use reference images and control videos to guide generations.
- Use [Spout](./docs/spout.md) (Windows only) to share real-time video between Scope and other local applications.
- Use the [API](./docs/server.md) to programatically control Scope.

## Firewalls

If you run Scope in a cloud environment with restrictive firewall settings (eg. Runpod), Scope supports using [TURN servers](https://webrtc.org/getting-started/turn-server) to establish a connection between your browser and the streaming server.

The easiest way to enable this feature is to create a HuggingFace account and a `read` [access token](https://huggingface.co/docs/hub/en/security-tokens). You can then set an environment variable before starting Scope:

```bash
# You should set this to your HuggingFace access token
export HF_TOKEN=your_token_here
```

When you start Scope, it will automatically use Cloudflare's TURN servers and you'll have 10GB of free streaming per month:

```
uv run daydream-scope
```

## Environment Variables

### `HF_TOKEN`
HuggingFace access token for using Cloudflare's TURN servers. See [Firewalls](#firewalls) for details.

### `RECORDING_ENABLED`
- **Default**: `true`
- **Description**: Enable/disable recording. When `false`, recording is disabled.

### `RECORDING_MAX_LENGTH`
- **Default**: `1h`
- **Description**: Maximum total recording length (sum of all segments). Recording stops when reached.
- **Format**: Supports `1h`, `30m`, `120s`, or plain seconds (e.g., `3600`)

### `RECORDING_STARTUP_CLEANUP_ENABLED`
- **Default**: `true`
- **Description**: Enable/disable cleanup of recording files from previous sessions at startup.

## Contributing

Read the [contribution guide](./docs/contributing.md).

## Troubleshooting

**Python.h: No such file or directory**

This error has been encountered on certain Linux machines when the Python header file is missing. If you encounter this error, make sure you install the [python3-dev package](https://packages.debian.org/bookworm/python3-dev).

## License

The alpha version of this project is licensed under [CC BY-NC-SA 4.0](./LICENSE).

You may use, modify, and share the code for non-commercial purposes only, provided that proper attribution is given.

We will consider re-licensing future versions under a more permissive license if/when non-commercial dependencies are refactored or replaced.

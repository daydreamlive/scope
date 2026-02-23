# Using LoRAs

The pipelines in Scope support using one or multiple LoRAs to customize concepts and styles used in generations.

## Compatibility

- The `streamdiffusionv2` pipeline is compatible with Wan2.1-T2V-1.3B LoRAs.
- The `longlive` pipeline is compatible with Wan2.1-T2V-1.3B LoRAs.
- The `reward-forcing` pipeline is compatible with Wan2.1-T2V-1.3B LoRAs.
- The `memflow` pipeline is compatible with Wan2.1-T2V-1.3B LoRAs.
- The `krea-realtime-video` pipeline is compatible with Wan2.1-T2V-14B LoRAs.

## Installing LoRAs

Scope supports using LoRAs that can be downloaded from popular hubs such as [HuggingFace](https://huggingface.co/) or [CivitAI](https://civitai.com/).

A few LoRAs that you can start with for `streamdiffusionv2`, `longlive`, `reward-forcing` and `memflow`:

- [Arcane Jinx](https://civitai.com/models/1332383/wan-lora-arcane-jinx-v1-wan-13b)
- [Genshin TCG](https://civitai.com/models/1728768/genshin-tcg-style-wan-13b)

A few LoRAs that you can start with for `krea-realtime-video`:

- [Origami](https://huggingface.co/shauray/Origami_WanLora/tree/main)
- [Film Noir](https://huggingface.co/Remade-AI/Film-Noir)
- [Pixar](https://huggingface.co/Remade-AI/Pixar)

### Using the Settings Dialog

The easiest way to install LoRAs is through the Settings dialog:

1. Click the **Settings** icon (gear) in the header
2. Select the **LoRAs** tab
3. Paste a LoRA URL from HuggingFace or CivitAI into the input field
4. Click **Install**

The LoRA will be downloaded and saved to your LoRA directory automatically. Once installed, you can select it from the LoRA Adapters section in the Settings panel.

#### CivitAI API Token

CivitAI requires an API token for programmatic downloads. You can configure this in one of two ways:

**Option 1: Settings Dialog**

1. Click the **Settings** icon (gear) in the header
2. Select the **API Keys** tab
3. Enter your CivitAI API token and click **Save**

**Option 2: Environment Variable**

```bash
export CIVITAI_API_TOKEN=your_civitai_token_here
```

> **Note:** The environment variable takes precedence over a token stored through the UI.

Get your API key at [civitai.com/user/account](https://civitai.com/user/account).

### Manual Installation

For manual installation, follow the steps below.

#### Local

If you are running Scope locally you can simply download the LoRA files to your computer and move them to the proper directory.

**HuggingFace**

<img width="1766" height="665" alt="CleanShot 2025-11-21 at 12 21 11" src="https://github.com/user-attachments/assets/12da7b9e-875e-4946-9ea6-10acfe97f057" />

Click the download button and move the file to the `~/.daydream-scope/models/lora` folder.

**CivitAI**

<img width="1778" height="756" alt="CleanShot 2025-11-21 at 12 20 28" src="https://github.com/user-attachments/assets/9b53f86e-0d21-4e58-b81a-00decc466f75" />

Click the download button and move the file to the `~/.daydream-scope/models/lora` folder.

#### Cloud

If you are running the Scope server on a remote machine in the cloud, then we recommend you programmatically download the LoRA files to the remote machine.

**HuggingFace**

First, copy the link address for the `.safetensors` file found under "Files and versions" of the LoRA page:

https://github.com/user-attachments/assets/f02134e7-3658-4390-a4ec-6083dedbf5e6

Then, navigate to `/workspace/models/lora` and run:

```
wget -O <file name> <link address>
```

An example:

```
wget -O pixar_10_epochs.safetensors https://huggingface.co/Remade-AI/Pixar/resolve/main/pixar_10_epochs.safetensors?download=true
```

> **Tip:** On some RunPod templates, `wget` may not be pre-installed. If so, run `apt-get install wget` first.

**CivitAI**

First, get a [CivitAI API key](https://developer.civitai.com/docs/getting-started/setup-profile) which is required for programmatic downloads.

Then, copy the link address from the LoRA page:

https://github.com/user-attachments/assets/2b3dd3ef-ba68-4254-b0dd-af2ee97bd8a9

Then, navigate to `/workspace/models/lora` and run:

```
wget -O <file name> "<link address>&token=<TOKEN>"
```

`<TOKEN>` is your API key.

An example:

```
wget -O arcane-jinx.safetensors "https://civitai.com/api/download/models/1679582?type=Model&format=SafeTensor&token=<TOKEN>"
```

> [!WARNING]
> Many LoRAs require a **trigger keyword** in your prompt to activate. Check the LoRA's model page (on HuggingFace or CivitAI) for the required trigger phrase and include it in your prompt.

> [!IMPORTANT]
> Make sure to surround the link address including the token with double quotes!

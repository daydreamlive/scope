# VAE Types

VAE (Variational Autoencoder) is used for encoding pixels to latents and decoding latents back to pixels. Different VAE types offer tradeoffs between quality, speed, and memory usage.

The comparison and descriptions below are sourced from the lightx2v [docs](https://huggingface.co/lightx2v/Autoencoders) which contain additional technical details about each implementation.

## Comparison

| Type       | Quality | Speed  | Description                        |
| ---------- | ------- | ------ | ---------------------------------- |
| `wan`      | Best    | Slow   | Full WanVAE (default)              |
| `lightvae` | High    | Medium | 75% pruned WanVAE                  |
| `tae`      | Average | Fast   | Tiny AutoEncoder                   |
| `lighttae` | High    | Fast   | TAE with WanVAE normalization      |

## WanVAE (`wan`)

The full WanVAE is the default and provides the best quality output. It uses the complete model architecture without any pruning, resulting in the highest fidelity encoding and decoding at the cost of speed.

## LightVAE (`lightvae`)

LightVAE is a 75% pruned version of WanVAE. It removes 75% of the model's channels, significantly reducing computation while maintaining high quality output. This takes a middle ground between quality and speed.

## TAE (`tae`)

TAE (Tiny AutoEncoder) is a completely different architecture from WanVAE. It's a lightweight model specifically designed for quick encoding/decoding previews. Key architectural differences include:

- Uses MemBlock for temporal memory
- Has TPool/TGrow blocks for temporal downsampling/upsampling
- Much simpler architecture with 64 channels throughout

TAE trades quality for speed.

## LightTAE (`lighttae`)

LightTAE combines the fast TAE architecture with WanVAE's normalization parameters. This provides a balance of high quality and fast performance.

## See Also

- [Load Pipeline API](api/load.md#vae-types) - Configure VAE type when loading a pipeline

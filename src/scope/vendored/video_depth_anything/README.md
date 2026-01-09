# Video Depth Anything (Vendored)

Minimal inference code vendored from [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything).

## Source

- **Repository:** https://github.com/DepthAnything/Video-Depth-Anything
- **Commit:** 4f5ae23172ba60fd7bc11ef671cca678842c7072
- **Date vendored:** 2025-12-28

## License

Apache-2.0 (see LICENSE file)

## Modifications

- Renamed `utils/` to `vda_utils/` to avoid import conflicts
- Updated import path in `video_depth_stream.py`
- Moved to `src/scope/vendored/` to be included in package

## Usage

```python
from scope.vendored.video_depth_anything import VideoDepthAnything

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
}

model = VideoDepthAnything(**model_configs['vits'])
model.load_state_dict(torch.load('~/.daydream-scope/models/vda/video_depth_anything_vits.pth', map_location='cpu'))
model = model.to('cuda').eval()

# Streaming inference (one frame at a time)
depth = model.infer_video_depth_one(frame_rgb, input_size=518, device='cuda')

# Reset cache on hard cuts
model.transform = None
model.frame_id_list = []
model.frame_cache_list = []
model.id = -1
```

## Checkpoints

Download to `~/.daydream-scope/models/vda/` (or `$DAYDREAM_SCOPE_MODELS_DIR/vda/`):
- `video_depth_anything_vits.pth` (VDA-Small, 28.4M params, Apache-2.0)

```bash
mkdir -p ~/.daydream-scope/models/vda
wget -O ~/.daydream-scope/models/vda/video_depth_anything_vits.pth \
  https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
```

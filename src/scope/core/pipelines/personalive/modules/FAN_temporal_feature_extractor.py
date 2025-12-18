import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PatchEmbed

from einops import rearrange
from .motion_module import TemporalTransformerBlock, zero_module, random_module

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.dropout = nn.Dropout(0.5)

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(256, 256))

        self.add_module('b2_' + str(level), ConvBlock(256, 256))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(256, 256))

        self.add_module('b3_' + str(level), ConvBlock(256, 256))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)
        up1 = self.dropout(up1)
        # Lower branch
        low1 = F.max_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)
        up1size = up1.size()
        rescale_size = (up1size[2], up1size[3])
        up2 = F.upsample(low3, size=rescale_size, mode='bilinear')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)

class TemporalTransformer3DModel(nn.Module):
    def __init__(
        self,
        in_channels,
        num_attention_heads=8,
        attention_head_dim=128,
        num_layers=1,
        attention_block_types=("Temporal_Self",),
        sample_size=56,
        patch_size=2,
        dropout=0.0,
        norm_num_groups=32,
        cross_attention_dim=0,
        activation_fn="geglu",
        attention_bias=False,
        upcast_attention=False,
        cross_frame_attention_mode=None,
        temporal_position_encoding=True,
        temporal_position_encoding_max_len=24,
        zero_initialize=True
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.norm = torch.nn.GroupNorm(
            num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
        # self.proj_in = nn.Linear(in_channels, inner_dim)  # patchEmbed中有
        if cross_attention_dim == 0 and any([block_name.endswith("_Cross") for block_name in attention_block_types]):
            cross_attention_dim = inner_dim
        self.cross_frame_attention_mode = cross_frame_attention_mode
        if cross_frame_attention_mode is not None:
            assert cross_frame_attention_mode in ['Spatial', 'Temporal'], cross_attention_dim
            assert any([block_name.endswith("_Cross") for block_name in attention_block_types])
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    attention_block_types=attention_block_types,
                    dropout=dropout,
                    norm_num_groups=norm_num_groups,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_frame_attention_mode=None,
                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim // (patch_size**2), in_channels)

        self.patch_size = patch_size
        self.sample_size = sample_size
        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=1,
        )
        # 在转onnx模型时，需要设置为False
        self.pos_embed.pos_embed.requires_grad = False
        self.proj_out = zero_module(self.proj_out) if zero_initialize else random_module(self.proj_out)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, skip=True):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        batch, _, video_length = hidden_states.shape[:3]
        residual = hidden_states
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        _, _, height, width = hidden_states.shape

        hidden_states = self.norm(hidden_states)  # 需要先norm，再加posEmb，否则Emb偏执太大了
        grid_h, grid_w =  height // self.patch_size, width // self.patch_size,

        hidden_states = self.pos_embed(hidden_states)  # [(bf), l, c]  # 偏执太大了
        # print(
        #     round(torch.abs(hidden_states).mean().item(), 6),
        #     round(torch.abs(norm_hidden_states).mean().item(), 6),
        # )
        if self.cross_frame_attention_mode is not None:
            assert encoder_hidden_states is None
            encoder_hidden_states = rearrange(hidden_states, "(b f) d c -> b f d c", b=batch)

            hidden_states = encoder_hidden_states[:, -1] # [b, d, c]
            residual = residual[:, :, -1:]
            video_length = 1
            if self.cross_frame_attention_mode == 'Temporal': # 用所有帧来condition最后一帧（当前帧）
                encoder_hidden_states = rearrange(encoder_hidden_states, "b f d c -> (b d) f c")
            elif self.cross_frame_attention_mode == 'Spatial':
                encoder_hidden_states = rearrange(encoder_hidden_states, "b f d c -> b (f d) c")

        # Transformer Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                video_length=video_length,
                att_flag=False
            )

        hidden_states = hidden_states.reshape(
            shape=(batch * video_length, grid_h, grid_w, self.patch_size, self.patch_size, -1)
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        hidden_states = hidden_states.reshape(shape=(batch * video_length, self.in_channels, height, width))

        hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", b=batch)
        # print(
        #     'out',
        #     round(torch.abs(hidden_states).mean().item(), 6),
        #     round(torch.abs(residual).mean().item(), 6),
        # )
        output = (hidden_states + residual) if skip else hidden_states

        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union
from diffusers.models.attention_processor import AttentionProcessor


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
                nn.BatchNorm2d(in_planes, eps=1e-4),
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


class FAN_use(nn.Module):
    def __init__(self):
        super(FAN_use, self).__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        hg_module = 0
        self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
        self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        self.add_module('conv_last' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
        self.add_module('l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0))
        self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

        if hg_module < self.num_modules - 1:
            self.add_module('bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('al' + str(hg_module), nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0))

        self.avgpool = nn.MaxPool2d((2, 2), 2)
        self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        self.fc = nn.Linear(28 * 28, 512)
        self.bn5 = nn.BatchNorm2d(68)
        self.relu = nn.ReLU(True)

    def forward(self, x, return_featmap=False):
        x = F.relu(self.bn1(self.conv1(x)), True)   # 112
        x = F.max_pool2d(self.conv2(x), 2)  # 56    # [B, 128, 112, 112]
        x = self.conv3(x)
        x = self.conv4(x)   # [B, 256, 56, 56]

        previous = x

        i = 0
        hg = self._modules['m' + str(i)](previous)

        ll = hg
        ll = self._modules['top_m_' + str(i)](ll)

        ll = self._modules['bn_end' + str(i)](self._modules['conv_last' + str(i)](ll))   # [B, 256, 56, 56]
        if return_featmap:
            return ll
        tmp_out = self._modules['l' + str(i)](F.relu(ll))

        net = self.relu(self.bn5(tmp_out))  # [B, 68, 56, 56]
        net = self.conv6(net)   # 28       # [B, 1, 28, 28]
        net = net.view(-1, net.shape[-2] * net.shape[-1])
        net = self.relu(net)
        net = self.fc(net)
        return net

from .FAN_temporal_feature_extractor import TemporalTransformer3DModel
from einops import rearrange


class FAN_SA(nn.Module):
    def __init__(self):
        super(FAN_SA, self).__init__()
        self.num_modules = 1

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        hg_module = 0
        self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
        self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
        self.add_module(
            'conv_last' + str(hg_module),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        )
        self.add_module(
            'l' + str(hg_module), nn.Conv2d(256, 68, kernel_size=1, stride=1, padding=0)
        )
        self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))

        if hg_module < self.num_modules - 1:
            self.add_module(
                'bl' + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            )
            self.add_module(
                'al' + str(hg_module),
                nn.Conv2d(68, 256, kernel_size=1, stride=1, padding=0),
            )

        self.avgpool = nn.MaxPool2d((2, 2), 2)
        self.conv6 = nn.Conv2d(68, 1, 3, 2, 1)
        self.fc = nn.Linear(28 * 28, 512)
        # self.conv6 = nn.Conv2d(68, 2, 3, 2, 1)
        # self.fc = nn.Linear(28 * 28 * 2, 1024)
        self.bn5 = nn.BatchNorm2d(68)
        self.relu = nn.ReLU(True)

        # Add by zxc
        self.att_1 = TemporalTransformer3DModel(
            in_channels=128,
            sample_size=112,
            patch_size=4,
            attention_block_types=("Spatial_Self",),
            zero_initialize=True,
        )
        self.att_2 = TemporalTransformer3DModel(
            in_channels=256,
            sample_size=56,
            patch_size=2,
            attention_block_types=("Spatial_Self",),
            zero_initialize=True,
        )
        self.att_3 = TemporalTransformer3DModel(
            in_channels=256,
            sample_size=56,
            patch_size=2,
            attention_block_types=("Spatial_Self",),
            zero_initialize=True,
        )

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)  # 112
        x = self.conv2(x)  # [B, 128, 112, 112]

        # Temp Self-Att: [B*h*w, T, 128*p*p] [B*28*28, T, 1024]
        x = rearrange(x, "(b f) c h w -> b c f h w", f=1)
        x = self.att_1(x, skip=True)[:, :, 0]

        x = F.max_pool2d(x, 2)  # 56
        x = self.conv3(x)
        x = self.conv4(x)  # [B, 256, 56, 56]

        # Temp Self-Att: [B*h*w, T, 256*p*p] [B*28*28, T, 1024]
        x = rearrange(x, "(b f) c h w -> b c f h w", f=1)
        x = self.att_2(x, skip=True)[:, :, 0]

        previous = x

        i = 0
        hg = self._modules['m' + str(i)](previous)

        ll = hg
        ll = self._modules['top_m_' + str(i)](ll)

        ll = self._modules['bn_end' + str(i)](
            self._modules['conv_last' + str(i)](ll)
        )  # [B, 256, 56, 56]
        # Temp Cross-Att: [B*28*28, 1, 1024]*[B*28*28, T, 1024]
        ll = rearrange(ll, "(b f) c h w -> b c f h w", f=1)
        ll = self.att_3(ll, skip=True)[:, :, 0]  # "b c 1 h w -> b c h w"
        # print('att3', torch.abs(ll).mean().item())

        tmp_out = self._modules['l' + str(i)](F.relu(ll))

        net = self.relu(self.bn5(tmp_out))  # [B, 68, 56, 56]
        net = self.conv6(net)  # 28       # [B, 1, 28, 28]
        net = net.view(-1, net.shape[-2] * net.shape[-1] * net.shape[1])
        net = self.relu(net)
        net = self.fc(net)
        return net

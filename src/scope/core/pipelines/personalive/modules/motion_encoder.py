import torch
import torch.nn as nn
import torch.nn.functional as F
from .FAN_feature_extractor import FAN_SA
from einops import rearrange
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid
from diffusers.models.modeling_utils import ModelMixin

def zero_module(module):
    # Zero out the parameters of a module and return it.
    assert isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear), type(module)
    for p in module.parameters():
        p.detach().zero_()
    return module

class MotEncoder(ModelMixin):
    def __init__(self, out_ch=16):
        super().__init__()
        self.model = FAN_SA()
        self.out_drop = None #nn.Dropout(p=0.4)
        self.out_ch = out_ch
        expr_dim = 512
        # Use torch.arange and output_type='pt' for diffusers >= 0.34.0 compatibility
        pos_grid = torch.arange(expr_dim // out_ch, dtype=torch.float32)
        extra_pos_embed = get_1d_sincos_pos_embed_from_grid(out_ch, pos_grid, output_type='pt')
        self.register_buffer("pe", extra_pos_embed.float().unsqueeze(0))
        self.final_proj = nn.Linear(expr_dim, expr_dim)
        self.out_bn = None

    def change_out_dim(self, out_ch):
        self.out_proj = nn.Linear(self.out_ch, out_ch)

    def set_attn_processor(self, processor):
        self.model.set_attn_processor(processor)

    def forward(self, x):
        x = x.to(self.dtype)
        latent = self.model(rearrange(x, "b c f h w -> (b f) c h w"))
        latent = self.final_proj(latent)
        latent = rearrange(latent, "b (l c) -> b l c", c=self.out_ch) + self.pe
        latent = rearrange(latent, "(b f) l c -> b f l c", f=x.shape[2])
        return latent

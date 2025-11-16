# Modified from https://github.com/guandeh17/Self-Forcing
import os

import torch

from ...utils import load_state_dict
from ..modules.t5 import umt5_xxl
from ..modules.tokenizers import HuggingfaceTokenizer


class WanTextEncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "Wan2.1-T2V-1.3B",
        model_dir: str | None = None,
        text_encoder_path: str | None = None,
        tokenizer_path: str | None = None,
    ) -> None:
        super().__init__()

        # Determine paths with priority: specific paths > model_dir > default
        if text_encoder_path is None:
            model_dir = model_dir if model_dir is not None else "wan_models"
            model_path = os.path.join(model_dir, model_name)
            text_encoder_path = os.path.join(
                model_path, "models_t5_umt5-xxl-enc-bf16.pth"
            )

        if tokenizer_path is None:
            model_dir = model_dir if model_dir is not None else "wan_models"
            model_path = os.path.join(model_dir, model_name)
            tokenizer_path = os.path.join(model_path, "google/umt5-xxl/")

        # Load weights first, then create model with those weights
        state_dict = load_state_dict(text_encoder_path)

        # Create model with meta device for fast initialization
        with torch.device("meta"):
            self.text_encoder = (
                umt5_xxl(
                    encoder_only=True,
                    return_tokenizer=False,
                    dtype=torch.float32,
                    device=torch.device("meta"),
                )
                .eval()
                .requires_grad_(False)
            )

        # Directly assign weights and materialize on CPU
        self.text_encoder.load_state_dict(state_dict, assign=True)
        self.text_encoder = self.text_encoder.to("cpu")

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, text_prompts: list[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        # ids = ids.to(torch.device('cpu'))
        # mask = mask.to(torch.device('cpu'))
        for u, v in zip(context, seq_lens, strict=False):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}

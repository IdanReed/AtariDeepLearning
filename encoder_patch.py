# image_encoder_paper_style.py
import torch
import torch.nn as nn

class AtariPatchEncoder(nn.Module):
    def __init__(self, img_size=84, patch_size=14, in_channels=1, d_model=768):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model

        self.num_patches_per_side = img_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

        self.patch_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, d_model)
        )

        nn.init.trunc_normal_(self.patch_pos_embed, std=0.02)

    def _encode_4d(self, x: torch.Tensor) -> torch.Tensor:

        x = self.proj(x)
        B, D, H_p, W_p = x.shape

        x = x.flatten(2).transpose(1, 2)

        x = x + self.patch_pos_embed
        return x

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            frames = frames.view(B * T, C, H, W)
            x = self._encode_4d(frames)
            x = x.view(B, T, self.num_patches, self.d_model)
            return x
        elif frames.dim() == 4:
            return self._encode_4d(frames)
        else:
            raise ValueError("frames must be 4D or 5D tensor")

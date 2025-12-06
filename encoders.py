from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    """
    Base class for observation encoders.

    Expected interface:
      - forward(frames): (B, T, C, H, W) -> (B, T, L_obs, d_model)
      - num_tokens_per_frame: number of tokens L_obs per image frame.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self._num_tokens_per_frame: Optional[int] = None

    @property
    def num_tokens_per_frame(self) -> int:
        if self._num_tokens_per_frame is None:
            raise ValueError(
                "num_tokens_per_frame is not set; encoder must infer it in __init__ "
                "or during first forward pass."
            )
        return self._num_tokens_per_frame

    @num_tokens_per_frame.setter
    def num_tokens_per_frame(self, value: int) -> None:
        self._num_tokens_per_frame = value

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B, T, L_obs, d_model)
        """
        raise NotImplementedError


class PatchEncoder(ObsEncoder):
    """
    Simple patch encoder: Conv2d with kernel_size = stride = patch_size
    to generate non-overlapping patches, then flatten.

    For example, with image_size=(84,84) and patch_size=14:
      H' = 84/14 = 6, W' = 6 -> L_obs = 36 tokens/frame.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int = 3,
        d_model: int = 512,
        patch_size: int = 14,
    ):
        super().__init__(d_model=d_model)

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        H, W = image_size
        H_p = H // patch_size
        W_p = W // patch_size
        self.num_tokens_per_frame = H_p * W_p

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B, T, L_obs, d_model)
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)          # (B*T, C, H, W)
        x = self.proj(x)                         # (B*T, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)         # (B*T, L_obs, d_model)
        L_obs = x.shape[1]
        x = x.view(B, T, L_obs, self.d_model)    # (B, T, L_obs, d_model)
        return x


class CNNEncoder(ObsEncoder):
    """
    Atari-style CNN encoder:
      - 3 conv layers
      - Each spatial cell becomes a token, then projected to d_model.

    We infer num_tokens_per_frame using a dummy pass with the given image_size.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int = 3,
        d_model: int = 512,
    ):
        super().__init__(d_model=d_model)

        self.image_size = image_size
        self.in_channels = in_channels

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        self.proj = nn.Linear(64, d_model)  # applied per spatial location

        # infer num_tokens_per_frame with a dummy forward
        H, W = image_size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)  # (1, C, H, W)
            feats = self.conv(dummy)                   # (1, 64, H', W')
            _, _, Hp, Wp = feats.shape
            self.num_tokens_per_frame = Hp * Wp

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B, T, C, H, W)
        returns: (B, T, L_obs, d_model)
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)         # (B*T, C, H, W)
        x = self.conv(x)                        # (B*T, 64, H', W')
        x = x.permute(0, 2, 3, 1)               # (B*T, H', W', 64)
        B_T, Hp, Wp, C_out = x.shape
        x = x.reshape(B_T, Hp * Wp, C_out)      # (B*T, L_obs, 64)
        x = self.proj(x)                        # (B*T, L_obs, d_model)
        L_obs = x.shape[1]
        x = x.view(B, T, L_obs, self.d_model)   # (B, T, L_obs, d_model)
        return x

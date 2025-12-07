from __future__ import annotations

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ObsEncoder(nn.Module):
    """
    Base class for obs encoders, defines forward and num_tokens_per_frame 
    """

    def __init__(self, emb_size: int):
        super().__init__()
        self.emb_size = emb_size
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
        frames: (batch_size, n_timesteps, n_channels, img_height, img_width)
        returns: (batch_size, n_timesteps, n_tokens_per_frame, emb_size)
        """
        raise NotImplementedError


class PatchEncoder(ObsEncoder):
    def __init__(
        self,
        image_size: Tuple[int, int],
        in_channels: int = 3,
        emb_size: int = 512,
        patch_size: int = 14,
    ):
        super().__init__(emb_size=emb_size)

        self.image_size = image_size
        self.in_channels = in_channels
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            emb_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        img_height, img_width = image_size
        patches_height = img_height // patch_size
        patches_width = img_width // patch_size
        self.num_tokens_per_frame = patches_height * patches_width

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (batch_size, n_timesteps, n_channels, img_height, img_width)
        returns: (batch_size, n_timesteps, n_tokens_per_frame, emb_size)
        """
        batch_size, n_timesteps, n_channels, img_height, img_width = frames.shape
        processed_frames = frames.view(batch_size * n_timesteps, n_channels, img_height, img_width)     # (batch_size * n_timesteps, n_channels, img_height, img_width)
        patch_features = self.proj(processed_frames)                                                    # (batch_size * n_timesteps, emb_size, patches_height, patches_width)
        obs_tokens = patch_features.flatten(2).transpose(1, 2)                                          # (batch_size * n_timesteps, num_obs_tokens, emb_size)
        num_obs_tokens = obs_tokens.shape[1]
        obs_tokens = obs_tokens.view(batch_size, n_timesteps, num_obs_tokens, self.emb_size)            # (batch_size, n_timesteps, num_obs_tokens, emb_size)
        return obs_tokens


class CNNEncoder(ObsEncoder):
    def __init__(   
        self,
        image_size: Tuple[int, int],
        in_channels: int = 3,
        emb_size: int = 512,
    ):
        super().__init__(emb_size=emb_size)

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

        self.proj = nn.Linear(64, emb_size) 

        # Just run forward to figure out num_tokens_per_frame
        img_height, img_width = image_size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, img_width)
            feats = self.conv(dummy)
            _, _, Hp, Wp = feats.shape
            self.num_tokens_per_frame = Hp * Wp

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (batch_size, n_timesteps, n_channels, img_height, img_width)
        returns: (batch_size, n_timesteps, n_tokens_per_frame, emb_size)
        """
        batch_size, n_timesteps, n_channels, img_height, img_width = frames.shape
        processed_frames = frames.view(batch_size * n_timesteps, n_channels, img_height, img_width)     # (batch_size * n_timesteps, n_channels, img_height, img_width)
        
        # Conv layers
        conv_features = self.conv(processed_frames)                                                     # (batch_size * n_timesteps, 64, feat_height, feat_width)
        spatial_features = conv_features.permute(0, 2, 3, 1)                                            # (batch_size * n_timesteps, feat_height, feat_width, 64)
        
        # Flatten
        flat_batch_size, feat_height, feat_width, out_channels = spatial_features.shape
        flattened_features = spatial_features.reshape(
            flat_batch_size, feat_height * feat_width, out_channels
        )                                                                                                   # (batch_size * n_timesteps, num_obs_tokens, 64)
        
        # Linear projection to emb_size
        obs_tokens = self.proj(flattened_features)                                                      # (batch_size * n_timesteps, num_obs_tokens, emb_size)

        # Add back in timestep dim
        num_obs_tokens = obs_tokens.shape[1]
        obs_tokens = obs_tokens.view(batch_size, n_timesteps, num_obs_tokens, self.emb_size)            # (batch_size, n_timesteps, num_obs_tokens, emb_size)

        return obs_tokens

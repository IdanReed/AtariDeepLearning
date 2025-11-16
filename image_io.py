# image_io.py

from __future__ import annotations

from pathlib import Path
from typing import Sequence, List, Callable

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms


class AtariImageLoader:
    """
    Utility to turn obs path sequences from AtariDataset into image tensors.

    - Expects obs entries to be paths that are valid from the current
      working directory (project root), e.g.:

        dataset/BeamRiderNoFrameskip-v4/BeamRiderNoFrameskip-v4/BeamRiderNoFrameskip-v4-recorded_images-0/0.png

      or whatever `fix_obs_paths` has written into the 'obs' array.

    - Outputs grayscale 84x84 tensors, matching the paper's preprocessing.
    """

    def __init__(
        self,
        img_size: int = 84,
        grayscale: bool = True,
    ) -> None:
        self.img_size = img_size
        self.grayscale = grayscale

        # build transform pipeline
        tfs: List[Callable] = []
        tfs.append(transforms.Resize((img_size, img_size)))
        if grayscale:
            tfs.append(transforms.Grayscale(num_output_channels=1))
        tfs.append(transforms.ToTensor())  # -> (C,H,W), values in [0,1]

        self.transform = transforms.Compose(tfs)

    # ---- low-level helpers -------------------------------------------------

    def _load_single_image(self, path: str) -> Tensor:
        """
        Load a single image given the path string from obs.

        The path is interpreted relative to the current working directory.
        """
        full_path = Path(path)
        img = Image.open(full_path).convert("RGB")
        img_t: Tensor = self.transform(img)
        return img_t

    # ---- main public API ---------------------------------------------------

    def load_sequence(self, obs_paths: Sequence[str]) -> Tensor:
        """
        Load a single sequence of frames:

        Args:
            obs_paths: iterable of strings (paths), length T

        Returns:
            frames: tensor of shape (T, C, H, W)
        """
        frames = [self._load_single_image(str(p)) for p in obs_paths]
        return torch.stack(frames, dim=0)  # (T,C,H,W)

    def load_batch(self, batch_obs_paths: Sequence[Sequence[str]]) -> Tensor:
        """
        Load a batch of sequences:

        Args:
            batch_obs_paths: iterable of sequences of strings, length B, each length T

        Returns:
            batch_frames: tensor of shape (B, T, C, H, W)
        """
        seq_tensors = [self.load_sequence(seq_paths) for seq_paths in batch_obs_paths]
        return torch.stack(seq_tensors, dim=0)  # (B,T,C,H,W)

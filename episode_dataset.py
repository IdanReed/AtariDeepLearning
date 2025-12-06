# episode_dataset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Any

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from episode import Episode, TimeStep


def load_frame(
    path: Path,
    resize_to: Tuple[int, int] = (84, 84),
) -> torch.Tensor:
    img = Image.open(path)
    img = img.convert("RGB")

    if resize_to is not None:
        img = img.resize(resize_to, resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0  # (height, width, channels)

    # Convert from (height, width, channels) to (channels, height, width)
    # PyTorch expects this apparently
    tensor = torch.from_numpy(arr).permute(2, 0, 1)

    return tensor


def reward_to_bin(reward: float) -> int:
    """
    Map reward -1, 0, 1 to class indices 0, 1, 2 respectively.
    If other values somehow appear, they are mapped by sign.
    """
    if reward > 0:
        return 2  # +1
    elif reward < 0:
        return 0  # -1
    else:
        return 1  # 0


class EpisodeSliceDataset(Dataset):
    def __init__(
        self,
        episodes: List[Episode],
        timestep_window_size: int = 4,
        image_size: Tuple[int, int] = (84, 84),
    ):
        self.episodes: List[Episode] = episodes
        self.timestep_window_size: int = timestep_window_size
        self.image_size: Tuple[int, int] = image_size

        # Find global RTG min/max
        rtg_values: List[float] = []
        for episode in self.episodes:
            for timestep in episode.timesteps:
                if timestep.rtg is not None:
                    rtg_values.append(float(timestep.rtg))

        if len(rtg_values) == 0:
            raise ValueError("No RTG values found")
        else:
            rtg_ints = [int(round(v)) for v in rtg_values]
            self.rtg_min_int = min(rtg_ints)
            self.rtg_max_int = max(rtg_ints)

        self.num_rtg_bins: int = self.rtg_max_int - self.rtg_min_int + 1

        # Precompute dataset (timestep) index -> timestep index
        self._slice_indexes: List[Tuple[int, int]] = []
        for episode_index, episode in enumerate(self.episodes):
            timesteps_len = len(episode.timesteps)

            if timesteps_len < self.timestep_window_size:
                continue

            for start_t in range(0, timesteps_len - self.timestep_window_size + 1):
                self._slice_indexes.append((episode_index, start_t))

    def __len__(self) -> int:
        return len(self._slice_indexes)

    def _rtg_to_bin(self, rtg_value: float) -> int:
        v_int = int(round(rtg_value))
        v_int = max(self.rtg_min_int, min(self.rtg_max_int, v_int))
        return v_int - self.rtg_min_int

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_index, start_t = self._slice_indexes[idx]
        episode = self.episodes[episode_index]
        end_t = start_t + self.timestep_window_size
        window_steps: List[TimeStep] = episode.timesteps[start_t:end_t]

        frames = []
        actions = []
        rewards = []
        rtg = []
        model_selected_actions = []
        repeated_actions = []
        reward_bins = []
        rtg_bins = []

        for ts in window_steps:
            frames.append(
                load_frame(
                    ts.obs,
                    resize_to=self.image_size,
                )
            )
            actions.append(int(ts.taken_action))
            rewards.append(float(ts.reward))
            rtg.append(float(ts.rtg))

            # binning
            reward_bins.append(reward_to_bin(float(ts.reward)))
            rtg_bins.append(self._rtg_to_bin(float(ts.rtg)))

            model_selected_actions.append(int(ts.model_selected_action))
            repeated_actions.append(bool(ts.repeated_action))

        frames_tensor = torch.stack(frames, dim=0)                                  # (timesteps, channels, height, width)
        actions_tensor = torch.tensor(actions, dtype=torch.long)                    # (timesteps)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)                 # (timesteps)
        rtg_tensor = torch.tensor(rtg, dtype=torch.float32)                         # (timesteps)
        reward_bins_tensor = torch.tensor(reward_bins, dtype=torch.long)           # (timesteps)
        rtg_bins_tensor = torch.tensor(rtg_bins, dtype=torch.long)                 # (timesteps)
        model_sel_tensor = torch.tensor(model_selected_actions, dtype=torch.long)   # (timesteps)
        repeated_tensor = torch.tensor(repeated_actions, dtype=torch.bool)          # (timesteps)

        return {
            "frames": frames_tensor,
            "actions": actions_tensor,              # taken actions (labels for action head)
            "rewards": rewards_tensor,              # raw reward scalars (for logging)
            "rtg": rtg_tensor,                      # raw RTG scalars (for logging)
            "reward_bins": reward_bins_tensor,      # class indices for reward head
            "rtg_bins": rtg_bins_tensor,            # class indices for return head
            "model_selected_actions": model_sel_tensor,
            "repeated_actions": repeated_tensor,
            "game_name": episode.game_name,
            "episode_index": episode_index,
            "start_t": start_t,
        }


def make_episode_dataloader(
    episodes: List[Episode],
    context_len: int = 4,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (84, 84),
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:

    dataset = EpisodeSliceDataset(
        episodes=episodes,
        timestep_window_size=context_len,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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
        # Not sure if this is needed but for sanity we'll force all images to the same size
        img = img.resize(resize_to, resample=Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor


def reward_to_bin(reward: float) -> int:
    if reward > 0:
        return 2  # +1
    elif reward < 0:
        return 0  # -1
    else:
        return 1  # 0


def compute_rtg_bin_range(episodes: List[Episode]) -> Tuple[int, int]:
    rtg_values: List[float] = []
    for episode in episodes:
        for ts in episode.timesteps:
            if ts.rtg is not None:
                rtg_values.append(float(ts.rtg))

    if not rtg_values:
        raise ValueError("No RTG values found")

    rtg_ints = [int(round(v)) for v in rtg_values]
    return min(rtg_ints), max(rtg_ints)


def compute_n_actions(episodes: List[Episode]) -> int:
    max_action: Optional[int] = None
    for episode in episodes:
        for ts in episode.timesteps:
            a = int(ts.taken_action)
            if max_action is None or a > max_action:
                max_action = a

    if max_action is None:
        raise ValueError("No actions found in episodes")

    return max_action + 1


@dataclass
class BinningInfo:
    rtg_min_int: int
    rtg_max_int: int
    num_rtg_bins: int
    n_actions: int


class EpisodeSliceDataset(Dataset):
    def __init__(
        self,
        episodes: List[Episode],
        timestep_window_size: int = 4,
        image_size: Tuple[int, int] = (84, 84),
        rtg_min_int: Optional[int] = None,
        rtg_max_int: Optional[int] = None,
    ):
        self.episodes: List[Episode] = episodes
        self.timestep_window_size: int = timestep_window_size
        self.image_size: Tuple[int, int] = image_size

        if rtg_min_int is None or rtg_max_int is None:
            rtg_min_int, rtg_max_int = compute_rtg_bin_range(episodes)

        self.rtg_min_int: int = int(rtg_min_int)
        self.rtg_max_int: int = int(rtg_max_int)
        self.num_rtg_bins: int = self.rtg_max_int - self.rtg_min_int + 1

        # Precompute slices
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
        taken_actions = []
        rewards = []
        rtg = []
        model_selected_actions = []
        repeated_actions = []
        reward_bins = []
        rtg_bins = []

        for ts in window_steps:
            frames.append(load_frame(ts.obs, resize_to=self.image_size))
            taken_actions.append(int(ts.taken_action))
            rewards.append(float(ts.reward))
            rtg.append(float(ts.rtg))

            reward_bins.append(reward_to_bin(float(ts.reward)))
            rtg_bins.append(self._rtg_to_bin(float(ts.rtg)))

            model_selected_actions.append(int(ts.model_selected_action))
            repeated_actions.append(bool(ts.repeated_action))

        frames_tensor = torch.stack(frames, dim=0)
        taken_actions_tensor = torch.tensor(taken_actions, dtype=torch.long)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        rtg_tensor = torch.tensor(rtg, dtype=torch.float32)
        reward_bins_tensor = torch.tensor(reward_bins, dtype=torch.long)
        rtg_bins_tensor = torch.tensor(rtg_bins, dtype=torch.long)
        model_sel_tensor = torch.tensor(model_selected_actions, dtype=torch.long)
        repeated_tensor = torch.tensor(repeated_actions, dtype=torch.bool)

        return {
            "frames": frames_tensor,
            "taken_actions": taken_actions_tensor,
            "rewards": rewards_tensor,
            "rtg": rtg_tensor,
            "reward_bins": reward_bins_tensor,
            "rtg_bins": rtg_bins_tensor,
            "model_selected_actions": model_sel_tensor,
            "repeated_actions": repeated_tensor,
            "game_name": episode.game_name,
            "episode_index": episode_index,
            "start_t": start_t,
        }
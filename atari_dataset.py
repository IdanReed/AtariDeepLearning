import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms


class AtariDataset(Dataset):
    def __init__(self, sequences_by_game, context_len: int = 16, img_size: int = 84, grayscale: bool = True):
        """
        sequences_by_game: dict[game, list[sequence_dict]]
          where each sequence_dict has keys:
            - "obs": (T,)  array of strings (image paths)
            - "taken actions": (T, 1) or (T,) int
            - "rewards": (T,) float
            - optionally "episode_starts": (T,) bool

        This dataset creates overlapping windows of length `context_len` over
        each sequence and returns:
          frames (tensor), action_window, return_to_go_window
        """
        self.context_len = context_len
        self.img_size = img_size
        self.grayscale = grayscale
        self.samples = []
        
        # Build transform pipeline for image loading
        tfs = [transforms.Resize((img_size, img_size))]
        if grayscale:
            tfs.append(transforms.Grayscale(num_output_channels=1))
        tfs.append(transforms.ToTensor())  # -> (C,H,W), values in [0,1]
        self.transform = transforms.Compose(tfs)

        # Precompute returns-to-go per sequence and build sliding windows
        for game, seqs in sequences_by_game.items():
            for seq in seqs:
                obs = seq["obs"]          # (T,)
                actions = seq["taken actions"]  # (T, 1) or (T,)
                rewards = seq["rewards"]  # (T,)

                T_len = int(obs.shape[0])

                if actions.shape[0] != T_len:
                    raise ValueError("obs and taken actions must have same length")
                if rewards.shape[0] != T_len:
                    raise ValueError("obs and rewards must have same length")

                # --- compute returns-to-go (per timestep) ---
                # If episode_starts exists, compute RTG separately per episode.
                episode_starts = seq.get("episode_starts", None)
                rtg = np.zeros_like(rewards, dtype=np.float32)

                if episode_starts is None:
                    # single episode: simple backward cumulative sum
                    rtg = np.cumsum(rewards[::-1])[::-1].astype(np.float32)
                else:
                    # multiple episodes packed into one sequence
                    starts = np.where(episode_starts)[0].tolist()
                    if 0 not in starts:
                        starts = [0] + starts
                    starts = sorted(set(starts))
                    starts.append(T_len)  # sentinel end

                    for s_idx in range(len(starts) - 1):
                        s = starts[s_idx]
                        e = starts[s_idx + 1]
                        seg = rewards[s:e]
                        seg_rtg = np.cumsum(seg[::-1])[::-1].astype(np.float32)
                        rtg[s:e] = seg_rtg

                # stash RTG into the seq dict so __getitem__ can access it
                seq["returns_to_go"] = rtg

                # build sliding windows over this sequence
                for start in range(0, T_len - context_len + 1):
                    self.samples.append((seq, start))

        self._n_actions = None

    def _load_image(self, path: str) -> torch.Tensor:
        """
        Load and transform a single image.
        
        Args:
            path: Path to image file
            
        Returns:
            Tensor of shape (C, H, W)
        """
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    """
    Determines the number of unique actions across all samples in the dataset
    """
    def n_actions(self):
        if self._n_actions is None:
            max_a = 0
            for seq, _ in self.samples:
                acts = seq["taken actions"]
                if acts.size:
                    if acts.ndim == 2 and acts.shape[1] == 1:
                        acts = acts[:, 0]
                    max_a = max(max_a, int(np.max(acts)))
            self._n_actions = max_a + 1
        return self._n_actions

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, start = self.samples[idx]
        end = start + self.context_len

        # Load images here (in worker process for parallel loading)
        obs_paths = seq["obs"][start:end]  # (context_len,) array of strings
        frames = torch.stack([self._load_image(str(p)) for p in obs_paths], dim=0)  # (T, C, H, W)

        # actions: (context_len, 1) or (context_len,) -> (context_len,)
        actions = seq["taken actions"][start:end]
        if actions.ndim == 2 and actions.shape[1] == 1:
            actions = actions[:, 0]
        actions = torch.from_numpy(actions).long()

        # returns-to-go: (context_len,)
        rtg = seq["returns_to_go"][start:end].astype(np.float32)
        rtg = torch.from_numpy(rtg).float()

        # Return: frames (tensor), actions, rtg
        return frames, actions, rtg

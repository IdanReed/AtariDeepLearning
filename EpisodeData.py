import torch.nn as nn
from torch.utils.data import Dataset
import random
import torch


class EpisodeData(Dataset):
    def __init__(self, episodes, context_length):
        super().__init__()
        self.episodes = episodes
        self.context_length = context_length

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        states, actions, rtg = self.episodes[idx]
        start_timestamp = random.randint(0, int(len(states)) - self.context_length)

        assert len(states) >= self.context_length, "Episode is too short for subsequence length"
        states = [s.flatten() for s in states[start_timestamp:start_timestamp + self.context_length]]
        actions = [a for a in actions[start_timestamp:start_timestamp + self.context_length]]
        rtg = [r for r in rtg[start_timestamp:start_timestamp + self.context_length]]

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long)
        rtg = torch.tensor(rtg, dtype=torch.float32)
        #print(states.shape)

        #print("State raw:", states[start_timestamp].shape)

        return states, actions, rtg
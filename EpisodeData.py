import torch.nn as nn


class EpisodeData(nn.Module):
    def __init__(self, dt, state_encoder):
        super().__init__()
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
"""
There's a couple confusing things, first is that each NPZ file is a gameplay sequence.
But each gameplay sequence is broken up into multiple episodes and 
"""

from pathlib import Path
from typing import List


class TimeStep():
    def __init__(self, obs: Path, model_selected_action: any, action_taken: any, repeated: bool, reward: any):
        self.obs = obs
        self.model_selected_action = model_selected_action
        self.taken_action = action_taken
        self.repeated_action = repeated
        self.reward = reward

        # Return to go (rtg) is the sum of all rewards from the current timestep to the end of the episode
        # Calculated once a episode is created
        self.rtg = None

class Episode():
    def __init__(self, game_name: str, timesteps: List[TimeStep]):
        self.game_name = game_name
        self.timesteps = timesteps
        self.calculate_rtg()

    def calculate_rtg(self):
        running_sum = 0
        for step in reversed(self.timesteps):
            running_sum += step.reward
            step.rtg = running_sum
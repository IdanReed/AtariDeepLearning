from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, List

import numpy as np
from torch.utils.data import Dataset, DataLoader


def discover_games(root_dir: Path) -> list[Path]:
    paths = root_dir.glob("*/*")
    return list(paths)

def discover_npz(game_dir: Path) -> list[Path]:
    npz_files = game_dir.glob("*.npz")
    return list(npz_files)

def discover_game_npz_paths(paths: list[Path]) -> Dict[str, Any]:
    game_npz_paths = {}
    for path in paths:
        game_npz_paths[path] = discover_npz(path)

    return game_npz_paths

class AtariData():
    def __init__(self, game_npz_paths: Dict[str, List[Path]]):
        # Game -> npz files each represents one sequence
        self._game_npz_paths = game_npz_paths
        self.game_to_sequences: Dict[str, List[np.ndarray]] = defaultdict(list)

        for game, npz_paths in self._game_npz_paths.items():
            for npz_path in npz_paths:
                self.game_to_sequences[game].append(np.load(npz_path))

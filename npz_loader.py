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

def get_sequences_by_game(game_npz_paths: Dict[str, List[Path]]):
    # Game -> dicts each represents one sequence
    # actually we should be splitting this by episode
    game_to_sequences: Dict[Path, List[dict]] = {}

    for game, npz_paths in game_npz_paths.items():
        game_path = Path(game)
        game_to_sequences[game_path] = []
        for npz_path in npz_paths:
            npz_file = np.load(npz_path)
            game_to_sequences[game_path].append(dict(npz_file))
    
    return game_to_sequences

def fix_obs_paths(
    game_to_sequences: Dict[Path, List[dict]],
    dataset_root: str | Path = "dataset",
):
    """
    The data set png paths are absolute paths from Brian's C drive, so we need to make them relative
    """
    dataset_root = Path(dataset_root)
    fixed: Dict[Path, List[dict]] = {}

    for game_path, seqs in game_to_sequences.items():
        game_path = Path(game_path)

        top_game_dir = game_path.parent.name 
        inner_game_dir = game_path.name

        fixed_seqs: List[dict] = []

        for seq_dict in seqs:
            fixed_seq_dict: dict = {}
            for key, arr in seq_dict.items():

                if key == "obs" and arr.dtype.kind in ("U", "S", "O"):
                    
                    new_obs = []
                    for s in arr:
                        p = Path(str(s))
                        parts = p.parts

                        if inner_game_dir in parts:
                            idx = parts.index(inner_game_dir)
                            rest = Path(*parts[idx + 1 :])
                        else:
                            rest = Path(*parts[-2:])

                        new_path = dataset_root / top_game_dir / inner_game_dir / rest
                        new_obs.append(str(new_path))

                    arr = np.array(new_obs, dtype=arr.dtype)

                fixed_seq_dict[key] = arr

            fixed_seqs.append(fixed_seq_dict)

        fixed[game_path] = fixed_seqs

    return fixed
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
    # Game -> npz files each represents one sequence
    game_to_sequences: Dict[Path, List[np.lib.npyio.NpzFile]] = {}

    for game, npz_paths in game_npz_paths.items():
        game_path = Path(game)
        game_to_sequences[game_path] = []
        for npz_path in npz_paths:
            game_to_sequences[game_path].append(np.load(npz_path))

    return fix_obs_paths(game_to_sequences)

def fix_obs_paths(
    game_to_sequences: Dict[Path, List[np.lib.npyio.NpzFile]],
    dataset_root: str | Path = "dataset",
):
    """
    The data set png paths are absolute paths from Brian's C drive, so we need to make them relative
    """
    dataset_root = Path(dataset_root)
    fixed: Dict[Path, List[dict]] = {}

    for game_path, seqs in game_to_sequences.items():
        game_path = Path(game_path)

        # Example game_path: dataset/raw/BeamRiderNoFrameskip-v4/BeamRiderNoFrameskip-v4
        top_game_dir = game_path.parent.name  # 'BeamRiderNoFrameskip-v4'
        inner_game_dir = game_path.name       # 'BeamRiderNoFrameskip-v4'

        fixed_seqs: List[dict] = []

        for npz in seqs:
            seq_dict: dict = {}
            for key in npz.files:
                arr = npz[key]

                if key == "obs" and arr.dtype.kind in ("U", "S", "O"):
                    # rewrite each path string
                    new_obs = []
                    for s in arr:
                        p = Path(str(s))
                        parts = p.parts

                        # Find first occurrence of the inner game dir in Brian's path
                        # e.g. .../BeamRiderNoFrameskip-v4/BeamRiderNoFrameskip-v4-recorded_images-0/0.png
                        if inner_game_dir in parts:
                            idx = parts.index(inner_game_dir)
                            # everything after the inner game dir
                            rest = Path(*parts[idx + 1 :])  # BeamRiderNoFrameskip-v4-recorded_images-0/0.png
                        else:
                            # fallback: keep last 2 components before file
                            rest = Path(*parts[-2:])

                        # Build new relative path rooted at 'dataset'
                        new_path = dataset_root / top_game_dir / inner_game_dir / rest
                        new_obs.append(str(new_path))

                    arr = np.array(new_obs, dtype=arr.dtype)

                seq_dict[key] = arr

            fixed_seqs.append(seq_dict)

        fixed[game_path] = fixed_seqs

    return fixed
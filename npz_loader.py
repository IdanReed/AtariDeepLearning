from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Iterable, List

import numpy as np
from torch.utils.data import Dataset, DataLoader


def _discover_games(root_dir: Path) -> list[Path]:
    paths = root_dir.glob("*/*")
    return list(paths)

def _discover_npz(game_dir: Path) -> list[Path]:
    npz_files = game_dir.glob("*.npz")
    return list(npz_files)

def _discover_game_npz_paths(paths: list[Path]) -> Dict[str, Any]:
    game_npz_paths = {}
    for path in paths:
        game_npz_paths[path] = _discover_npz(path)

    return game_npz_paths

def _get_sequences_by_game(game_npz_paths: Dict[str, List[Path]]):
    # Game -> dicts each represents one gameplay sequence
    # actually we should be splitting this by episode
    game_to_sequences: Dict[Path, List[dict]] = {}

    for game, npz_paths in game_npz_paths.items():
        game_path = Path(game)
        game_to_sequences[game_path] = []
        for npz_path in npz_paths:
            npz_file = np.load(npz_path)
            game_to_sequences[game_path].append(dict(npz_file))
    
    return game_to_sequences

def _fix_obs_paths(
    game_to_sequences: Dict[Path, List[dict]],
    dataset_root: str | Path = "dataset",
    is_collab: bool = False,
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
                        if is_collab:
                            split_path = str(s).split("\\")
                            p = Path(split_path[0]).joinpath(*split_path[1:])
                        else:
                            p = Path(s)

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

from episode import Episode, TimeStep

def _build_episodes_from_sequences(
    sequences_by_game: Dict[Path, List[dict]],
) -> List[Episode]:
    episodes: List[Episode] = []

    for game_path, seq_list in sequences_by_game.items():
        game_name = Path(game_path).name  # e.g. "BeamRiderNoFrameskip-v4"

        for seq_dict in seq_list:
            episode_starts = np.asarray(seq_dict["episode_starts"], dtype=bool)
            
            timestep_count = episode_starts.shape[0]
            
            num_episode_starts = np.sum(episode_starts)

            # indices where a new episode begins within this gameplay sequence
            start_indices = np.where(episode_starts)[0]

            # If there are no episode_starts, we skip this sequence
            if start_indices.size == 0:
                continue

            for i, start in enumerate(start_indices):
                end = start_indices[i + 1] if i + 1 < start_indices.size else timestep_count

                timesteps: List[TimeStep] = []
                for t in range(start, end):
                    obs_str = seq_dict["obs"][t]
                    obs_path = Path(str(obs_str))

                    # arrays are shaped (T, 1) for actions
                    model_selected_action = seq_dict["model selected actions"][t][0]
                    action_taken = seq_dict["taken actions"][t][0]

                    repeated = bool(seq_dict["repeated"][t])
                    reward = float(seq_dict["rewards"][t])

                    ts = TimeStep(
                        obs=obs_path,
                        model_selected_action=model_selected_action,
                        action_taken=action_taken,
                        repeated=repeated,
                        reward=reward,
                    )
                    timesteps.append(ts)

                if timesteps:
                    episodes.append(Episode(game_name=game_name, timesteps=timesteps))

    return episodes

def load_episodes(main_game_dirs: List[Path], holdout_game_dirs: List[Path], is_collab: bool = False) -> List[Episode]:
    all_game_dirs = main_game_dirs + holdout_game_dirs

    npz_paths_by_game = _discover_game_npz_paths(all_game_dirs)
    game_to_sequences = _get_sequences_by_game(npz_paths_by_game)
    if is_collab:
        dataset_root = "/content/dataset/dataset/"
    else:
        dataset_root = "dataset"

    sequences_by_game = _fix_obs_paths(game_to_sequences, dataset_root=dataset_root, is_collab=is_collab)
    episodes = _build_episodes_from_sequences(sequences_by_game)

    # Add game ids to episode
    game_names = sorted({episode.game_name for episode in episodes})
    name_to_id = {name: i for i, name in enumerate(game_names)}

    for episode in episodes:
        episode.assign_game_id(name_to_id)

    print(f"Loaded {len(episodes)} episodes")
    return episodes

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from episode import Episode
from episode_dataset import EpisodeSliceDataset, BinningInfo, compute_rtg_bin_range, compute_n_actions


def make_episode_dataloader(
    episodes: List[Episode],
    context_len: int = 4,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (84, 84),
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[EpisodeSliceDataset, DataLoader]:
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
    return dataset, loader


@dataclass
class DataLoaderBundle:
    """Container for train/val dataloaders and datasets."""
    train_loader: DataLoader
    val_loader: DataLoader
    train_dataset: EpisodeSliceDataset
    val_dataset: EpisodeSliceDataset


def make_train_val_dataloaders(
    episodes: List[Episode],
    holdout_game_dirs: Optional[List[Path]] = None,
    train_frac: float = 0.8,
    timestep_window_size: int = 4,
    image_size: Tuple[int, int] = (84, 84),
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[
    DataLoaderBundle,
    Optional[DataLoaderBundle],
    BinningInfo,
]:
    episodes = list(episodes)
    if len(episodes) == 0:
        raise ValueError("No episodes provided")

    # Calc bins from all episodes (other wise bins with be wrong for new data)
    rtg_min_int, rtg_max_int = compute_rtg_bin_range(episodes)
    n_actions = compute_n_actions(episodes)

    bins = BinningInfo(
        rtg_min_int=rtg_min_int,
        rtg_max_int=rtg_max_int,
        num_rtg_bins=rtg_max_int - rtg_min_int + 1,
        n_actions=n_actions,
    )

    # Split holdout game episodes out
    if holdout_game_dirs is not None:
        holdout_game_names = [game.name for game in holdout_game_dirs]
        holdout_set = set(holdout_game_names)
        main_episodes = [ep for ep in episodes if ep.game_name not in holdout_set]
        holdout_episodes = [ep for ep in episodes if ep.game_name in holdout_set]
        
        if len(main_episodes) == 0:
            raise ValueError("No main episodes")
        if len(holdout_episodes) == 0:
            raise ValueError(f"No holdout episodes")
    else:
        main_episodes = episodes
        holdout_episodes = []

    # Create main bundle
    main_bundle = _create_bundle(
        main_episodes,
        train_frac=train_frac,
        timestep_window_size=timestep_window_size,
        image_size=image_size,
        rtg_min_int=rtg_min_int,
        rtg_max_int=rtg_max_int,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Create holdout bundle if applicable
    holdout_bundle: Optional[DataLoaderBundle] = None
    if holdout_game_names is not None and len(holdout_episodes) > 0:
        holdout_bundle = _create_bundle(
            holdout_episodes,
            train_frac=train_frac,
            timestep_window_size=timestep_window_size,
            image_size=image_size,
            rtg_min_int=rtg_min_int,
            rtg_max_int=rtg_max_int,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return main_bundle, holdout_bundle, bins

def _create_bundle(
    eps: List[Episode],
    train_frac: float,
    timestep_window_size: int,
    image_size: Tuple[int, int],
    rtg_min_int: int,
    rtg_max_int: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoaderBundle:
    n_total = len(eps)
    indices = torch.randperm(n_total).tolist()
    n_train = int(train_frac * n_total)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_eps = [eps[i] for i in train_indices]
    val_eps = [eps[i] for i in val_indices]

    train_dataset = EpisodeSliceDataset(
        train_eps,
        timestep_window_size=timestep_window_size,
        image_size=image_size,
        rtg_min_int=rtg_min_int,
        rtg_max_int=rtg_max_int,
    )

    val_dataset = EpisodeSliceDataset(
        val_eps,
        timestep_window_size=timestep_window_size,
        image_size=image_size,
        rtg_min_int=rtg_min_int,
        rtg_max_int=rtg_max_int,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DataLoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

"""
Shared utilities for vast.ai experiment scripts.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils import seed_random_generators, sample_list
from npz_loader import load_episodes
from epsiode_dataloader import make_train_val_dataloaders, DataLoaderBundle
from episode_dataset import BinningInfo


# Best baseline params (from optuna tuning)
BEST_BASELINE_PARAMS = {
    'lr': 0.002226768831180977,
    'emb_size': 128,
    'n_layers': 2,
    'n_heads': 2,
    'num_epochs': 2,
}


@dataclass
class ExperimentArgs:
    """Parsed command line arguments for experiments."""
    dataset_root: Path
    output_dir: Path
    seed: int
    sample_fraction: Optional[float] = None


@dataclass 
class ExperimentData:
    """Loaded data for experiments."""
    main_bundle: DataLoaderBundle
    holdout_bundle: DataLoaderBundle
    bins: BinningInfo


def parse_experiment_args(
    description: str,
    include_sample_fraction: bool = False,
) -> ExperimentArgs:
    """Parse common experiment arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset-root", type=str, default="dataset")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=666)
    
    if include_sample_fraction:
        parser.add_argument(
            "--sample-fraction", 
            type=float, 
            default=0.1,
            help="Fraction of dataset to use (default: 0.1 = 10%)"
        )
    
    args = parser.parse_args()
    
    return ExperimentArgs(
        dataset_root=Path(args.dataset_root),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        sample_fraction=getattr(args, 'sample_fraction', None),
    )


def get_game_dirs(dataset_root: Path):
    """Get the holdout and main game directories."""
    holdout_game_dirs = [
        dataset_root / "BeamRiderNoFrameskip-v4" / "BeamRiderNoFrameskip-v4",
        dataset_root / "BreakoutNoFrameskip-v4" / "BreakoutNoFrameskip-v4",
    ]
    
    main_game_dirs = [
        dataset_root / "EnduroNoFrameskip-v4" / "EnduroNoFrameskip-v4",
        dataset_root / "MsPacmanNoFrameskip-v4" / "MsPacmanNoFrameskip-v4",
        dataset_root / "PongNoFrameskip-v4" / "PongNoFrameskip-v4",
        dataset_root / "QbertNoFrameskip-v4" / "QbertNoFrameskip-v4",
        dataset_root / "SeaquestNoFrameskip-v4" / "SeaquestNoFrameskip-v4",
        dataset_root / "SpaceInvadersNoFrameskip-v4" / "SpaceInvadersNoFrameskip-v4",
    ]
    
    return holdout_game_dirs, main_game_dirs


def load_experiment_data(
    dataset_root: Path,
    timestep_window_size: int = 4,
    sample_fraction: Optional[float] = None,
) -> ExperimentData:
    """Load episodes and create dataloaders."""
    holdout_game_dirs, main_game_dirs = get_game_dirs(dataset_root)
    
    episodes = load_episodes(main_game_dirs, holdout_game_dirs, dataset_root=dataset_root)
    
    if sample_fraction is not None:
        episodes = sample_list(episodes, fraction=sample_fraction)
    
    main_bundle, holdout_bundle, bins = make_train_val_dataloaders(
        episodes=episodes,
        holdout_game_dirs=holdout_game_dirs,
        train_frac=0.8,
        timestep_window_size=timestep_window_size,
    )
    
    return ExperimentData(
        main_bundle=main_bundle,
        holdout_bundle=holdout_bundle,
        bins=bins,
    )


def setup_experiment(
    name: str,
    description: str,
    timestep_window_size: int = 4,
    include_sample_fraction: bool = False,
) -> tuple[ExperimentArgs, ExperimentData]:
    """
    Common setup for all experiments.
    
    Returns:
        Tuple of (args, data)
    """
    args = parse_experiment_args(description, include_sample_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    seed_random_generators(args.seed)
    
    print(f"Running experiment: {name}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Output dir: {args.output_dir}")
    if args.sample_fraction:
        print(f"Sample fraction: {args.sample_fraction * 100:.0f}%")
    print()
    
    data = load_experiment_data(
        args.dataset_root,
        timestep_window_size=timestep_window_size,
        sample_fraction=args.sample_fraction,
    )
    
    return args, data
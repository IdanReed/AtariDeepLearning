from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TypeVar
import random

import optuna
import torch

@dataclass
class ModelCheckpoint:
    model: torch.nn.Module
    main_train_stats: List[Dict[str, Any]]
    main_val_stats: List[Dict[str, Any]]
    holdout_train_stats: List[Dict[str, Any]]
    holdout_val_stats: List[Dict[str, Any]]
    
    study: Optional[optuna.Study] = None,

def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    main_train_stats: List[Dict[str, Any]],
    main_val_stats: List[Dict[str, Any]],
    holdout_train_stats: List[Dict[str, Any]],
    holdout_val_stats: List[Dict[str, Any]],
    study: Optional[optuna.Study] = None,
) -> Path:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        model=model,
        main_train_stats=main_train_stats,
        main_val_stats=main_val_stats,
        holdout_train_stats=holdout_train_stats,
        holdout_val_stats=holdout_val_stats,
        study=study,
    )

    # No timestamp so we only save the latest checkpoint
    model_path = output_dir / f"model_checkpoint.pt"

    torch.save(checkpoint, model_path)
    print(f"Model and stats saved to {model_path}")
    return model_path

def safe_clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        safe_extensions = {".png", ".pt"}
        all_files = list(output_dir.iterdir())
        unsafe_files = [f for f in all_files if f.is_file() and f.suffix not in safe_extensions]

        if unsafe_files:
            raise ValueError(f"Found unexpected files in {output_dir}: {unsafe_files}")

        for f in all_files:
            if f.is_file():
                f.unlink()
        print(f"Cleared {len(all_files)} files from {output_dir}")

def sample_list(items: List[Any], fraction: float = 0.1) -> List[Any]:
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be between 0 and 1")

    sample_size = max(1, int(len(items) * fraction))
    sampled = random.sample(items, sample_size)
    print(f"Sampled {len(sampled)} items ({len(sampled)/len(items)*100:.1f}% of {len(items)} total)")
    return sampled

def seed_random_generators(seed: int = 666):
    random.seed(seed)
    torch.manual_seed(seed)
    return
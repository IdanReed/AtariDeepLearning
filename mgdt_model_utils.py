from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Any

import torch

@dataclass
class ModelCheckpoint:
    model: torch.nn.Module
    train_stats: List[Dict[str, Any]]
    val_stats: List[Dict[str, Any]]

def save_checkpoint(
    model: torch.nn.Module,
    train_stats: List[Dict[str, Any]],
    val_stats: List[Dict[str, Any]],
    output_dir: Path = Path("output"),
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    checkpoint = ModelCheckpoint(
        model=model,
        train_stats=train_stats,
        val_stats=val_stats,
    )
    model_path = output_dir / f"model_{timestamp}.pt"
    torch.save(checkpoint, model_path)
    print(f"Model and stats saved to {model_path}")
    return model_path
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")


def _extract(stats: List[Dict[str, Any]], key: str) -> np.ndarray:
    """Extract a key from all stats dictionaries into a numpy array."""
    return np.array([s[key] for s in stats], dtype=float)


def _compute_ema(values: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    """Compute exponential moving average of values."""
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * values[i]
    return ema


def _extract_loss_data(
    train_stats: List[Dict[str, Any]], val_stats: Optional[List[Dict[str, Any]]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Extract loss data from stats dictionaries.
    
    Returns:
        train_steps: Training step numbers
        train_losses: Dict with keys 'loss', 'loss_return', 'loss_action', 'loss_reward'
        val_steps: Validation step numbers (None if no validation)
        val_losses: Dict with same keys as train_losses (None if no validation)
    """
    train_steps = _extract(train_stats, "step")
    train_losses = {
        "loss": _extract(train_stats, "loss"),
        "loss_return": _extract(train_stats, "loss_return"),
        "loss_action": _extract(train_stats, "loss_action"),
        "loss_reward": _extract(train_stats, "loss_reward"),
    }
    
    has_val = val_stats is not None and len(val_stats) > 0
    val_steps = None
    val_losses = None
    if has_val:
        val_steps = _extract(val_stats, "step")
        val_losses = {
            "loss": _extract(val_stats, "loss"),
            "loss_return": _extract(val_stats, "loss_return"),
            "loss_action": _extract(val_stats, "loss_action"),
            "loss_reward": _extract(val_stats, "loss_reward"),
        }
    
    return train_steps, train_losses, val_steps, val_losses


def _plot_single_loss(
    ax: plt.Axes,
    train_steps: np.ndarray,
    train_values: np.ndarray,
    val_steps: Optional[np.ndarray],
    val_values: Optional[np.ndarray],
    color: Union[str, Tuple[float, float, float]],
    title: str,
    xlabel: str = "Step",
    ylabel: str = "Loss",
) -> None:
    """Plot a single loss subplot with train and optionally validation."""
    ax.plot(train_steps, train_values, label="Train", color=color, linewidth=1.5, alpha=0.8)
    if val_steps is not None and val_values is not None:
        ax.plot(val_steps, val_values, label="Val", color=color, linewidth=1.5, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_per_head_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    title_prefix: str,
) -> None:
    """Create 2x2 subplot grid showing per-head losses."""
    # Get seaborn colors - use consistent colors for each loss type
    colors = sns.color_palette("husl", 4)
    loss_configs = [
        ("loss", "Total Loss", colors[0]),
        ("loss_return", "Return Loss", colors[1]),
        ("loss_action", "Action Loss", colors[2]),
        ("loss_reward", "Reward Loss", colors[3]),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    main_title = (title_prefix + " ").strip() + "Losses"
    fig.suptitle(main_title, fontsize=16)
    
    for idx, (loss_key, title, color) in enumerate(loss_configs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        val_values = val_losses[loss_key] if val_losses is not None else None
        _plot_single_loss(
            ax,
            train_steps,
            train_losses[loss_key],
            val_steps,
            val_values,
            color,
            title,
        )
    
    plt.tight_layout()
    plt.show()


def _plot_combined_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    title_prefix: str,
) -> None:
    """Create a single plot showing all losses together."""
    colors = sns.color_palette("husl", 4)
    loss_configs = [
        ("loss", "Total", colors[0]),
        ("loss_return", "Return", colors[1]),
        ("loss_action", "Action", colors[2]),
        ("loss_reward", "Reward", colors[3]),
    ]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot training losses
    for loss_key, label_suffix, color in loss_configs:
        ax.plot(
            train_steps,
            train_losses[loss_key],
            label=f"{label_suffix} (Train)",
            color=color,
            linewidth=2.0,
            alpha=0.8,
        )
    
    # Plot validation losses
    if val_steps is not None and val_losses is not None:
        for loss_key, label_suffix, color in loss_configs:
            ax.plot(
                val_steps,
                val_losses[loss_key],
                label=f"{label_suffix} (Val)",
                color=color,
                linewidth=2.0,
                alpha=0.6,
            )
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title((title_prefix + " All Losses").strip())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def _plot_ema_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    title_prefix: str,
    alpha: float = 0.99,
) -> None:
    """Create plots showing EMA smoothed losses."""
    colors = sns.color_palette("husl", 4)
    loss_configs = [
        ("loss", "Total Loss", colors[0]),
        ("loss_return", "Return Loss", colors[1]),
        ("loss_action", "Action Loss", colors[2]),
        ("loss_reward", "Reward Loss", colors[3]),
    ]
    
    # Compute EMAs
    train_emas = {key: _compute_ema(values, alpha) for key, values in train_losses.items()}
    val_emas = None
    if val_losses is not None:
        val_emas = {key: _compute_ema(values, alpha) for key, values in val_losses.items()}
    
    # Per-head EMA
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    main_title = (title_prefix + " ").strip() + "Losses (EMA)"
    fig.suptitle(main_title, fontsize=16)
    
    for idx, (loss_key, title, color) in enumerate(loss_configs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        val_ema = val_emas[loss_key] if val_emas is not None else None
        _plot_single_loss(
            ax,
            train_steps,
            train_emas[loss_key],
            val_steps,
            val_ema,
            color,
            title,
        )
    
    plt.tight_layout()
    plt.show()
    
    # Combined EMA plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for loss_key, label_suffix, color in loss_configs:
        ax.plot(
            train_steps,
            train_emas[loss_key],
            label=f"{label_suffix} (Train)",
            color=color,
            linewidth=2.0,
            alpha=0.8,
        )
    
    if val_steps is not None and val_emas is not None:
        for loss_key, label_suffix, color in loss_configs:
            ax.plot(
                val_steps,
                val_emas[loss_key],
                label=f"{label_suffix} (Val)",
                color=color,
                linewidth=2.0,
                alpha=0.6,
            )
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (EMA)")
    ax.set_title((title_prefix + " All Losses (EMA)").strip())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_losses(
    train_stats: List[Dict[str, Any]],
    val_stats: Optional[List[Dict[str, Any]]] = None,
    title_prefix: str = "",
) -> None:

    if not train_stats:
        raise ValueError("train_stats is empty")
    
    train_steps, train_losses, val_steps, val_losses = _extract_loss_data(train_stats, val_stats)
    
    _plot_per_head_losses(train_steps, train_losses, val_steps, val_losses, title_prefix)
    _plot_combined_losses(train_steps, train_losses, val_steps, val_losses, title_prefix)
    _plot_ema_losses(train_steps, train_losses, val_steps, val_losses, title_prefix)
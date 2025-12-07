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


def _extract_optional(stats: List[Dict[str, Any]], key: str, default: Any = 0) -> np.ndarray:
    """Extract a key from stats, using default if key doesn't exist."""
    return np.array([s.get(key, default) for s in stats], dtype=float)


def _compute_ema(values: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    """Compute exponential moving average of values."""
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * values[i]
    return ema


def _get_epoch_boundaries(train_stats: List[Dict[str, Any]]) -> List[int]:
    """Get global_step values where epochs end."""
    boundaries = []
    if not train_stats:
        return boundaries
    
    prev_epoch = train_stats[0].get("epoch", 1)
    for s in train_stats:
        curr_epoch = s.get("epoch", 1)
        if curr_epoch != prev_epoch:
            # Previous entry was end of epoch
            boundaries.append(s.get("global_step", s.get("step", 0)) - 1)
            prev_epoch = curr_epoch
    
    # Add final step as last boundary
    last = train_stats[-1]
    boundaries.append(last.get("global_step", last.get("step", len(train_stats))))
    
    return boundaries


def _extract_loss_data(
    train_stats: List[Dict[str, Any]], val_stats: Optional[List[Dict[str, Any]]] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], Optional[np.ndarray]]:
    """
    Extract loss data from stats dictionaries.
    
    Returns:
        train_steps: Training step numbers (global_step if available, else step)
        train_losses: Dict with keys 'loss', 'loss_return', 'loss_action', 'loss_reward'
        val_steps: Validation step numbers (None if no validation)
        val_losses: Dict with same keys as train_losses (None if no validation)
        val_is_mid_epoch: Boolean array indicating mid-epoch validation (None if no validation)
    """
    # Use global_step if available, fallback to step
    if train_stats and "global_step" in train_stats[0]:
        train_steps = _extract(train_stats, "global_step")
    else:
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
    val_is_mid_epoch = None
    
    if has_val:
        # Use global_step for validation positioning on same axis as training
        val_steps = _extract(val_stats, "global_step")
        val_losses = {
            "loss": _extract(val_stats, "loss"),
            "loss_return": _extract(val_stats, "loss_return"),
            "loss_action": _extract(val_stats, "loss_action"),
            "loss_reward": _extract(val_stats, "loss_reward"),
        }
        val_is_mid_epoch = np.array([s.get("is_mid_epoch", False) for s in val_stats], dtype=bool)
    
    return train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch


def _aggregate_validation_stats(val_stats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate per-batch validation stats into per-validation-run stats.
    Groups by (epoch, global_step, is_mid_epoch) and averages losses/accuracies.
    """
    if not val_stats:
        return []
    
    from collections import defaultdict
    
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in val_stats:
        key = (s.get("epoch", 1), s.get("global_step", 0), s.get("is_mid_epoch", False))
        grouped[key].append(s)
    
    aggregated = []
    for (epoch, global_step, is_mid_epoch), batch_stats in sorted(grouped.items()):
        agg = {
            "epoch": epoch,
            "global_step": global_step,
            "is_mid_epoch": is_mid_epoch,
            "loss": float(np.mean([s["loss"] for s in batch_stats])),
            "loss_return": float(np.mean([s["loss_return"] for s in batch_stats])),
            "loss_action": float(np.mean([s["loss_action"] for s in batch_stats])),
            "loss_reward": float(np.mean([s["loss_reward"] for s in batch_stats])),
            "return_acc": float(np.mean([s["return_acc"] for s in batch_stats])),
            "action_acc": float(np.mean([s["action_acc"] for s in batch_stats])),
            "reward_acc": float(np.mean([s["reward_acc"] for s in batch_stats])),
        }
        aggregated.append(agg)
    
    return aggregated


def _add_epoch_boundaries(ax: plt.Axes, boundaries: List[int], max_y: float) -> None:
    """Add vertical lines at epoch boundaries."""
    for i, boundary in enumerate(boundaries[:-1]):  # Skip last boundary (end of training)
        ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)


def _plot_single_loss(
    ax: plt.Axes,
    train_steps: np.ndarray,
    train_values: np.ndarray,
    val_steps: Optional[np.ndarray],
    val_values: Optional[np.ndarray],
    val_is_mid_epoch: Optional[np.ndarray],
    color: Union[str, Tuple[float, float, float]],
    title: str,
    epoch_boundaries: Optional[List[int]] = None,
    xlabel: str = "Step",
    ylabel: str = "Loss",
) -> None:
    """Plot a single loss subplot with train and optionally validation."""
    ax.plot(train_steps, train_values, label="Train", color=color, linewidth=1.5, alpha=0.8)
    
    if val_steps is not None and val_values is not None:
        if val_is_mid_epoch is not None and len(val_is_mid_epoch) > 0:
            # Plot mid-epoch validation with different marker
            mid_mask = val_is_mid_epoch
            end_mask = ~val_is_mid_epoch
            
            if np.any(mid_mask):
                ax.scatter(
                    val_steps[mid_mask], val_values[mid_mask],
                    label="Val (mid-epoch)", color=color, marker='o', s=30, alpha=0.6
                )
            if np.any(end_mask):
                ax.scatter(
                    val_steps[end_mask], val_values[end_mask],
                    label="Val (end-epoch)", color=color, marker='s', s=50, alpha=0.8
                )
        else:
            ax.plot(val_steps, val_values, label="Val", color=color, linewidth=1.5, alpha=0.6)
    
    # Add epoch boundaries
    if epoch_boundaries:
        _add_epoch_boundaries(ax, epoch_boundaries, train_values.max())
    
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
    val_is_mid_epoch: Optional[np.ndarray],
    title_prefix: str,
    epoch_boundaries: Optional[List[int]] = None,
) -> None:
    """Create 2x2 subplot grid showing per-head losses."""
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
            val_is_mid_epoch,
            color,
            title,
            epoch_boundaries,
        )
    
    plt.tight_layout()
    plt.show()


def _plot_combined_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    val_is_mid_epoch: Optional[np.ndarray],
    title_prefix: str,
    epoch_boundaries: Optional[List[int]] = None,
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
    
    # Plot validation losses with mid/end-epoch distinction
    if val_steps is not None and val_losses is not None:
        if val_is_mid_epoch is not None and len(val_is_mid_epoch) > 0:
            mid_mask = val_is_mid_epoch
            end_mask = ~val_is_mid_epoch
            
            for loss_key, label_suffix, color in loss_configs:
                if np.any(mid_mask):
                    ax.scatter(
                        val_steps[mid_mask], val_losses[loss_key][mid_mask],
                        label=f"{label_suffix} (Val mid)",
                        color=color, marker='o', s=30, alpha=0.5
                    )
                if np.any(end_mask):
                    ax.scatter(
                        val_steps[end_mask], val_losses[loss_key][end_mask],
                        label=f"{label_suffix} (Val end)",
                        color=color, marker='s', s=50, alpha=0.8
                    )
        else:
            for loss_key, label_suffix, color in loss_configs:
                ax.plot(
                    val_steps,
                    val_losses[loss_key],
                    label=f"{label_suffix} (Val)",
                    color=color,
                    linewidth=2.0,
                    alpha=0.6,
                )
    
    # Add epoch boundaries
    if epoch_boundaries:
        _add_epoch_boundaries(ax, epoch_boundaries, max(train_losses["loss"]))
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title((title_prefix + " All Losses").strip())
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def _plot_ema_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    val_is_mid_epoch: Optional[np.ndarray],
    title_prefix: str,
    epoch_boundaries: Optional[List[int]] = None,
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
    
    # Compute EMAs for training
    train_emas = {key: _compute_ema(values, alpha) for key, values in train_losses.items()}
    
    # For validation, we don't apply EMA since these are sparse points
    val_emas = val_losses  # Keep as-is
    
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
            val_is_mid_epoch,
            color,
            title,
            epoch_boundaries,
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
        if val_is_mid_epoch is not None and len(val_is_mid_epoch) > 0:
            mid_mask = val_is_mid_epoch
            end_mask = ~val_is_mid_epoch
            
            for loss_key, label_suffix, color in loss_configs:
                if np.any(mid_mask):
                    ax.scatter(
                        val_steps[mid_mask], val_emas[loss_key][mid_mask],
                        label=f"{label_suffix} (Val mid)",
                        color=color, marker='o', s=30, alpha=0.5
                    )
                if np.any(end_mask):
                    ax.scatter(
                        val_steps[end_mask], val_emas[loss_key][end_mask],
                        label=f"{label_suffix} (Val end)",
                        color=color, marker='s', s=50, alpha=0.8
                    )
        else:
            for loss_key, label_suffix, color in loss_configs:
                ax.plot(
                    val_steps,
                    val_emas[loss_key],
                    label=f"{label_suffix} (Val)",
                    color=color,
                    linewidth=2.0,
                    alpha=0.6,
                )
    
    # Add epoch boundaries
    if epoch_boundaries:
        _add_epoch_boundaries(ax, epoch_boundaries, max(train_emas["loss"]))
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (EMA)")
    ax.set_title((title_prefix + " All Losses (EMA)").strip())
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_losses(
    train_stats: List[Dict[str, Any]],
    val_stats: Optional[List[Dict[str, Any]]] = None,
    title_prefix: str = "",
    aggregate_validation: bool = True,
) -> None:
    """
    Plot training and validation losses.
    
    Args:
        train_stats: List of training stats dictionaries
        val_stats: Optional list of validation stats dictionaries
        title_prefix: Prefix for plot titles
        aggregate_validation: If True, aggregate per-batch validation stats 
                             into per-validation-run averages
    """
    if not train_stats:
        raise ValueError("train_stats is empty")
    
    # Aggregate validation stats if requested
    processed_val_stats = val_stats
    if aggregate_validation and val_stats:
        processed_val_stats = _aggregate_validation_stats(val_stats)
    
    # Get epoch boundaries for visualization
    epoch_boundaries = _get_epoch_boundaries(train_stats)
    
    train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch = _extract_loss_data(
        train_stats, processed_val_stats
    )
    
    _plot_per_head_losses(
        train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch, 
        title_prefix, epoch_boundaries
    )
    _plot_combined_losses(
        train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch,
        title_prefix, epoch_boundaries
    )
    _plot_ema_losses(
        train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch,
        title_prefix, epoch_boundaries
    )

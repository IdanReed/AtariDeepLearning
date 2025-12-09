from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_palette("husl")


def _save_and_show(fig: plt.Figure, output_dir: Optional[Path], filename: str, no_show: bool = False) -> None:
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename}.png"
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {filepath}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def _extract(stats: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([s[key] for s in stats], dtype=float)


def _extract_optional(stats: List[Dict[str, Any]], key: str, default: Any = 0) -> np.ndarray:
    return np.array([s.get(key, default) for s in stats], dtype=float)


def _compute_ema(values: np.ndarray, alpha: float = 0.99) -> np.ndarray:
    ema = np.zeros_like(values)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * values[i]
    return ema


def _get_epoch_boundaries(train_stats: List[Dict[str, Any]]) -> List[int]:
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
        # Include F1 scores if present (only computed during validation)
        if "action_f1" in batch_stats[0]:
            agg["action_f1"] = float(np.mean([s["action_f1"] for s in batch_stats]))
        if "return_f1" in batch_stats[0]:
            agg["return_f1"] = float(np.mean([s["return_f1"] for s in batch_stats]))
        if "reward_f1" in batch_stats[0]:
            agg["reward_f1"] = float(np.mean([s["reward_f1"] for s in batch_stats]))
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
    output_dir: Optional[Path] = None,
    no_show: bool = False,
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
    safe_prefix = title_prefix.replace(" ", "_").replace("/", "_").lower() if title_prefix else "train"
    _save_and_show(fig, output_dir, f"model_{safe_prefix}_losses_per_head", no_show)


def _plot_combined_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    val_is_mid_epoch: Optional[np.ndarray],
    title_prefix: str,
    epoch_boundaries: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    no_show: bool = False,
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
    safe_prefix = title_prefix.replace(" ", "_").replace("/", "_").lower() if title_prefix else "train"
    _save_and_show(fig, output_dir, f"model_{safe_prefix}_losses_combined", no_show)


def _plot_ema_losses(
    train_steps: np.ndarray,
    train_losses: Dict[str, np.ndarray],
    val_steps: Optional[np.ndarray],
    val_losses: Optional[Dict[str, np.ndarray]],
    val_is_mid_epoch: Optional[np.ndarray],
    title_prefix: str,
    epoch_boundaries: Optional[List[int]] = None,
    alpha: float = 0.99,
    output_dir: Optional[Path] = None,
    no_show: bool = False,
) -> None:
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
    
    safe_prefix = title_prefix.replace(" ", "_").replace("/", "_").lower() if title_prefix else "train"
    
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
    _save_and_show(fig, output_dir, f"model_{safe_prefix}_losses_ema_per_head", no_show)
    
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
    _save_and_show(fig2, output_dir, f"model_{safe_prefix}_losses_ema_combined", no_show)


def plot_losses(
    train_stats: List[Dict[str, Any]],
    val_stats: Optional[List[Dict[str, Any]]] = None,
    title_prefix: str = "",
    aggregate_validation: bool = True,
    output_dir: Optional[Path] = None,
    no_show: bool = False,
) -> None:
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
        title_prefix, epoch_boundaries, output_dir, no_show
    )
    _plot_combined_losses(
        train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch,
        title_prefix, epoch_boundaries, output_dir, no_show
    )
    _plot_ema_losses(
        train_steps, train_losses, val_steps, val_losses, val_is_mid_epoch,
        title_prefix, epoch_boundaries, output_dir=output_dir, no_show=no_show
    )


def plot_holdout_comparison(
    main_train_stats: List[Dict[str, Any]],
    main_val_stats: Optional[List[Dict[str, Any]]],
    holdout_train_stats: List[Dict[str, Any]],
    holdout_val_stats: Optional[List[Dict[str, Any]]],
    title_prefix: str = "",
    aggregate_validation: bool = True,
    ema_alpha: float = 0.9,
    output_dir: Optional[Path] = None,
    no_show: bool = False,
) -> None:
    if not main_train_stats:
        raise ValueError("main_train_stats is empty")
    if not holdout_train_stats:
        raise ValueError("holdout_train_stats is empty")
    
    safe_prefix = title_prefix.replace(" ", "_").replace("/", "_").lower() if title_prefix else "holdout"
    
    # Aggregate validation stats
    proc_main_val = main_val_stats
    proc_holdout_val = holdout_val_stats
    if aggregate_validation:
        if main_val_stats:
            proc_main_val = _aggregate_validation_stats(main_val_stats)
        if holdout_val_stats:
            proc_holdout_val = _aggregate_validation_stats(holdout_val_stats)
    
    # Extract data
    main_steps = _extract(main_train_stats, "global_step") if "global_step" in main_train_stats[0] else _extract(main_train_stats, "step")
    main_loss = _extract(main_train_stats, "loss")
    main_loss_ema = _compute_ema(main_loss, ema_alpha)
    
    holdout_steps = _extract(holdout_train_stats, "global_step") if "global_step" in holdout_train_stats[0] else _extract(holdout_train_stats, "step")
    holdout_loss = _extract(holdout_train_stats, "loss")
    holdout_loss_ema = _compute_ema(holdout_loss, ema_alpha)
    
    # Get colors
    main_color = sns.color_palette("husl", 4)[0]
    holdout_color = sns.color_palette("husl", 4)[2]
    
    # Plot 1: Side-by-side loss comparison (normalized steps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle((title_prefix + " Main vs Holdout Training").strip(), fontsize=14)
    
    # Left: Absolute steps
    ax1 = axes[0]
    ax1.plot(main_steps, main_loss_ema, label="Main Training", color=main_color, linewidth=2, alpha=0.8)
    ax1.plot(holdout_steps, holdout_loss_ema, label="Holdout Fine-tune", color=holdout_color, linewidth=2, alpha=0.8)
    
    if proc_main_val:
        main_val_steps = _extract(proc_main_val, "global_step")
        main_val_loss = _extract(proc_main_val, "loss")
        ax1.scatter(main_val_steps, main_val_loss, label="Main Val", color=main_color, marker='s', s=40, alpha=0.6)
    
    if proc_holdout_val:
        holdout_val_steps = _extract(proc_holdout_val, "global_step")
        holdout_val_loss = _extract(proc_holdout_val, "loss")
        ax1.scatter(holdout_val_steps, holdout_val_loss, label="Holdout Val", color=holdout_color, marker='s', s=40, alpha=0.6)
    
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Loss (EMA)")
    ax1.set_title("Training Progress (Absolute Steps)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Normalized to show relative learning speed
    ax2 = axes[1]
    # Normalize steps to [0, 1] for each phase
    main_steps_norm = (main_steps - main_steps.min()) / (main_steps.max() - main_steps.min() + 1e-8)
    holdout_steps_norm = (holdout_steps - holdout_steps.min()) / (holdout_steps.max() - holdout_steps.min() + 1e-8)
    
    ax2.plot(main_steps_norm, main_loss_ema, label="Main Training", color=main_color, linewidth=2, alpha=0.8)
    ax2.plot(holdout_steps_norm, holdout_loss_ema, label="Holdout Fine-tune", color=holdout_color, linewidth=2, alpha=0.8)
    ax2.set_xlabel("Normalized Progress (0-1)")
    ax2.set_ylabel("Loss (EMA)")
    ax2.set_title("Learning Speed Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(fig, output_dir, f"comparison_{safe_prefix}_main_vs_holdout", no_show)
    
    # Plot 2: Per-head loss comparison
    loss_keys = ["loss_return", "loss_action", "loss_reward"]
    loss_titles = ["Return Loss", "Action Loss", "Reward Loss"]
    colors = sns.color_palette("husl", 3)
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle((title_prefix + " Per-Head Loss: Main vs Holdout").strip(), fontsize=14)
    
    for idx, (loss_key, title) in enumerate(zip(loss_keys, loss_titles)):
        ax = axes2[idx]
        
        main_loss_head = _compute_ema(_extract(main_train_stats, loss_key), ema_alpha)
        holdout_loss_head = _compute_ema(_extract(holdout_train_stats, loss_key), ema_alpha)
        
        ax.plot(main_steps_norm, main_loss_head, label="Main", color=colors[idx], linewidth=2, alpha=0.8)
        ax.plot(holdout_steps_norm, holdout_loss_head, label="Holdout", color=colors[idx], linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel("Normalized Progress")
        ax.set_ylabel("Loss (EMA)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_and_show(fig2, output_dir, f"comparison_{safe_prefix}_per_head_loss", no_show)
    
    # Plot 3: Accuracy comparison
    acc_keys = ["return_acc", "action_acc", "reward_acc"]
    acc_titles = ["Return Accuracy", "Action Accuracy", "Reward Accuracy"]
    
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))
    fig3.suptitle((title_prefix + " Accuracy: Main vs Holdout").strip(), fontsize=14)
    
    for idx, (acc_key, title) in enumerate(zip(acc_keys, acc_titles)):
        ax = axes3[idx]
        
        main_acc = _compute_ema(_extract(main_train_stats, acc_key), ema_alpha)
        holdout_acc = _compute_ema(_extract(holdout_train_stats, acc_key), ema_alpha)
        
        ax.plot(main_steps_norm, main_acc, label="Main", color=colors[idx], linewidth=2, alpha=0.8)
        ax.plot(holdout_steps_norm, holdout_acc, label="Holdout", color=colors[idx], linewidth=2, alpha=0.8, linestyle='--')
        
        ax.set_xlabel("Normalized Progress")
        ax.set_ylabel("Accuracy (EMA)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    _save_and_show(fig3, output_dir, f"comparison_{safe_prefix}_accuracy", no_show)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("HOLDOUT ADAPTATION SUMMARY")
    print("="*60)
    print(f"Main training steps: {len(main_train_stats)}")
    print(f"Holdout fine-tune steps: {len(holdout_train_stats)}")
    print(f"\nMain training - Final loss (EMA): {main_loss_ema[-1]:.4f}")
    print(f"Holdout fine-tune - Initial loss (EMA): {holdout_loss_ema[0]:.4f}")
    print(f"Holdout fine-tune - Final loss (EMA): {holdout_loss_ema[-1]:.4f}")
    print(f"Holdout loss reduction: {holdout_loss_ema[0] - holdout_loss_ema[-1]:.4f}")
    
    # Compute how quickly holdout reaches main's final loss
    main_final = main_loss_ema[-1]
    steps_to_match = None
    for i, loss in enumerate(holdout_loss_ema):
        if loss <= main_final:
            steps_to_match = i + 1
            break
    
    if steps_to_match:
        pct = steps_to_match / len(holdout_loss_ema) * 100
        print(f"\nHoldout reached main's final loss at step {steps_to_match} ({pct:.1f}% of fine-tuning)")
    else:
        print(f"\nHoldout did not reach main's final loss of {main_final:.4f}")
    print("="*60)

@dataclass
class ExperimentData:
    name: str
    main_train_stats: List[Dict[str, Any]]
    main_val_stats: List[Dict[str, Any]]
    holdout_train_stats: List[Dict[str, Any]]
    holdout_val_stats: List[Dict[str, Any]]


def _steps_to_reach_threshold(
    stats: List[Dict[str, Any]], 
    metric_key: str, 
    threshold: float,
    higher_is_better: bool = True,
) -> Optional[int]:
    """Find the first step where metric reaches/exceeds threshold."""
    for s in stats:
        val = s.get(metric_key)
        if val is None:
            continue
        if higher_is_better and val >= threshold:
            return s.get("global_step", s.get("step", 0))
        elif not higher_is_better and val <= threshold:
            return s.get("global_step", s.get("step", 0))
    return None


def experiment_comparison(
    experiments: List[ExperimentData],
    f1_thresholds: List[float] = [0.3, 0.4, 0.5, 0.6],
    output_dir: Optional[Path] = None,
    no_show: bool = False,
) -> Dict[str, Any]:

    colors = sns.color_palette("husl", len(experiments))
    
    processed_experiments = []
    for exp in experiments:
        agg_holdout_val = _aggregate_validation_stats(exp.holdout_val_stats) if exp.holdout_val_stats else []
        agg_main_val = _aggregate_validation_stats(exp.main_val_stats) if exp.main_val_stats else []
        processed_experiments.append({
            "name": exp.name,
            "holdout_train": exp.holdout_train_stats,
            "holdout_val": agg_holdout_val,
            "main_train": exp.main_train_stats,
            "main_val": agg_main_val,
        })
    
    # Holdout validation loss comparison
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    for idx, exp in enumerate(processed_experiments):
        if not exp["holdout_val"]:
            continue
        steps = _extract(exp["holdout_val"], "global_step")
        loss = _extract(exp["holdout_val"], "loss")
        ax1.plot(steps, loss, label=exp["name"], color=colors[idx], linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel("Global Step")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Holdout Validation Loss Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_and_show(fig1, output_dir, "experiment_comparison_holdout_val_loss", no_show)
    
    # Holdout validation F1 comparison
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    has_f1_data = False
    for idx, exp in enumerate(processed_experiments):
        if not exp["holdout_val"]:
            continue
        # Check if F1 data exists
        if "action_f1" not in exp["holdout_val"][0]:
            continue
        has_f1_data = True
        steps = _extract(exp["holdout_val"], "global_step")
        f1 = _extract(exp["holdout_val"], "action_f1")
        ax2.plot(steps, f1, label=exp["name"], color=colors[idx], linewidth=2, marker='o', markersize=4)
    
    if has_f1_data:
        # Add threshold lines
        for thresh in f1_thresholds:
            ax2.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax2.set_xlabel("Global Step")
        ax2.set_ylabel("Action F1 Score (macro)")
        ax2.set_title("Holdout Validation Action F1 Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        plt.tight_layout()
        _save_and_show(fig2, output_dir, "experiment_comparison_holdout_val_f1", no_show)
    else:
        plt.close(fig2)
    
    # F1 steps to reach thresholds (using validation data since F1 is only computed during eval)
    steps_to_f1_threshold: Dict[str, Dict[float, Optional[int]]] = {}
    
    for exp in processed_experiments:
        exp_name = exp["name"]
        steps_to_f1_threshold[exp_name] = {}
        
        # Use validation stats for F1 since it's only computed during eval
        stats_to_use = exp["holdout_val"] if exp["holdout_val"] else []
        
        for thresh in f1_thresholds:
            if stats_to_use and "action_f1" in stats_to_use[0]:
                steps = _steps_to_reach_threshold(stats_to_use, "action_f1", thresh, higher_is_better=True)
            else:
                steps = None
            steps_to_f1_threshold[exp_name][thresh] = steps
    
    # Holdout validation accuracy comparison
    fig_acc_val, ax_acc_val = plt.subplots(1, 1, figsize=(10, 6))
    
    has_acc_data = False
    for idx, exp in enumerate(processed_experiments):
        if not exp["holdout_val"]:
            continue
        if "action_acc" not in exp["holdout_val"][0]:
            continue
        has_acc_data = True
        steps = _extract(exp["holdout_val"], "global_step")
        acc = _extract(exp["holdout_val"], "action_acc")
        ax_acc_val.plot(steps, acc, label=exp["name"], color=colors[idx], linewidth=2, marker='o', markersize=4)
    
    if has_acc_data:
        # Add threshold lines for accuracy
        acc_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for thresh in acc_thresholds:
            ax_acc_val.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax_acc_val.set_xlabel("Global Step")
        ax_acc_val.set_ylabel("Action Accuracy")
        ax_acc_val.set_title("Holdout Validation Action Accuracy Comparison")
        ax_acc_val.legend()
        ax_acc_val.grid(True, alpha=0.3)
        ax_acc_val.set_ylim(0, 1)
        plt.tight_layout()
        _save_and_show(fig_acc_val, output_dir, "experiment_comparison_holdout_val_acc", no_show)
    else:
        plt.close(fig_acc_val)
    
    # Holdout training accuracy over steps
    fig_acc_train, ax_acc_train = plt.subplots(1, 1, figsize=(10, 6))
    
    has_train_acc = False
    for idx, exp in enumerate(processed_experiments):
        if not exp["holdout_train"]:
            continue
        if "action_acc" not in exp["holdout_train"][0]:
            continue
        has_train_acc = True
        steps = _extract(exp["holdout_train"], "global_step")
        acc = _extract(exp["holdout_train"], "action_acc")
        acc_ema = _compute_ema(acc, alpha=0.9)
        ax_acc_train.plot(steps, acc_ema, label=exp["name"], color=colors[idx], linewidth=2, alpha=0.8)
    
    if has_train_acc:
        acc_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for thresh in acc_thresholds:
            ax_acc_train.axhline(y=thresh, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        ax_acc_train.set_xlabel("Global Step")
        ax_acc_train.set_ylabel("Action Accuracy (EMA)")
        ax_acc_train.set_title("Holdout Training Action Accuracy Comparison")
        ax_acc_train.legend()
        ax_acc_train.grid(True, alpha=0.3)
        ax_acc_train.set_ylim(0, 1)
        plt.tight_layout()
        _save_and_show(fig_acc_train, output_dir, "experiment_comparison_holdout_train_acc", no_show)
    else:
        plt.close(fig_acc_train)
    
    # Accuracy steps to reach thresholds
    acc_thresholds_for_steps = [0.3, 0.4, 0.5, 0.6, 0.7]
    steps_to_acc_threshold: Dict[str, Dict[float, Optional[int]]] = {}
    
    for exp in processed_experiments:
        exp_name = exp["name"]
        steps_to_acc_threshold[exp_name] = {}
        
        stats_to_use = exp["holdout_train"] if exp["holdout_train"] else []
        
        for thresh in acc_thresholds_for_steps:
            if stats_to_use and "action_acc" in stats_to_use[0]:
                steps = _steps_to_reach_threshold(stats_to_use, "action_acc", thresh, higher_is_better=True)
            else:
                steps = None
            steps_to_acc_threshold[exp_name][thresh] = steps
    
    # Steps to accuracy threshold bar chart
    if has_train_acc:
        fig_acc_steps, ax_acc_steps = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(acc_thresholds_for_steps))
        width = 0.8 / len(experiments)
        
        for idx, exp in enumerate(processed_experiments):
            exp_name = exp["name"]
            values = []
            for thresh in acc_thresholds_for_steps:
                steps = steps_to_acc_threshold[exp_name].get(thresh)
                values.append(steps if steps is not None else 0)
            
            offset = (idx - len(experiments) / 2 + 0.5) * width
            ax_acc_steps.bar(x + offset, values, width, label=exp_name, color=colors[idx])
        
        ax_acc_steps.set_xlabel("Accuracy Threshold")
        ax_acc_steps.set_ylabel("Steps to Reach Threshold")
        ax_acc_steps.set_title("Steps to Reach Action Accuracy Thresholds (Holdout Training)")
        ax_acc_steps.set_xticks(x)
        ax_acc_steps.set_xticklabels([f'Acc >= {t}' for t in acc_thresholds_for_steps])
        ax_acc_steps.legend()
        ax_acc_steps.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        _save_and_show(fig_acc_steps, output_dir, "experiment_comparison_steps_to_acc", no_show)
    
    # F1 steps to threshold bar chart (using validation F1)
    if has_f1_data:
        fig4, ax4 = plt.subplots(1, 1, figsize=(12, 6))
        
        x = np.arange(len(f1_thresholds))
        width = 0.8 / len(experiments)
        
        for idx, exp in enumerate(processed_experiments):
            exp_name = exp["name"]
            values = []
            for thresh in f1_thresholds:
                steps = steps_to_f1_threshold[exp_name].get(thresh)
                values.append(steps if steps is not None else 0)
            
            offset = (idx - len(experiments) / 2 + 0.5) * width
            ax4.bar(x + offset, values, width, label=exp_name, color=colors[idx])
        
        ax4.set_xlabel("F1 Threshold")
        ax4.set_ylabel("Steps to Reach Threshold")
        ax4.set_title("Steps to Reach Action F1 Thresholds (Holdout Validation)")
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'F1 >= {t}' for t in f1_thresholds])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        _save_and_show(fig4, output_dir, "experiment_comparison_steps_to_f1", no_show)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 70)
    
    for exp in processed_experiments:
        exp_name = exp["name"]
        print(f"\n{exp_name}:")
        print("-" * 40)
        
        if exp["holdout_train"]:
            final_step = exp["holdout_train"][-1].get("global_step", len(exp["holdout_train"]))
            print(f"  Total holdout training steps: {final_step}")
        
        if exp["holdout_val"]:
            final_loss = exp["holdout_val"][-1].get("loss", float('nan'))
            print(f"  Final holdout val loss: {final_loss:.4f}")
            if "action_f1" in exp["holdout_val"][-1]:
                final_f1 = exp["holdout_val"][-1].get("action_f1", float('nan'))
                print(f"  Final holdout val F1: {final_f1:.4f}")
            if "action_acc" in exp["holdout_val"][-1]:
                final_acc = exp["holdout_val"][-1].get("action_acc", float('nan'))
                print(f"  Final holdout val accuracy: {final_acc:.4f}")
        
        if has_f1_data:
            print(f"  Steps to reach F1 thresholds (validation):")
            for thresh in f1_thresholds:
                steps = steps_to_f1_threshold[exp_name].get(thresh)
                if steps is not None:
                    print(f"    F1 >= {thresh}: {steps} steps")
                else:
                    print(f"    F1 >= {thresh}: not reached")
        
        if has_train_acc:
            print(f"  Steps to reach accuracy thresholds:")
            for thresh in acc_thresholds_for_steps:
                steps = steps_to_acc_threshold[exp_name].get(thresh)
                if steps is not None:
                    print(f"    Acc >= {thresh}: {steps} steps")
                else:
                    print(f"    Acc >= {thresh}: not reached")
    
    print("\n" + "=" * 70)
    
    return {
        "steps_to_f1_threshold": steps_to_f1_threshold,
        "steps_to_acc_threshold": steps_to_acc_threshold,
    }
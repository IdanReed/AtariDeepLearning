from __future__ import annotations

from typing import List, Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np


def _extract(stats: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.array([s[key] for s in stats], dtype=float)


def plot_losses(
    train_stats: List[Dict[str, Any]],
    val_stats: Optional[List[Dict[str, Any]]] = None,
    title_prefix: str = "",
) -> None:
    """
    Plot per-head losses for training (and optionally validation).

    Expects each stats dict to contain:
      - "step", "loss", "loss_return", "loss_action", "loss_reward"
    """
    if not train_stats:
        raise ValueError("train_stats is empty")

    # ----- training series -----
    train_steps = _extract(train_stats, "step")
    train_loss = _extract(train_stats, "loss")
    train_loss_R = _extract(train_stats, "loss_return")
    train_loss_A = _extract(train_stats, "loss_action")
    train_loss_r = _extract(train_stats, "loss_reward")

    # ----- optional validation series -----
    has_val = val_stats is not None and len(val_stats) > 0
    if has_val:
        # use their internal step index (1..len(val_stats))
        val_steps = _extract(val_stats, "step")
        val_loss = _extract(val_stats, "loss")
        val_loss_R = _extract(val_stats, "loss_return")
        val_loss_A = _extract(val_stats, "loss_action")
        val_loss_r = _extract(val_stats, "loss_reward")

    # ========================
    # Per-head loss subplots
    # ========================

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    main_title = (title_prefix + " ").strip() + "Losses"
    fig.suptitle(main_title, fontsize=16)

    # 1) Total loss
    axes[0, 0].plot(
        train_steps, train_loss, label="Train", color="black", linewidth=1.5
    )
    if has_val:
        axes[0, 0].plot(
            val_steps,
            val_loss,
            label="Val",
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2) Return loss
    axes[0, 1].plot(
        train_steps, train_loss_R, label="Train", color="blue", linewidth=1.5
    )
    if has_val:
        axes[0, 1].plot(
            val_steps,
            val_loss_R,
            label="Val",
            color="blue",
            linestyle="--",
            linewidth=1.0,
        )
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Return Loss")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3) Action loss
    axes[1, 0].plot(
        train_steps, train_loss_A, label="Train", color="green", linewidth=1.5
    )
    if has_val:
        axes[1, 0].plot(
            val_steps,
            val_loss_A,
            label="Val",
            color="green",
            linestyle="--",
            linewidth=1.0,
        )
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Action Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4) Reward loss
    axes[1, 1].plot(
        train_steps, train_loss_r, label="Train", color="red", linewidth=1.5
    )
    if has_val:
        axes[1, 1].plot(
            val_steps,
            val_loss_r,
            label="Val",
            color="red",
            linestyle="--",
            linewidth=1.0,
        )
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Reward Loss")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

    # ========================
    # Combined loss plot
    # ========================

    fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(train_steps, train_loss, label="Total (Train)", color="black", linewidth=2)
    ax.plot(
        train_steps,
        train_loss_R,
        label="Return (Train)",
        color="blue",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        train_steps,
        train_loss_A,
        label="Action (Train)",
        color="green",
        linewidth=1.5,
        alpha=0.7,
    )
    ax.plot(
        train_steps,
        train_loss_r,
        label="Reward (Train)",
        color="red",
        linewidth=1.5,
        alpha=0.7,
    )

    if has_val:
        ax.plot(
            val_steps,
            val_loss,
            label="Total (Val)",
            color="black",
            linestyle="--",
            linewidth=1.5,
        )
        ax.plot(
            val_steps,
            val_loss_R,
            label="Return (Val)",
            color="blue",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.plot(
            val_steps,
            val_loss_A,
            label="Action (Val)",
            color="green",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.plot(
            val_steps,
            val_loss_r,
            label="Reward (Val)",
            color="red",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title((title_prefix + " All Losses").strip())
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

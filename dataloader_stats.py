from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Match style from mgdt_model_stats
sns.set_style("whitegrid")
sns.set_palette("husl")


def inspect_dataloader(
    dataloader,
    title: str,
    max_batches: int = 50,
    output_dir: Optional[Path] = None,
    max_bars_for_labels: int = 20,
) -> dict[str, Any]:
    all_actions = []
    all_rewards = []
    all_rtg = []
    all_reward_bins = []
    all_rtg_bins = []
    all_game_names = []

    first_shapes = None
    n_batches_used = 0

    for b_idx, batch in enumerate(dataloader):
        if b_idx >= max_batches:
            break

        n_batches_used += 1

        frames = batch["frames"]
        actions = batch["model_selected_actions"]
        rewards = batch["rewards"]
        rtg = batch["rtg"]
        reward_bins = batch["reward_bins"]
        rtg_bins = batch["rtg_bins"]
        game_names = batch["game_name"]

        if first_shapes is None:
            first_shapes = {
                "frames": tuple(frames.shape),
                "actions": tuple(actions.shape),
                "rewards": tuple(rewards.shape),
                "rtg": tuple(rtg.shape),
            }

        all_actions.append(actions.reshape(-1).cpu())
        all_rewards.append(rewards.reshape(-1).cpu())
        all_rtg.append(rtg.reshape(-1).cpu())
        all_reward_bins.append(reward_bins.reshape(-1).cpu())
        all_rtg_bins.append(rtg_bins.reshape(-1).cpu())
        all_game_names.extend(list(game_names))

    if n_batches_used == 0:
        raise ValueError("inspect_dataloader: dataloader was empty or max_batches=0")

    actions_tensor = torch.cat(all_actions, dim=0)
    rewards_tensor = torch.cat(all_rewards, dim=0)
    rtg_tensor = torch.cat(all_rtg, dim=0)
    reward_bins_tensor = torch.cat(all_reward_bins, dim=0)
    rtg_bins_tensor = torch.cat(all_rtg_bins, dim=0)

    actions_np = actions_tensor.numpy()
    rewards_np = rewards_tensor.numpy()
    rtg_np = rtg_tensor.numpy()
    reward_bins_np = reward_bins_tensor.numpy()
    rtg_bins_np = rtg_bins_tensor.numpy()

    game_counts = Counter(all_game_names)

    print(f"=== Dataloader Sanity Check: {title} ===")
    print(f"Used batches: {n_batches_used}")
    print(f"First batch shapes: {first_shapes}")
    print(f"Total samples (timesteps): {actions_np.shape[0]}")
    print(f"Games in subset: {len(game_counts)} -> {dict(game_counts)}")
    print(f"Actions: min={actions_np.min()}, max={actions_np.max()}, "
          f"unique={len(np.unique(actions_np))}")
    print(f"Rewards: min={rewards_np.min():.3f}, max={rewards_np.max():.3f}")
    print(f"RTG: min={rtg_np.min():.3f}, max={rtg_np.max():.3f}")
    print(f"Reward bins: unique={np.unique(reward_bins_np)}")
    print(f"RTG bins: min={rtg_bins_np.min()}, max={rtg_bins_np.max()}")

    # NaN / inf checks
    print(f"NaNs in rewards? {np.isnan(rewards_np).any()}")
    print(f"NaNs in RTG? {np.isnan(rtg_np).any()}")

    colors = sns.color_palette("husl", 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    fig.suptitle(f"Dataloader Sanity Plots: {title}", fontsize=16)

    # Action frequency
    ax = axes[0, 0]
    unique_actions, counts_actions = np.unique(actions_np, return_counts=True)
    bars = ax.bar(unique_actions, counts_actions, color=colors[0])
    if len(unique_actions) <= max_bars_for_labels:
        ax.bar_label(bars, fmt='%d')
    ax.set_title("Action Frequencies")
    ax.set_xlabel("Action ID")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Reward distribution
    ax = axes[0, 1]
    reward_unique, reward_counts = np.unique(rewards_np, return_counts=True)
    bars = ax.bar(reward_unique, reward_counts, color=colors[1], width=0.1)
    if len(reward_unique) <= max_bars_for_labels:
        ax.bar_label(bars, fmt='%d')
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # RTG distribution
    ax = axes[0, 2]
    ax.hist(rtg_np, bins=50, color=colors[2], edgecolor='white', alpha=0.8)
    ax.set_title("RTG Distribution")
    ax.set_xlabel("RTG")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Reward bin counts
    ax = axes[1, 0]
    rb_unique, rb_counts = np.unique(reward_bins_np, return_counts=True)
    bars = ax.bar(rb_unique, rb_counts, color=colors[3])
    if len(rb_unique) <= max_bars_for_labels:
        ax.bar_label(bars, fmt='%d')
    ax.set_title("Reward Bin Counts")
    ax.set_xlabel("Reward bin (0=-1,1=0,2=+1)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # RTG bin counts
    ax = axes[1, 1]
    rtb_unique, rtb_counts = np.unique(rtg_bins_np, return_counts=True)
    bars = ax.bar(rtb_unique, rtb_counts, color=colors[4])
    if len(rtb_unique) <= max_bars_for_labels:
        ax.bar_label(bars, fmt='%d')
    ax.set_title("RTG Bin Counts")
    ax.set_xlabel("RTG bin index")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    # Samples per game
    ax = axes[1, 2]
    if game_counts:
        games = list(game_counts.keys())
        counts = [game_counts[g] for g in games]
        bars = ax.bar(range(len(games)), counts, color=colors[5])
        if len(games) <= max_bars_for_labels:
            ax.bar_label(bars, fmt='%d')
        ax.set_xticks(range(len(games)))
        ax.set_xticklabels(games, rotation=45, ha="right")
        ax.set_title("Timesteps per Game (subset)")
        ax.set_ylabel("Timesteps")
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_title = title.replace(" ", "_").replace("/", "_").lower()
        fig.savefig(output_dir / f"dataloader_{safe_title}.png", dpi=150)
        print(f"Saved plot to {output_dir / f'dataloader_{safe_title}.png'}")
    else:
        plt.show()

    plt.close(fig)

    stats = {
        "first_shapes": first_shapes,
        "n_batches_used": n_batches_used,
        "n_timesteps": int(actions_np.shape[0]),
        "game_counts": dict(game_counts),
        "actions_min": float(actions_np.min()),
        "actions_max": float(actions_np.max()),
        "n_unique_actions": int(len(np.unique(actions_np))),
        "rewards_min": float(rewards_np.min()),
        "rewards_max": float(rewards_np.max()),
        "rtg_min": float(rtg_np.min()),
        "rtg_max": float(rtg_np.max()),
        "reward_bins_unique": np.unique(reward_bins_np),
        "rtg_bins_min": int(rtg_bins_np.min()),
        "rtg_bins_max": int(rtg_bins_np.max()),
    }

    print(stats)
    return stats
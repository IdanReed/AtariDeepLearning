from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt


def inspect_dataloader(
    dataloader,
    title: str,
    max_batches: int = 50,
) -> dict[str, Any]:
    """
    Iterate over up to `max_batches` from a dataloader built on EpisodeSliceDataset
    and produce basic sanity-check plots + stats.

    Expects each batch to be a dict with keys:
      - frames: (B, T, C, H, W)
      - actions: (B, T)
      - rewards: (B, T)
      - rtg: (B, T)
      - reward_bins: (B, T)
      - rtg_bins: (B, T)
      - game_name: list[str]
      - repeated_actions, model_selected_actions, ...

    Returns a dict of aggregated stats in case you want to inspect numerically.
    """

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

        frames = batch["frames"]           # (B, T, C, H, W)
        actions = batch["model_selected_actions"]
        rewards = batch["rewards"]
        rtg = batch["rtg"]
        reward_bins = batch["reward_bins"]
        rtg_bins = batch["rtg_bins"]
        game_names = batch["game_name"]    # list[str]

        if first_shapes is None:
            first_shapes = {
                "frames": tuple(frames.shape),
                "actions": tuple(actions.shape),
                "rewards": tuple(rewards.shape),
                "rtg": tuple(rtg.shape),
            }

        # flatten (B, T) -> (-1,)
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

    # ---- basic prints ----
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

    # ---- plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 12))
    fig.suptitle(f"Dataloader Sanity Plots: {title}")

    # 1. Action frequency
    ax = axes[0, 0]
    unique_actions, counts_actions = np.unique(actions_np, return_counts=True)
    bars = ax.bar(unique_actions, counts_actions)
    ax.bar_label(bars, fmt='%d')
    ax.set_title("Action Frequencies")
    ax.set_xlabel("Action ID")
    ax.set_ylabel("Count")

    # 2. Reward distribution
    ax = axes[0, 1]
    ax.hist(rewards_np, bins=50)
    ax.set_title("Reward Distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")

    # 3. RTG distribution
    ax = axes[0, 2]
    ax.hist(rtg_np, bins=50)
    ax.set_title("RTG Distribution")
    ax.set_xlabel("RTG")
    ax.set_ylabel("Count")

    # 4. Reward bin counts
    ax = axes[1, 0]
    rb_unique, rb_counts = np.unique(reward_bins_np, return_counts=True)
    bars = ax.bar(rb_unique, rb_counts)
    ax.bar_label(bars, fmt='%d')
    ax.set_title("Reward Bin Counts")
    ax.set_xlabel("Reward bin (0=-1,1=0,2=+1)")
    ax.set_ylabel("Count")

    # 5. RTG bin counts
    ax = axes[1, 1]
    rtb_unique, rtb_counts = np.unique(rtg_bins_np, return_counts=True)
    bars = ax.bar(rtb_unique, rtb_counts)
    ax.bar_label(bars, fmt='%d')
    ax.set_title("RTG Bin Counts")
    ax.set_xlabel("RTG bin index")
    ax.set_ylabel("Count")

    # 6. Samples per game
    ax = axes[1, 2]
    if game_counts:
        games = list(game_counts.keys())
        counts = [game_counts[g] for g in games]
        bars = ax.bar(range(len(games)), counts)
        ax.bar_label(bars, fmt='%d')
        ax.set_xticks(range(len(games)))
        ax.set_xticklabels(games, rotation=45, ha="right")
        ax.set_title("Timesteps per Game (subset)")
        ax.set_ylabel("Timesteps")
    else:
        ax.set_visible(False)
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
    plt.show()
    
    return

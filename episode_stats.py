from __future__ import annotations

from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from episode import Episode, TimeStep


@dataclass
class GameEpisodeStats:
    """Aggregated metrics for a single game."""
    game_name: str
    episode_lengths: np.ndarray          # (N_episodes,)
    episode_returns: np.ndarray          # (N_episodes,)
    per_step_rewards: np.ndarray         # (N_steps_total,)
    action_counts: Dict[int, int]        # action -> count
    repeated_action_fraction: float      # fraction of steps with repeated_action == True


def _episode_return(ep: Episode) -> float:
    return float(sum(ts.reward for ts in ep.timesteps))


def summarize_episodes_by_game(episodes: List[Episode]) -> Dict[str, GameEpisodeStats]:
    """
    Aggregate useful metrics by game from a list of Episode objects.

    Returns:
        Dict mapping game_name -> GameEpisodeStats
    """
    lengths_by_game: Dict[str, List[int]] = defaultdict(list)
    returns_by_game: Dict[str, List[float]] = defaultdict(list)
    rewards_by_game: Dict[str, List[float]] = defaultdict(list)
    actions_by_game: Dict[str, List[int]] = defaultdict(list)
    repeated_by_game: Dict[str, List[bool]] = defaultdict(list)

    for ep in episodes:
        game = ep.game_name
        ep_len = len(ep.timesteps)
        ep_ret = _episode_return(ep)

        lengths_by_game[game].append(ep_len)
        returns_by_game[game].append(ep_ret)

        for ts in ep.timesteps:
            rewards_by_game[game].append(float(ts.reward))
            actions_by_game[game].append(int(ts.taken_action))
            repeated_by_game[game].append(bool(ts.repeated_action))

    stats_by_game: Dict[str, GameEpisodeStats] = {}

    for game in sorted(lengths_by_game.keys()):
        ep_lengths = np.asarray(lengths_by_game[game], dtype=np.int32)
        ep_returns = np.asarray(returns_by_game[game], dtype=np.float32)
        per_step_rewards = np.asarray(rewards_by_game[game], dtype=np.float32)

        action_counts = dict(Counter(actions_by_game[game]))
        repeated_flags = np.asarray(repeated_by_game[game], dtype=bool)
        repeated_fraction = float(repeated_flags.mean()) if repeated_flags.size > 0 else 0.0

        stats_by_game[game] = GameEpisodeStats(
            game_name=game,
            episode_lengths=ep_lengths,
            episode_returns=ep_returns,
            per_step_rewards=per_step_rewards,
            action_counts=action_counts,
            repeated_action_fraction=repeated_fraction,
        )

    return stats_by_game


# ----------------
# Plotting helpers
# ----------------

def plot_episode_length_distribution(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    For each game, plot a histogram of episode lengths.
    """
    for game, stats in stats_by_game.items():
        fig, ax = plt.subplots()
        ax.hist(stats.episode_lengths, bins=min(50, len(stats.episode_lengths)))
        ax.set_title(f"Episode Lengths – {game}")
        ax.set_xlabel("Episode length (timesteps)")
        ax.set_ylabel("Count")
        fig.tight_layout()


def plot_episode_return_distribution(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    For each game, plot a histogram of total episode returns.
    """
    for game, stats in stats_by_game.items():
        fig, ax = plt.subplots()
        ax.hist(stats.episode_returns, bins=min(50, len(stats.episode_returns)))
        ax.set_title(f"Episode Returns – {game}")
        ax.set_xlabel("Episode return (sum of rewards)")
        ax.set_ylabel("Count")
        fig.tight_layout()


def plot_episode_return_vs_length(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    For each game, scatter plot: episode length vs episode return.
    Useful to see if longer episodes correlate with higher rewards.
    """
    for game, stats in stats_by_game.items():
        fig, ax = plt.subplots()
        ax.scatter(stats.episode_lengths, stats.episode_returns, alpha=0.5)
        ax.set_title(f"Return vs Length – {game}")
        ax.set_xlabel("Episode length (timesteps)")
        ax.set_ylabel("Episode return")
        fig.tight_layout()


def plot_reward_histograms(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    For each game, plot histogram of per-step rewards.
    This will often be mostly zeros with a few positive/negative rewards.
    """
    for game, stats in stats_by_game.items():
        fig, ax = plt.subplots()
        ax.hist(stats.per_step_rewards, bins=50)
        ax.set_title(f"Per-step Rewards – {game}")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        fig.tight_layout()


def plot_action_frequencies(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    For each game, bar plot of action frequencies over all timesteps.
    """
    for game, stats in stats_by_game.items():
        if not stats.action_counts:
            continue

        actions = sorted(stats.action_counts.keys())
        counts = [stats.action_counts[a] for a in actions]

        fig, ax = plt.subplots()
        ax.bar(actions, counts)
        ax.set_title(f"Action Frequencies – {game}")
        ax.set_xlabel("Action ID")
        ax.set_ylabel("Count")
        fig.tight_layout()


def plot_repeated_action_fraction(stats_by_game: Dict[str, GameEpisodeStats]) -> None:
    """
    Single bar chart comparing repeated-action fraction across games.
    (How often a step is flagged as 'repeated_action' due to sticky actions.)
    """
    games = list(stats_by_game.keys())
    fractions = [stats_by_game[g].repeated_action_fraction for g in games]

    fig, ax = plt.subplots()
    ax.bar(range(len(games)), fractions)
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games, rotation=45, ha="right")
    ax.set_ylabel("Fraction of timesteps with repeated_action=True")
    ax.set_title("Sticky / Repeated Action Fraction by Game")
    fig.tight_layout()


def create_all_plots(episodes: List[Episode]) -> Dict[str, GameEpisodeStats]:
    """
    Convenience function:
      - summarize episodes
      - create a suite of plots per game

    Returns the stats_by_game dict in case you want to inspect numbers too.
    """
    stats_by_game = summarize_episodes_by_game(episodes)

    plot_episode_length_distribution(stats_by_game)
    plot_episode_return_distribution(stats_by_game)
    plot_episode_return_vs_length(stats_by_game)
    plot_reward_histograms(stats_by_game)
    plot_action_frequencies(stats_by_game)
    plot_repeated_action_fraction(stats_by_game)

    return stats_by_game

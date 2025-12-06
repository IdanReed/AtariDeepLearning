# train_mgdt.py

from __future__ import annotations

from enum import Enum
from typing import Any, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from episode import Episode
from episode_dataset import EpisodeSliceDataset, BinningInfo
from encoders import PatchEncoder, CNNEncoder
from mgdt_model import MGDTModel


def _ensure_device(device: Optional[torch.device]) -> torch.device:
    if device is not None:
        return device
    if not torch.cuda.is_available():
        raise ValueError("No GPU found and I think you want one")
    return torch.device("cuda")


def evaluate_mgdt(
    model: MGDTModel,
    dataloader: DataLoader,
    device: torch.device,
) -> List[dict[str, Any]]:
    model.eval()
    eval_stats: List[dict[str, Any]] = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Validation"), start=1):
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            rtg_bins = batch["rtg_bins"].to(device)
            reward_bins = batch["reward_bins"].to(device)

            loss, stats = model.compute_loss(frames, rtg_bins, actions, reward_bins)

            out = model(frames, rtg_bins, actions, reward_bins)
            ret_pred = out["return_logits"].argmax(dim=-1)
            act_pred = out["action_logits"].argmax(dim=-1)
            rew_pred = out["reward_logits"].argmax(dim=-1)

            ret_acc = (ret_pred == rtg_bins).float().mean().item()
            act_acc = (act_pred == actions).float().mean().item()
            rew_acc = (rew_pred == reward_bins).float().mean().item()

            stats = {
                "step": step,
                "loss": float(loss.item()),
                "loss_return": float(stats["loss_return"]),
                "loss_action": float(stats["loss_action"]),
                "loss_reward": float(stats["loss_reward"]),
                "return_acc": ret_acc,
                "action_acc": act_acc,
                "reward_acc": rew_acc,
            }
            eval_stats.append(stats)

    return eval_stats


class Encoder(Enum):
    Patch = "patch"
    CNN = "cnn"


def train_mgdt(
    bins: BinningInfo,
    dataloader_train: DataLoader,
    dataloader_val: Optional[DataLoader] = None,
    *,
    encoder_type: Encoder = Encoder.Patch,
    image_size: Tuple[int, int] = (84, 84),
    d_model: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    max_timestep_window_size: int = 32,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
) -> Tuple[MGDTModel, List[dict[str, Any]], List[dict[str, Any]]]:
    device = _ensure_device(device)

    # Encoder
    if encoder_type == Encoder.CNN:
        encoder = CNNEncoder(image_size=image_size, in_channels=3, d_model=d_model)
    elif encoder_type == Encoder.Patch:
        encoder = PatchEncoder(image_size=image_size, in_channels=3, d_model=d_model)

    n_actions = bins.n_actions
    n_return_bins = bins.num_rtg_bins

    model = MGDTModel(
        obs_encoder=encoder,
        n_actions=n_actions,
        n_return_bins=n_return_bins,
        n_reward_bins=3,
        emb_size=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_timestep_window_size=max_timestep_window_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_stats: List[dict[str, Any]] = []

    for step, batch in enumerate(tqdm(dataloader_train, desc="Training"), start=1):
        frames = batch["frames"].to(device)           # (B, T, C, H, W)
        actions = batch["actions"].to(device)         # (B, T)
        rtg_bins = batch["rtg_bins"].to(device)       # (B, T)
        reward_bins = batch["reward_bins"].to(device) # (B, T)

        model.train()
        optimizer.zero_grad()

        loss, stats = model.compute_loss(frames, rtg_bins, actions, reward_bins)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # gradient norm
        total_grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** 0.5

        optimizer.step()

        # accuracies
        with torch.no_grad():
            out = model(frames, rtg_bins, actions, reward_bins)
            ret_pred = out["return_logits"].argmax(dim=-1)
            act_pred = out["action_logits"].argmax(dim=-1)
            rew_pred = out["reward_logits"].argmax(dim=-1)

            ret_acc = (ret_pred == rtg_bins).float().mean().item()
            act_acc = (act_pred == actions).float().mean().item()
            rew_acc = (rew_pred == reward_bins).float().mean().item()

        train_stats.append(
            {
                "step": step,
                "loss": float(loss.item()),
                "loss_return": float(stats["loss_return"]),
                "loss_action": float(stats["loss_action"]),
                "loss_reward": float(stats["loss_reward"]),
                "grad_norm": float(total_grad_norm),
                "return_acc": ret_acc,
                "action_acc": act_acc,
                "reward_acc": rew_acc,
            }
        )

    val_stats: List[dict[str, Any]] = []
    if dataloader_val is not None:
        val_stats = evaluate_mgdt(model, dataloader_val, device)

    return model, train_stats, val_stats

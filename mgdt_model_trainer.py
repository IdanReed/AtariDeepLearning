from __future__ import annotations

from enum import Enum
from typing import Any, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
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

class Encoder(Enum):
    Patch = "patch"
    CNN = "cnn"

def _create_subset_dataloader(
    dataloader: DataLoader,
    fraction: float,
    seed: int,
) -> DataLoader:
    """
    We want to be able to see validation loss during an epoch, because our dataset is quite large.
    So we'll sample a subset of the validation dataset to periodically test on mid-epoch.
    """
    dataset = dataloader.dataset
    n_samples = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:n_samples].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=getattr(dataloader, 'num_workers', 0),
        pin_memory=getattr(dataloader, 'pin_memory', False),
    )

def train_mgdt(
    bins: BinningInfo,
    dataloader_train: DataLoader,
    dataloader_val: Optional[DataLoader] = None,
    *,
    model: Optional[MGDTModel] = None,
    num_epochs: int = 1,
    val_every_pct: float = 0.2,
    mid_epoch_val_fraction: float = 0.25,
    val_seed: int = 42,
    encoder_type: Encoder = Encoder.Patch,
    image_size: Tuple[int, int] = (84, 84),
    emb_size: int = 512,
    n_layers: int = 6,
    n_heads: int = 8,
    max_timestep_window_size: int = 32,
    lr: float = 3e-4,
    finetune_lr_factor: float = 0.1,
    weight_decay: float = 0.01,
    device: Optional[torch.device] = None,
) -> Tuple[MGDTModel, List[dict[str, Any]], List[dict[str, Any]]]:

    device = _ensure_device(device)

    is_finetuning = model is not None
    if not is_finetuning:
        if encoder_type == Encoder.CNN:
            encoder = CNNEncoder(image_size=image_size, in_channels=3, emb_size=emb_size)
        elif encoder_type == Encoder.Patch:
            encoder = PatchEncoder(image_size=image_size, in_channels=3, emb_size=emb_size)

        n_actions = bins.n_actions
        n_return_bins = bins.num_rtg_bins

        dataset = dataloader_train.dataset
        all_game_ids = [ep.game_id for ep in dataset.episodes]
        n_games = max(all_game_ids) + 1

        model = MGDTModel(
            obs_encoder=encoder,
            n_actions=n_actions,
            n_return_bins=n_return_bins,
            n_games=n_games,
            n_reward_bins=3,
            emb_size=emb_size,
            n_layers=n_layers,
            n_heads=n_heads,
            max_timestep_window_size=max_timestep_window_size,
        ).to(device)
    else:
        model = model.to(device)

    # Reduce LR for finetuning
    effective_lr = lr * finetune_lr_factor if is_finetuning else lr

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=effective_lr, 
        weight_decay=weight_decay
    )

    train_stats: List[dict[str, Any]] = []
    val_stats: List[dict[str, Any]] = []

    # Create mid-epoch validation dataloader if needed
    mid_epoch_val_loader = None
    if dataloader_val is not None and mid_epoch_val_fraction < 1.0:
        mid_epoch_val_loader = _create_subset_dataloader(
            dataloader_val, mid_epoch_val_fraction, val_seed
        )

    global_step = 0
    total_batches = len(dataloader_train)
    val_interval = max(1, int(total_batches * val_every_pct))

    for epoch in range(1, num_epochs + 1):
        desc_prefix = "Finetune" if is_finetuning else "Epoch"
        epoch_pbar = tqdm(
            enumerate(dataloader_train, start=1),
            total=total_batches,
            desc=f"{desc_prefix} {epoch}/{num_epochs}",
        )

        for step_in_epoch, batch in epoch_pbar:
            global_step += 1

            frames = batch["frames"].to(device)
            actions = batch["model_selected_actions"].to(device)
            rtg_bins = batch["rtg_bins"].to(device)
            reward_bins = batch["reward_bins"].to(device)
            game_ids = batch["game_ids"].to(device)


            model.train()
            optimizer.zero_grad()

            # Forward
            out, loss, stats = model.forward_and_compute_loss(frames, rtg_bins, actions, reward_bins, game_ids)

            # Backward
            loss.backward()

            # Clip grad and calc grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # Optimize step
            optimizer.step()

            # Calc accuracy
            with torch.no_grad():
                ret_pred = out["return_logits"].argmax(dim=-1)
                act_pred = out["action_logits"].argmax(dim=-1)
                rew_pred = out["reward_logits"].argmax(dim=-1)

                ret_acc = (ret_pred == rtg_bins).float().mean().item()
                act_acc = (act_pred == actions).float().mean().item()
                rew_acc = (rew_pred == reward_bins).float().mean().item()

            train_stats.append(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "step_in_epoch": step_in_epoch,
                    "is_finetune": is_finetuning,

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

            # Mid-epoch validation
            if (
                dataloader_val is not None
                and step_in_epoch % val_interval == 0
                and step_in_epoch < total_batches
            ):
                loader_to_use = mid_epoch_val_loader if mid_epoch_val_loader else dataloader_val
                mid_val_stats = _evaluate_mgdt(
                    model, loader_to_use, device,
                    epoch=epoch,
                    global_step=global_step,
                    is_mid_epoch=True,
                    is_finetune=is_finetuning,
                )
                val_stats.extend(mid_val_stats)

        # End-of-epoch: full validation
        if dataloader_val is not None:
            end_epoch_val_stats = _evaluate_mgdt(
                model, dataloader_val, device,
                epoch=epoch,
                global_step=global_step,
                is_mid_epoch=False,
                is_finetune=is_finetuning,
            )
            val_stats.extend(end_epoch_val_stats)

    return model, train_stats, val_stats

def _evaluate_mgdt(
    model: MGDTModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    epoch: int = 1,
    global_step: int = 0,
    is_mid_epoch: bool = False,
    is_finetune: bool = False,
) -> List[dict[str, Any]]:
    model.eval()
    eval_stats: List[dict[str, Any]] = []

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False), start=1):
            frames = batch["frames"].to(device)
            actions = batch["model_selected_actions"].to(device)
            rtg_bins = batch["rtg_bins"].to(device)
            reward_bins = batch["reward_bins"].to(device)
            game_ids = batch["game_ids"].to(device)

            out, loss, stats = model.forward_and_compute_loss(frames, rtg_bins, actions, reward_bins)

            ret_pred = out["return_logits"].argmax(dim=-1)
            act_pred = out["action_logits"].argmax(dim=-1)
            rew_pred = out["reward_logits"].argmax(dim=-1)

            ret_acc = (ret_pred == rtg_bins).float().mean().item()
            act_acc = (act_pred == actions).float().mean().item()
            rew_acc = (rew_pred == reward_bins).float().mean().item()

            eval_stats.append({
                "epoch": epoch,
                "global_step": global_step,
                "step": step,
                "is_mid_epoch": is_mid_epoch,
                "is_finetune": is_finetune,

                "loss": float(loss.item()),
                "loss_return": float(stats["loss_return"]),
                "loss_action": float(stats["loss_action"]),
                "loss_reward": float(stats["loss_reward"]),

                "return_acc": ret_acc,
                "action_acc": act_acc,
                "reward_acc": rew_acc,
            })

    return eval_stats

from __future__ import annotations

from pathlib import Path
import multiprocessing
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from npz_loader import (
    discover_games,
    discover_game_npz_paths,
    get_sequences_by_game,
    fix_obs_paths,
)
from atari_dataset import AtariDataset
from image_io import AtariImageLoader
from baseline_encoder import AtariPatchEncoder
from tokenizer import MGDTTokenizer
from baseline_model import MultiGameDecisionTransformer

# Set multiprocessing start method for Windows compatibility
if __name__ == "__main__" or hasattr(multiprocessing, '_parent_pid'):
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, ignore
        pass


# ------------------------------------------------------------
# 1. Build sequences_by_game and dataset
# ------------------------------------------------------------
def build_sequences_by_game(raw_root: str | Path = "dataset/raw") -> dict:
    raw_root = Path(raw_root)
    train_game_dirs = [
        Path(r"dataset\BeamRiderNoFrameskip-v4\BeamRiderNoFrameskip-v4"),
        Path(r"dataset\BreakoutNoFrameskip-v4\BreakoutNoFrameskip-v4"),
    ]
    game_npz_paths = discover_game_npz_paths(train_game_dirs)
    game_to_sequences = get_sequences_by_game(game_npz_paths)
    sequences_by_game = fix_obs_paths(game_to_sequences, dataset_root="dataset")
    return sequences_by_game


def collate_fn(batch):
    """
    batch: list of (frames, actions, rtg) from AtariDataset

    frames: tensor (T, C, H, W) - already loaded images
    actions: tensor (T,)
    rtg: tensor (T,)
    """
    frames, actions, rtg = zip(*batch)
    # Stack frames: (B, T, C, H, W)
    return torch.stack(frames, dim=0), torch.stack(actions, dim=0), torch.stack(rtg, dim=0)


def create_train_test_dataloaders(
    sequences_by_game: dict,
    train_fraction: float = 0.8,
    total_frac: float = 1.0,
    context_len: int = 32,
    batch_size: int = 32,
    num_workers: int = 16,
    shuffle_train: bool = True,
    seed: int | None = None,
    img_size: int = 84,
    grayscale: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Split sequences_by_game into train and test sets and create DataLoaders.
    
    Args:
        sequences_by_game: dict mapping game paths to lists of sequence dicts
        train_fraction: fraction of sequences to use for training (default 0.8)
        total_frac: fraction of total sequences to use (default 1.0, use all).
                    Useful for quick training on a smaller subset.
        context_len: context length for sliding windows
        batch_size: batch size for both dataloaders
        num_workers: number of worker processes for both dataloaders
        shuffle_train: whether to shuffle the training dataloader
        seed: random seed for reproducibility (optional)
        img_size: image size for loading (default 84)
        grayscale: whether to load images as grayscale (default True)
    
    Returns:
        tuple of (train_loader, test_loader)
    """
    if seed is not None:
        random.seed(seed)
    
    # Split sequences for each game
    train_sequences_by_game = {}
    test_sequences_by_game = {}
    
    for game, sequences in sequences_by_game.items():
        # Shuffle sequences for this game
        sequences_shuffled = sequences.copy()
        random.shuffle(sequences_shuffled)
        
        # Split based on train_fraction first (before applying total_frac)
        n_train = int(len(sequences_shuffled) * train_fraction)
        train_seqs = sequences_shuffled[:n_train]
        test_seqs = sequences_shuffled[n_train:]
        
        # Optionally reduce total dataset size (apply total_frac proportionally to both splits)
        if total_frac < 1.0:
            n_train_reduced = int(len(train_seqs) * total_frac)
            n_test_reduced = int(len(test_seqs) * total_frac)
            # Randomly sample instead of taking first N to avoid bias
            train_sequences_by_game[game] = random.sample(train_seqs, n_train_reduced)
            test_sequences_by_game[game] = random.sample(test_seqs, n_test_reduced)
        else:
            train_sequences_by_game[game] = train_seqs
            test_sequences_by_game[game] = test_seqs
    
    # Create datasets (with image loading parameters)
    train_dataset = AtariDataset(
        train_sequences_by_game, 
        context_len=context_len,
        img_size=img_size,
        grayscale=grayscale,
    )
    test_dataset = AtariDataset(
        test_sequences_by_game, 
        context_len=context_len,
        img_size=img_size,
        grayscale=grayscale,
    )
    
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    # Create dataloaders with optimizations for parallel image loading
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        pin_memory=True,  # Faster GPU transfer
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches in workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test set
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    
    return train_loader, test_loader


# ------------------------------------------------------------
# 2. Main training function
# ------------------------------------------------------------
def train(
    model: MultiGameDecisionTransformer,
    tokenizer: MGDTTokenizer,
    train_loader: DataLoader,
    test_loader: DataLoader | None = None,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    device: torch.device | None = None,
    n_actions: int | None = None,
    n_games: int = 1,
    grad_clip_norm: float = 1.0,
    stop_file_path: str | Path = "stop",
    eval_every_n_epochs: int = 1,
) -> MultiGameDecisionTransformer:
    """
    Train a Multi-Game Decision Transformer model.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer to use for encoding inputs
        train_loader: DataLoader for training data (images are loaded in Dataset workers)
        test_loader: Optional DataLoader for test/validation data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on (defaults to cuda if available, else cpu)
        n_actions: Number of actions (inferred from train_loader.dataset if not provided)
        n_games: Number of games (for game_id generation)
        grad_clip_norm: Gradient clipping norm
        stop_file_path: Path to stop file for graceful stopping
        eval_every_n_epochs: Evaluate on test set every N epochs (0 to disable)
    
    Returns:
        Tuple of (trained_model, train_losses, val_losses)
        - trained_model: The trained model
        - train_losses: List of training losses per epoch
        - val_losses: List of validation losses per epoch (None if not evaluated that epoch)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Move model to device
    model = model.to(device)
    tokenizer = tokenizer.to(device)
    
    # Get n_actions from dataset if not provided
    if n_actions is None:
        n_actions = train_loader.dataset.n_actions()
    
    # ---- optimization ----
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Stop file path
    stop_file = Path(stop_file_path)

    # Track losses for plotting
    train_losses = []
    val_losses = []

    # --------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------
    # Outer loop: epochs
    for epoch in range(1, num_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{num_epochs}")
        # Check for stop file at the start of each epoch
        if stop_file.exists():
            tqdm.write(f"Stop file detected. Stopping training after epoch {epoch-1}.")
            break
            
        model.train()
        tokenizer.train()
        total_loss = 0.0
        total_tokens = 0

        # Inner loop: batches
        batch_pbar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )
        
        for step, (frames_batch, actions_batch, rtg_batch) in batch_pbar:
            # Check for stop file periodically during batch processing
            if stop_file.exists():
                tqdm.write(f"Stop file detected. Stopping training at epoch {epoch}, step {step}.")
                break

            # frames_batch: (B, T, C, H, W) - already loaded images from Dataset
            # actions_batch: (B, T)
            # rtg_batch: (B, T)

            B, T = actions_batch.shape

            # 1) Move to device (images already loaded in Dataset workers)
            frames = frames_batch.to(device, non_blocking=True)

            actions = actions_batch.to(device, non_blocking=True)
            rtg = rtg_batch.to(device, non_blocking=True)

            # single-game for now -> game id 0
            game_ids = torch.zeros(B, dtype=torch.long, device=device)

            # 2) Tokenize
            tok_out = tokenizer(
                frames=frames,
                actions=actions,
                rtg=rtg,
                game_ids=game_ids,
            )
            tokens = tok_out.tokens  # (B, L, d_model)
            S = tok_out.tokens_per_step

            # 3) Transformer forward
            out = model(tokens, tokens_per_step=S, T=T)
            logits = out.logits  # (B,T,n_actions)

            # 4) Loss: predict the action at each timestep
            loss = criterion(
                logits.view(B * T, n_actions),
                actions.view(B * T),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            total_loss += loss.item() * B * T
            total_tokens += B * T

            # Update batch progress bar with current metrics
            avg_loss = total_loss / max(total_tokens, 1)
            batch_pbar.set_postfix({
                "batch_loss": f"{loss.item():.4f}",
                "avg_loss": f"{avg_loss:.4f}",
            })
        
        # Check again after batch loop in case stop file was created during processing
        if stop_file.exists():
            tqdm.write(f"Stop file detected. Stopping training after epoch {epoch}.")
            break

        epoch_loss = total_loss / max(total_tokens, 1)
        train_losses.append(epoch_loss)
        tqdm.write(f"Epoch {epoch} done. Train loss: {epoch_loss:.4f}")
        
        # Evaluate on test set if provided
        if test_loader is not None and eval_every_n_epochs > 0 and epoch % eval_every_n_epochs == 0:
            model.eval()
            tokenizer.eval()
            test_total_loss = 0.0
            test_total_tokens = 0
            
            with torch.no_grad():
                for frames_batch, actions_batch, rtg_batch in test_loader:
                    B, T = actions_batch.shape
                    
                    # Move to device (images already loaded in Dataset workers)
                    frames = frames_batch.to(device, non_blocking=True)
                    actions = actions_batch.to(device, non_blocking=True)
                    rtg = rtg_batch.to(device, non_blocking=True)
                    game_ids = torch.zeros(B, dtype=torch.long, device=device)
                    
                    # Tokenize
                    tok_out = tokenizer(
                        frames=frames,
                        actions=actions,
                        rtg=rtg,
                        game_ids=game_ids,
                    )
                    tokens = tok_out.tokens
                    S = tok_out.tokens_per_step
                    
                    # Forward pass
                    out = model(tokens, tokens_per_step=S, T=T)
                    logits = out.logits
                    
                    # Loss
                    loss = criterion(
                        logits.view(B * T, n_actions),
                        actions.view(B * T),
                    )
                    
                    test_total_loss += loss.item() * B * T
                    test_total_tokens += B * T
            
            test_loss = test_total_loss / max(test_total_tokens, 1)
            val_losses.append(test_loss)
            tqdm.write(f"Epoch {epoch} test loss: {test_loss:.4f}")
        else:
            # If not evaluating this epoch, append None to maintain alignment
            val_losses.append(None)
    
    # Clean up stop file if it exists
    if stop_file.exists():
        tqdm.write("Removing stop file.")
        stop_file.unlink()

    return model, train_losses, val_losses

from __future__ import annotations

from pathlib import Path
import multiprocessing

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
    batch: list of (obs_paths, actions, rtg) from AtariDataset

    obs_paths: tuple of length T of strings
    actions:  tensor (T,)
    rtg:      tensor (T,)
    """
    obs_paths, actions, rtg = zip(*batch)
    # obs_paths stays a list[tuple[str]], we don't touch it
    return list(obs_paths), torch.stack(actions, dim=0), torch.stack(rtg, dim=0)


# ------------------------------------------------------------
# 2. Main training function
# ------------------------------------------------------------
def train(sequences_by_game):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---- data ----
    # sequences_by_game = build_sequences_by_game("dataset/raw")
    context_len = 32

    dataset = AtariDataset(sequences_by_game, context_len=context_len)
    print("Dataset length:", len(dataset))

    batch_size = 32

    # Configure DataLoader for Windows multiprocessing
    # Use spawn context explicitly for Windows compatibility
    import multiprocessing as mp
    # mp_context = mp.get_context('spawn')
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn,
        persistent_workers=False,  # Don't persist workers (Windows compatibility)
        # pin_memory=False,  # Disable pin_memory for Windows
        # multiprocessing_context=mp_context,  # Use spawn method explicitly
    )

    # Image loader (Phase 1)
    img_loader = AtariImageLoader(img_size=84, grayscale=True)

    # Patch encoder (Phase 2)
    d_model = 128
    patch_encoder = AtariPatchEncoder(
        img_size=84,
        patch_size=14,
        in_channels=1,
        d_model=d_model,
    ).to(device)

    # Tokenizer (Phase 3)
    n_actions = dataset.n_actions()
    tokenizer = MGDTTokenizer(
        patch_encoder=patch_encoder,
        n_actions=n_actions,
        n_games=1,      # single-game setup for now
        rtg_min=-20,
        rtg_max=100,
    ).to(device)

    # Transformer (Phase 4)
    model = MultiGameDecisionTransformer(
        d_model=d_model,
        n_actions=n_actions,
        n_layers=2,      # small to start
        n_heads=4,
        dim_feedforward=4 * d_model,
        dropout=0.1,
        max_seq_len=2048,
    ).to(device)

    # ---- optimization ----
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------
    num_epochs = 1  # bump this later
    
    # Stop file path - check for this file to gracefully stop training
    stop_file = Path("stop")

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
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=False,
        )

        for step, (obs_paths_batch, actions_batch, rtg_batch) in batch_pbar:
            # Check for stop file periodically during batch processing
            if stop_file.exists():
                tqdm.write(f"Stop file detected. Stopping training at epoch {epoch}, step {step}.")
                break
            # obs_paths_batch: list length B, each length T (strings)
            # actions_batch:   (B,T)
            # rtg_batch:       (B,T)

            B, T = actions_batch.shape

            # 1) Load images
            frames = img_loader.load_batch(obs_paths_batch)  # (B,T,1,84,84)
            frames = frames.to(device)

            actions = actions_batch.to(device)
            rtg = rtg_batch.to(device)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        tqdm.write(f"Epoch {epoch} done. Avg loss: {epoch_loss:.4f}")
    
    # Clean up stop file if it exists
    if stop_file.exists():
        tqdm.write("Removing stop file.")
        stop_file.unlink()

    return model
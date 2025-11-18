"""
Diagnostic visualization module for verifying tokenization correctness and data integrity.
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

from tokenizer import MGDTTokenizer, TokenizerOutput


def generate_diagnostics(
    train_loader: DataLoader,
    test_loader: DataLoader,
    tokenizer: MGDTTokenizer,
    device: torch.device | None = None,
    max_samples: int = 1000,
    save_dir: str | Path = "diagnostics",
    show_plots: bool = True,
) -> dict:
    """
    Generate comprehensive diagnostic plots and statistics.
    
    Args:
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        tokenizer: MGDTTokenizer instance
        device: Device to run tokenization on (defaults to CPU)
        max_samples: Maximum number of samples to process (for efficiency)
        save_dir: Directory to save plots
        show_plots: Whether to display plots interactively
    
    Returns:
        Dictionary containing statistics and plot paths
    """
    if device is None:
        device = torch.device('cpu')
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    tokenizer = tokenizer.to(device)
    tokenizer.eval()
    
    # Collect data from dataloaders
    print("Collecting data from train loader...")
    train_data = _collect_data(train_loader, tokenizer, device, max_samples // 2, "train")
    
    print("Collecting data from test loader...")
    test_data = _collect_data(test_loader, tokenizer, device, max_samples // 2, "test")
    
    # Generate all diagnostics
    stats = {}
    
    print("Generating action distribution analysis...")
    stats['action'] = _analyze_actions(train_data, test_data, save_dir, show_plots)
    
    print("Generating RTG distribution analysis...")
    stats['rtg'] = _analyze_rtg(train_data, test_data, tokenizer, save_dir, show_plots)
    
    print("Generating image encoding analysis...")
    stats['image'] = _analyze_images(train_data, test_data, tokenizer, save_dir, show_plots)
    
    print("Verifying tokenization...")
    stats['tokenization'] = _verify_tokenization(train_data, test_data, tokenizer)
    
    print("Checking data consistency...")
    stats['consistency'] = _check_consistency(train_data, test_data)
    
    print("Comparing train/test splits...")
    stats['split'] = _compare_splits(train_data, test_data, save_dir, show_plots)
    
    print("Generating sample visualizations...")
    stats['samples'] = _visualize_samples(train_data, test_data, save_dir, show_plots)
    
    # Save summary statistics
    summary_path = save_dir / "summary_stats.txt"
    with open(summary_path, 'w') as f:
        f.write("=== Diagnostic Summary ===\n\n")
        for category, data in stats.items():
            f.write(f"\n{category.upper()}:\n")
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float, bool, str)):
                        f.write(f"  {key}: {value}\n")
    
    print(f"\nDiagnostics complete! Results saved to {save_dir}")
    print(f"Summary: {summary_path}")
    
    return stats


def _collect_data(
    loader: DataLoader,
    tokenizer: MGDTTokenizer,
    device: torch.device,
    max_samples: int,
    split_name: str,
) -> dict:
    """Collect and process data from a dataloader."""
    all_frames = []
    all_actions = []
    all_rtg = []
    all_tok_outputs = []
    all_game_ids = []
    
    samples_processed = 0
    
    with torch.no_grad():
        for frames_batch, actions_batch, rtg_batch in loader:
            if samples_processed >= max_samples:
                break
            
            B = frames_batch.shape[0]
            batch_size = min(B, max_samples - samples_processed)
            
            frames = frames_batch[:batch_size].to(device, non_blocking=True)
            actions = actions_batch[:batch_size].to(device, non_blocking=True)
            rtg = rtg_batch[:batch_size].to(device, non_blocking=True)
            
            # Generate game_ids (default to 0 for single-game)
            game_ids = torch.zeros(frames.shape[0], dtype=torch.long, device=device)
            
            # Tokenize
            tok_out = tokenizer(
                frames=frames,
                actions=actions,
                rtg=rtg,
                game_ids=game_ids,
            )
            
            all_frames.append(frames.cpu())
            all_actions.append(actions.cpu())
            all_rtg.append(rtg.cpu())
            all_tok_outputs.append(tok_out)
            all_game_ids.append(game_ids.cpu())
            
            samples_processed += batch_size
    
    # Handle empty data case
    if not all_frames:
        raise ValueError(f"No data collected from {split_name} loader. Check dataloader configuration.")
    
    return {
        'frames': torch.cat(all_frames, dim=0),
        'actions': torch.cat(all_actions, dim=0),
        'rtg': torch.cat(all_rtg, dim=0),
        'tok_outputs': all_tok_outputs,
        'game_ids': torch.cat(all_game_ids, dim=0),
        'split_name': split_name,
    }


def _analyze_actions(train_data: dict, test_data: dict, save_dir: Path, show_plots: bool) -> dict:
    """Analyze action distributions."""
    train_actions = train_data['actions']
    test_actions = test_data['actions']
    
    # Flatten actions
    train_actions_flat = train_actions.flatten().numpy()
    test_actions_flat = test_actions.flatten().numpy()
    
    # Statistics
    n_actions = int(max(train_actions_flat.max(), test_actions_flat.max()) + 1)
    train_unique = len(np.unique(train_actions_flat))
    test_unique = len(np.unique(test_actions_flat))
    
    # Validation: check all actions are in valid range
    train_valid = np.all((train_actions_flat >= 0) & (train_actions_flat < n_actions))
    test_valid = np.all((test_actions_flat >= 0) & (test_actions_flat < n_actions))
    
    # Calculate action change statistics (number of times action changes per sequence)
    train_action_changes = []
    for i in range(train_actions.shape[0]):
        seq = train_actions[i].numpy()
        # Count number of times action changes (diff != 0)
        changes = np.sum(np.diff(seq) != 0)
        train_action_changes.append(changes)
    
    test_action_changes = []
    for i in range(test_actions.shape[0]):
        seq = test_actions[i].numpy()
        # Count number of times action changes (diff != 0)
        changes = np.sum(np.diff(seq) != 0)
        test_action_changes.append(changes)
    
    train_action_changes = np.array(train_action_changes)
    test_action_changes = np.array(test_action_changes)
    
    stats = {
        'n_actions': n_actions,
        'train_min': int(train_actions_flat.min()),
        'train_max': int(train_actions_flat.max()),
        'train_unique': train_unique,
        'test_min': int(test_actions_flat.min()),
        'test_max': int(test_actions_flat.max()),
        'test_unique': test_unique,
        'train_valid': bool(train_valid),
        'test_valid': bool(test_valid),
        # Action change statistics
        'train_action_changes_mean': float(train_action_changes.mean()),
        'train_action_changes_std': float(train_action_changes.std()),
        'train_action_changes_min': int(train_action_changes.min()),
        'train_action_changes_max': int(train_action_changes.max()),
        'test_action_changes_mean': float(test_action_changes.mean()),
        'test_action_changes_std': float(test_action_changes.std()),
        'test_action_changes_min': int(test_action_changes.min()),
        'test_action_changes_max': int(test_action_changes.max()),
    }
    
    # Plot 1: Action frequency histogram (train vs test)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1, ax2, ax3 = axes
    
    ax1.hist(train_actions_flat, bins=n_actions, alpha=0.7, label='Train', color='blue', edgecolor='black')
    ax1.hist(test_actions_flat, bins=n_actions, alpha=0.7, label='Test', color='red', edgecolor='black')
    ax1.set_xlabel('Action ID')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Action Distribution (Train vs Test)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Action counts bar chart
    train_counts = Counter(train_actions_flat)
    test_counts = Counter(test_actions_flat)
    action_ids = sorted(set(list(train_counts.keys()) + list(test_counts.keys())))
    
    train_counts_list = [train_counts.get(aid, 0) for aid in action_ids]
    test_counts_list = [test_counts.get(aid, 0) for aid in action_ids]
    
    x = np.arange(len(action_ids))
    width = 0.35
    ax2.bar(x - width/2, train_counts_list, width, label='Train', color='blue', alpha=0.7)
    ax2.bar(x + width/2, test_counts_list, width, label='Test', color='red', alpha=0.7)
    ax2.set_xlabel('Action ID')
    ax2.set_ylabel('Count')
    ax2.set_title('Action Counts per Action ID')
    ax2.set_xticks(x)
    ax2.set_xticklabels(action_ids)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Action change distribution (number of action changes per sequence)
    max_changes = max(train_action_changes.max(), test_action_changes.max())
    bins = min(30, int(max_changes) + 1)  # Limit bins for readability
    ax3.hist(train_action_changes, bins=bins, alpha=0.7, label='Train', color='blue', edgecolor='black', density=True)
    ax3.hist(test_action_changes, bins=bins, alpha=0.7, label='Test', color='red', edgecolor='black', density=True)
    ax3.set_xlabel('Number of Action Changes per Sequence')
    ax3.set_ylabel('Density')
    ax3.set_title('Action Change Frequency Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = save_dir / "action_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Print action change statistics
    print("\n=== Action Change Statistics ===")
    print(f"Train - Mean changes per sequence: {stats['train_action_changes_mean']:.2f} ± {stats['train_action_changes_std']:.2f}")
    print(f"Train - Range: [{stats['train_action_changes_min']}, {stats['train_action_changes_max']}]")
    print(f"Test  - Mean changes per sequence: {stats['test_action_changes_mean']:.2f} ± {stats['test_action_changes_std']:.2f}")
    print(f"Test  - Range: [{stats['test_action_changes_min']}, {stats['test_action_changes_max']}]")
    
    stats['plot_path'] = str(plot_path)
    return stats


def _analyze_rtg(
    train_data: dict,
    test_data: dict,
    tokenizer: MGDTTokenizer,
    save_dir: Path,
    show_plots: bool,
) -> dict:
    """Analyze RTG distributions."""
    train_rtg = train_data['rtg']
    test_rtg = test_data['rtg']
    
    # Flatten RTG
    train_rtg_flat = train_rtg.flatten().numpy()
    test_rtg_flat = test_rtg.flatten().numpy()
    
    # Statistics
    train_min, train_max = float(train_rtg_flat.min()), float(train_rtg_flat.max())
    test_min, test_max = float(test_rtg_flat.min()), float(test_rtg_flat.max())
    train_mean, train_std = float(train_rtg_flat.mean()), float(train_rtg_flat.std())
    test_mean, test_std = float(test_rtg_flat.mean()), float(test_rtg_flat.std())
    
    # Clipping analysis
    rtg_min, rtg_max = tokenizer.rtg_min, tokenizer.rtg_max
    train_clipped = np.sum((train_rtg_flat < rtg_min) | (train_rtg_flat > rtg_max))
    test_clipped = np.sum((test_rtg_flat < rtg_min) | (test_rtg_flat > rtg_max))
    train_clip_pct = 100 * train_clipped / len(train_rtg_flat)
    test_clip_pct = 100 * test_clipped / len(test_rtg_flat)
    
    # Quantized RTG distribution
    train_rtg_ids = []
    test_rtg_ids = []
    for tok_out in train_data['tok_outputs']:
        train_rtg_ids.append(tok_out.rtg_ids.cpu().numpy().flatten())
    for tok_out in test_data['tok_outputs']:
        test_rtg_ids.append(tok_out.rtg_ids.cpu().numpy().flatten())
    
    train_rtg_ids_flat = np.concatenate(train_rtg_ids) if train_rtg_ids else np.array([])
    test_rtg_ids_flat = np.concatenate(test_rtg_ids) if test_rtg_ids else np.array([])
    
    # Bin usage
    n_rtg_bins = tokenizer.n_rtg_bins
    train_bins_used = len(np.unique(train_rtg_ids_flat)) if len(train_rtg_ids_flat) > 0 else 0
    test_bins_used = len(np.unique(test_rtg_ids_flat)) if len(test_rtg_ids_flat) > 0 else 0
    
    stats = {
        'train_min': train_min,
        'train_max': train_max,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_min': test_min,
        'test_max': test_max,
        'test_mean': test_mean,
        'test_std': test_std,
        'rtg_min': rtg_min,
        'rtg_max': rtg_max,
        'n_rtg_bins': n_rtg_bins,
        'train_clipped': int(train_clipped),
        'train_clip_pct': train_clip_pct,
        'test_clipped': int(test_clipped),
        'test_clip_pct': test_clip_pct,
        'train_bins_used': train_bins_used,
        'test_bins_used': test_bins_used,
    }
    
    # Plot 1: RTG value distribution (before quantization)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(train_rtg_flat, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 0].hist(test_rtg_flat, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
    axes[0, 0].axvline(rtg_min, color='green', linestyle='--', label=f'RTG min={rtg_min}')
    axes[0, 0].axvline(rtg_max, color='green', linestyle='--', label=f'RTG max={rtg_max}')
    axes[0, 0].set_xlabel('RTG Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('RTG Distribution (Before Quantization)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: RTG quantized bin distribution
    if len(train_rtg_ids_flat) > 0 and len(test_rtg_ids_flat) > 0:
        axes[0, 1].hist(train_rtg_ids_flat, bins=n_rtg_bins, alpha=0.7, label='Train', color='blue', edgecolor='black')
        axes[0, 1].hist(test_rtg_ids_flat, bins=n_rtg_bins, alpha=0.7, label='Test', color='red', edgecolor='black')
        axes[0, 1].set_xlabel('RTG Bin ID')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title(f'RTG Quantized Bin Distribution (Total bins: {n_rtg_bins})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: RTG over time for sample sequences
    n_samples = min(5, train_rtg.shape[0])
    for i in range(n_samples):
        axes[1, 0].plot(train_rtg[i].numpy(), alpha=0.6, label=f'Train seq {i}' if i < 3 else '')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('RTG Value')
    axes[1, 0].set_title('RTG Trajectories (Sample Train Sequences)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    n_samples = min(5, test_rtg.shape[0])
    for i in range(n_samples):
        axes[1, 1].plot(test_rtg[i].numpy(), alpha=0.6, label=f'Test seq {i}' if i < 3 else '')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('RTG Value')
    axes[1, 1].set_title('RTG Trajectories (Sample Test Sequences)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / "rtg_distribution.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    stats['plot_path'] = str(plot_path)
    return stats


def _analyze_images(
    train_data: dict,
    test_data: dict,
    tokenizer: MGDTTokenizer,
    save_dir: Path,
    show_plots: bool,
) -> dict:
    """Analyze image encoding distributions."""
    train_frames = train_data['frames']
    test_frames = test_data['frames']
    
    # Frame pixel statistics
    train_pixels = train_frames.numpy().flatten()
    test_pixels = test_frames.numpy().flatten()
    
    train_pixel_min, train_pixel_max = float(train_pixels.min()), float(train_pixels.max())
    train_pixel_mean, train_pixel_std = float(train_pixels.mean()), float(train_pixels.std())
    test_pixel_min, test_pixel_max = float(test_pixels.min()), float(test_pixels.max())
    test_pixel_mean, test_pixel_std = float(test_pixels.mean()), float(test_pixels.std())
    
    # Check normalization (should be in [0, 1])
    train_normalized = (train_pixel_min >= 0) and (train_pixel_max <= 1)
    test_normalized = (test_pixel_min >= 0) and (test_pixel_max <= 1)
    
    # Patch token statistics (sample a few batches)
    train_patch_stats = []
    test_patch_stats = []
    
    # Get device from tokenizer
    device = next(tokenizer.parameters()).device
    
    tokenizer.eval()
    with torch.no_grad():
        # Sample train patches
        sample_frames = train_frames[:min(10, train_frames.shape[0])]
        if sample_frames.shape[0] > 0:
            sample_frames = sample_frames.to(device)
            patch_tokens = tokenizer.patch_encoder(sample_frames)
            train_patch_stats = {
                'mean': float(patch_tokens.mean().item()),
                'std': float(patch_tokens.std().item()),
                'min': float(patch_tokens.min().item()),
                'max': float(patch_tokens.max().item()),
            }
        
        # Sample test patches
        sample_frames = test_frames[:min(10, test_frames.shape[0])]
        if sample_frames.shape[0] > 0:
            sample_frames = sample_frames.to(device)
            patch_tokens = tokenizer.patch_encoder(sample_frames)
            test_patch_stats = {
                'mean': float(patch_tokens.mean().item()),
                'std': float(patch_tokens.std().item()),
                'min': float(patch_tokens.min().item()),
                'max': float(patch_tokens.max().item()),
            }
    
    stats = {
        'train_pixel_min': train_pixel_min,
        'train_pixel_max': train_pixel_max,
        'train_pixel_mean': train_pixel_mean,
        'train_pixel_std': train_pixel_std,
        'test_pixel_min': test_pixel_min,
        'test_pixel_max': test_pixel_max,
        'test_pixel_mean': test_pixel_mean,
        'test_pixel_std': test_pixel_std,
        'train_normalized': train_normalized,
        'test_normalized': test_normalized,
        'train_patch_stats': train_patch_stats,
        'test_patch_stats': test_patch_stats,
    }
    
    # Plot 1: Pixel value distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(train_pixels, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black')
    axes[0, 0].hist(test_pixels, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black')
    axes[0, 0].axvline(0, color='green', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(1, color='green', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Pixel Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Pixel Value Distribution (should be in [0, 1])')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Sample images with patch grid
    num_patches_per_side = tokenizer.patch_encoder.num_patches_per_side
    patch_size = tokenizer.patch_encoder.patch_size
    
    # Show sample train images (2 samples)
    for idx in range(min(2, train_frames.shape[0])):
        ax = axes[0, 1] if idx == 0 else axes[1, 0]
        frame = train_frames[idx, 0, 0].numpy()  # First timestep, first channel
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Train Sample {idx} (with patch grid)')
        ax.axis('off')
        
        # Draw patch grid
        for i in range(num_patches_per_side + 1):
            y = i * patch_size
            ax.axhline(y, color='red', linewidth=0.5, alpha=0.5)
        for j in range(num_patches_per_side + 1):
            x = j * patch_size
            ax.axvline(x, color='red', linewidth=0.5, alpha=0.5)
    
    # Plot 3: Patch token magnitude heatmap (spatial distribution)
    if train_patch_stats:
        # Get patch tokens for one sample
        sample_frame = train_frames[0:1, 0:1].to(device)  # (1, 1, C, H, W)
        with torch.no_grad():
            patch_tokens = tokenizer.patch_encoder(sample_frame)  # (1, 1, M, d_model)
            patch_magnitudes = patch_tokens.cpu().norm(dim=-1).squeeze().numpy()  # (M,)
            
            # Reshape to spatial grid
            patch_grid = patch_magnitudes.reshape(num_patches_per_side, num_patches_per_side)
            
            im = axes[1, 1].imshow(patch_grid, cmap='viridis')
            axes[1, 1].set_title('Patch Token Magnitude Heatmap (Spatial)')
            axes[1, 1].axis('off')
            plt.colorbar(im, ax=axes[1, 1])
    
    plt.tight_layout()
    plot_path = save_dir / "image_encoding.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    stats['plot_path'] = str(plot_path)
    return stats


def _verify_tokenization(
    train_data: dict,
    test_data: dict,
    tokenizer: MGDTTokenizer,
) -> dict:
    """Verify tokenization correctness."""
    all_checks = []
    
    # Check token structure for a few samples
    for split_name, data in [('train', train_data), ('test', test_data)]:
        for tok_out in data['tok_outputs'][:5]:  # Check first 5 batches
            B, T = tok_out.B, tok_out.T
            tokens_per_step = tok_out.tokens_per_step
            num_patches = tok_out.num_patches
            
            # Expected structure: [RTG, GAME, PATCH_1..M, ACTION]
            expected_tokens_per_step = 3 + num_patches  # RTG + GAME + PATCHES + ACTION
            tokens_per_step_valid = (tokens_per_step == expected_tokens_per_step)
            
            # Check sequence length
            expected_L = T * tokens_per_step
            actual_L = tok_out.tokens.shape[1]
            seq_len_valid = (actual_L == expected_L)
            
            # Verify action positions (should be at positions: tokens_per_step - 1, 2*tokens_per_step - 1, ...)
            action_positions = []
            for t in range(T):
                expected_pos = t * tokens_per_step + (tokens_per_step - 1)
                action_positions.append(expected_pos)
            
            # Extract action tokens and verify they match action_ids
            action_tokens_match = True
            for b in range(min(2, B)):  # Check first 2 samples in batch
                for t_idx, pos in enumerate(action_positions):
                    if pos < actual_L:
                        # Get the token at action position
                        action_token = tok_out.tokens[b, pos]
                        # Get the expected action embedding
                        action_id = tok_out.action_ids[b, t_idx].item()
                        expected_embed = tokenizer.action_embed(torch.tensor(action_id, device=action_token.device))
                        
                        # Check if they match (allowing for small numerical differences)
                        if not torch.allclose(action_token, expected_embed, atol=1e-5):
                            action_tokens_match = False
                            break
                if not action_tokens_match:
                    break
            
            # Verify RTG quantization - check that RTG IDs are in valid range
            rtg_ids_valid = True
            rtg_ids_flat = tok_out.rtg_ids.cpu().numpy().flatten()
            if len(rtg_ids_flat) > 0:
                # Check all RTG IDs are in valid range [0, n_rtg_bins-1]
                valid_range = (rtg_ids_flat >= 0) & (rtg_ids_flat < tokenizer.n_rtg_bins)
                rtg_ids_valid = bool(np.all(valid_range))
            
            all_checks.append({
                'split': split_name,
                'tokens_per_step_valid': tokens_per_step_valid,
                'seq_len_valid': seq_len_valid,
                'action_tokens_match': action_tokens_match,
                'rtg_ids_valid': rtg_ids_valid,
            })
    
    # Aggregate results
    all_tokens_per_step_valid = all(c['tokens_per_step_valid'] for c in all_checks)
    all_seq_len_valid = all(c['seq_len_valid'] for c in all_checks)
    all_action_tokens_match = all(c['action_tokens_match'] for c in all_checks)
    all_rtg_ids_valid = all(c['rtg_ids_valid'] for c in all_checks)
    
    return {
        'tokens_per_step_valid': all_tokens_per_step_valid,
        'seq_len_valid': all_seq_len_valid,
        'action_tokens_match': all_action_tokens_match,
        'rtg_ids_valid': all_rtg_ids_valid,
        'all_checks_passed': all_tokens_per_step_valid and all_seq_len_valid and all_action_tokens_match and all_rtg_ids_valid,
        'num_checks': len(all_checks),
    }


def _check_consistency(train_data: dict, test_data: dict) -> dict:
    """Check data consistency."""
    checks = {}
    
    # Shape validation
    for split_name, data in [('train', train_data), ('test', test_data)]:
        frames = data['frames']
        actions = data['actions']
        rtg = data['rtg']
        
        # Check batch dimension matches
        batch_match = (frames.shape[0] == actions.shape[0] == rtg.shape[0])
        
        # Check sequence dimension matches (T dimension)
        if frames.shape[1] == actions.shape[1] == rtg.shape[1]:
            seq_match = True
            T = frames.shape[1]
        else:
            seq_match = False
            T = None
        
        # NaN/Inf checks
        frames_has_nan = torch.isnan(frames).any().item()
        frames_has_inf = torch.isinf(frames).any().item()
        actions_has_nan = torch.isnan(actions.float()).any().item()
        actions_has_inf = torch.isinf(actions.float()).any().item()
        rtg_has_nan = torch.isnan(rtg).any().item()
        rtg_has_inf = torch.isinf(rtg).any().item()
        
        # RTG monotonicity check (should generally decrease, but allow for episode boundaries)
        rtg_monotonic = True
        if T is not None and rtg.shape[0] > 0:
            # Check a sample of sequences
            for i in range(min(10, rtg.shape[0])):
                rtg_seq = rtg[i].numpy()
                # Count how many times RTG increases (should be rare, mostly at episode boundaries)
                increases = np.sum(np.diff(rtg_seq) > 0.1)  # Allow small increases due to floating point
                # If more than 20% of steps increase, something might be wrong
                if increases > len(rtg_seq) * 0.2:
                    rtg_monotonic = False
                    break
        
        checks[split_name] = {
            'batch_match': batch_match,
            'seq_match': seq_match,
            'frames_has_nan': frames_has_nan,
            'frames_has_inf': frames_has_inf,
            'actions_has_nan': actions_has_nan,
            'actions_has_inf': actions_has_inf,
            'rtg_has_nan': rtg_has_nan,
            'rtg_has_inf': rtg_has_inf,
            'rtg_monotonic': rtg_monotonic,
        }
    
    # Overall consistency
    all_consistent = (
        checks['train']['batch_match'] and checks['train']['seq_match'] and
        checks['test']['batch_match'] and checks['test']['seq_match'] and
        not checks['train']['frames_has_nan'] and not checks['train']['frames_has_inf'] and
        not checks['test']['frames_has_nan'] and not checks['test']['frames_has_inf'] and
        not checks['train']['actions_has_nan'] and not checks['train']['actions_has_inf'] and
        not checks['test']['actions_has_nan'] and not checks['test']['actions_has_inf'] and
        not checks['train']['rtg_has_nan'] and not checks['train']['rtg_has_inf'] and
        not checks['test']['rtg_has_nan'] and not checks['test']['rtg_has_inf']
    )
    
    return {
        'checks': checks,
        'all_consistent': all_consistent,
    }


def _compare_splits(train_data: dict, test_data: dict, save_dir: Path, show_plots: bool) -> dict:
    """Compare train/test splits."""
    # Distribution comparisons are already done in other functions
    # Here we just create a summary comparison plot
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Action distribution comparison
    train_actions = train_data['actions'].flatten().numpy()
    test_actions = test_data['actions'].flatten().numpy()
    n_actions = int(max(train_actions.max(), test_actions.max()) + 1)
    
    axes[0].hist(train_actions, bins=n_actions, alpha=0.7, label='Train', color='blue', edgecolor='black', density=True)
    axes[0].hist(test_actions, bins=n_actions, alpha=0.7, label='Test', color='red', edgecolor='black', density=True)
    axes[0].set_xlabel('Action ID')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Action Distribution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RTG distribution comparison
    train_rtg = train_data['rtg'].flatten().numpy()
    test_rtg = test_data['rtg'].flatten().numpy()
    
    axes[1].hist(train_rtg, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black', density=True)
    axes[1].hist(test_rtg, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black', density=True)
    axes[1].set_xlabel('RTG Value')
    axes[1].set_ylabel('Density')
    axes[1].set_title('RTG Distribution Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Pixel value distribution comparison
    train_pixels = train_data['frames'].numpy().flatten()
    test_pixels = test_data['frames'].numpy().flatten()
    
    axes[2].hist(train_pixels, bins=50, alpha=0.7, label='Train', color='blue', edgecolor='black', density=True)
    axes[2].hist(test_pixels, bins=50, alpha=0.7, label='Test', color='red', edgecolor='black', density=True)
    axes[2].set_xlabel('Pixel Value')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Pixel Value Distribution Comparison')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / "train_test_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # Compute distribution similarity metrics (simple overlap)
    # For actions
    train_action_dist = np.bincount(train_actions.astype(int), minlength=n_actions)
    test_action_dist = np.bincount(test_actions.astype(int), minlength=n_actions)
    train_action_dist = train_action_dist / train_action_dist.sum()
    test_action_dist = test_action_dist / test_action_dist.sum()
    action_overlap = np.minimum(train_action_dist, test_action_dist).sum()
    
    return {
        'action_distribution_overlap': float(action_overlap),
        'plot_path': str(plot_path),
    }


def _visualize_samples(train_data: dict, test_data: dict, save_dir: Path, show_plots: bool) -> dict:
    """Generate sample visualizations."""
    import random

    # Image grid
    n_samples = 4
    train_frames = train_data['frames']
    test_frames = test_data['frames']
    
    # Calculate grid dimensions for frames
    # We want to show n_samples frames total (n_samples//2 train + n_samples//2 test)
    n_frames_per_split = n_samples // 2
    # Use a grid that can accommodate all frames (e.g., 5 columns for 20 frames = 4 rows)
    n_cols = 5
    n_rows_frames = (n_samples + n_cols - 1) // n_cols  # Ceiling division
    # Add rows for action/RTG plots (2 rows)
    n_rows_total = n_rows_frames + 2
    
    fig = plt.figure(figsize=(16, 4 * n_rows_total))
    
    # Show sample frames
    for i in range(min(n_frames_per_split, train_frames.shape[0])):
        row = i // n_cols
        col = i % n_cols
        ax = plt.subplot(n_rows_total, n_cols, row * n_cols + col + 1)
        frame = train_frames[i, 0, 0].numpy()  # First timestep
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Train Frame {i}', fontsize=8)
        ax.axis('off')
    
    for i in range(min(n_frames_per_split, test_frames.shape[0])):
        row = (i + n_frames_per_split) // n_cols
        col = (i + n_frames_per_split) % n_cols
        ax = plt.subplot(n_rows_total, n_cols, row * n_cols + col + 1)
        frame = test_frames[i, 0, 0].numpy()  # First timestep
        ax.imshow(frame, cmap='gray')
        ax.set_title(f'Test Frame {i}', fontsize=8)
        ax.axis('off')
    
    # Action sequences (spanning 2 columns in the second-to-last row, starting at column 1)
    seq_count = 50
    ax1 = plt.subplot(n_rows_total, n_cols, (n_rows_total - 1) * n_cols + 1)
    train_actions = train_data['actions']
    for i in range(min(seq_count, train_actions.shape[0])):
        ax1.plot(random.choice(train_actions).numpy(), alpha=0.6, label=f'Train seq {i}')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Action ID')
    ax1.set_title('Action Sequences (Train)')
    # ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(n_rows_total, n_cols, (n_rows_total - 1) * n_cols + 2)
    test_actions = test_data['actions']
    for i in range(min(seq_count, test_actions.shape[0])):
        ax2.plot(random.choice(test_actions).numpy(), alpha=0.6, label=f'Test seq {i}')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Action ID')
    ax2.set_title('Action Sequences (Test)')
    # ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # RTG trajectories (spanning 2 columns in the last row, starting at column 1)
    ax3 = plt.subplot(n_rows_total, n_cols, n_rows_total * n_cols - 1)
    train_rtg = train_data['rtg']
    for i in range(min(seq_count, train_rtg.shape[0])):
        ax3.plot(random.choice(train_rtg).numpy(), alpha=0.6, label=f'Train seq {i}')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('RTG Value')
    ax3.set_title('RTG Trajectories (Train)')
    # ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(n_rows_total, n_cols, n_rows_total * n_cols)
    test_rtg = test_data['rtg']
    for i in range(min(seq_count, test_rtg.shape[0])):
        ax4.plot(random.choice(test_rtg).numpy(), alpha=0.6, label=f'Test seq {i}')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('RTG Value')
    ax4.set_title('RTG Trajectories (Test)')
    # ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = save_dir / "sample_visualizations.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    return {
        'plot_path': str(plot_path),
    }


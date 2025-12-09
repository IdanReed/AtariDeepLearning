from typing import Any, Dict, List, Optional
from episode_dataset import BinningInfo
from epsiode_dataloader import DataLoaderBundle
from mgdt_model_trainer import Encoder, train_mgdt
from pathlib import Path

def run_experiment_basic(
    title_prefix: str,
    main_bundle: DataLoaderBundle, 
    holdout_bundle: DataLoaderBundle, 
    bins: BinningInfo,
    experiment_dir: Path,
    encoder_type: Encoder,
    best_params: Optional[Dict[str, Any]] = None,
    ):
    
    from utils import safe_clear_output_dir
    safe_clear_output_dir(experiment_dir)

    # Best params
    if best_params is None:
        from optuna_tuning import run_optuna
        study = run_optuna(main_bundle.train_loader, main_bundle.val_loader, bins, encoder_type=encoder_type)
        best_params = study.best_params
    
    # Train with best params
    model, main_train_stats, main_val_stats = train_mgdt(
        bins=bins,
        dataloader_train=main_bundle.train_loader,
        dataloader_val=main_bundle.val_loader,
        encoder_type=encoder_type,
        **best_params,
    )

    # Train holdout
    model, holdout_train_stats, holdout_val_stats = train_mgdt(
        model=model,
        bins=bins,
        dataloader_train=holdout_bundle.train_loader,
        dataloader_val=holdout_bundle.val_loader,
        **best_params,
    )

    # Save checkpoint
    from utils import save_checkpoint
    save_checkpoint(
        output_dir=experiment_dir,
        model=model, 
        main_train_stats=main_train_stats, 
        main_val_stats=main_val_stats, 
        holdout_train_stats=holdout_train_stats, 
        holdout_val_stats=holdout_val_stats,
        params=best_params,
    )

    # Main plots
    from mgdt_model_stats import plot_losses, plot_holdout_comparison
    plot_losses(main_train_stats, main_val_stats, output_dir=experiment_dir, title_prefix=f"{title_prefix} - Main", no_show=True)
    plot_losses(holdout_train_stats, holdout_val_stats, output_dir=experiment_dir, title_prefix=f"{title_prefix} - Holdout", no_show=True)
    plot_holdout_comparison(main_train_stats, main_val_stats, holdout_train_stats, holdout_val_stats, output_dir=experiment_dir, title_prefix=f"{title_prefix} - Comparison", no_show=True)

    return best_params
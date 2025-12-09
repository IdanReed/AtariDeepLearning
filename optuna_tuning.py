import optuna
from typing import List, Tuple
from mgdt_model_trainer import Encoder, train_mgdt


def run_optuna(
    train_loader,
    val_loader,
    bins,
    *,
    n_trials: int = 20,
    lr_range: Tuple[float, float] = (1e-5, 1e-2),
    emb_size_choices: List[int] = [64, 128, 256, 512, 1024],
    n_layers_range: Tuple[int, int] = (2, 6),
    n_heads_range: Tuple[int, int] = (1, 4),
    num_epochs_range: Tuple[int, int] = (1, 5),

    encoder_type: Encoder = Encoder.Patch,
):
    def objective(trial):
        # Use same param names as train_mgdt so best_params can be expanded directly
        lr = trial.suggest_float('lr', lr_range[0], lr_range[1], log=True)
        emb_size = trial.suggest_categorical('emb_size', emb_size_choices)
        n_layers = trial.suggest_int('n_layers', n_layers_range[0], n_layers_range[1])
        n_heads = trial.suggest_int('n_heads', n_heads_range[0], n_heads_range[1])
        num_epochs = trial.suggest_int('num_epochs', num_epochs_range[0], num_epochs_range[1])
        print("Trial params:", trial.params)
        
        # Ensure that emb_size is divisible by n_heads
        if emb_size % n_heads != 0:
            raise optuna.exceptions.TrialPruned()
        
        model, main_train_stats, main_val_stats = train_mgdt(
            bins=bins,
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            lr=lr,
            n_layers=n_layers,
            n_heads=n_heads,
            num_epochs=num_epochs,
            emb_size=emb_size,
            encoder_type=encoder_type
        )
        return main_val_stats[-1]['loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    return study
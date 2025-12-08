import optuna
from mgdt_model_trainer import train_mgdt

def run_optuna(train_loader, val_loader, bins):

    def objective(trial):
        # Define the hyperparameter search space
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        emb_size = trial.suggest_categorical('embedding_size', [64, 128, 256, 512, 1024])
        n_layers = trial.suggest_int('num_layers', 5, 10)
        n_heads = trial.suggest_int('num_heads', 2, 8)
        num_epochs = trial.suggest_int('num_epochs', 10, 100)

        # Train the model with the suggested hyperparameters
        model, main_train_stats, main_val_stats = train_mgdt(
            lr=lr,
            n_layers=n_layers,
            n_heads=n_heads,
            num_epochs=num_epochs,
            emb_size=emb_size
        )
        return main_val_stats[-1]['loss']

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    return study
        
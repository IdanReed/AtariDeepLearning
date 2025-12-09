#!/usr/bin/env python3
"""
Window Size 8 experiment for vast.ai deployment.

Usage:
    python vast_experiment_window_8.py --dataset-root /workspace/dataset
"""
from vast_utils import setup_experiment
from experiment_basic import run_experiment_basic
from mgdt_model_trainer import Encoder


def main():
    args, data = setup_experiment(
        name="window_8",
        description="Run Window Size 8 experiment",
        timestep_window_size=8,
    )

    # None triggers optuna tuning
    run_experiment_basic(
        title_prefix="Window Size 8",
        main_bundle=data.main_bundle,
        holdout_bundle=data.holdout_bundle,
        bins=data.bins,
        experiment_dir=args.output_dir / "window_size_8",
        encoder_type=Encoder.Patch,
        best_params=None,
    )

    print("Experiment window_8 completed!")


if __name__ == "__main__":
    main()

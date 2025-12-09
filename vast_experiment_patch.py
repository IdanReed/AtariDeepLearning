#!/usr/bin/env python3
"""
Patch Encoder experiment for vast.ai deployment.

Usage:
    python vast_experiment_patch.py --dataset-root /workspace/dataset
"""
from vast_utils import setup_experiment
from experiment_basic import run_experiment_basic
from mgdt_model_trainer import Encoder


def main():
    args, data = setup_experiment(
        name="patch",
        description="Run Patch experiment",
    )

    # None triggers optuna tuning
    run_experiment_basic(
        title_prefix="Patch",
        main_bundle=data.main_bundle,
        holdout_bundle=data.holdout_bundle,
        bins=data.bins,
        experiment_dir=args.output_dir / "patch",
        encoder_type=Encoder.Patch,
        best_params=None,
    )

    print("Experiment patch completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test experiment for vast.ai deployment.
Uses 10% of dataset to quickly verify the setup works before expensive runs.

Usage:
    python vast_experiment_test.py --dataset-root /workspace/dataset
"""
from vast_utils import setup_experiment
from experiment_basic import run_experiment_basic
from mgdt_model_trainer import Encoder


# Reduced params for faster testing
TEST_PARAMS = {
    'lr': 0.002226768831180977,
    'emb_size': 128,
    'n_layers': 2,
    'n_heads': 2,
    'num_epochs': 1,  # Only 1 epoch for testing
}


def main():
    print("=" * 60)
    print("TEST EXPERIMENT - Quick validation run")
    print("=" * 60)
    
    args, data = setup_experiment(
        name="test",
        description="Run Test experiment (10% dataset)",
        include_sample_fraction=True,
    )

    # Run with fixed params (no optuna tuning for speed)
    run_experiment_basic(
        title_prefix="Test",
        main_bundle=data.main_bundle,
        holdout_bundle=data.holdout_bundle,
        bins=data.bins,
        experiment_dir=args.output_dir / "test",
        encoder_type=Encoder.Patch,
        best_params=TEST_PARAMS,
    )

    print()
    print("=" * 60)
    print("TEST EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Your vast.ai setup is working correctly.")


if __name__ == "__main__":
    main()

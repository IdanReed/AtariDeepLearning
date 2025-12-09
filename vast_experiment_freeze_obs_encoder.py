#!/usr/bin/env python3
"""
Freeze Obs Encoder experiment for vast.ai deployment.

Usage:
    python vast_experiment_freeze_obs_encoder.py --dataset-root /workspace/dataset
"""
from vast_utils import setup_experiment, BEST_BASELINE_PARAMS
from experiment_freeze import run_experiment_freeze
from mgdt_model import Freezeable


def main():
    args, data = setup_experiment(
        name="freeze_obs_encoder",
        description="Run Freeze Obs Encoder experiment",
    )

    run_experiment_freeze(
        title_prefix="Freeze Obs Encoder",
        main_bundle=data.main_bundle,
        holdout_bundle=data.holdout_bundle,
        bins=data.bins,
        freeze_components=[Freezeable.ObsEncoder],
        experiment_dir=args.output_dir / "freeze_obs_encoder",
        best_params=BEST_BASELINE_PARAMS,
    )

    print("Experiment freeze_obs_encoder completed!")


if __name__ == "__main__":
    main()

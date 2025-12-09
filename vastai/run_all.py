#!/usr/bin/env python3
"""
Launch all experiments in parallel on separate vast.ai instances.

This script:
1. Launches a GPU instance for each experiment
2. Uploads project code
3. Starts experiments
4. Monitors progress
5. Downloads results when complete
6. Cleans up instances

Usage:
    python vastai/run_all.py --config vastai/config.yaml
    python vastai/run_all.py --config vastai/config.yaml --experiments freeze_transformer cnn
    python vastai/run_all.py --config vastai/config.yaml --dry-run
"""
import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vastai.orchestrate import VastAIOrchestrator, VastConfig


@dataclass
class ExperimentRun:
    """Tracks the state of a running experiment."""
    experiment_name: str
    instance_id: Optional[int] = None
    status: str = "pending"  # pending, launching, running, completed, failed
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class ParallelExperimentRunner:
    """Runs multiple experiments in parallel on vast.ai."""
    
    def __init__(
        self,
        config_path: str | Path,
        experiments: Optional[List[str]] = None,
        results_dir: str | Path = "./results",
        max_concurrent: int = 10,
    ):
        self.config = VastConfig.from_yaml(config_path)
        self.orchestrator = VastAIOrchestrator(config=self.config)
        self.results_dir = Path(results_dir)
        self.max_concurrent = max_concurrent
        
        # Use specified experiments or all from config
        self.experiments = experiments or self.config.experiments
        
        # Track experiment runs
        self.runs: Dict[str, ExperimentRun] = {
            exp: ExperimentRun(experiment_name=exp)
            for exp in self.experiments
        }
        
        self._lock = threading.Lock()
    
    def _log(self, message: str):
        """Thread-safe logging."""
        with self._lock:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")
    
    def _run_single_experiment(self, experiment_name: str) -> ExperimentRun:
        """Run a single experiment on vast.ai (called in thread)."""
        run = self.runs[experiment_name]
        run.status = "launching"
        run.start_time = time.time()
        
        try:
            # Step 1: Launch instance
            self._log(f"[{experiment_name}] Launching instance...")
            instance_id = self.orchestrator.launch_experiment(experiment_name)
            run.instance_id = instance_id
            
            # Step 2: Wait for instance to be ready
            self._log(f"[{experiment_name}] Waiting for instance {instance_id} to be ready...")
            self.orchestrator.wait_for_instance_ready(instance_id, timeout=600)
            
            # Step 3: Upload project files
            self._log(f"[{experiment_name}] Uploading project files...")
            self.orchestrator.upload_project(instance_id)
            
            # Step 4: Run experiment
            self._log(f"[{experiment_name}] Starting experiment...")
            run.status = "running"
            process = self.orchestrator.run_experiment_on_instance(instance_id, experiment_name)
            
            # Step 5: Wait for completion and stream output
            for line in process.stdout:
                self._log(f"[{experiment_name}] {line.rstrip()}")
            
            return_code = process.wait()
            
            if return_code != 0:
                raise RuntimeError(f"Experiment exited with code {return_code}")
            
            # Step 6: Download results
            self._log(f"[{experiment_name}] Downloading results...")
            self.orchestrator.download_results(instance_id, self.results_dir)
            
            run.status = "completed"
            run.end_time = time.time()
            self._log(f"[{experiment_name}] Completed successfully!")
            
        except Exception as e:
            run.status = "failed"
            run.error = str(e)
            run.end_time = time.time()
            self._log(f"[{experiment_name}] Failed: {e}")
        
        finally:
            # Cleanup: destroy instance
            if run.instance_id:
                try:
                    self._log(f"[{experiment_name}] Cleaning up instance {run.instance_id}...")
                    self.orchestrator.destroy_instance(run.instance_id)
                except Exception as e:
                    self._log(f"[{experiment_name}] Failed to cleanup: {e}")
        
        return run
    
    def run_all(self, dry_run: bool = False) -> Dict[str, ExperimentRun]:
        """
        Run all experiments in parallel.
        
        Args:
            dry_run: If True, just print what would be done
            
        Returns:
            Dictionary of experiment name -> ExperimentRun
        """
        print("=" * 60)
        print("vast.ai Parallel Experiment Runner")
        print("=" * 60)
        print(f"Experiments to run: {', '.join(self.experiments)}")
        print(f"Results directory: {self.results_dir}")
        print(f"Max concurrent: {self.max_concurrent}")
        print()
        
        if dry_run:
            print("DRY RUN - Would launch the following experiments:")
            for exp in self.experiments:
                print(f"  - {exp}")
            print()
            
            # Check for available instances
            print("Checking for available GPU instances...")
            try:
                offers = self.orchestrator.find_gpu_instances()
                print(f"Found {len(offers)} suitable instances")
                if offers:
                    print(f"Cheapest: ${offers[0]['dph_total']:.3f}/hr on {offers[0]['gpu_name']}")
            except Exception as e:
                print(f"Error checking instances: {e}")
            
            return self.runs
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Run experiments in parallel using ThreadPoolExecutor
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # Submit all experiments
            futures = {
                executor.submit(self._run_single_experiment, exp): exp
                for exp in self.experiments
            }
            
            # Wait for all to complete
            for future in as_completed(futures):
                exp_name = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self._log(f"[{exp_name}] Unexpected error: {e}")
        
        total_time = time.time() - start_time
        
        # Print summary
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        completed = [r for r in self.runs.values() if r.status == "completed"]
        failed = [r for r in self.runs.values() if r.status == "failed"]
        
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Completed: {len(completed)}/{len(self.runs)}")
        print(f"Failed: {len(failed)}/{len(self.runs)}")
        print()
        
        for exp_name, run in self.runs.items():
            duration = ""
            if run.start_time and run.end_time:
                duration = f" ({(run.end_time - run.start_time) / 60:.1f} min)"
            
            status_icon = "✓" if run.status == "completed" else "✗"
            print(f"  {status_icon} {exp_name}: {run.status}{duration}")
            if run.error:
                print(f"      Error: {run.error}")
        
        print()
        print(f"Results saved to: {self.results_dir}")
        
        return self.runs


def main():
    parser = argparse.ArgumentParser(
        description="Run all experiments in parallel on vast.ai"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vastai/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        help="Specific experiments to run (default: all from config)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just print what would be done",
    )
    
    args = parser.parse_args()
    
    runner = ParallelExperimentRunner(
        config_path=args.config,
        experiments=args.experiments,
        results_dir=args.results_dir,
        max_concurrent=args.max_concurrent,
    )
    
    runs = runner.run_all(dry_run=args.dry_run)
    
    # Exit with error if any experiments failed
    if any(r.status == "failed" for r in runs.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()


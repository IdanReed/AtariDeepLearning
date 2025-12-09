#!/usr/bin/env python3
"""
vast.ai orchestration for running parallel experiments.

This module provides functions to:
- Search for suitable GPU instances
- Launch instances with experiments
- Monitor experiment progress
- Download results
- Cleanup instances

Prerequisites:
    pip install vastai
    Set VAST_API_KEY environment variable or use ~/.vast_api_key file

Usage:
    from vastai.orchestrate import VastAIOrchestrator
    
    orchestrator = VastAIOrchestrator(config_path="vastai/config.yaml")
    instance_id = orchestrator.launch_experiment("freeze_transformer")
    orchestrator.monitor_instance(instance_id)
    orchestrator.download_results(instance_id, local_dir="./results")
    orchestrator.destroy_instance(instance_id)
"""
import json
import os
import subprocess
import time
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import shutil


@dataclass
class InstanceInfo:
    """Information about a running vast.ai instance."""
    instance_id: int
    experiment_name: str
    status: str
    ssh_host: Optional[str] = None
    ssh_port: Optional[int] = None
    start_time: float = field(default_factory=time.time)


@dataclass
class VastConfig:
    """Configuration for vast.ai deployment."""
    gcs_dataset_url: str
    gpu_types: List[str]
    disk_space_gb: int
    experiments: List[str]
    docker_image: str = "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
    min_gpu_ram_gb: float = 16.0
    max_price_per_hour: float = 1.0
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> "VastConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            gcs_dataset_url=data.get("gcs_dataset_url", ""),
            gpu_types=data.get("gpu_types", ["RTX_3090", "RTX_4090"]),
            disk_space_gb=data.get("disk_space_gb", 50),
            experiments=data.get("experiments", []),
            docker_image=data.get("docker_image", cls.docker_image),
            min_gpu_ram_gb=data.get("min_gpu_ram_gb", 16.0),
            max_price_per_hour=data.get("max_price_per_hour", 1.0),
        )


class VastAIOrchestrator:
    """Orchestrator for vast.ai experiments."""
    
    def __init__(
        self, 
        config: Optional[VastConfig] = None,
        config_path: Optional[str | Path] = None,
        project_dir: Optional[Path] = None,
    ):
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = VastConfig.from_yaml(config_path)
        else:
            raise ValueError("Must provide either config or config_path")
        
        self.project_dir = project_dir or Path(__file__).parent.parent
        self.instances: Dict[int, InstanceInfo] = {}
        
        # Verify vastai CLI is available
        self._check_vastai_cli()
    
    def _check_vastai_cli(self):
        """Check that vastai CLI is installed and configured."""
        try:
            result = subprocess.run(
                ["vastai", "show", "user"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "vastai CLI not configured. Run: vastai set api-key YOUR_API_KEY"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "vastai CLI not found. Install with: pip install vastai"
            )
    
    def _run_vastai(self, *args, parse_json: bool = False) -> Any:
        """Run a vastai CLI command."""
        cmd = ["vastai"] + list(args)
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            raise RuntimeError(f"vastai command failed: {result.stderr}")
        
        if parse_json:
            return json.loads(result.stdout)
        return result.stdout
    
    def find_gpu_instances(self, num_gpus: int = 1) -> List[Dict]:
        """
        Search for suitable GPU instances.
        
        Returns list of available offers sorted by price.
        """
        # Build search query
        gpu_query = " OR ".join([f"gpu_name={g}" for g in self.config.gpu_types])
        query = f"({gpu_query}) num_gpus>={num_gpus} gpu_ram>={self.config.min_gpu_ram_gb} disk_space>={self.config.disk_space_gb} dph<={self.config.max_price_per_hour}"
        
        print(f"Searching for instances: {query}")
        
        offers = self._run_vastai(
            "search", "offers",
            "--raw",
            query,
            parse_json=True,
        )
        
        # Sort by price
        offers = sorted(offers, key=lambda x: x.get("dph_total", float("inf")))
        
        print(f"Found {len(offers)} matching instances")
        return offers
    
    def launch_experiment(
        self,
        experiment_name: str,
        offer_id: Optional[int] = None,
    ) -> int:
        """
        Launch an instance to run a specific experiment.
        
        Args:
            experiment_name: Name of the experiment (e.g., "freeze_transformer")
            offer_id: Specific offer to use, or None to auto-select cheapest
            
        Returns:
            Instance ID
        """
        # Find an offer if not specified
        if offer_id is None:
            offers = self.find_gpu_instances()
            if not offers:
                raise RuntimeError("No suitable GPU instances available")
            offer_id = offers[0]["id"]
            print(f"Selected offer {offer_id} at ${offers[0]['dph_total']:.3f}/hr")
        
        # Create the onstart script that will be executed when instance starts
        onstart_script = self._create_onstart_script(experiment_name)
        
        # Write script to temp file
        script_path = Path("/tmp/onstart.sh") if os.name != "nt" else Path("./onstart_temp.sh")
        script_path.write_text(onstart_script)
        
        # Create instance
        result = self._run_vastai(
            "create", "instance",
            str(offer_id),
            "--image", self.config.docker_image,
            "--disk", str(self.config.disk_space_gb),
            "--onstart-cmd", f"bash -c '{onstart_script}'",
            "--raw",
            parse_json=True,
        )
        
        instance_id = result.get("new_contract")
        if not instance_id:
            raise RuntimeError(f"Failed to create instance: {result}")
        
        print(f"Created instance {instance_id} for experiment {experiment_name}")
        
        self.instances[instance_id] = InstanceInfo(
            instance_id=instance_id,
            experiment_name=experiment_name,
            status="starting",
        )
        
        return instance_id
    
    def _create_onstart_script(self, experiment_name: str) -> str:
        """Create the onstart script for an instance."""
        return f"""
#!/bin/bash
set -e

# Setup workspace
mkdir -p /workspace/dataset /workspace/project /workspace/output

# Download dataset
cd /workspace
if [ ! -d "dataset/BeamRiderNoFrameskip-v4" ]; then
    echo "Downloading dataset..."
    wget -O dataset.zip "{self.config.gcs_dataset_url.replace('gs://', 'https://storage.googleapis.com/')}"
    unzip -q dataset.zip -d dataset/
    rm dataset.zip
fi

# Project files should be uploaded via scp after instance starts
echo "Waiting for project files..."
"""
    
    def get_instance_status(self, instance_id: int) -> Dict:
        """Get the current status of an instance."""
        instances = self._run_vastai("show", "instances", "--raw", parse_json=True)
        
        for inst in instances:
            if inst.get("id") == instance_id:
                return inst
        
        return {"status": "not_found"}
    
    def wait_for_instance_ready(
        self, 
        instance_id: int, 
        timeout: int = 600,
        poll_interval: int = 10,
    ) -> Dict:
        """Wait for an instance to be ready for SSH."""
        start = time.time()
        
        while time.time() - start < timeout:
            status = self.get_instance_status(instance_id)
            actual_status = status.get("actual_status", "")
            
            print(f"Instance {instance_id} status: {actual_status}")
            
            if actual_status == "running":
                ssh_host = status.get("ssh_host")
                ssh_port = status.get("ssh_port")
                
                if ssh_host and ssh_port:
                    if instance_id in self.instances:
                        self.instances[instance_id].ssh_host = ssh_host
                        self.instances[instance_id].ssh_port = ssh_port
                        self.instances[instance_id].status = "running"
                    return status
            
            elif actual_status in ("exited", "error"):
                raise RuntimeError(f"Instance {instance_id} failed: {status}")
            
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Instance {instance_id} did not become ready in {timeout}s")
    
    def upload_project(self, instance_id: int):
        """Upload project files to an instance via scp."""
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")
        
        if not ssh_host or not ssh_port:
            raise RuntimeError(f"Instance {instance_id} SSH not available")
        
        # Files to upload
        files_to_upload = [
            "requirements-linux.txt",
            "utils.py",
            "npz_loader.py",
            "episode.py",
            "episode_dataset.py",
            "epsiode_dataloader.py",
            "encoders.py",
            "mgdt_model.py",
            "mgdt_model_trainer.py",
            "mgdt_model_stats.py",
            "experiment_basic.py",
            "experiment_freeze.py",
            "optuna_tuning.py",
            "vast_utils.py",
            # Standalone experiment scripts
            "vast_experiment_test.py",
            "vast_experiment_freeze_transformer.py",
            "vast_experiment_freeze_obs_encoder.py",
            "vast_experiment_cnn.py",
            "vast_experiment_patch.py",
            "vast_experiment_window_8.py",
            "vast_experiment_window_16.py",
            "vast_experiment_window_32.py",
        ]
        
        print(f"Uploading project files to instance {instance_id}...")
        
        for file in files_to_upload:
            src = self.project_dir / file
            if src.exists():
                cmd = [
                    "scp",
                    "-P", str(ssh_port),
                    "-o", "StrictHostKeyChecking=no",
                    str(src),
                    f"root@{ssh_host}:/workspace/project/",
                ]
                subprocess.run(cmd, check=True)
                print(f"  Uploaded {file}")
        
        print("Project files uploaded successfully!")
    
    def run_experiment_on_instance(
        self,
        instance_id: int,
        experiment_name: str,
    ) -> subprocess.Popen:
        """Start an experiment on an instance (async)."""
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")
        
        if not ssh_host or not ssh_port:
            raise RuntimeError(f"Instance {instance_id} SSH not available")
        
        # Build the command to run on the remote instance
        # experiment_name is expected to be like "vast_experiment_freeze_transformer"
        remote_cmd = f"""
cd /workspace/project && \
pip install -q -r requirements-linux.txt && \
python {experiment_name}.py \
    --dataset-root /workspace/dataset \
    --output-dir /workspace/output
"""
        
        cmd = [
            "ssh",
            "-p", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            f"root@{ssh_host}",
            remote_cmd,
        ]
        
        print(f"Starting experiment {experiment_name} on instance {instance_id}")
        
        # Run in background so we can monitor
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        
        return process
    
    def download_results(
        self,
        instance_id: int,
        local_dir: str | Path = "./results",
    ):
        """Download experiment results from an instance."""
        status = self.get_instance_status(instance_id)
        ssh_host = status.get("ssh_host")
        ssh_port = status.get("ssh_port")
        
        if not ssh_host or not ssh_port:
            raise RuntimeError(f"Instance {instance_id} SSH not available")
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Get experiment name for this instance
        exp_name = self.instances.get(instance_id, InstanceInfo(0, "unknown", "")).experiment_name
        
        # Create experiment-specific output directory
        exp_output_dir = local_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading results from instance {instance_id} to {exp_output_dir}")
        
        cmd = [
            "scp",
            "-r",
            "-P", str(ssh_port),
            "-o", "StrictHostKeyChecking=no",
            f"root@{ssh_host}:/workspace/output/*",
            str(exp_output_dir),
        ]
        
        subprocess.run(cmd, check=True)
        print(f"Results downloaded to {exp_output_dir}")
    
    def destroy_instance(self, instance_id: int):
        """Destroy an instance."""
        print(f"Destroying instance {instance_id}")
        self._run_vastai("destroy", "instance", str(instance_id))
        
        if instance_id in self.instances:
            self.instances[instance_id].status = "destroyed"
    
    def cleanup_all(self):
        """Destroy all tracked instances."""
        for instance_id in list(self.instances.keys()):
            try:
                self.destroy_instance(instance_id)
            except Exception as e:
                print(f"Failed to destroy instance {instance_id}: {e}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="vast.ai orchestration")
    parser.add_argument("--config", default="vastai/config.yaml", help="Config file path")
    parser.add_argument("--action", choices=["search", "launch", "status", "destroy"], required=True)
    parser.add_argument("--experiment", help="Experiment name for launch action")
    parser.add_argument("--instance-id", type=int, help="Instance ID for status/destroy actions")
    
    args = parser.parse_args()
    
    orchestrator = VastAIOrchestrator(config_path=args.config)
    
    if args.action == "search":
        offers = orchestrator.find_gpu_instances()
        for offer in offers[:10]:
            print(f"ID: {offer['id']}, GPU: {offer['gpu_name']}, Price: ${offer['dph_total']:.3f}/hr")
    
    elif args.action == "launch":
        if not args.experiment:
            print("Error: --experiment required for launch action")
            return
        instance_id = orchestrator.launch_experiment(args.experiment)
        print(f"Launched instance {instance_id}")
    
    elif args.action == "status":
        if not args.instance_id:
            print("Error: --instance-id required for status action")
            return
        status = orchestrator.get_instance_status(args.instance_id)
        print(json.dumps(status, indent=2))
    
    elif args.action == "destroy":
        if not args.instance_id:
            print("Error: --instance-id required for destroy action")
            return
        orchestrator.destroy_instance(args.instance_id)


if __name__ == "__main__":
    main()


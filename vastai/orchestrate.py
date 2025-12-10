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
import re
import subprocess
import sys
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
            stdout = result.stdout.strip()
            if not stdout:
                print("Warning: vastai returned empty response")
                return []
            try:
                return json.loads(stdout)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON response: {stdout[:200]}")
                raise RuntimeError(f"Invalid JSON from vastai: {e}")
        return result.stdout
    
    def find_gpu_instances(self, num_gpus: int = 1) -> List[Dict]:
        """
        Search for suitable GPU instances.
        
        Returns list of available offers sorted by price.
        """
        # Build search query - simpler format without OR (search for first available GPU type)
        # vast.ai CLI has issues with complex queries, so we search each GPU type separately
        all_offers = []
        
        for gpu_type in self.config.gpu_types:
            query = f"gpu_name={gpu_type} num_gpus>={num_gpus} gpu_ram>={self.config.min_gpu_ram_gb} disk_space>={self.config.disk_space_gb} dph<={self.config.max_price_per_hour}"
            
            print(f"Searching for {gpu_type}: {query}")
            
            try:
                offers = self._run_vastai_search(query)
                all_offers.extend(offers)
                if offers:
                    print(f"  Found {len(offers)} {gpu_type} instances")
            except Exception as e:
                print(f"  Warning: Search failed for {gpu_type}: {e}")
        
        # Sort by price
        all_offers = sorted(all_offers, key=lambda x: x.get("dph_total", float("inf")))
        
        print(f"Total: {len(all_offers)} matching instances")
        return all_offers
    
    def _run_vastai_search(self, query: str) -> List[Dict]:
        """Run vastai search with proper quoting."""
        # Use shell=True to properly handle the query string
        cmd = f'vastai search offers --raw "{query}"'
        print(f"Running: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"Search failed: {result.stderr}")
        
        stdout = result.stdout.strip()
        if not stdout or stdout == "[]":
            return []
        
        try:
            return json.loads(stdout)
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON: {stdout[:200]}")
            return []
    
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
        
        # Use a simple onstart command - just create workspace directories
        # The actual setup (dataset download) happens via SSH after instance is ready
        simple_onstart = "mkdir -p /workspace/dataset /workspace/project /workspace/output && touch /workspace/.instance_ready"
        
        # Create instance
        result = self._run_vastai(
            "create", "instance",
            str(offer_id),
            "--image", self.config.docker_image,
            "--disk", str(self.config.disk_space_gb),
            "--onstart-cmd", simple_onstart,
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
    
    def _create_setup_script(self) -> str:
        """Create the setup script content to run via SSH."""
        gcs_url = self.config.gcs_dataset_url
        if gcs_url.startswith('gs://'):
            gcs_url = gcs_url.replace('gs://', 'https://storage.googleapis.com/')
        
        return f'''#!/bin/bash
set -e

echo "=== VAST.AI SETUP SCRIPT ==="
echo "URL: {gcs_url}"

cd /workspace

# Check if dataset already exists (double-nested structure expected)
# The code expects: dataset_root/GameName/GameName/files
if [ -d "dataset/BeamRiderNoFrameskip-v4/BeamRiderNoFrameskip-v4" ]; then
    echo "Dataset already exists, skipping download."
    ls -la dataset/
else
    # Install unzip if not available
    if ! command -v unzip &> /dev/null; then
        echo "Installing unzip..."
        apt-get update -qq && apt-get install -qq -y unzip
    fi

    echo "Downloading dataset..."
    wget -q --show-progress -O dataset.zip "{gcs_url}" || {{
        echo "ERROR: wget failed with code $?"
        echo "Trying with curl..."
        curl -L -o dataset.zip "{gcs_url}"
    }}
    
    echo "Download complete. File size:"
    ls -lh dataset.zip
    
    echo "Extracting dataset..."
    unzip -q dataset.zip
    rm dataset.zip
    
    echo "Dataset extracted. Structure:"
    ls -la dataset/
    echo ""
    echo "Checking for expected game directories..."
    for game in BeamRiderNoFrameskip-v4 BreakoutNoFrameskip-v4 EnduroNoFrameskip-v4; do
        if [ -d "dataset/$game" ]; then
            echo "  Found: dataset/$game"
            ls dataset/$game/ | head -3
        else
            echo "  MISSING: dataset/$game"
        fi
    done
fi

# Verify the expected structure exists
if [ ! -d "dataset/BeamRiderNoFrameskip-v4" ]; then
    echo "ERROR: Dataset structure is incorrect!"
    echo "Expected: dataset/BeamRiderNoFrameskip-v4/"
    echo "Actual contents:"
    find dataset -maxdepth 2 -type d
    exit 1
fi

echo "SETUP_COMPLETE" > /workspace/.onstart_done
echo "Setup complete!"
'''
    
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
    
    def run_setup_script(
        self,
        instance_id: int,
        timeout: int = 900,  # 15 min for dataset download
    ):
        """Run setup script via SSH to download dataset."""
        user, host, port = self._get_ssh_info(instance_id)
        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30"
        
        print(f"Running setup script on instance {instance_id}...")
        
        # Create the setup script content and save locally
        setup_script = self._create_setup_script()
        local_script_path = Path("./setup_temp.sh") if os.name == "nt" else Path("/tmp/setup_vast.sh")
        # Force Unix line endings (critical for bash scripts)
        local_script_path.write_text(setup_script, newline='\n')
        
        # Upload the script via SCP
        scp_cmd = f'scp -P {port} {ssh_opts} "{local_script_path}" {user}@{host}:/workspace/setup.sh'
        print("Uploading setup script...")
        result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to upload setup script: {result.stderr}")
        
        # Execute the script via SSH
        ssh_cmd = f'ssh -p {port} {ssh_opts} {user}@{host} "chmod +x /workspace/setup.sh && /workspace/setup.sh"'
        print("Downloading dataset (this may take several minutes)...")
        
        # Run with output streaming
        # Use UTF-8 encoding to handle remote output properly (Windows default is cp1252)
        process = subprocess.Popen(
            ssh_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid bytes instead of crashing
        )
        
        # Stream output
        for line in process.stdout:
            print(f"  {line.rstrip()}")
        
        return_code = process.wait(timeout=timeout)
        
        # Clean up local temp file
        try:
            local_script_path.unlink()
        except Exception:
            pass
        
        if return_code != 0:
            raise RuntimeError(f"Setup script failed with code {return_code}")
        
        # Verify dataset was downloaded
        ssh_base = f'ssh -p {port} {ssh_opts} {user}@{host}'
        verify_cmd = f'{ssh_base} "ls -la /workspace/dataset/ 2>&1 | head -20"'
        verify_result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True, timeout=30)
        print(f"Dataset contents:\n{verify_result.stdout}")
        
        # Check the done marker
        check_cmd = f'{ssh_base} "test -f /workspace/.onstart_done && echo SUCCESS || echo FAILED"'
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)
        
        if "SUCCESS" not in check_result.stdout:
            raise RuntimeError("Setup script did not complete successfully")
        
        print("Setup completed successfully!")
    
    def wait_for_onstart_complete(
        self,
        instance_id: int,
        timeout: int = 900,  # 15 min for dataset download
        poll_interval: int = 15,
    ):
        """
        Wait for instance to be ready then run setup script.
        
        This replaces the old approach of running a complex onstart script.
        Now we just wait for SSH to be ready, then run setup via SSH.
        """
        user, host, port = self._get_ssh_info(instance_id)
        start = time.time()
        ssh_base = f'ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 {user}@{host}'
        
        print(f"Waiting for instance to be ready for setup...")
        
        # Wait for the simple onstart to complete (just creates directories)
        while time.time() - start < 120:  # 2 min timeout for directory creation
            elapsed = int(time.time() - start)
            
            check_cmd = f'{ssh_base} "test -f /workspace/.instance_ready && echo READY || echo WAITING"'
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if "READY" in result.stdout:
                print(f"Instance ready after {elapsed}s")
                break
            
            print(f"  [{elapsed}s] Waiting for instance initialization...")
            time.sleep(poll_interval)
        else:
            raise TimeoutError("Instance did not become ready in 120s")
        
        # Now run the actual setup script via SSH
        self.run_setup_script(instance_id, timeout=timeout)
    
    def _get_ssh_info(self, instance_id: int) -> tuple[str, str, str]:
        """Get SSH connection info (user, host, port) for an instance."""
        ssh_url_result = subprocess.run(
            f'vastai ssh-url {instance_id}',
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if ssh_url_result.returncode != 0:
            raise RuntimeError(f"Failed to get SSH URL: {ssh_url_result.stderr}")
        
        ssh_url = ssh_url_result.stdout.strip()
        match = re.match(r'ssh://(\w+)@([^:]+):(\d+)', ssh_url)
        if not match:
            raise RuntimeError(f"Failed to parse SSH URL: {ssh_url}")
        
        return match.groups()  # (user, host, port)
    
    def upload_project(self, instance_id: int):
        """Upload project files to an instance via SCP."""
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
        
        # Get SSH connection info
        user, host, port = self._get_ssh_info(instance_id)
        print(f"Uploading project files to instance {instance_id} via SCP ({user}@{host}:{port})...")
        
        # Use SCP to upload files directly (more reliable than vastai copy on Windows)
        for file in files_to_upload:
            src = self.project_dir / file
            if src.exists():
                dst_path = f"/workspace/project/{file}"
                
                # scp -P port -o options local_file user@host:remote_path
                cmd = f'scp -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "{src}" {user}@{host}:{dst_path}'
                print(f"  Uploading {file}...")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                if result.returncode != 0:
                    stderr = result.stderr.strip()
                    print(f"    Error: {stderr[:200]}")
                    raise RuntimeError(f"Failed to upload {file}: {stderr}")
        
        print("Project files uploaded successfully!")
        
        # Verify files exist via SSH
        verify_cmd = f'ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {user}@{host} "ls -la /workspace/project/"'
        result = subprocess.run(verify_cmd, shell=True, capture_output=True, text=True, timeout=60)
        print(f"Files on instance:\n{result.stdout}")
    
    def run_experiment_on_instance(
        self,
        instance_id: int,
        experiment_name: str,
    ) -> subprocess.Popen:
        """Start an experiment on an instance via SSH."""
        user, host, port = self._get_ssh_info(instance_id)
        print(f"SSH URL: ssh://{user}@{host}:{port}")
        
        # Build the command to run on the remote instance
        remote_cmd = f"cd /workspace/project && pip install -q -r requirements-linux.txt && python {experiment_name}.py --dataset-root /workspace/dataset --output-dir /workspace/output"
        
        # Construct full SSH command
        ssh_cmd = f'ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {user}@{host} "{remote_cmd}"'
        
        print(f"Starting experiment {experiment_name} on instance {instance_id}")
        print(f"Running: ssh -p {port} {user}@{host} ...")
        
        # Run in background so we can monitor
        # Use UTF-8 encoding to handle remote output properly (Windows default is cp1252)
        process = subprocess.Popen(
            ssh_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # Replace invalid bytes instead of crashing
        )
        
        return process
    
    def download_results(
        self,
        instance_id: int,
        local_dir: str | Path = "./results",
        experiment_name: Optional[str] = None,
    ):
        """Download experiment results from an instance."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        # Get experiment name for this instance
        if experiment_name is None:
            exp_name = self.instances.get(instance_id, InstanceInfo(0, "unknown", "")).experiment_name
        else:
            exp_name = experiment_name
        
        # Create experiment-specific output directory
        exp_output_dir = local_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading results from instance {instance_id} to {exp_output_dir}")
        
        # Use SCP to download results (more reliable than vastai copy on Windows)
        user, host, port = self._get_ssh_info(instance_id)
        
        # scp -r to copy directory recursively
        # Note: scp uses -P (uppercase) for port, not -p
        cmd = f'scp -r -P {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {user}@{host}:/workspace/output/* "{exp_output_dir}/"'
        print(f"Running: scp -r -P {port} {user}@{host}:/workspace/output/* ...")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Don't fail if no files to download
            if "No such file or directory" not in stderr:
                print(f"Warning: {stderr[:200]}")
                raise RuntimeError(f"Failed to download results: {stderr}")
            else:
                print("No output files to download")
        else:
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
    parser.add_argument("--action", choices=["search", "launch", "status", "destroy", "test-setup", "run-on-existing"], required=True)
    parser.add_argument("--experiment", help="Experiment name for launch/run-on-existing action")
    parser.add_argument("--instance-id", type=int, help="Instance ID for status/destroy/test-setup/run-on-existing actions")
    
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
    
    elif args.action == "test-setup":
        if not args.instance_id:
            print("Error: --instance-id required for test-setup action")
            return
        print(f"Testing setup on instance {args.instance_id}...")
        try:
            orchestrator.wait_for_instance_ready(args.instance_id, timeout=300)
            orchestrator.wait_for_onstart_complete(args.instance_id, timeout=900)
            print("Setup completed successfully!")
        except Exception as e:
            print(f"Setup failed: {e}")
    
    elif args.action == "run-on-existing":
        if not args.instance_id or not args.experiment:
            print("Error: --instance-id and --experiment required for run-on-existing action")
            return
        print(f"Running experiment {args.experiment} on existing instance {args.instance_id}...")
        try:
            # Register instance with orchestrator so download_results can find it
            if args.instance_id not in orchestrator.instances:
                orchestrator.instances[args.instance_id] = InstanceInfo(
                    instance_id=args.instance_id,
                    experiment_name=args.experiment,
                    status="running",
                )
            
            # Ensure instance is ready
            orchestrator.wait_for_instance_ready(args.instance_id, timeout=300)
            
            # Check if dataset is already downloaded (skip if .onstart_done exists)
            user, host, port = orchestrator._get_ssh_info(args.instance_id)
            ssh_base = f'ssh -p {port} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null {user}@{host}'
            check_cmd = f'{ssh_base} "test -f /workspace/.onstart_done && echo EXISTS || echo MISSING"'
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True, timeout=30)
            
            if "EXISTS" not in result.stdout:
                print("Dataset not found, running setup...")
                orchestrator.wait_for_onstart_complete(args.instance_id, timeout=900)
            else:
                print("Dataset already downloaded, skipping setup.")
            
            # Upload project files
            print("Uploading project files...")
            orchestrator.upload_project(args.instance_id)
            
            # Run experiment
            print(f"Starting experiment {args.experiment}...")
            process = orchestrator.run_experiment_on_instance(args.instance_id, args.experiment)
            
            # Stream output
            for line in process.stdout:
                print(line.rstrip())
            
            return_code = process.wait()
            if return_code != 0:
                print(f"Experiment failed with code {return_code}")
                sys.exit(1)
            
            # Download results
            print("Downloading results...")
            orchestrator.download_results(args.instance_id, "./results", experiment_name=args.experiment)
            print("Experiment completed successfully!")
            
        except Exception as e:
            print(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()



#!/bin/bash
set -e

echo "=== VAST.AI EXPERIMENT RUNNER ==="
echo "Experiment: vast_experiment_test"

# Setup workspace
mkdir -p /workspace/dataset /workspace/project /workspace/output

# Download dataset (zip already contains dataset/ folder at root)
cd /workspace
if [ ! -d "dataset/BeamRiderNoFrameskip-v4" ]; then
    echo "Downloading dataset..."
    wget -O dataset.zip "https://storage.googleapis.com/atari_dataset_rehost/dataset.zip"
    unzip -q dataset.zip
    rm dataset.zip
    echo "Dataset downloaded and extracted!"
    ls -la dataset/
fi

# Signal that onstart is done (project files will be uploaded separately)
echo "ONSTART_COMPLETE" > /workspace/.onstart_done
echo "Waiting for project files..."

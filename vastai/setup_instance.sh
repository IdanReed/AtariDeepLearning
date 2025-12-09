#!/bin/bash
# Setup script for vast.ai instances
# This script is executed when an instance starts

set -e  # Exit on error

echo "============================================"
echo "vast.ai Instance Setup Script"
echo "============================================"

# Configuration - these can be overridden by environment variables
GCS_DATASET_URL="${GCS_DATASET_URL:-https://storage.googleapis.com/your-bucket/dataset.zip}"
WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
DATASET_DIR="${DATASET_DIR:-$WORKSPACE_DIR/dataset}"
PROJECT_DIR="${PROJECT_DIR:-$WORKSPACE_DIR/project}"

echo "GCS Dataset URL: $GCS_DATASET_URL"
echo "Workspace: $WORKSPACE_DIR"
echo "Dataset dir: $DATASET_DIR"
echo "Project dir: $PROJECT_DIR"

# Create directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$DATASET_DIR"
mkdir -p "$PROJECT_DIR"

# Step 1: Download and extract dataset from GCS
echo ""
echo "Step 1: Downloading dataset from GCS..."
cd "$WORKSPACE_DIR"

if [ ! -d "$DATASET_DIR/BeamRiderNoFrameskip-v4" ]; then
    echo "Downloading dataset..."
    
    # For public GCS bucket, use wget/curl
    # For private bucket, use gsutil with service account
    if command -v gsutil &> /dev/null; then
        echo "Using gsutil..."
        gsutil cp "$GCS_DATASET_URL" dataset.zip
    else
        echo "Using wget..."
        # Convert gs:// URL to https:// for public buckets
        HTTP_URL=$(echo "$GCS_DATASET_URL" | sed 's|gs://|https://storage.googleapis.com/|')
        wget -O dataset.zip "$HTTP_URL"
    fi
    
    echo "Extracting dataset..."
    unzip -q dataset.zip -d "$DATASET_DIR"
    rm dataset.zip
    echo "Dataset extracted successfully!"
else
    echo "Dataset already exists, skipping download."
fi

# Step 2: Clone or copy project code
echo ""
echo "Step 2: Setting up project code..."
cd "$PROJECT_DIR"

# If project files don't exist, they should be uploaded via scp or git clone
# This is handled by the orchestration script
if [ ! -f "requirements-linux.txt" ]; then
    echo "WARNING: Project files not found. Please upload via scp or git clone."
    echo "Expected files in $PROJECT_DIR"
fi

# Step 3: Install Python dependencies
echo ""
echo "Step 3: Installing Python dependencies..."
if [ -f "requirements-linux.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements-linux.txt
    echo "Dependencies installed successfully!"
else
    echo "requirements-linux.txt not found, skipping dependency installation."
fi

# Step 4: Verify GPU availability
echo ""
echo "Step 4: Verifying GPU..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "To run an experiment:"
echo "  cd $PROJECT_DIR"
echo "  python vast_experiment_freeze_transformer.py --dataset-root $DATASET_DIR"
echo ""


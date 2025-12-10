# vast.ai Experiment Runner

Run MGDT experiments in parallel on vast.ai GPU instances.

## Prerequisites

1. **Create a vast.ai account** at https://vast.ai

2. **Add credits** to your account (GPU instances cost ~$0.50-2.00/hr)

3. **Get your API key** from https://cloud.vast.ai/api/

## Setup

### 1. Install the vast.ai CLI

```bash
pip install vastai
```

### 2. Set your API key

```bash
vastai set api-key YOUR_API_KEY_HERE
```

Verify it works:
```bash
vastai show user
```

### 3. Set up SSH key (required for running experiments)

Generate an SSH key if you don't have one:
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Add your public key to vast.ai:
```bash
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
```

Or on Windows (PowerShell):
```powershell
vastai create ssh-key (Get-Content ~/.ssh/id_ed25519.pub)
```

Verify it was added:
```bash
vastai show ssh-keys
```

### 4. Update the config

Edit `vastai/config.yaml`:

```yaml
# Update with your GCS bucket URL
gcs_dataset_url: "https://storage.googleapis.com/YOUR_BUCKET/dataset.zip"
```

## Running Experiments

### Quick Test (recommended first)

Run a single test experiment to verify everything works:

```bash
python vastai/run_all.py --experiments vast_experiment_test
```

This uses 10% of the dataset and 1 epoch - should complete in ~10-15 minutes.

### Dry Run

See what would happen without spending money:

```bash
python vastai/run_all.py --dry-run
```

### Run All Experiments

Launch all experiments in parallel:

```bash
python vastai/run_all.py
```

### Run Specific Experiments

```bash
python vastai/run_all.py --experiments vast_experiment_freeze_transformer vast_experiment_cnn
```

### Available Experiments

| Experiment | Description | Optuna? |
|------------|-------------|---------|
| `vast_experiment_test` | Quick test (10% data, 1 epoch) | No |
| `vast_experiment_freeze_transformer` | Freeze transformer layers | No |
| `vast_experiment_freeze_obs_encoder` | Freeze observation encoder | No |
| `vast_experiment_cnn` | CNN encoder | Yes |
| `vast_experiment_patch` | Patch encoder | Yes |
| `vast_experiment_window_8` | Window size 8 | Yes |
| `vast_experiment_window_16` | Window size 16 | Yes |
| `vast_experiment_window_32` | Window size 32 | Yes |

## Results

Results are downloaded to `./results/` by default:

```
results/
├── vast_experiment_test/
│   └── test/
│       ├── model_checkpoint.pt
│       └── *.png (plots)
├── vast_experiment_freeze_transformer/
│   └── freeze_transformer/
│       └── ...
└── ...
```

## Configuration Options

Edit `vastai/config.yaml`:

```yaml
# GPU types to search for (in order of preference)
gpu_types:
  - "A100"
  - "RTX_4090"

# Maximum price per hour
max_price_per_hour: 1.50

# Minimum GPU RAM
min_gpu_ram_gb: 16.0

# Disk space (dataset ~10GB + checkpoints)
disk_space_gb: 50
```

## Manual Operations

### Search for available GPUs

```bash
python vastai/orchestrate.py --action search
```

### Check instance status

```bash
python vastai/orchestrate.py --action status --instance-id 12345
```

### Destroy an instance

```bash
python vastai/orchestrate.py --action destroy --instance-id 12345
```

## Troubleshooting

### "No suitable GPU instances available"

- Increase `max_price_per_hour` in config
- Add more GPU types (e.g., RTX_3090, RTX_A5000)
- Try again later (availability fluctuates)

### Instance fails to start

- Check vast.ai dashboard for error messages
- Verify your account has sufficient credits

### SSH connection fails

- Wait a few minutes - instances take time to initialize
- Check if instance is in "running" state on dashboard

### Dataset download fails

- Verify GCS URL is correct and publicly accessible
- Test the URL in a browser: should download the zip file

## Cost Estimates

| Experiment | ~Duration | ~Cost (A100) |
|------------|-----------|--------------|
| test | 10-15 min | $0.30 |
| freeze_* | 1-2 hours | $2-4 |
| cnn/patch (with optuna) | 4-8 hours | $8-16 |
| window_* (with optuna) | 4-8 hours | $8-16 |

**Full suite**: ~$50-100 depending on GPU availability and optuna trials.


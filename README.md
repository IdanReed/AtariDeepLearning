# Venv - Create
uv venv .venv_atari

uv pip install -r requirements.txt

# Venv - Create
source .venv_atari/Scripts/activate

# Venv - Update
uv pip freeze > requirements.txt

uv pip install -r requirements.txt

# Data
Download from here (the UMich download is annoying):
https://storage.googleapis.com/atari_dataset_rehost/dataset.zip

Extract into root level of this repo:
- ./dataset/gamename/gamename/gamename-images-seq
- ./dataset/gamename/gamename/gamename-seq.npz
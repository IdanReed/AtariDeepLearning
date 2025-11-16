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

# Baseline Goal (What MGDT Paper Did)
1 - Load NPZ files, each file is a sequence for a game played by RL agent
    - Atari games are deterministic so the ALE envs make actions "sticky" so there's a % chance that an action will be ignored   
    - (this is so model's can't memorize paths)
2 - Encode images
    - Take image convert to 84x84 grayscale
    - Encode into 36 patch tokens via conv
3 - 
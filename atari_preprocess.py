import os
import cv2
import numpy as np
import torch

IMG_SIZE = 84
PATCH_SIZE = 14
PATCHES_PER_FRAME = (IMG_SIZE // PATCH_SIZE) ** 2    # 36 patches
RETURN_MIN = -20
RETURN_MAX = 100


# ----------------------------------------------
# 1. Load an image from the dataset root
# ----------------------------------------------

from pathlib import Path
import cv2
import numpy as np
def find_image_folders(game_root: Path):
    folders = [
        p for p in game_root.iterdir()
        if p.is_dir() and ("image" in p.name.lower())
    ]

    def folder_index(name):
        # extract the last number: e.g. "-12" → 12
        return int(name.split("-")[-1])

    folders.sort(key=lambda p: folder_index(p.name))
    return folders



def find_npz_files(game_root: Path):
    files = sorted(list(game_root.glob("*.npz")))
    return files


def load_frame(dataset_root: Path, rel_path: str):
    """
    dataset_root: Path object pointing to the folder that contains the image folders.
    rel_path: relative path from the NPZ file (string)
    """
    full_path = dataset_root / rel_path  # ← OS-independent joining

    full_path = full_path.resolve()      # normalize path
    if not full_path.exists():
        raise FileNotFoundError(f"Image not found: {full_path}")

    img = cv2.imread(str(full_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (84, 84))
    return img.astype(np.float32) / 255.0



# ----------------------------------------------
# 2. Convert frame to 36 patches (14×14)
# ----------------------------------------------

def frame_to_patches(frame):
    patches = []
    for i in range(0, IMG_SIZE, PATCH_SIZE):
        for j in range(0, IMG_SIZE, PATCH_SIZE):
            patch = frame[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
            patches.append(patch)
    return np.stack(patches)  # shape (36,14,14)


# ----------------------------------------------
# 3. Discretize reward into {-1,0,+1}
# ----------------------------------------------

def discretize_reward(r):
    if r > 0:
        return 1
    elif r < 0:
        return -1
    return 0


# ----------------------------------------------
# 4. Compute RTG and quantize
# ----------------------------------------------

def compute_rtg(rewards):
    rtg = np.zeros_like(rewards)
    running = 0
    for t in reversed(range(len(rewards))):
        running += rewards[t]
        rtg[t] = running

    rtg = np.clip(rtg, RETURN_MIN, RETURN_MAX)
    rtg = rtg - RETURN_MIN  # shift to 0..120
    return rtg.astype(int)


# ----------------------------------------------
# 5. Build sequence of (patch, rtg, action, reward) tokens
# ----------------------------------------------

def build_token_sequence(patch_seq, rtg_seq, actions, reward_seq):
    """
    Returns a list of tokens, each token is:
    {
        "type": "patch"/"rtg"/"action"/"reward",
        "value": np.array(14x14) OR integer
    }
    """
    T = len(actions)
    seq = []

    for t in range(T):
        # 36 patch tokens
        for m in range(PATCHES_PER_FRAME):
            seq.append({
                "type": "patch",
                "value": patch_seq[t, m]
            })

        # RTG token
        seq.append({
            "type": "rtg",
            "value": int(rtg_seq[t])
        })

        # Action token
        seq.append({
            "type": "action",
            "value": int(actions[t])
        })

        # Reward token
        seq.append({
            "type": "reward",
            "value": int(reward_seq[t])
        })

    return seq


# ----------------------------------------------
# 6. Full preprocessing for ONE NPZ sequence
# ----------------------------------------------
def preprocess_one_sequence(npz_seq, game_root: Path, seq_index: int):
    """
    Loads NPZ file #seq_index and aligns it with image folder #seq_index.
    Trims the episode using T = min(#frames, #actions).
    """
    game_root = Path(game_root).resolve()

    # ---- Find matching image folder ----
    image_folders = find_image_folders(game_root)
    image_folder = image_folders[seq_index]

    # ---- Load actions ----
    if "taken actions" in npz_seq:
        actions = npz_seq["taken actions"].reshape(-1)
    else:
        actions = npz_seq["model selected actions"].reshape(-1)

    rewards = npz_seq["rewards"].astype(float)

    # ---- Load frames (sorted by name) ----
    image_paths = sorted(image_folder.glob("*"))
    frames = np.stack([
        load_frame(image_folder, p.name) for p in image_paths
    ])

    num_frames = frames.shape[0]
    num_steps = len(actions)

    # ---- ALIGN LENGTH ----
    T = min(num_frames, num_steps)

    # trim all sequences
    frames = frames[:T]
    actions = actions[:T]
    rewards = rewards[:T]

    # ---- Build patches ----
    patch_seq = np.stack([frame_to_patches(f) for f in frames])

    # ---- Reward & RTG ----
    reward_seq = np.array([discretize_reward(r) for r in rewards])
    rtg_seq = compute_rtg(rewards)[:T]

    # ---- Build final tokens ----
    tokens = build_token_sequence(
        patch_seq=patch_seq,
        rtg_seq=rtg_seq,
        actions=actions,
        reward_seq=reward_seq
    )

    return tokens
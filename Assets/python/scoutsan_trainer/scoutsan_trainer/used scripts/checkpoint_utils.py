

import os
import re
import glob

_PY = "/home/nyctophilia/UnityProjects/RL_Scoutsan/python"

ARM_MODEL_DIR = os.path.join(_PY, "model_16arm_ratepenalty")
DEFAULT_OVERRIDE = os.path.join(ARM_MODEL_DIR, "ppo_16arm_ratepenalty_interrupted.zip")
DEFAULT_CHECKPOINT_SOURCES = [
    (os.path.join(ARM_MODEL_DIR, "checkpoints"), "ppo_16arm_ratepenalty"),
]

DEFAULT_FALLBACK_SAVES = [
    os.path.join(ARM_MODEL_DIR, "ppo_16arm_ratepenalty_interrupted.zip"),
    os.path.join(ARM_MODEL_DIR, "ppo_16arm_ratepenalty.zip"),
]

def find_latest_checkpoint(ckpt_dir, prefix):
    pattern = os.path.join(ckpt_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None

    def step_of(path):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(files, key=step_of)

def resolve_pretrained_path(override=DEFAULT_OVERRIDE,
                            checkpoint_sources=DEFAULT_CHECKPOINT_SOURCES,
                            fallback_saves=DEFAULT_FALLBACK_SAVES):
    if override and os.path.exists(override):
        return override

    for ckpt_dir, prefix in checkpoint_sources:
        latest = find_latest_checkpoint(ckpt_dir, prefix)
        if latest is not None:
            return latest

    for save in fallback_saves:
        if os.path.exists(save):
            return save

    return None

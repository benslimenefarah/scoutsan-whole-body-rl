#!/usr/bin/env python

import os
import sys
import glob
import runpy

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
HERE = os.path.dirname(os.path.abspath(__file__))

BUILD_DIR = os.path.join(REPO, "s9_np")
MODEL_DIR = os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9_np")
TB_DIR    = os.path.join(REPO, "scoutsan_Baseline", "tb", "tb_scoutsan_s9_np")
PREFIX    = "ppo_scoutsan_s9_np"

final_zip = os.path.join(MODEL_DIR, PREFIX + ".zip")
if os.path.exists(final_zip) and os.environ.get("ARM_FORCE_RERUN", "").strip() != "1":
    raise SystemExit(
        f"[s9_np] a final model already exists: {final_zip}\n"
        f"  reset_num_timesteps=True would restart checkpoint numbering and clobber it.\n"
        f"  Archive/rename {MODEL_DIR} first, or set ARM_FORCE_RERUN=1.")

hits = sorted(glob.glob(os.path.join(BUILD_DIR, "*.x86_64")))
if not hits:
    raise SystemExit(
        f"[s9_np] no .x86_64 in {BUILD_DIR}\n"
        f"  Build the 8-robot s9.unity scene (fixed target targetSpawnOffset (0,0.25,2), useEvalSamples 0,\n"
        f"  randomize pos/orient OFF, supportBox None, cube deleted) headless into s9_np/ first.\n"
        f"  Verify 8 robots:  grep -c 'm_BehaviorName:' Assets/scoutsan_Baseline/scenes/s9.unity  # == 8")

os.environ["ARM_BUILD"]        = hits[0]
os.environ["ARM_MODEL_DIR"]    = MODEL_DIR
os.environ["ARM_TB_DIR"]       = TB_DIR
os.environ["ARM_MODEL_PREFIX"] = PREFIX
os.environ["ARM_WORKER_ID"]    = "14"
os.environ.setdefault("ARM_TRANSPLANT", "1")
os.environ.setdefault("ARM_TIME_SCALE", "10")
os.environ.setdefault("ARM_TIMESTEPS", "10000000")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print(f"[s9_np] build: {hits[0]}")
print(f"[s9_np] model: {MODEL_DIR}   port 5019   GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[s9_np] warm start: {os.environ.get('ARM_WARMSTART') or 'padded ARM transplant (model_16arm_ratepenalty)'}")
print(f"[s9_np] !! VERIFY baked: 8 robots, obs14/act8, fixed target (0,0.25,2), useEvalSamples 0, no cube.")

runpy.run_path(os.path.join(HERE, "train_scoutsan_s9.py"), run_name="__main__")

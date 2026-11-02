#!/usr/bin/env python

import os
import sys
import glob
import re
import runpy

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
HERE = os.path.dirname(os.path.abspath(__file__))

BUILD_DIR   = os.path.join(REPO, "s9_tarc")
MODEL_DIR   = os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9_tarc")
TB_DIR      = os.path.join(REPO, "scoutsan_Baseline", "tb", "tb_scoutsan_s9_tarc")
PREFIX      = "ppo_scoutsan_s9_tarc"
S9NP_DIR    = os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9_np")
S9NP_PREFIX = "ppo_scoutsan_s9_np"

def latest_ckpt(model_dir, prefix):
    cands = [os.path.join(model_dir, f"{prefix}_interrupted.zip")]
    cands += glob.glob(os.path.join(model_dir, "checkpoints", f"{prefix}_*_steps.zip"))
    cands = [p for p in cands if os.path.exists(p)]
    if not cands:
        return None
    def steps_of(p):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(p)); return int(m.group(1)) if m else -1
    ck = [p for p in cands if p.endswith("_steps.zip")]
    return max(ck, key=steps_of) if ck else max(cands, key=os.path.getmtime)

final_zip = os.path.join(MODEL_DIR, PREFIX + ".zip")
if os.path.exists(final_zip) and os.environ.get("ARM_FORCE_RERUN", "").strip() != "1":
    raise SystemExit(
        f"[s9_tarc] a final model already exists: {final_zip}\n"
        f"  reset_num_timesteps=True would restart checkpoint numbering and clobber it.\n"
        f"  Archive/rename {MODEL_DIR} first, or set ARM_FORCE_RERUN=1.")

hits = sorted(glob.glob(os.path.join(BUILD_DIR, "*.x86_64")))
if not hits:
    raise SystemExit(
        f"[s9_tarc] no .x86_64 in {BUILD_DIR}\n"
        f"  Build the 8-robot s9 scene with useTArcCurriculum=1 (tArc: distanceFullSteps 500000, arcFullSteps 1000000,\n"
        f"  minRadius 0.75, lineRadius 2, arcTotalDeg 40, height 0.25) headless into s9_tarc/.\n"
        f"  Verify 8 robots:  grep -c 'm_BehaviorName:' Assets/scoutsan_Baseline/scenes/s9.unity  # == 8")

warm = os.environ.get("ARM_WARMSTART", "").strip()
if warm:
    note = f"WARM (env override): {warm}"
else:
    own = latest_ckpt(MODEL_DIR, PREFIX)
    if own:
        warm, note = own, f"RESUME own state: {os.path.basename(own)} (re-launch)"
    else:
        warm = latest_ckpt(S9NP_DIR, S9NP_PREFIX)
        if not warm:
            raise SystemExit(f"[s9_tarc] no s9_np checkpoint to warm-start from in {S9NP_DIR} "
                             f"(set ARM_WARMSTART=<14/8 .zip>).")
        note = f"WARM-START from s9_np: {os.path.basename(warm)}"

os.environ["ARM_BUILD"]        = hits[0]
os.environ["ARM_MODEL_DIR"]    = MODEL_DIR
os.environ["ARM_TB_DIR"]       = TB_DIR
os.environ["ARM_MODEL_PREFIX"] = PREFIX
os.environ["ARM_WARMSTART"]    = warm
os.environ["ARM_TRANSPLANT"]   = "0"
os.environ["ARM_WORKER_ID"]    = "20"
os.environ.setdefault("ARM_TIME_SCALE", "10")
os.environ.setdefault("ARM_TIMESTEPS", "4000000")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

print(f"[s9_tarc] build: {hits[0]}")
print(f"[s9_tarc] model: {MODEL_DIR}   port 5025   GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
print(f"[s9_tarc] {note}")
print(f"[s9_tarc] !! VERIFY baked: 8 robots, obs14/act8, useTArcCurriculum=1, tArc distance500k/arc1M, DLL newer than .cs.")

runpy.run_path(os.path.join(HERE, "train_scoutsan_s9.py"), run_name="__main__")

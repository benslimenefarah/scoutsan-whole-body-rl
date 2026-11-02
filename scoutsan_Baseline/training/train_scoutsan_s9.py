#!/usr/bin/env python

import os
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/nyctophilia/UnityProjects/RL_Scoutsan/Assets/python/scoutsan_trainer/scoutsan_trainer/used scripts")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor

from unity_multiagent_env import UnityMultiAgentVecEnv
from graph_callback import GraphSnapshotCallback

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
BASE = os.path.join(REPO, "python")

OBS_EXPECTED, ACT_EXPECTED, N_EXPECTED = 14, 8, 8

BUILD_DIR = os.path.join(REPO, "s9")
WORKER_ID = int(os.environ.get("ARM_WORKER_ID", "11"))
BASE_PORT = 5005

OUT_DIR   = os.environ.get("ARM_MODEL_DIR", "").strip()    or os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9")
TB_DIR    = os.environ.get("ARM_TB_DIR", "").strip()       or os.path.join(REPO, "scoutsan_Baseline", "tb", "tb_scoutsan_s9")
PREFIX    = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_scoutsan_s9"
CKPT_DIR  = os.path.join(OUT_DIR, "checkpoints")
GRAPH_DIR = os.path.join(OUT_DIR, "graphs")
UNITY_LOG = os.path.join(OUT_DIR, "unity_logs")

TOTAL_TIMESTEPS = int(os.environ.get("ARM_TIMESTEPS", "20000000"))
TIME_SCALE      = float(os.environ.get("ARM_TIME_SCALE", "10"))
DEVICE          = os.environ.get("ARM_DEVICE", "cuda")

LR         = float(os.environ.get("ARM_LR", "3e-4"))
N_STEPS    = int(os.environ.get("ARM_N_STEPS", "2048"))
BATCH_SIZE = int(os.environ.get("ARM_BATCH", "64"))
ENT_COEF   = float(os.environ.get("ARM_ENT_COEF", "0.01"))
CLIP_RANGE = float(os.environ.get("ARM_CLIP", "0.2"))
GAMMA      = float(os.environ.get("ARM_GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("ARM_GAE", "0.95"))

CS_SRC = os.path.join(REPO, "Assets", "scoutsan_Baseline", "scripts", "ScoutsanV1_1b.cs")

def newest_mtime_in(build):
    d = os.path.dirname(build)
    m = os.path.getmtime(build)
    for root, _, files in os.walk(d):
        for f in files:
            try:
                m = max(m, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return m

def assert_build_fresh(build):
    if os.environ.get("ARM_SKIP_BUILD_CHECK", "").strip() == "1":
        print("[s9] WARNING: build-freshness check SKIPPED (ARM_SKIP_BUILD_CHECK=1).")
        return
    if os.path.exists(CS_SRC) and os.path.getmtime(CS_SRC) > newest_mtime_in(build):
        raise SystemExit(
            f"[s9] STALE BUILD -- ScoutsanV1_1b.cs is NEWER than the build:\n"
            f"    src   : {CS_SRC}\n    build : {build}\n"
            f"  Rebuild s9/ (ScoutsanV1_1b: fixed target, KEEP cube, useBaseSlew=true), then relaunch "
            f"(override: ARM_SKIP_BUILD_CHECK=1).")
    print("[s9] build freshness OK.")

def find_build():
    override = os.environ.get("ARM_BUILD", "").strip()
    if override:
        return override
    hits = sorted(glob.glob(os.path.join(BUILD_DIR, "*.x86_64")))
    if not hits:
        raise SystemExit(f"[s9] no .x86_64 build in {BUILD_DIR} (build the ScoutsanV1_1b fixed-target WITH-CUBE scene there).")
    return hits[0]

def main():
    for d in (CKPT_DIR, GRAPH_DIR, UNITY_LOG):
        os.makedirs(d, exist_ok=True)

    build = find_build()
    assert_build_fresh(build)
    print(f"[s9] HEADLESS build: {build}  worker_id={WORKER_ID} -> port {BASE_PORT + WORKER_ID}  time_scale={TIME_SCALE}")
    os.environ["SCOUTSAN_REWARD_CSV"] = os.path.join(OUT_DIR, "rewards.csv")
    print(f"[s9] reward-term CSV -> {os.environ['SCOUTSAN_REWARD_CSV']}")

    env = UnityMultiAgentVecEnv(file_name=build, no_graphics=True, time_scale=TIME_SCALE,
                                worker_id=WORKER_ID, log_folder=UNITY_LOG)
    n = env.num_envs
    print(f"[s9] detected {n} agent(s)  obs={env.observation_space.shape} act={env.action_space.shape}")
    if env.observation_space.shape[0] != OBS_EXPECTED or env.action_space.shape[0] != ACT_EXPECTED:
        print(f"[WARNING] expected {OBS_EXPECTED}/{ACT_EXPECTED}, got "
              f"{env.observation_space.shape[0]}/{env.action_space.shape[0]}.")
    if n != N_EXPECTED:
        print(f"[WARNING] expected {N_EXPECTED} agents; got {n}. Config still runs (buffer = n_steps*n).")
    env = VecMonitor(env)

    if (N_STEPS * n) % BATCH_SIZE != 0:
        print(f"[WARNING] n_steps*n ({N_STEPS*n}) not divisible by batch_size ({BATCH_SIZE}).")

    warm       = os.environ.get("ARM_WARMSTART", "").strip()
    armsrc     = os.environ.get("ARM_ARMSRC", "").strip()
    transplant = os.environ.get("ARM_TRANSPLANT", "1").strip() != "0"
    print(f"[s9] PPO (s8 working config): lr={LR} n_steps={N_STEPS} (buffer={N_STEPS*n}) batch={BATCH_SIZE} "
          f"ent={ENT_COEF} clip={CLIP_RANGE}")

    def _fresh_ppo():
        return PPO("MlpPolicy", env, verbose=1, tensorboard_log=TB_DIR,
                   learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                   gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE, ent_coef=ENT_COEF, device=DEVICE)

    if armsrc:
        from weight_transfer import transfer_arm_keep_zero_base
        if not os.path.exists(armsrc):
            raise SystemExit(f"[s9] ARM_ARMSRC set but not found: {armsrc}")
        print(f"[s9] ARM-KEEP transplant from whole-body {armsrc} (keep arm weights, ZERO base -> learn grippy drive fresh)")
        model = _fresh_ppo()
        transfer_arm_keep_zero_base(armsrc, model)
    elif warm:
        if not os.path.exists(warm):
            raise SystemExit(f"[s9] ARM_WARMSTART set but not found: {warm}")
        print(f"[s9] RESUME (PPO.load): {warm}")
        model = PPO.load(warm, env=env, device=DEVICE, verbose=1, tensorboard_log=TB_DIR,
                         learning_rate=LR, n_steps=N_STEPS, batch_size=BATCH_SIZE,
                         gamma=GAMMA, gae_lambda=GAE_LAMBDA, clip_range=CLIP_RANGE, ent_coef=ENT_COEF)
    elif transplant:
        from weight_transfer import transfer_weights_padded
        from checkpoint_utils import resolve_pretrained_path
        arm_model = resolve_pretrained_path()
        if not arm_model or not os.path.exists(arm_model):
            raise SystemExit("[s9] ARM transplant: model_16arm_ratepenalty not found; "
                             "set ARM_WARMSTART=<14/8 .zip> or ARM_TRANSPLANT=0.")
        print(f"[s9] FRESH ARM TRANSPLANT (14/6 -> 14/8 padded) donor: {arm_model}  (arm skill, base learns grippy drive fresh)")
        model = _fresh_ppo()
        transfer_weights_padded(arm_model, model)
    else:
        print("[s9] PURE FROM SCRATCH (ARM_TRANSPLANT=0) -- arm confondu from random init is HARD; watch for life by ~2-3M.")
        model = _fresh_ppo()

    checkpoint_cb = CheckpointCallback(save_freq=max(50_000 // n, 1), save_path=CKPT_DIR,
                                       name_prefix=PREFIX, save_replay_buffer=False, save_vecnormalize=False)
    graph_cb = GraphSnapshotCallback(save_dir=GRAPH_DIR, snapshot_freq=50_000, success_radius=0.03)
    callbacks = CallbackList([checkpoint_cb, graph_cb])

    print(f"[s9] output dir: {OUT_DIR}   TB: {TB_DIR}   timesteps: {TOTAL_TIMESTEPS:,}")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks,
                    reset_num_timesteps=True, tb_log_name=PREFIX)
        model.save(os.path.join(OUT_DIR, PREFIX))
        graph_cb.render_final()
        print("[s9] complete.")
    except KeyboardInterrupt:
        print("\n[s9] interrupted -- saving partial model + graphs...")
        model.save(os.path.join(OUT_DIR, f"{PREFIX}_interrupted"))
        graph_cb.render_final()
    finally:
        env.close()

if __name__ == "__main__":
    main()

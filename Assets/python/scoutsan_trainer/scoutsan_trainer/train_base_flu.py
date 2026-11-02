#!/usr/bin/env python

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "used scripts"))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor

from unity_multiagent_env import UnityMultiAgentVecEnv

REPO      = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
MODEL_DIR = os.environ.get("ARM_MODEL_DIR", "").strip()    or os.path.join(REPO, "python", "model_bearing_angle_FLU_new")
TB_DIR    = os.environ.get("ARM_TB_DIR", "").strip()       or os.path.join(MODEL_DIR, "tb")
PREFIX    = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_bearing_angle_FLU_new"
CKPT_DIR  = os.path.join(MODEL_DIR, "checkpoints")
UNITY_LOG = os.path.join(MODEL_DIR, "unity_logs")

OBS_EXPECTED, ACT_EXPECTED = 4, 2
EDITOR     = os.environ.get("ARM_EDITOR", "1").strip() != "0"
BUILD      = os.environ.get("ARM_BUILD", "").strip()
WORKER_ID  = int(os.environ.get("ARM_WORKER_ID", "15"))
TIME_SCALE = float(os.environ.get("ARM_TIME_SCALE", "20"))
DEVICE     = os.environ.get("ARM_DEVICE", "cpu")
TOTAL      = int(os.environ.get("ARM_TIMESTEPS", "1500000"))

LR    = float(os.environ.get("ARM_LR", "3e-4"))
NSTEP = int(os.environ.get("ARM_N_STEPS", "2048"))
BATCH = int(os.environ.get("ARM_BATCH", "64"))
ENT   = float(os.environ.get("ARM_ENT_COEF", "0.005"))
GAMMA = float(os.environ.get("ARM_GAMMA", "0.99"))
GAE   = float(os.environ.get("ARM_GAE", "0.95"))
CLIP  = float(os.environ.get("ARM_CLIP", "0.2"))

def main():
    for d in (CKPT_DIR, TB_DIR, UNITY_LOG):
        os.makedirs(d, exist_ok=True)

    final_zip = os.path.join(MODEL_DIR, PREFIX + ".zip")
    if os.path.exists(final_zip) and os.environ.get("ARM_FORCE_RERUN", "").strip() != "1":
        raise SystemExit(
            f"[base] a final model already exists: {final_zip}\n"
            f"  reset_num_timesteps=True would restart checkpoint numbering and clobber it.\n"
            f"  Archive/rename {MODEL_DIR} first, or set ARM_FORCE_RERUN=1.")

    if EDITOR:
        print("[base] EDITOR mode -> open Assets/Scenes/base.unity, Behavior Type=Default (Space Size 4 / Cont. Actions 2), "
              f"then press PLAY (port 5004). time_scale={TIME_SCALE}, device={DEVICE}.")
        env = UnityMultiAgentVecEnv(file_name=None, no_graphics=False, time_scale=TIME_SCALE,
                                    worker_id=0, log_folder=UNITY_LOG)
    else:
        if not BUILD or not os.path.exists(BUILD):
            raise SystemExit(f"[base] ARM_EDITOR=0 but ARM_BUILD not found: {BUILD!r}")
        print(f"[base] BUILD mode: {BUILD}  worker={WORKER_ID} -> port {5005 + WORKER_ID}  time_scale={TIME_SCALE}  device={DEVICE}")
        env = UnityMultiAgentVecEnv(file_name=BUILD, no_graphics=True, time_scale=TIME_SCALE,
                                    worker_id=WORKER_ID, log_folder=UNITY_LOG)

    n = env.num_envs
    print(f"[base] detected {n} agent(s)  obs={env.observation_space.shape}  act={env.action_space.shape}")
    if env.observation_space.shape[0] != OBS_EXPECTED or env.action_space.shape[0] != ACT_EXPECTED:
        print(f"[WARNING] expected {OBS_EXPECTED}/{ACT_EXPECTED}. base.unity Behavior Params must be Space Size=4, "
              f"Continuous Actions=2 (got {env.observation_space.shape[0]}/{env.action_space.shape[0]}).")
    env = VecMonitor(env)

    if (NSTEP * n) % BATCH != 0:
        print(f"[WARNING] n_steps*n ({NSTEP*n}) not divisible by batch_size ({BATCH}); PPO will still run.")

    print(f"[base] PPO: lr={LR} n_steps={NSTEP} (buffer={NSTEP*n}) batch={BATCH} ent={ENT} gamma={GAMMA} device={DEVICE}")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=TB_DIR,
                learning_rate=LR, n_steps=NSTEP, batch_size=BATCH,
                gamma=GAMMA, gae_lambda=GAE, clip_range=CLIP, ent_coef=ENT, device=DEVICE)

    ckpt = CheckpointCallback(save_freq=max(25_000 // n, 1), save_path=CKPT_DIR, name_prefix=PREFIX,
                              save_replay_buffer=False, save_vecnormalize=False)

    print(f"[base] out: {MODEL_DIR}   TB: {TB_DIR}   timesteps: {TOTAL:,}   (Ctrl-C to stop -> {PREFIX}_interrupted.zip)")
    try:
        model.learn(total_timesteps=TOTAL, callback=ckpt, tb_log_name=PREFIX, reset_num_timesteps=True)
        model.save(final_zip[:-4])
        print("[base] complete.")
    except KeyboardInterrupt:
        print("\n[base] interrupted -- saving partial model...")
        model.save(os.path.join(MODEL_DIR, PREFIX + "_interrupted"))
    finally:
        env.close()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import os, sys, glob, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/nyctophilia/UnityProjects/RL_Scoutsan/Assets/python/scoutsan_trainer/scoutsan_trainer/used scripts")
from stable_baselines3 import PPO
from unity_multiagent_env import UnityMultiAgentVecEnv

MODEL_DIR   = "/home/nyctophilia/UnityProjects/RL_Scoutsan/scoutsan_Baseline/models/model_scoutsan_s9r"
PREFIX      = "ppo_scoutsan_s9r"
CKPT_DIR    = os.path.join(MODEL_DIR, "checkpoints")
WORKER_ID   = int(os.environ.get("ARM_WORKER_ID", "0"))
TIME_SCALE  = float(os.environ.get("ARM_TIME_SCALE", "1.0"))
SUCCESS_R   = float(os.environ.get("ARM_SUCCESS_RADIUS", "0.03"))
ORIENT_TOL  = float(os.environ.get("ARM_ORIENT_TOL_RAD", "0.05"))
PRINT_EVERY = int(os.environ.get("ARM_PRINT_EVERY", "1"))

def resolve_checkpoint():
    override = os.environ.get("ARM_EVAL_MODEL", "").strip()
    if override:
        if not os.path.exists(override):
            raise SystemExit(f"[watch-s9r] ARM_EVAL_MODEL not found: {override}")
        return override
    cands = glob.glob(os.path.join(CKPT_DIR, "*_steps.zip"))
    def steps_of(p):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(p)); return int(m.group(1)) if m else -1
    best = max(cands, key=steps_of) if cands else None
    for extra in (f"{PREFIX}_interrupted.zip", f"{PREFIX}.zip"):
        c = os.path.join(MODEL_DIR, extra)
        if os.path.exists(c) and (best is None or os.path.getmtime(c) > os.path.getmtime(best)):
            best = c
    if best is None:
        raise SystemExit(f"[watch-s9r] no checkpoints in {CKPT_DIR} (set ARM_EVAL_MODEL).")
    return best

def pose_errors(o):
    gp, gq = o[0:3], o[3:7]; tp, tq = o[7:10], o[10:14]
    d = float(np.linalg.norm(gp - tp))
    phi = 2.0 * np.arccos(min(1.0, abs(float(np.dot(gq, tq)))))
    return d, phi

def classify(r):
    if r >= 15:  return "SUCCESS"
    if r <= -5:  return "FAIL (box/collision/fall)"
    return "timeout/other"

def main():
    ck = resolve_checkpoint()
    print(f"[watch-s9r] checkpoint: {os.path.basename(ck)}")
    print(f"[watch-s9r] gate: d<{SUCCESS_R} m, phi<{ORIENT_TOL} rad ({np.degrees(ORIENT_TOL):.1f} deg)")
    print(f"[watch-s9r] MODEL model_scoutsan_s9r  <->  SCENE s9.unity (ScoutsanV1_1b: box + FIXED target)")
    print(f"[watch-s9r] >>> OPEN 's9', Behavior Type = Default, PRESS PLAY (port {5004+WORKER_ID}) <<<\n")

    env = UnityMultiAgentVecEnv(file_name=None, no_graphics=False, time_scale=TIME_SCALE, worker_id=WORKER_ID)
    n = env.num_envs
    print(f"[watch-s9r] connected: {n} agent(s), obs={env.observation_space.shape}, act={env.action_space.shape}")
    model = PPO.load(ck, env=env)

    obs = env.reset()
    ep_rew = np.zeros(n); ep_len = np.zeros(n, dtype=int)
    min_d = np.full(n, 1e9); min_phi = np.full(n, 1e9)
    ep = 0; succ = 0; pos_ok_orient_miss = 0
    print("[watch-s9r] deterministic. Per-step = AGENT 0; per-episode = all agents (minD/minPhi = closest approach).")
    print("-" * 92)
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ep_rew += rewards; ep_len += 1
            for i in range(n):
                d_i, phi_i = pose_errors(np.asarray(obs[i]))
                min_d[i] = min(min_d[i], d_i); min_phi[i] = min(min_phi[i], phi_i)

            for i in range(n):
                if dones[i]:
                    ep += 1; outcome = classify(ep_rew[i])
                    if outcome == "SUCCESS": succ += 1

                    posok = min_d[i] < 0.06; orientmiss = min_phi[i] > ORIENT_TOL
                    flag = ""
                    if outcome != "SUCCESS" and posok and orientmiss:
                        pos_ok_orient_miss += 1
                        flag = "  <== pos-OK / orient-MISS (reached sphere, wrist not aligned)"
                    print(f"  >> EP {ep:03d} [a{i}] {outcome:24s} steps={int(ep_len[i]):3d} "
                          f"return={ep_rew[i]:+7.1f}  minD={min_d[i]:.3f}m minPhi={np.degrees(min_phi[i]):4.1f}deg{flag}")
                    ep_rew[i] = 0.0; ep_len[i] = 0; min_d[i] = 1e9; min_phi[i] = 1e9
    except KeyboardInterrupt:
        pass
    finally:
        if ep:
            print("-" * 92)
            print(f"[watch-s9r] {ep} episodes | success {succ}/{ep} = {100*succ/ep:.0f}% | "
                  f"pos-OK/orient-MISS {pos_ok_orient_miss}/{ep} = {100*pos_ok_orient_miss/ep:.0f}% "
                  f"(reached the sphere, wrist not aligned)")
        env.close(); print("[watch-s9r] closed.")

if __name__ == "__main__":
    main()

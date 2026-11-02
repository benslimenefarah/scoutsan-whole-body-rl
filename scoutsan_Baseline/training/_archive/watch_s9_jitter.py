#!/usr/bin/env python3

import os, sys, glob, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/nyctophilia/UnityProjects/RL_Scoutsan/Assets/python/scoutsan_trainer/scoutsan_trainer/used scripts")
from stable_baselines3 import PPO
from unity_multiagent_env import UnityMultiAgentVecEnv

MODEL_DIR = os.environ.get("ARM_MODEL_DIR", "").strip() or \
    "/home/nyctophilia/UnityProjects/RL_Scoutsan/scoutsan_Baseline/models/model_scoutsan_s9"
PREFIX    = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_scoutsan_s9"
CKPT_DIR  = os.path.join(MODEL_DIR, "checkpoints")
WORKER_ID = int(os.environ.get("ARM_WORKER_ID", "0"))
TIME_SCALE = float(os.environ.get("ARM_TIME_SCALE", "1.0"))
PRINT_EVERY = int(os.environ.get("ARM_PRINT_EVERY", "20"))
N_EPISODES  = int(os.environ.get("ARM_EPISODES", "10"))

W_R  = float(os.environ.get("ARM_ACTION_RATE", "0.05"))
NORM = 32.0
DEAD = float(os.environ.get("ARM_REV_DEADBAND", "0.05"))
TARGET_FRAC = float(os.environ.get("ARM_TARGET_FRAC", "0.15"))

def resolve_checkpoint():
    override = os.environ.get("ARM_EVAL_MODEL", "").strip()
    if override:
        if not os.path.exists(override):
            raise SystemExit(f"[jitter] ARM_EVAL_MODEL not found: {override}")
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
        raise SystemExit(f"[jitter] no checkpoints in {CKPT_DIR} (set ARM_EVAL_MODEL).")
    return best

def main():
    ck = resolve_checkpoint()
    print(f"[jitter] checkpoint: {os.path.basename(ck)}   (measuring on the deterministic policy)")
    print(f"[jitter] r8 model: r8 = -{W_R} * sum(da^2)/{int(NORM)} over 8 actions ; reversal deadband={DEAD}")
    print(f"[jitter] >>> OPEN the matching scene, Behavior Type = Default, PRESS PLAY (port {5004+WORKER_ID}) <<<\n")

    env = UnityMultiAgentVecEnv(file_name=None, no_graphics=False, time_scale=TIME_SCALE, worker_id=WORKER_ID)
    n = env.num_envs
    print(f"[jitter] connected: {n} agent(s), obs={env.observation_space.shape}, act={env.action_space.shape}")
    model = PPO.load(ck, env=env)

    obs = env.reset()
    prev = np.zeros((n, 8)); have_prev = np.zeros(n, dtype=bool)

    g_steps = 0; g_arm_jerk = 0.0; g_tot_jerk = 0.0; g_rev = 0; g_rev_steps = 0; g_pos_rew = 0.0

    e_steps = np.zeros(n, dtype=int); e_arm_jerk = np.zeros(n); e_rev = np.zeros(n, dtype=int)
    ep = 0
    print("[jitter] per-step (agent 0): armJerk, armReversals(/6), base v/w. Per-episode: mean jerk + reversal rate.")
    print("-" * 92)
    try:
        while ep < N_EPISODES:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            a = np.asarray(action)
            disp_jerk = 0.0; disp_rev = 0
            for i in range(n):
                if have_prev[i]:
                    da = a[i] - prev[i]
                    arm_jerk = float(np.sum(da[0:6] ** 2))
                    tot_jerk = float(np.sum(da ** 2))

                    rev = int(np.sum([(a[i, k] * prev[i, k] < 0) and (abs(a[i, k]) > DEAD) and (abs(prev[i, k]) > DEAD)
                                      for k in range(6)]))
                    g_steps += 1; g_arm_jerk += arm_jerk; g_tot_jerk += tot_jerk
                    g_rev += rev; g_rev_steps += (1 if rev > 0 else 0)
                    g_pos_rew += max(float(rewards[i]), 0.0)
                    e_steps[i] += 1; e_arm_jerk[i] += arm_jerk; e_rev[i] += rev
                    if i == 0:
                        disp_jerk = arm_jerk; disp_rev = rev
                prev[i] = a[i]; have_prev[i] = True
            if e_steps[0] > 0 and e_steps[0] % PRINT_EVERY == 0:
                print(f"a0 t={e_steps[0]:3d}  armJerk={disp_jerk:.4f}  armReversals={disp_rev}/6"
                      f"  base v={a[0,6]:+.2f} w={a[0,7]:+.2f}  arm|max|={float(np.max(np.abs(a[0,0:6]))):.2f}")
            for i in range(n):
                if dones[i]:
                    ep += 1
                    st = max(int(e_steps[i]), 1)
                    print(f"  >> EP {ep:03d} [a{i}] steps={int(e_steps[i]):3d}  "
                          f"mean armJerk={e_arm_jerk[i]/st:.4f}  reversals={int(e_rev[i])} "
                          f"({100*e_rev[i]/(st*6):.0f}% of arm-action-steps)")
                    e_steps[i] = 0; e_arm_jerk[i] = 0.0; e_rev[i] = 0
                    have_prev[i] = False
    except KeyboardInterrupt:
        pass
    finally:
        print("-" * 92)
        if g_steps > 0:
            mean_arm_jerk = g_arm_jerk / g_steps
            mean_tot_jerk = g_tot_jerk / g_steps
            rev_rate      = g_rev_steps / g_steps
            mean_r8       = W_R * mean_tot_jerk / NORM
            mean_pos      = g_pos_rew / g_steps
            frac          = (mean_r8 / mean_pos) if mean_pos > 1e-9 else float("inf")
            sugg          = (W_R * TARGET_FRAC / frac) if (frac > 1e-9 and np.isfinite(frac)) else W_R
            sugg          = float(np.clip(sugg, W_R, 1.0))
            verdict = ("SMOOTH -- keep w_r=0.05" if rev_rate < 0.10 else
                       "MILD shake -- try w_r ~ 0.10" if rev_rate < 0.30 else
                       "STRONG shake -- try w_r ~ 0.20")
            print(f"[jitter] over {g_steps} agent-steps:")
            print(f"  mean ARM jerk sum(da^2)   = {mean_arm_jerk:.4f}")
            print(f"  arm REVERSAL rate         = {rev_rate*100:.0f}% of steps have >=1 arm sign-flip "
                  f"({g_rev} flips total)")
            print(f"  current r8 bite           = {mean_r8:.4f}/step = {frac*100:.1f}% of the mean +reward "
                  f"({mean_pos:.3f}/step)")
            print(f"  suggested w_r (for r8 ~ {int(TARGET_FRAC*100)}% of +reward at THIS jitter) = {sugg:.2f}  "
                  f"[starting point -- raising w_r also lowers the jerk, so re-measure after]")
            print(f"  VERDICT: {verdict}")
        env.close(); print("[jitter] closed.")

if __name__ == "__main__":
    main()

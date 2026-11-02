#!/usr/bin/env python

import os, sys, glob, math, json, csv, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "used scripts"))

import numpy as np
from stable_baselines3 import PPO
from unity_multiagent_env import UnityMultiAgentVecEnv

REPO      = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
MODEL_DIR = os.environ.get("ARM_MODEL_DIR", "").strip()    or os.path.join(REPO, "python", "model_bearing_angle_FLU_new")
PREFIX    = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_bearing_angle_FLU_new"
OUT_DIR   = os.environ.get("ARM_OUT_DIR", "").strip()      or os.path.join(REPO, "base_model")
TIME_SCALE= float(os.environ.get("ARM_TIME_SCALE", "1.0"))
DEVICE    = os.environ.get("ARM_DEVICE", "cpu")
EPISODES  = int(os.environ.get("ARM_WATCH_EPISODES", "12"))
SUCC_D    = float(os.environ.get("ARM_SUCCESS_RADIUS", "0.6"))
TAG       = "base-eval-4obs"

def resolve_last_model():
    explicit = os.environ.get("ARM_WATCH_MODEL", "").strip()
    if explicit:
        return explicit if os.path.exists(explicit) else None
    cands = [os.path.join(MODEL_DIR, f"{PREFIX}.zip"), os.path.join(MODEL_DIR, f"{PREFIX}_interrupted.zip")]
    cands += glob.glob(os.path.join(MODEL_DIR, "checkpoints", f"{PREFIX}_*_steps.zip"))
    cands = [p for p in cands if os.path.exists(p)]
    return max(cands, key=os.path.getmtime) if cands else None

def d_bearing(o):
    fwd, lat = float(o[0]), float(o[1])
    d = math.hypot(fwd, lat)
    bearing = math.atan2(lat, fwd)
    return d, bearing

def main():
    mp = resolve_last_model()
    if not mp:
        raise SystemExit(f"[{TAG}] no model in {MODEL_DIR} (set ARM_WATCH_MODEL to a .zip)")
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path  = os.path.join(OUT_DIR, "base_eval_steps.csv")
    json_path = os.path.join(OUT_DIR, "base_eval.json")

    print("=" * 78)
    print(f"[{TAG}] model : {mp}")
    print(f"[{TAG}] gate  : d < {SUCC_D} m   episodes: {EPISODES}   out: {OUT_DIR}")
    print(f"[{TAG}] Open Assets/Scenes/base.unity (Space Size=4, Cont. Actions=2), press PLAY. time_scale={TIME_SCALE}x")
    print("=" * 78)

    env = UnityMultiAgentVecEnv(file_name=None, time_scale=TIME_SCALE, log_folder=None)
    n = env.num_envs
    print(f"[{TAG}] connected: {n} agent(s)  obs={env.observation_space.shape}  act={env.action_space.shape}")
    model = PPO.load(mp, env=env, device=DEVICE)

    steps   = [0] * n
    min_d   = [float("inf")] * n
    sum_bear= [0.0] * n
    reach_t = [None] * n
    ep_idx  = [1] * n

    skip_next = [False] * n
    ep_rows = []
    step_rows = []
    total = 0

    try:
        obs = env.reset()
        while total < EPISODES:
            cur = obs
            action, _ = model.predict(cur, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            for i in range(n):
                if skip_next[i]:
                    skip_next[i] = False
                    if dones[i]:
                        skip_next[i] = True
                    continue
                steps[i] += 1
                d, bear = d_bearing(cur[i])
                min_d[i] = min(min_d[i], d)
                sum_bear[i] += abs(bear)
                if reach_t[i] is None and d < SUCC_D:
                    reach_t[i] = steps[i]
                a_v, a_w = float(action[i][0]), float(action[i][1])
                step_rows.append([ep_idx[i], i, steps[i],
                                  round(d, 4), round(bear, 4), round(math.degrees(bear), 2),
                                  round(float(cur[i][0]), 4), round(float(cur[i][1]), 4),
                                  round(float(cur[i][2]), 4), round(float(cur[i][3]), 4),
                                  round(a_v, 4), round(a_w, 4), round(float(rewards[i]), 4)])
                if i == 0:
                    flag = "  <== IN GATE" if d < SUCC_D else ("  <near" if d < 1.0 else "")
                    print(f"[a0] t{steps[i]:>3} d={d:6.3f}m  bearing={math.degrees(bear):6.1f}deg{flag}")
                if dones[i]:
                    r = float(rewards[i])
                    end = "SUCCESS" if r > 0.5 else ("FALL" if r < -0.5 else "TIMEOUT")
                    term = infos[i].get("terminal_observation") if isinstance(infos[i], dict) else None
                    final_d = d_bearing(term)[0] if term is not None else min_d[i]
                    rec = {"episode": ep_idx[i], "agent": i, "steps": steps[i], "end": end,
                           "reached": end == "SUCCESS",
                           "decisions_to_reach": reach_t[i],
                           "min_d": round(min_d[i], 4), "final_d": round(final_d, 4),
                           "mean_abs_bearing_rad": round(sum_bear[i] / max(steps[i], 1), 4)}
                    ep_rows.append(rec)
                    total += 1
                    print(f"  === ep#{total} END t{steps[i]} | {end} | min_d={min_d[i]:.3f} "
                          f"final_d={final_d:.3f} | decisions_to_gate={reach_t[i]}")
                    steps[i], min_d[i], sum_bear[i], reach_t[i] = 0, float("inf"), 0.0, None
                    ep_idx[i] += 1
                    skip_next[i] = True
    except KeyboardInterrupt:
        print(f"\n[{TAG}] stopped early by user.")
    finally:
        env.close()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "agent", "step", "d_m", "bearing_rad", "bearing_deg",
                    "forwardDist", "lateralDist", "linVelForward", "angVel", "act_v", "act_w", "reward"])
        w.writerows(step_rows)

    reached = [e for e in ep_rows if e["reached"]]
    n_ep = len(ep_rows)
    succ_rate = (len(reached) / n_ep) if n_ep else 0.0
    dtr = [e["decisions_to_reach"] for e in reached if e["decisions_to_reach"] is not None]
    summary = {
        "model": mp,
        "date": datetime.date.today().isoformat(),
        "success_gate_m": SUCC_D,
        "episodes": n_ep,
        "successes": len(reached),
        "success_rate": round(succ_rate, 3),
        "mean_decisions_to_reach": round(float(np.mean(dtr)), 1) if dtr else None,
        "median_decisions_to_reach": round(float(np.median(dtr)), 1) if dtr else None,
        "mean_steps_all": round(float(np.mean([e["steps"] for e in ep_rows])), 1) if n_ep else None,
        "mean_min_d": round(float(np.mean([e["min_d"] for e in ep_rows])), 4) if n_ep else None,
        "mean_final_d": round(float(np.mean([e["final_d"] for e in ep_rows])), 4) if n_ep else None,
        "end_reasons": {k: sum(1 for e in ep_rows if e["end"] == k) for k in ("SUCCESS", "FALL", "TIMEOUT")},
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "episodes": ep_rows}, f, indent=2)

    print("-" * 78)
    print(f"[{TAG}] episodes={n_ep}  success={len(reached)}/{n_ep} ({succ_rate*100:.0f}%)  "
          f"mean_decisions_to_reach={summary['mean_decisions_to_reach']}  mean_min_d={summary['mean_min_d']}")
    print(f"[{TAG}] wrote {json_path}")
    print(f"[{TAG}] wrote {csv_path}  ({len(step_rows)} steps)")

if __name__ == "__main__":
    main()

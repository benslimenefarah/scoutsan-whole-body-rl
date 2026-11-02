#!/usr/bin/env python

import os, sys, glob, re, json, csv, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "used scripts"))

import numpy as np
from stable_baselines3 import PPO
from unity_multiagent_env import UnityMultiAgentVecEnv

REPO      = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
MODEL_DIR = os.environ.get("ARM_MODEL_DIR", "").strip()    or os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9_np")
PREFIX    = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_scoutsan_s9_np"
OUT_DIR   = os.environ.get("ARM_OUT_DIR", "").strip()      or os.path.join(REPO, "s9_np_model")
TIME_SCALE= float(os.environ.get("ARM_TIME_SCALE", "1.0"))
DEVICE    = os.environ.get("ARM_DEVICE", "cpu")
EPISODES  = int(os.environ.get("ARM_WATCH_EPISODES", "5"))
SUCC_D    = float(os.environ.get("ARM_SUCCESS_RADIUS", "0.03"))
SUCC_PHI  = float(os.environ.get("ARM_ORIENT_TOL_RAD", "0.05"))
TAG       = "s9np-eval"

def resolve_model():
    explicit = os.environ.get("ARM_EVAL_MODEL", "").strip()
    if explicit:
        return explicit if os.path.exists(explicit) else None
    cands = [os.path.join(MODEL_DIR, f"{PREFIX}.zip"), os.path.join(MODEL_DIR, f"{PREFIX}_interrupted.zip")]
    cands += glob.glob(os.path.join(MODEL_DIR, "checkpoints", f"{PREFIX}_*_steps.zip"))
    cands = [p for p in cands if os.path.exists(p)]
    if not cands:
        return None
    def steps_of(p):
        m = re.search(r"_(\d+)_steps\.zip$", os.path.basename(p)); return int(m.group(1)) if m else -1

    ck = [p for p in cands if p.endswith("_steps.zip")]
    return max(ck, key=steps_of) if ck else max(cands, key=os.path.getmtime)

def metrics(o):
    gp, gq, tp, tq = o[0:3], o[3:7], o[7:10], o[10:14]
    d     = float(np.linalg.norm(gp - tp))
    baseD = float(np.linalg.norm(tp))
    phi   = float(2.0 * np.arccos(min(1.0, abs(float(np.dot(gq, tq))))))
    return d, baseD, phi

def main():
    mp = resolve_model()
    if not mp:
        raise SystemExit(f"[{TAG}] no model in {MODEL_DIR} (set ARM_EVAL_MODEL=<zip>)")
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path, json_path = os.path.join(OUT_DIR, "s9np_eval_steps.csv"), os.path.join(OUT_DIR, "s9np_eval.json")

    print("=" * 80)
    print(f"[{TAG}] model : {mp}")
    print(f"[{TAG}] gate  : d<{SUCC_D} m AND phi<{SUCC_PHI} rad   episodes: {EPISODES}   out: {OUT_DIR}")
    print(f"[{TAG}] Open the s9 scene (s9_np FIXED config, Behavior=Default, 14/8), PRESS PLAY. time_scale={TIME_SCALE}x")
    print("=" * 80)

    env = UnityMultiAgentVecEnv(file_name=None, time_scale=TIME_SCALE, log_folder=None)
    n = env.num_envs
    print(f"[{TAG}] connected: {n} agent(s)  obs={env.observation_space.shape}  act={env.action_space.shape}")
    model = PPO.load(mp, env=env, device=DEVICE)

    steps = [0]*n; min_d = [1e9]*n; min_phi = [1e9]*n; min_base = [1e9]*n; reach_t = [None]*n
    ep_idx = [1]*n; skip_next = [False]*n
    ep_rows, step_rows, total = [], [], 0

    try:
        obs = env.reset()
        while total < EPISODES:
            cur = obs
            action, _ = model.predict(cur, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            for i in range(n):
                if skip_next[i]:
                    skip_next[i] = False
                    if dones[i]: skip_next[i] = True
                    continue
                steps[i] += 1
                d, baseD, phi = metrics(np.asarray(cur[i], float))
                min_d[i] = min(min_d[i], d); min_phi[i] = min(min_phi[i], phi); min_base[i] = min(min_base[i], baseD)
                pos_ok, orient_ok = d < SUCC_D, phi < SUCC_PHI
                if reach_t[i] is None and pos_ok and orient_ok:
                    reach_t[i] = steps[i]
                step_rows.append([ep_idx[i], i, steps[i], round(d, 4), round(baseD, 4),
                                  round(phi, 4), round(float(np.degrees(phi)), 2),
                                  int(pos_ok), int(orient_ok), round(float(rewards[i]), 3)])
                if i == 0:
                    fl = "  <== CONFONDU" if (pos_ok and orient_ok) else ("  <near" if d < 0.10 else "")
                    print(f"[a0] t{steps[i]:>3} d={d:6.3f}m baseD={baseD:5.2f}m phi={np.degrees(phi):5.1f}deg{fl}")
                if dones[i]:
                    r = float(rewards[i])
                    end = "SUCCESS" if r > 15 else ("FAIL" if steps[i] < 3 else "TIMEOUT")
                    reached = (reach_t[i] is not None) or end == "SUCCESS"
                    ep_rows.append({"episode": ep_idx[i], "steps": steps[i], "end": end, "reached": reached,
                                    "decisions_to_reach": reach_t[i], "min_d": round(min_d[i], 4),
                                    "min_phi_rad": round(min_phi[i], 4), "min_baseD": round(min_base[i], 4)})
                    total += 1
                    print(f"  === ep#{total} END t{steps[i]} | {end} | min_d={min_d[i]:.3f} "
                          f"min_phi={np.degrees(min_phi[i]):.1f}deg | reach@{reach_t[i]}")
                    steps[i], min_d[i], min_phi[i], min_base[i], reach_t[i] = 0, 1e9, 1e9, 1e9, None
                    ep_idx[i] += 1; skip_next[i] = True
    except KeyboardInterrupt:
        print(f"\n[{TAG}] stopped early.")
    finally:
        env.close()

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","agent","step","d_m","baseD_m","phi_rad","phi_deg","pos_ok","orient_ok","reward"])
        w.writerows(step_rows)

    reached = [e for e in ep_rows if e["reached"]]
    dtr = [e["decisions_to_reach"] for e in reached if e["decisions_to_reach"] is not None]
    summary = {
        "model": mp, "date": datetime.date.today().isoformat(),
        "gate_pos_m": SUCC_D, "gate_orient_rad": SUCC_PHI,
        "episodes": len(ep_rows), "successes": len(reached),
        "success_rate": round(len(reached)/len(ep_rows), 3) if ep_rows else 0.0,
        "mean_decisions_to_reach": round(float(np.mean(dtr)), 1) if dtr else None,
        "mean_min_d": round(float(np.mean([e["min_d"] for e in ep_rows])), 4) if ep_rows else None,
        "mean_min_phi_rad": round(float(np.mean([e["min_phi_rad"] for e in ep_rows])), 4) if ep_rows else None,
        "end_reasons": {k: sum(1 for e in ep_rows if e["end"] == k) for k in ("SUCCESS","FAIL","TIMEOUT")},
    }
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "episodes": ep_rows}, f, indent=2)

    print("-"*80)
    print(f"[{TAG}] episodes={summary['episodes']} success={summary['successes']} "
          f"({summary['success_rate']*100:.0f}%) mean_min_d={summary['mean_min_d']} "
          f"mean_min_phi={summary['mean_min_phi_rad']}rad")
    print(f"[{TAG}] wrote {json_path}\n[{TAG}] wrote {csv_path} ({len(step_rows)} steps)")

if __name__ == "__main__":
    main()

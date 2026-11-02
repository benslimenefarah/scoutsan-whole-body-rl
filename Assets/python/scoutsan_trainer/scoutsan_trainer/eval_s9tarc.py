#!/usr/bin/env python

import os, sys, glob, re, json, csv, math, socket, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "used scripts"))
import numpy as np
from stable_baselines3 import PPO
from unity_multiagent_env import UnityMultiAgentVecEnv

REPO       = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
EXP_TAG    = os.environ.get("ARM_EXP_TAG", "").strip() or "s9_tarc"
DESC       = os.environ.get("ARM_LAMBDA_DESC", "").strip() or "s9 direct-velocity whole-body (14/8), T-arc curriculum"
MODEL_DIR  = os.environ.get("ARM_MODEL_DIR", "").strip()    or os.path.join(REPO, "scoutsan_Baseline", "models", "model_scoutsan_s9_tarc")
PREFIX     = os.environ.get("ARM_MODEL_PREFIX", "").strip() or "ppo_scoutsan_s9_tarc"
CKPT_DIR   = os.path.join(MODEL_DIR, "checkpoints")
OUT_DIR    = os.environ.get("ARM_OUT_DIR", "").strip()      or os.path.join(REPO, "deliverables", "s9_tarc_model")
TIME_SCALE = float(os.environ.get("ARM_TIME_SCALE", "1.0"))
COUNT      = int(os.environ.get("ARM_LINEARC_COUNT", "5"))
SUCC_D     = float(os.environ.get("ARM_SUCCESS_RADIUS", "0.02"))
SUCC_PHI   = float(os.environ.get("ARM_ORIENT_TOL_RAD", "0.02"))
HOLD_STEPS = int(os.environ.get("ARM_HOLD_STEPS", "10"))
AGENT_D    = float(os.environ.get("ARM_AGENT_GATE_D", "0.02"))
AGENT_PHI  = float(os.environ.get("ARM_AGENT_GATE_PHI", "0.02"))
DEVICE     = os.environ.get("ARM_DEVICE", "cpu")
DT         = 0.1

def resolve_ckpt():
    ov = os.environ.get("ARM_EVAL_MODEL", "").strip()
    if ov:
        if not os.path.exists(ov): raise SystemExit(f"[{EXP_TAG}] ARM_EVAL_MODEL not found: {ov}")
        return ov
    cands = glob.glob(os.path.join(CKPT_DIR, f"{PREFIX}_*_steps.zip"))
    stepn = lambda p: int(re.search(r"_(\d+)_steps\.zip$", os.path.basename(p)).group(1)) if re.search(r"_(\d+)_steps\.zip$", p) else -1
    best = max(cands, key=stepn) if cands else None
    for extra in (f"{PREFIX}.zip", f"{PREFIX}_interrupted.zip"):
        c = os.path.join(MODEL_DIR, extra)
        if os.path.exists(c) and (best is None or os.path.getmtime(c) > os.path.getmtime(best)): best = c
    if best is None: raise SystemExit(f"[{EXP_TAG}] no checkpoint in {CKPT_DIR} (set ARM_EVAL_MODEL).")
    return best

def ckpt_steps(name):
    m = re.search(r"_(\d+)_steps", name)
    return int(m.group(1)) if m else None

def pose_errors(o):
    gp, gq, tp, tq = o[0:3], o[3:7], o[7:10], o[10:14]
    d = float(np.linalg.norm(np.asarray(gp) - np.asarray(tp)))
    dot = min(1.0, max(-1.0, abs(float(np.dot(gq, tq)))))
    return d, 2.0 * math.acos(dot)

def qinv_rot(q, v):
    u = -np.asarray(q[:3], float); w = float(q[3]); v = np.asarray(v, float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)

def flu_to_unity(v):
    v = np.asarray(v, float)
    return np.array([-v[1], v[2], v[0]])

def spawn_of(o):
    tp, tq, gp = np.asarray(o[7:10], float), np.asarray(o[10:14], float), np.asarray(o[0:3], float)
    fwd, lat = float(tp[0]), float(tp[1])
    rel  = qinv_rot(tq, flu_to_unity(gp - tp))
    lat0 = float(-rel[0])
    key = "axis" if abs(lat0) < 0.04 else ("arc+" if lat0 > 0 else "arc-")
    return dict(dist=round(float(np.linalg.norm(tp)), 3), az=round(math.degrees(math.atan2(lat, fwd)), 1),
                lat0=round(lat0, 4), key=key)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ck = resolve_ckpt()
    out_json = os.path.join(OUT_DIR, "s9tarc_eval.json")
    out_csv  = os.path.join(OUT_DIR, "s9tarc_eval_steps.csv")
    print("=" * 104)
    print(f"[{EXP_TAG}] {DESC}")
    print(f"[{EXP_TAG}] checkpoint = {os.path.basename(ck)}   device={DEVICE}   time_scale={TIME_SCALE}x")
    print(f"[{EXP_TAG}] EVAL gate d<{SUCC_D} m AND phi<{SUCC_PHI} rad (= the wb gate) | {COUNT} episodes of the line+arc set")
    print(f"[{EXP_TAG}] >>> s9 scene, ONE robot, Behavior=Default, tArcLineArc=ON (count 4 / lineN 1); PRESS PLAY (5004) <<<")
    print("=" * 104)

    env = UnityMultiAgentVecEnv(file_name=None, no_graphics=False, time_scale=TIME_SCALE, log_folder=None)
    n = env.num_envs
    print(f"[{EXP_TAG}] connected: {n} agent(s)  obs={env.observation_space.shape}  act={env.action_space.shape}")
    model = PPO.load(ck, env=env, device=DEVICE)

    def fresh(): return dict(min_d=float("inf"), min_phi=float("inf"), steps=0, gate_first=-1,
                             hold=0, max_hold=0, last_d=None, last_phi=None, spawn=None)
    per = {i: fresh() for i in range(n)}
    results, step_rows, ep = [], [], 0

    obs = env.reset()
    print("-" * 104)
    try:
        while ep < COUNT:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            for i in range(n):
                o = np.asarray(obs[i], float); r = per[i]; r["steps"] += 1
                if r["spawn"] is None: r["spawn"] = spawn_of(o)
                d, phi = pose_errors(o)
                r["last_d"], r["last_phi"] = d, phi
                r["min_d"] = min(r["min_d"], d); r["min_phi"] = min(r["min_phi"], phi)
                in_gate = (d < SUCC_D and phi < SUCC_PHI)
                if in_gate:
                    if r["gate_first"] < 0: r["gate_first"] = r["steps"]
                    r["hold"] += 1; r["max_hold"] = max(r["max_hold"], r["hold"])
                else:
                    r["hold"] = 0
                if ep < COUNT:
                    step_rows.append([ep, r["steps"], round(r["steps"] * DT, 3), round(d, 5),
                                      round(phi, 5), round(math.degrees(phi), 3), int(in_gate)]
                                     + [round(float(v), 5) for v in o[0:14]]
                                     + [round(float(rewards[i]), 4)])
                if dones[i] and ep < COUNT:
                    sp = r["spawn"]
                    reached = r["gate_first"] > 0
                    rec = dict(idx=ep, target_key=sp["key"], spawn_lat0=sp["lat0"],
                               spawn_dist=sp["dist"], spawn_az=sp["az"],
                               min_d=round(r["min_d"], 4), min_phi_deg=round(math.degrees(r["min_phi"]), 2),
                               reached=bool(reached), steps_to_gate=(r["gate_first"] if reached else None),
                               secs_to_gate=(round(r["gate_first"] * DT, 2) if reached else None),
                               max_hold=r["max_hold"], held=bool(r["max_hold"] >= HOLD_STEPS), steps=r["steps"],
                               final_d=round(r["last_d"], 4), final_phi_deg=round(math.degrees(r["last_phi"]), 2))
                    results.append(rec)
                    print(f"[{EXP_TAG}] tgt {ep:02d} {sp['key']:5s} d0={sp['dist']:.2f} az={sp['az']:+6.1f} | "
                          f"minD={rec['min_d']:.3f} minPhi={rec['min_phi_deg']:4.1f}deg "
                          f"{'REACH@'+str(rec['steps_to_gate'])+' ('+str(rec['secs_to_gate'])+'s)' if reached else 'MISS':>20s}")
                    ep += 1
                    per[i] = fresh()
    except KeyboardInterrupt:
        print(f"[{EXP_TAG}] interrupted -- writing partial results.")
    finally:
        env.close()

    results.sort(key=lambda x: x["idx"])
    reached = [r for r in results if r["reached"]]
    nr = max(1, len(results))
    summary = dict(
        n_targets=len(results),
        reached_rate=round(sum(r["reached"] for r in results) / nr, 4),
        mean_min_d=round(sum(r["min_d"] for r in results) / nr, 4),
        mean_min_phi_deg=round(sum(r["min_phi_deg"] for r in results) / nr, 2),
        mean_steps_to_gate=round(sum(r["steps_to_gate"] for r in reached) / max(1, len(reached)), 1) if reached else None,
        mean_secs_to_gate=round(sum(r["secs_to_gate"] for r in reached) / max(1, len(reached)), 2) if reached else None,
    )
    doc = dict(
        experiment=EXP_TAG, lambda_desc=DESC, weight_floor_watch=None,
        checkpoint=os.path.basename(ck), checkpoint_steps=ckpt_steps(os.path.basename(ck)), model_prefix=PREFIX,
        gate=dict(d=SUCC_D, phi=SUCC_PHI, hold_steps=HOLD_STEPS),
        agent_gate=dict(d=AGENT_D, phi=AGENT_PHI, hold_steps=0), success_terminates_episode=True,
        decision_dt_s=DT, linearc_count=COUNT,
        target_set=dict(distinct=3, radius_m=2.0, arc_half_deg=20.0, height_m=0.25,
                        note="scene tArc: lineRadius 2, arcTotalDeg 40, height 0.25; tArcLineArcCount 4 / lineN 1 "
                             "-> on-axis + arc -20/0/+20 (index 2 repeats on-axis). Same set as the wb_1am eval."),
        date=datetime.date.today().isoformat(), host=socket.gethostname(),
        targets=results, summary=summary)
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=2)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "step", "time", "d", "phi_rad", "phi_deg", "in_gate",
                    "gx", "gy", "gz", "gqx", "gqy", "gqz", "gqw",
                    "tx", "ty", "tz", "tqx", "tqy", "tqz", "tqw", "reward"])
        w.writerows(step_rows)

    print("-" * 104)
    print(f"[{EXP_TAG}] reached {summary['reached_rate']*100:.0f}%  mean minD {summary['mean_min_d']:.4f} m  "
          f"mean minPhi {summary['mean_min_phi_deg']:.2f} deg  mean steps->gate {summary['mean_steps_to_gate']} "
          f"({summary['mean_secs_to_gate']} s)")
    print(f"[{EXP_TAG}] targets hit: " + ", ".join(f"{r['idx']}:{r['target_key']}(lat0 {r['spawn_lat0']:+.2f})"
                                                   for r in results))
    print(f"[{EXP_TAG}] wrote {out_json}")
    print(f"[{EXP_TAG}] wrote {out_csv}  ({len(step_rows)} steps)")

if __name__ == "__main__":
    main()

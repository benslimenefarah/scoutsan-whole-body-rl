#!/usr/bin/env python

import os, glob, subprocess, tempfile, warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
import torch.nn as nn

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
def latest(p):
    if os.path.isfile(p): return p
    c = sorted(glob.glob(p)); return c[-1] if c else None

SPECS = [
 dict(folder="base_model", mode="policy",
      ckpt=f"{REPO}/python/model_bearing_angle_FLU_new/ppo_bearing_angle_FLU_new_interrupted.zip",
      title="Base driver policy  (FLU, direct velocity)",
      obs="Observation (4)\\lforwardDist, lateralDist,\\llinVelFwd, angVel",
      act="action head (2)\\lv_ros, w_ros   (FLU)",
      stages=[("FLU → Unity +\\ldifferential drive\\l(v ≤ 0.5, ω ≤ 0.8)", "#fff4d6"),
              ("4 wheels", "#eeeeee")]),
 dict(folder="s9_np_model/fixed target", mode="policy",
      ckpt=f"{REPO}/scoutsan_Baseline/models/model_scoutsan_s9_np/checkpoints/*_steps.zip",
      title="s9_np policy  (direct velocity, whole-body)",
      obs="Observation (14)\\lgripper pose (7) +\\ltarget pose (7)",
      act="action head (8)\\lΔJ1..ΔJ6, base v, ω",
      stages=[("direct velocity + clamps\\l(joint 30°/s, base 0.3/0.4,\\lwheel 150°/s)", "#fff4d6"),
              ("arm joints + base wheels", "#eeeeee")]),
 dict(folder="s9_tarc_model", mode="policy",
      ckpt=f"{REPO}/scoutsan_Baseline/models/model_scoutsan_s9_tarc/ppo_scoutsan_s9_tarc.zip",
      title="s9_tarc policy  (direct velocity, whole-body, T-arc curriculum)",
      obs="Observation (14)\\lgripper pose (7) +\\ltarget pose (7)",
      act="action head (8)\\lΔJ1..ΔJ6, base v, ω",
      stages=[("direct velocity + clamps\\l(joint 30°/s, base 0.3/0.4,\\lwheel 150°/s)", "#fff4d6"),
              ("arm joints + base wheels", "#eeeeee")]),
]

def arch(ckpt):
    m = PPO.load(ckpt, device="cpu")
    hid = [mod.out_features for mod in m.policy.mlp_extractor.policy_net if isinstance(mod, nn.Linear)]
    return m.observation_space.shape[0], m.action_space.shape[0], hid, m.policy.activation_fn.__name__

def dot_src(spec, act_n=None, hid=None, actf=None):
    L = ['digraph P {', '  rankdir=LR; bgcolor="white";',
         '  labelloc="t"; fontname="Times"; fontsize=18;', f'  label="{spec["title"]}";',
         '  node [fontname="Times", fontsize=12, shape=box, style="rounded,filled", margin="0.12,0.08"];',
         '  edge [color="#555555", penwidth=1.4, arrowsize=0.8];',
         f'  obs [label="{spec["obs"]}", fillcolor="#e8f0fe"];']
    if spec["mode"] == "controller":
        prev = "obs"
        for i, (lab, col) in enumerate(spec["stages"]):
            L.append(f'  s{i} [label="{lab}", fillcolor="{col}"];')
            L.append(f'  {prev} -> s{i};'); prev = f's{i}'
    else:
        L.append(f'  h1 [label="FC {hid[0]}\\l{actf}\\l", fillcolor="#dfeae0"];')
        L.append(f'  h2 [label="FC {hid[1]}\\l{actf}\\l", fillcolor="#dfeae0"];')
        L.append(f'  ah [label="{spec["act"]}", fillcolor="#fde8e8"];')
        L.append('  obs -> h1 -> h2 -> ah -> s0;')
        for i, (lab, col) in enumerate(spec["stages"]):
            L.append(f'  s{i} [label="{lab}", fillcolor="{col}"];')
        for i in range(len(spec["stages"]) - 1):
            L.append(f'  s{i} -> s{i+1};')
    L.append('}')
    return '\n'.join(L)

def render(spec):
    figdir = os.path.join(REPO, "deliverables", spec["folder"], "figures"); os.makedirs(figdir, exist_ok=True)
    outname = spec.get("outname", "policy_graph")
    if spec["mode"] == "controller":
        src, info = dot_src(spec), "analytic controller (no policy)"
    else:
        ck = latest(spec["ckpt"])
        if not ck:
            print(f"[{spec['folder']}] checkpoint not found -- skip"); return
        obs_n, act_n, hid, actf = arch(ck)
        if len(hid) < 2: hid = (hid + [64, 64])[:2]
        src, info = dot_src(spec, act_n, hid, actf), f"obs {obs_n} -> {hid} {actf} -> act {act_n}"
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(src); dotf = f.name
    for ext in ("pdf", "png"):
        args = ["dot", f"-T{ext}"] + (["-Gdpi=200"] if ext == "png" else []) + \
               ["-o", os.path.join(figdir, f"{outname}.{ext}"), dotf]
        subprocess.run(args)
    os.unlink(dotf)
    print(f"[{spec['folder']}] {outname}.{{pdf,png}}  ({info})")

if __name__ == "__main__":
    for s in SPECS:
        render(s)

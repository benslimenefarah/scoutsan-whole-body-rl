#!/usr/bin/env python

import os, subprocess, tempfile

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
GRN, LIME, RED, YEL, GRY, BLU = "#dfeae0", "#eaf3d8", "#fde8e8", "#fff4d6", "#eeeeee", "#e8f0fe"

SPECS = [
 dict(folder="base_model", title="Base driver — reward",
   step=[("r1  progress\\l+0.1·(prevD − d)\\l", GRN),
         ("r2  time\\l−0.001 / step\\l", RED),
         ("r3  bearing\\l−0.005·β·fade(0.8→1.8 m)\\l", RED)],
   term=[("+1  success\\l(d < 0.6 m)\\l", GRN),
         ("−1  fall\\l(y < −1.0 m)\\l", RED)]),
 dict(folder="s9_np_model/fixed target", title="s9_np — reward (confondu)",
   step=[("r1  progress\\l+10·clamp(prevD−d, ±0.25)\\l", GRN),
         ("r3  position directional\\l±0.01   (0 inside gate)\\l", LIME),
         ("r5  orientation attraction\\l0.5·(e^−φ − 1)\\l", GRN),
         ("r6  orientation directional\\l±0.01   (0 inside gate)\\l", LIME),
         ("r8  jerk penalty\\l−0.05·Σ(Δa)² / 32\\l", RED)],
   term=[("+100  reach\\l(d<0.03 m AND φ<0.05 rad)\\l", GRN),
         ("−50  fall\\l", RED)]),
 dict(folder="s9_tarc_model", title="s9_tarc — reward (confondu; SAME agent/reward as s9_np, only the target curriculum changed)",
   step=[("r1  progress\\l+10·clamp(prevD−d, ±0.25)\\l", GRN),
         ("r3  position directional\\l±0.01   (0 inside gate)\\l", LIME),
         ("r5  orientation attraction\\l0.5·(e^−φ − 1)\\l", GRN),
         ("r6  orientation directional\\l±0.01   (0 inside gate)\\l", LIME),
         ("r8  jerk penalty\\l−0.05·Σ(Δa)² / 32\\l", RED)],
   term=[("+100  reach\\l(d<0.03 m AND φ<0.05 rad)\\l", GRN),
         ("−50  fall\\l", RED)]),
]

def dot(spec):
    L = ['digraph R {', '  rankdir=LR; bgcolor="white";',
         '  labelloc="t"; fontname="Times"; fontsize=17;', f'  label="{spec["title"]}";',
         '  node [fontname="Times", fontsize=11, shape=box, style="rounded,filled", margin="0.13,0.07"];',
         '  edge [color="#666666", penwidth=1.3, arrowsize=0.7];']
    L.append('  subgraph cluster_s { label="per-step shaping (summed each decision)"; fontname="Times"; fontsize=12; color="#bbbbbb"; style="rounded";')
    for i, (lab, col) in enumerate(spec["step"]):
        L.append(f'    s{i} [label="{lab}", fillcolor="{col}"];')
    L.append('  }')
    L.append(f'  sum [label="Σ  step reward", fillcolor="{YEL}"];')
    for i in range(len(spec["step"])):
        L.append(f'  s{i} -> sum;')
    L.append('  subgraph cluster_t { label="terminal (one-shot, ends the episode)"; fontname="Times"; fontsize=12; color="#bbbbbb"; style="rounded";')
    for i, (lab, col) in enumerate(spec["term"]):
        L.append(f'    t{i} [label="{lab}", fillcolor="{col}"];')
    L.append('  }')
    L.append(f'  ret [label="episode\\lreturn\\l", fillcolor="{GRY}"];')
    L.append('  sum -> ret;')
    for i in range(len(spec["term"])):
        L.append(f'  t{i} -> ret;')
    L.append('}')
    return '\n'.join(L)

def main():
    for spec in SPECS:
        if spec["folder"].endswith("/wI"):
            continue
        figdir = os.path.join(REPO, "deliverables", spec["folder"], "figures"); os.makedirs(figdir, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
            f.write(dot(spec)); dotf = f.name
        ok = True
        for ext in ("pdf", "png"):
            args = ["dot", f"-T{ext}"] + (["-Gdpi=200"] if ext == "png" else []) + \
                   ["-o", os.path.join(figdir, f"reward_graph.{ext}"), dotf]
            if subprocess.run(args).returncode != 0: ok = False
        os.unlink(dotf)
        print(f"[{spec['folder']}] reward_graph.{{pdf,png}}" + ("" if ok else "  <ERR>"))

if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sns.set_theme(context="paper", style="ticks", font_scale=1.3, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.9,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

SPECS = [
    ("base_model", "python/model_bearing_angle_FLU_new/tb", "Base driver — training reward", "#2a6f97"),
    ("s9_np_model/fixed target", "scoutsan_Baseline/tb/tb_scoutsan_s9_np", "s9_np — training reward", "#2a6f97"),
    ("s9_tarc_model", "scoutsan_Baseline/tb/tb_scoutsan_s9_tarc", "s9_tarc — training reward (T-arc curriculum)", "#2a6f97"),
]

def newest_run(tbdir):
    runs = {}
    for ev in glob.glob(os.path.join(REPO, tbdir, "**", "events.out.tfevents*"), recursive=True):
        runs[os.path.dirname(ev)] = max(runs.get(os.path.dirname(ev), 0), os.path.getmtime(ev))
    if not runs:
        return None
    return max(runs, key=runs.get)

def read_reward(rundir):
    ea = EventAccumulator(rundir, size_guidance={"scalars": 0}); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag = "rollout/ep_rew_mean"
    if tag not in tags:
        cand = [t for t in tags if "ep_rew_mean" in t]
        if not cand: return None, None, tags
        tag = cand[0]
    sc = ea.Scalars(tag)
    return np.array([s.step for s in sc]), np.array([s.value for s in sc]), tag

def main():
    for folder, tbdir, title, col in SPECS:
        if folder.endswith("/wI"):
            continue
        run = newest_run(tbdir)
        if not run:
            print(f"[{folder}] no TB run under {tbdir} -- skip"); continue
        steps, vals, tag = read_reward(run)
        if steps is None or len(steps) == 0:
            print(f"[{folder}] no ep_rew_mean (tags: {tag}) -- skip"); continue
        figdir = os.path.join(REPO, "deliverables", folder, "figures"); os.makedirs(figdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.6, 4.4))
        xm = steps / 1e6
        ax.plot(xm, vals, color=col, alpha=0.35, lw=1.0)
        if len(vals) >= 9:
            k = max(3, len(vals) // 40) | 1
            sm = pd.Series(vals).rolling(k, center=True, min_periods=1).mean().to_numpy()
            ax.plot(xm, sm, color=col, lw=2.0, label="smoothed")
        ax.scatter([xm[-1]], [vals[-1]], color="#c0392b", s=30, zorder=5)
        ax.annotate(f"final {vals[-1]:.1f}", xy=(xm[-1], vals[-1]), xytext=(-8, 8),
                    textcoords="offset points", ha="right", fontsize="small", color="#c0392b")
        ax.set_xlabel("training steps (millions)"); ax.set_ylabel("episode reward  (ep_rew_mean)")
        ax.set_title(title)
        sns.despine(); fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(figdir, f"reward_curve.{ext}"))
        plt.close(fig)
        print(f"[{folder}] reward_curve.{{pdf,png}}  ({len(steps)} pts, {steps[-1]/1e6:.2f}M steps, final {vals[-1]:.1f})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns

HERE  = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(HERE, "base_eval_steps.csv")
JSONP = os.path.join(HERE, "base_eval.json")
FIG   = os.path.join(HERE, "figures")
GATE  = 0.6
CMAP  = sns.color_palette("crest", as_cmap=True)
GATECOL = "#c0392b"

sns.set_theme(context="paper", style="ticks", font_scale=1.3, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.7,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

def load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run eval_base_flu.py first")
    df = pd.read_csv(CSV)
    spawn = df.groupby("episode")["d_m"].first().rename("spawn_d")
    df = df.join(spawn, on="episode")

    dtr = df.groupby("episode")["step"].max()
    if os.path.exists(JSONP):
        eps = json.load(open(JSONP)).get("episodes", [])
        d = {e["episode"]: e.get("decisions_to_reach") for e in eps if e.get("decisions_to_reach")}
        dtr = pd.Series({ep: d.get(ep, dtr[ep]) for ep in dtr.index})
    per_ep = pd.DataFrame({"spawn_d": spawn, "decisions": dtr})
    return df, per_ep, Normalize(spawn.min(), spawn.max())

def _colorbar(fig, ax, norm, label="spawn distance (m)"):
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(label); cb.outline.set_linewidth(0.6)
    return cb

def plot_convergence(ax, df, norm):
    for ep, g in df.groupby("episode"):
        ax.plot(g["step"], g["d_m"], color=CMAP(norm(g["spawn_d"].iloc[0])), alpha=0.9)
    ax.axhline(GATE, ls="--", lw=1.4, color=GATECOL, label=f"success gate ({GATE} m)")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"distance to target  $d$ (m)")
    ax.set_title("Convergence to the target")
    ax.set_ylim(bottom=0); ax.legend(frameon=False, loc="upper right")

def plot_approach(ax, df, norm):
    for ep, g in df.groupby("episode"):
        c = CMAP(norm(g["spawn_d"].iloc[0]))
        ax.plot(g["lateralDist"], g["forwardDist"], color=c, alpha=0.85, lw=1.4)
        ax.scatter(g["lateralDist"].iloc[0], g["forwardDist"].iloc[0], color=c, s=22, zorder=3,
                   edgecolor="white", linewidth=0.5)
    th = np.linspace(0, 2 * np.pi, 240)
    ax.plot(GATE * np.cos(th), GATE * np.sin(th), ls="--", lw=1.4, color=GATECOL, label=f"gate ({GATE} m)")
    ax.scatter([0], [0], marker="^", s=90, color="black", zorder=4, label="robot (facing $+$fwd)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("lateral offset (m, $+$left)"); ax.set_ylabel("forward offset (m)")
    ax.set_title("Target approach in the robot frame")
    ax.legend(frameon=False, loc="upper left", fontsize="small")

def plot_scaling(ax, per_ep):
    x = per_ep["spawn_d"].to_numpy(float); y = per_ep["decisions"].to_numpy(float)
    order = np.argsort(x)
    ax.scatter(x, y, s=55, c=x, cmap=CMAP, edgecolor="white", linewidth=0.6, zorder=3)
    if len(x) >= 2:
        b1, b0 = np.polyfit(x, y, 1)
        xf = np.linspace(x.min(), x.max(), 50)
        ax.plot(xf, b0 + b1 * xf, color=GATECOL, lw=1.7, zorder=2)
        r = np.corrcoef(x, y)[0, 1]
        ax.annotate(f"$r = {r:.2f}$\nslope $= {b1:.0f}$ steps/m",
                    xy=(0.04, 0.96), xycoords="axes fraction", va="top", ha="left",
                    fontsize="small", bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    ax.set_xlabel("spawn distance (m)"); ax.set_ylabel("decisions to reach gate")
    ax.set_title("Effort scales with spawn distance")

def plot_bearing(ax, df, norm):
    for ep, g in df.groupby("episode"):
        ax.plot(g["step"], g["bearing_deg"].abs(), color=CMAP(norm(g["spawn_d"].iloc[0])), alpha=0.9)
    ax.set_xlabel("decision step"); ax.set_ylabel(r"$|\beta|$  heading error (deg)")
    ax.set_title("Alignment: heading error over time")
    ax.set_ylim(bottom=0)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)

def main():
    os.makedirs(FIG, exist_ok=True)
    df, per_ep, norm = load()

    for name, fn in [("convergence", plot_convergence), ("approach", plot_approach), ("bearing", plot_bearing)]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6) if name != "approach" else (5.4, 5.4))
        fn(ax, df, norm); _colorbar(fig, ax, norm)
        fig.tight_layout(); save(fig, f"base_eval_{name}")

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plot_scaling(ax, per_ep); fig.tight_layout(); save(fig, "base_eval_scaling")

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.8))
    plot_convergence(axes[0, 0], df, norm)
    plot_approach(axes[0, 1], df, norm)
    plot_scaling(axes[1, 0], per_ep)
    plot_bearing(axes[1, 1], df, norm)
    fig.suptitle("Base driver — deterministic evaluation", fontsize=17, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.85, top=0.92, bottom=0.08, wspace=0.28, hspace=0.32)
    sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
    cax = fig.add_axes([0.905, 0.12, 0.016, 0.76])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("spawn distance (m)"); cb.outline.set_linewidth(0.6)
    save(fig, "base_eval_overview")

    outs = sorted(f for f in os.listdir(FIG) if f.endswith((".pdf", ".png")))
    print(f"wrote {len(outs)} files to {FIG}:")
    for f in outs:
        print("  ", f)

if __name__ == "__main__":
    main()

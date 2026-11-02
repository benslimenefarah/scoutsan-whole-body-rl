#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE  = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(HERE, "s9np_eval_steps.csv")
FIG   = os.path.join(HERE, "figures")
GATE_D, GATE_PHI = 0.03, np.degrees(0.05)
GATECOL = "#c0392b"

sns.set_theme(context="paper", style="ticks", font_scale=1.3, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

def load():
    if not os.path.exists(CSV):
        raise SystemExit(f"missing {CSV} -- run eval_s9np.py first")
    df = pd.read_csv(CSV)
    eps = sorted(df["episode"].unique())
    pal = dict(zip(eps, sns.color_palette("crest", n_colors=max(2, len(eps)))))
    return df, eps, pal

def _lines(ax, df, eps, pal, ycol):
    for ep in eps:
        g = df[df["episode"] == ep]
        ax.plot(g["step"], g[ycol], color=pal[ep], alpha=0.9, label=f"ep {ep}")

def plot_reach(ax, df, eps, pal):
    _lines(ax, df, eps, pal, "d_m")
    ax.axhline(GATE_D, ls="--", lw=1.4, color=GATECOL, label=f"gate ({GATE_D} m)")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"gripper$\to$target  $d$ (m)")
    ax.set_title("Reach: distance to target"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", ncol=2)

def plot_orient(ax, df, eps, pal):
    _lines(ax, df, eps, pal, "phi_deg")
    ax.axhline(GATE_PHI, ls="--", lw=1.4, color=GATECOL, label=f"gate ({GATE_PHI:.1f}$^\\circ$)")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"orientation error $\varphi$ (deg)")
    ax.set_title("Confondu: orientation error"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", loc="upper right")

def plot_basedrive(ax, df, eps, pal):
    _lines(ax, df, eps, pal, "baseD_m")
    ax.axhline(0.9, ls=":", lw=1.2, color="0.4", label="arm reach $\\approx$0.9 m")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"base$\to$target  baseD (m)")
    ax.set_title("Base drives into arm reach"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", loc="upper right")

def plot_phase(ax, df, eps, pal):
    for ep in eps:
        g = df[df["episode"] == ep]
        ax.plot(g["d_m"], g["phi_deg"], color=pal[ep], alpha=0.9)
        ax.scatter(g["d_m"].iloc[0], g["phi_deg"].iloc[0], color=pal[ep], s=22, zorder=3, edgecolor="white", lw=0.5)
    ax.axvline(GATE_D, ls="--", lw=1.2, color=GATECOL); ax.axhline(GATE_PHI, ls="--", lw=1.2, color=GATECOL)
    ax.set_xlabel(r"distance $d$ (m)"); ax.set_ylabel(r"orientation $\varphi$ (deg)")
    ax.set_title("Joint convergence into the gate corner")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)

def main():
    os.makedirs(FIG, exist_ok=True)
    df, eps, pal = load()
    for name, fn in [("reach", plot_reach), ("orient", plot_orient), ("basedrive", plot_basedrive), ("phase", plot_phase)]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        fn(ax, df, eps, pal); fig.tight_layout(); save(fig, f"s9np_{name}")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.4))
    plot_reach(axes[0, 0], df, eps, pal)
    plot_orient(axes[0, 1], df, eps, pal)
    plot_basedrive(axes[1, 0], df, eps, pal)
    plot_phase(axes[1, 1], df, eps, pal)
    fig.suptitle("s9_np — fixed-target deterministic evaluation", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96]); save(fig, "s9np_overview")

    outs = sorted(f for f in os.listdir(FIG) if f.startswith("s9np_") and f.endswith((".pdf", ".png")))
    print(f"wrote {len(outs)} files to {FIG}:")
    for f in outs: print("  ", f)

if __name__ == "__main__":
    main()

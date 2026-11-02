#!/usr/bin/env python

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE  = os.path.dirname(os.path.abspath(__file__))
CSV   = os.path.join(HERE, "s9tarc_eval_steps.csv")
JSONP = os.path.join(HERE, "s9tarc_eval.json")
FIG   = os.path.join(HERE, "figures")
GATECOL, AGENTCOL = "#c0392b", "#7f8c8d"
KEYS   = ["arc-", "axis", "arc+"]
KEYLAB = {"arc-": "arc $-20^\\circ$", "axis": "on-axis", "arc+": "arc $+20^\\circ$"}

sns.set_theme(context="paper", style="ticks", font_scale=1.3, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

def load():
    if not (os.path.exists(CSV) and os.path.exists(JSONP)):
        raise SystemExit(f"missing s9tarc_eval.json / s9tarc_eval_steps.csv in {HERE} -- run eval_s9tarc.py first")
    meta = json.load(open(JSONP))
    df = pd.read_csv(CSV)
    df["baseD"] = np.linalg.norm(df[["tx", "ty", "tz"]].to_numpy(float), axis=1)
    key = {t["idx"]: t.get("target_key", "axis") for t in meta["targets"]}
    df["key"] = df["episode"].map(key)
    pal = dict(zip(KEYS, sns.color_palette("crest", n_colors=3)))
    return meta, df, pal

def _lines(ax, df, pal, ycol, seen=None):
    seen = set() if seen is None else seen
    for ep in sorted(df["episode"].unique()):
        g = df[df["episode"] == ep]; k = g["key"].iloc[0]
        lab = KEYLAB.get(k, k) if k not in seen else None
        seen.add(k)
        ax.plot(g["step"], g[ycol], color=pal.get(k, "0.3"), alpha=0.9, label=lab)

def plot_reach(ax, meta, df, pal):
    _lines(ax, df, pal, "d")
    ax.axhline(meta["gate"]["d"], ls="--", lw=1.4, color=GATECOL, label=f"gate ({meta['gate']['d']} m)")
    if abs(meta["agent_gate"]["d"] - meta["gate"]["d"]) > 1e-9:
        ax.axhline(meta["agent_gate"]["d"], ls=":", lw=1.3, color=AGENTCOL, label=f"agent gate ({meta['agent_gate']['d']} m)")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"gripper$\to$target  $d$ (m)")
    ax.set_title("Reach: distance to target"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", ncol=2)

def plot_orient(ax, meta, df, pal):
    _lines(ax, df, pal, "phi_deg")
    ge, ga = np.degrees(meta["gate"]["phi"]), np.degrees(meta["agent_gate"]["phi"])
    ax.axhline(ge, ls="--", lw=1.4, color=GATECOL, label=f"gate ({ge:.1f}$^\\circ$)")
    if abs(ga - ge) > 1e-9:
        ax.axhline(ga, ls=":", lw=1.3, color=AGENTCOL, label=f"agent gate ({ga:.1f}$^\\circ$)")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"orientation error $\varphi$ (deg)")
    ax.set_title("Confondu: orientation error"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", loc="upper right")

def plot_basedrive(ax, meta, df, pal):
    _lines(ax, df, pal, "baseD")
    ax.axhline(0.9, ls=":", lw=1.2, color="0.4", label="arm reach $\\approx$0.9 m")
    ax.set_xlabel("decision step"); ax.set_ylabel(r"base$\to$target  (m)")
    ax.set_title("Base drives into arm reach"); ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize="small", loc="upper right")

def plot_phase(ax, meta, df, pal):
    for ep in sorted(df["episode"].unique()):
        g = df[df["episode"] == ep]; k = g["key"].iloc[0]
        ax.plot(g["d"], g["phi_deg"], color=pal.get(k, "0.3"), alpha=0.9)
        ax.scatter(g["d"].iloc[0], g["phi_deg"].iloc[0], color=pal.get(k, "0.3"), s=24, zorder=3, edgecolor="white", lw=0.6)
    ax.axvline(meta["gate"]["d"], ls="--", lw=1.2, color=GATECOL)
    ax.axhline(np.degrees(meta["gate"]["phi"]), ls="--", lw=1.2, color=GATECOL)
    ax.set_xlabel(r"distance $d$ (m)"); ax.set_ylabel(r"orientation $\varphi$ (deg)")
    ax.set_title("Joint convergence into the gate corner ($\\circ$ = spawn)")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)

def plot_speed(ax, meta, df, pal):
    tg = meta["targets"]
    idx = [t["idx"] for t in tg]
    stg = [t["steps_to_gate"] if t["steps_to_gate"] else 0 for t in tg]
    cols = [pal.get(t.get("target_key", "axis"), "0.5") for t in tg]
    ax.bar(idx, stg, color=cols, edgecolor="white", linewidth=0.7)
    for t, s in zip(tg, stg):
        ax.text(t["idx"], s, f" {s*meta['decision_dt_s']:.1f}s", ha="center", va="bottom", fontsize="x-small")
    ax.set_xlabel("episode (target: " + ", ".join(f"{t['idx']}={t.get('target_key','?')}" for t in tg) + ")")
    ax.set_ylabel("decisions to gate")
    m = meta["summary"]["mean_steps_to_gate"]
    ax.axhline(m, ls="--", lw=1.2, color=GATECOL)
    ax.set_title(f"Speed per target (mean {m:.0f} steps = {meta['summary']['mean_secs_to_gate']:.1f} s)")
    ax.set_xticks(idx); ax.set_ylim(top=max(stg) * 1.18 if max(stg) else 1)

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)

def main():
    os.makedirs(FIG, exist_ok=True)
    meta, df, pal = load()
    panels = [("reach", plot_reach), ("orient", plot_orient), ("basedrive", plot_basedrive),
              ("phase", plot_phase), ("speed", plot_speed)]
    for name, fn in panels:
        fig, ax = plt.subplots(figsize=(6.4, 4.6)); fn(ax, meta, df, pal); fig.tight_layout(); save(fig, f"s9tarc_{name}")

    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.4))
    for (name, fn), ax in zip(panels, axes.ravel()):
        fn(ax, meta, df, pal)
    axes[1, 2].axis("off")
    s = meta["summary"]
    axes[1, 2].text(0.0, 0.95, "s9_tarc — line+arc evaluation", fontsize=15, fontweight="bold", va="top")
    rows = [f"checkpoint: {meta['checkpoint']}",
            f"targets: {s['n_targets']} passes over 3 curriculum targets (2.0 m, $\\pm20^\\circ$)",
            f"reached the eval gate: {s['reached_rate']*100:.0f}%",
            f"mean min $d$: {s['mean_min_d']*1000:.1f} mm     mean min $\\varphi$: {s['mean_min_phi_deg']:.2f}$^\\circ$",
            f"mean time to gate: {s['mean_steps_to_gate']:.0f} decisions = {s['mean_secs_to_gate']:.1f} s",
            f"gate: $d<${meta['gate']['d']} m, $\\varphi<${np.degrees(meta['gate']['phi']):.2f}$^\\circ$"
            f"  (= the scene's successRadius / orientToleranceRad)",
            "the agent ENDS the episode the instant the gate fires (no hold),",
            "so a settle-hold cannot be measured for s9 — unlike the wb regimes."]
    for i, r in enumerate(rows):
        axes[1, 2].text(0.0, 0.80 - 0.095 * i, r, fontsize=11, va="top")
    fig.suptitle(f"s9_tarc — deterministic line+arc evaluation "
                 f"({s['reached_rate']*100:.0f}% reached, mean {s['mean_secs_to_gate']:.1f} s to the gate)",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); save(fig, "s9tarc_overview")

    outs = sorted(f for f in os.listdir(FIG) if f.startswith("s9tarc_") and f.endswith((".pdf", ".png")))
    print(f"wrote {len(outs)} files to {FIG}:")
    for f in outs: print("  ", f)

if __name__ == "__main__":
    main()

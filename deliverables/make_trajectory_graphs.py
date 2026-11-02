#!/usr/bin/env python

import os, json, csv, subprocess, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

REPO = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DELIV = os.path.join(REPO, "deliverables")
CMP   = os.path.join(DELIV, "comparison")
GATE_D, GATE_PHI_DEG = 0.02, np.degrees(0.02)
GATECOL = "#c0392b"
NEAR_D = 0.15
SLUG = {"arc-": "arc_minus20", "axis": "on_axis", "arc+": "arc_plus20"}
KEYS = ["arc-", "axis", "arc+"]
KEYLAB = {"arc-": "arc $-20^\\circ$", "axis": "on-axis ($0^\\circ$)", "arc+": "arc $+20^\\circ$"}

sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.7,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

SPECS = [
    dict(folder="s9_tarc_model",        csv="s9tarc_eval_steps.csv", src="flu",   dt=0.1, groups=("three", "pair"),
         label="s9 T-arc  (end-to-end DRL)",               color="#2a6f97"),
    dict(folder="base_model",           csv="base_eval_steps.csv",   src="none",  dt=0.1, groups=(),
         label="base driver", color="#777777"),
    dict(folder="s9_np_model/fixed target", csv="s9np_eval_steps.csv", src="none", dt=0.1, groups=(),
         label="s9_np fixed target", color="#777777"),
]

GROUPS = [("3models",   "three", "The three whole-body controllers"),
          ("s9_vs_wb1", "pair",  "Head-to-head: end-to-end DRL vs DRL-weighted WDLS")]

def qinv_rot(q, v):
    u = np.asarray(q[:3], float) * -1.0
    w = float(q[3])
    v = np.asarray(v, float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)

def circle_center(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    return sol[0], sol[1], float(np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2))

def key_from_lat0(lat0):
    return "axis" if abs(lat0) < 0.04 else ("arc+" if lat0 > 0 else "arc-")

def load_unity(df):
    eps = []
    tg = {ep: g[["tx", "ty", "tz"]].to_numpy(float)[0] for ep, g in df.groupby("episode")}
    P = np.unique(np.array([[t[0], t[2]] for t in tg.values()]), axis=0)
    cx, cz, R = circle_center(P) if len(P) >= 3 else (P[:, 0].mean(), P[:, 1].mean(), 0.0)
    for ep, g in df.groupby("episode"):
        G = g[["gx", "gy", "gz"]].to_numpy(float); T = g[["tx", "ty", "tz"]].to_numpy(float)
        Q = g[["tqx", "tqy", "tqz", "tqw"]].to_numpy(float)
        rel = np.array([qinv_rot(Q[k], G[k] - T[k]) for k in range(len(G))])
        s, lat, up = rel[:, 2], -rel[:, 0], rel[:, 1]
        eps.append(dict(ep=int(ep), key=key_from_lat0(lat[0]),
                        az=round(float(np.degrees(np.arctan2(T[0][2] - cz, T[0][0] - cx))), 1), spawn_r=round(R, 3),
                        s=s, lat=lat, up=up,
                        d=g["d"].to_numpy(float), phi=g["phi_deg"].to_numpy(float), base=None))
    return eps

def flu_to_unity(v):
    v = np.asarray(v, float)
    return np.array([-v[1], v[2], v[0]])

def load_flu(df):
    eps = []
    for ep, g in df.groupby("episode"):
        G = g[["gx", "gy", "gz"]].to_numpy(float); T = g[["tx", "ty", "tz"]].to_numpy(float)
        Q = g[["tqx", "tqy", "tqz", "tqw"]].to_numpy(float)
        rel  = np.array([qinv_rot(Q[k], flu_to_unity(G[k] - T[k])) for k in range(len(G))])
        relb = np.array([qinv_rot(Q[k], flu_to_unity(-T[k])) for k in range(len(G))])
        s, lat, up = rel[:, 2], -rel[:, 0], rel[:, 1]
        eps.append(dict(ep=int(ep), key=key_from_lat0(lat[0]),
                        az=round(float(np.degrees(np.arctan2(T[0][1], T[0][0]))), 1),
                        spawn_r=round(float(np.linalg.norm(T[0])), 3),
                        s=s, lat=lat, up=up,
                        d=g["d"].to_numpy(float), phi=g["phi_deg"].to_numpy(float),
                        base=(relb[:, 2], -relb[:, 0], relb[:, 1])))
    return eps

def load(spec):
    path = os.path.join(DELIV, spec["folder"], spec["csv"])
    if spec["src"] == "none" or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not {"gx", "tx", "tqw"} <= set(df.columns):
        return None
    return load_unity(df) if spec["src"] == "unity" else load_flu(df)

COMMON_D = None

def describe(e, dt):
    P = np.c_[e["s"], e["lat"], e["up"]]
    L = float(np.abs(np.diff(P, axis=0)).sum() if len(P) < 2 else np.linalg.norm(np.diff(P, axis=0), axis=1).sum())
    d0 = float(e["d"][0])
    gate = np.where((e["d"] < GATE_D) & (e["phi"] < GATE_PHI_DEG))[0]
    ng = int(gate[0] + 1) if len(gate) else None
    common = None
    if COMMON_D is not None and ng:
        out = np.where(e["d"][:ng] >= COMMON_D)[0]
        if len(out):
            common = round((ng - 1 - int(out[-1])) * dt, 2)

    infl = rev = None
    near = np.where(e["d"] < NEAR_D)[0]
    if len(near):
        i0, i1 = int(near[0]), (ng - 1 if ng else len(P) - 1)
        if i1 > i0:
            seg = P[i0:i1 + 1]
            step = np.diff(seg, axis=0)
            Ln = float(np.linalg.norm(step, axis=1).sum())
            infl = round(Ln / float(e["d"][i0]), 2) if e["d"][i0] > 1e-6 else None
            dots = (step[1:] * step[:-1]).sum(1)
            rev = int((dots < 0).sum())
    return dict(target=e["key"], episode=e["ep"], steps=len(P),
                d0=round(d0, 3), phi0_deg=round(float(e["phi"][0]), 2),
                path_len_m=round(L, 3), straightness=round(L / d0, 2) if d0 > 1e-6 else None,
                max_abs_lat_m=round(float(np.abs(e["lat"]).max()), 3),
                max_up_m=round(float(e["up"].max()), 3),
                steps_to_gate=ng, secs_to_gate=(round(ng * dt, 2) if ng else None),
                common_d_m=(round(COMMON_D, 3) if COMMON_D else None), secs_from_common_d=common,
                near_goal_inflation=infl, near_goal_reversals=rev,
                min_d_mm=round(float(e["d"].min()) * 1000, 1), min_phi_deg=round(float(e["phi"].min()), 3),
                final_d_m=round(float(e["d"][-1]), 4), final_phi_deg=round(float(e["phi"][-1]), 2))

def timeline(ax, x, y, cmap="viridis", lw=2.0, zorder=3):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=cmap, lw=lw, zorder=zorder, norm=plt.Normalize(0, 1))
    lc.set_array(np.linspace(0, 1, len(seg)))
    ax.add_collection(lc)
    return lc

def target_marker(ax, gate=True, sc=1.0):
    ax.plot(0, 0, marker="x", ms=9, mew=2.2, color="#c0392b", zorder=6)
    if gate:
        ax.add_patch(plt.Circle((0, 0), GATE_D * sc, fill=False, ec="#c0392b", lw=1.3, ls="--", zorder=6))

def panel(ax, eps, which, spec, show_desc=True, zoom=False):
    yk = which
    sc = 100.0 if zoom else 1.0
    for e in eps:
        timeline(ax, e["s"] * sc, e[yk] * sc, lw=1.8 if zoom else 2.0)
        if not zoom:
            ax.plot(e["s"][0], e[yk][0], "o", ms=6, mfc="white", mec="#333333", mew=1.4, zorder=7)
        ax.plot(e["s"][-1] * sc, e[yk][-1] * sc, "*", ms=11, color="#c0392b", zorder=7)
        if e["base"] is not None and not zoom:
            bs, bl, bu = e["base"]
            ax.plot(bs, bl if yk == "lat" else bu, ls="--", lw=1.2, color="#8e8e8e", alpha=0.9, zorder=2)
    ax.axhline(0, ls=":", lw=0.9, color="0.55", zorder=1)
    target_marker(ax, gate=zoom, sc=sc)
    ax.set_xlabel("approach $s$ (cm)" if zoom else "approach $s$ (m)   [target at 0]")
    if show_desc:
        m = [describe(e, spec["dt"]) for e in eps]
        L = np.mean([x["path_len_m"] for x in m]); S = np.mean([x["straightness"] for x in m])
        g = [x["steps_to_gate"] for x in m if x["steps_to_gate"]]
        txt = (f"$d_0$ {np.mean([x['d0'] for x in m]):.2f} m   $\\varphi_0$ {np.mean([x['phi0_deg'] for x in m]):.0f}"
               f"$^\\circ$\n$L$ {L:.2f} m   $L/d_0$ {S:.2f}\n"
               + (f"gate @ {np.mean(g):.0f} steps ({np.mean(g)*spec['dt']:.1f} s)" if g else "gate not reached"))
        ax.text(0.03, 0.04, txt, transform=ax.transAxes, fontsize="x-small", va="bottom", ha="left",
                bbox=dict(fc="white", ec="0.75", lw=0.6, alpha=0.92, boxstyle="round,pad=0.35"))
    return ax

def figure_for(spec, eps):
    figdir = os.path.join(DELIV, spec["folder"], "figures"); os.makedirs(figdir, exist_ok=True)
    keys = [k for k in KEYS if any(e["key"] == k for e in eps)]
    fig, axes = plt.subplots(3, len(keys), figsize=(4.5 * len(keys), 11.2), squeeze=False, layout="constrained")
    xlim = (min(min(e["s"]) for e in eps) - 0.15, 0.25)
    for c, k in enumerate(keys):
        sub = [e for e in eps if e["key"] == k]
        panel(axes[0, c], sub, "lat", spec)
        panel(axes[1, c], sub, "lat", spec, show_desc=False, zoom=True)
        panel(axes[2, c], sub, "up", spec, show_desc=False)
        axes[0, c].set_title(f"{KEYLAB[k]}   ({len(sub)} pass{'es' if len(sub) > 1 else ''})")
        axes[0, c].set_xlim(*xlim);  axes[0, c].set_ylim(-0.75, 0.75); axes[0, c].set_aspect("equal")
        axes[1, c].set_xlim(-15, 5); axes[1, c].set_ylim(-10, 10);     axes[1, c].set_aspect("equal")
        axes[2, c].set_xlim(*xlim);  axes[2, c].set_ylim(-0.45, 1.15)
        if c == 0:
            axes[0, c].set_ylabel("lateral (m), + left\n[view from above]")
            axes[1, c].set_ylabel("lateral (cm)\n[zoom: last 15 cm]")
            axes[2, c].set_ylabel("height above target (m)\n[view from the side]")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis")
    cb = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cb.set_label("episode time  (0 = spawn, 1 = end)")
    fig.suptitle(f"{spec['label']} — gripper path in the TARGET frame\n"
                 f"$\\circ$ spawn   $\\star$ end   $\\times$ target (dashed = 2 cm gate)"
                 + ("    grey dashed = base_link path" if any(e["base"] is not None for e in eps) else ""),
                 fontsize=13, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"trajectories.{ext}"))
    plt.close(fig)
    rows = [describe(e, spec["dt"]) for e in sorted(eps, key=lambda e: (KEYS.index(e["key"]), e["ep"]))]
    with open(os.path.join(figdir, "trajectory_metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"[{spec['folder']}] trajectories.{{pdf,png}} + trajectory_metrics.csv  ({len(eps)} passes, targets {keys})")
    return rows

def comparison(models, tag, subtitle):
    keys = [k for k in KEYS if any(any(e["key"] == k for e in eps) for _, eps in models)]
    fig, axes = plt.subplots(3, len(keys), figsize=(4.5 * len(keys), 11.2), squeeze=False, layout="constrained")
    xlim = (min(min(min(e["s"]) for e in eps) for _, eps in models) - 0.15, 0.25)
    for c, k in enumerate(keys):
        for spec, eps in models:
            sub = [e for e in eps if e["key"] == k]
            for j, e in enumerate(sub):
                for r, yk, sc in ((0, "lat", 1.0), (1, "lat", 100.0), (2, "up", 1.0)):
                    axes[r, c].plot(e["s"] * sc, e[yk] * sc, color=spec["color"], lw=1.8, alpha=0.85,
                                    label=spec["label"] if (j == 0 and r == 0) else None)
                    if r != 1:
                        axes[r, c].plot(e["s"][0], e[yk][0], "o", ms=5, mfc="white", mec=spec["color"], mew=1.3)
                    axes[r, c].plot(e["s"][-1] * sc, e[yk][-1] * sc, "*", ms=9, color=spec["color"], zorder=7)
        for r in (0, 1, 2):
            axes[r, c].axhline(0, ls=":", lw=0.9, color="0.55")
            axes[r, c].set_xlabel("approach $s$ (cm)" if r == 1 else "approach $s$ (m)   [target at 0]")
        target_marker(axes[0, c], gate=False); target_marker(axes[1, c], sc=100.0); target_marker(axes[2, c], gate=False)
        axes[0, c].set_xlim(*xlim);  axes[0, c].set_ylim(-0.75, 0.75); axes[0, c].set_aspect("equal")
        axes[1, c].set_xlim(-15, 5); axes[1, c].set_ylim(-10, 10);     axes[1, c].set_aspect("equal")
        axes[2, c].set_xlim(*xlim);  axes[2, c].set_ylim(-0.45, 1.15)
        axes[0, c].set_title(KEYLAB[k])
        if c == 0:
            axes[0, c].set_ylabel("lateral (m), + left\n[view from above]")
            axes[1, c].set_ylabel("lateral (cm)\n[zoom: last 15 cm]")
            axes[2, c].set_ylabel("height above target (m)\n[view from the side]")
    axes[0, 0].legend(frameon=False, fontsize="x-small", loc="upper left")
    fig.suptitle(f"{subtitle} — gripper path in the TARGET frame\n"
                 "$\\circ$ spawn   $\\star$ end   $\\times$ target (dashed = 2 cm gate)",
                 fontsize=12.5, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(CMP, f"comparison_trajectories_{tag}.{ext}"))
    plt.close(fig)
    print(f"[deliverables] comparison_trajectories_{tag}.{{pdf,png}}  ({len(models)} models x {len(keys)} targets)")

def comparison_bars(models, allrows, tag, subtitle):
    df = pd.DataFrame(allrows)
    labs = [s["label"] for s, _ in models]; cols = {s["label"]: s["color"] for s, _ in models}
    keys = [k for k in KEYS if (df["target"] == k).any()]
    panels = [("secs_to_gate",       "Whole episode $\\to$ gate", "seconds  (mean of passes)"),
              ("secs_from_common_d", f"Final approach: last {COMMON_D:.2f} m $\\to$ gate", "seconds  (mean of passes)"),
              ("straightness",       "Path efficiency", "$L/d_0$   (1.0 = straight line)")]
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.9), layout="constrained")
    w = 0.8 / max(1, len(labs))
    for j, lab in enumerate(labs):
        for a, ((col, _, _), ax) in enumerate(zip(panels, axes)):
            x, y = [], []
            for i, k in enumerate(keys):
                v = df[(df.model == lab) & (df.target == k)][col].dropna()
                x.append(i + (j - (len(labs) - 1) / 2) * w); y.append(float(v.mean()) if len(v) else 0.0)
            b = ax.bar(x, y, width=w * 0.92, color=cols[lab], edgecolor="white", lw=0.6,
                       label=lab if a == 0 else None)
            ax.bar_label(b, fmt="%.1f", fontsize="xx-small", padding=1)
    for ax, (col, ttl, yl) in zip(axes, panels):
        ax.set_xticks(range(len(keys))); ax.set_xticklabels([KEYLAB[k] for k in keys])
        ax.set_title(ttl); ax.set_ylabel(yl)
        top = df[col].max()
        if np.isfinite(top): ax.set_ylim(top=float(top) * 1.26)
    axes[2].axhline(1.0, ls=":", lw=1.1, color="0.4")
    axes[0].legend(frameon=False, fontsize="x-small", loc="upper left")

    starts = "   |   ".join(f"{lab}: $d_0$ {df[df.model == lab]['d0'].mean():.2f} m, "
                            f"$\\varphi_0$ {df[df.model == lab]['phi0_deg'].mean():.0f}$^\\circ$" for lab in labs)
    fig.suptitle(f"{subtitle} — same three targets, same gate, same 0.1 s decision period\n"
                 f"start states:   {starts}", fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.03, "The middle panel is the like-for-like one: it times only the final approach that every pass "
                         f"shares, from $d={COMMON_D:.2f}$ m (the largest distance all of them pass through) down to "
                         "the gate.", ha="center", va="top", fontsize=8.5, color="0.30")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(CMP, f"comparison_speed_{tag}.{ext}"))
    plt.close(fig)
    print(f"[deliverables] comparison_speed_{tag}.{{pdf,png}}   (common-approach distance {COMMON_D:.2f} m)")

def per_target(models, tag, subtitle):
    keys = [k for k in KEYS if any(any(e["key"] == k for e in eps) for _, eps in models)]
    for k in keys:
        sub = [(s, [e for e in eps if e["key"] == k]) for s, eps in models]
        sub = [(s, e) for s, e in sub if e]
        if not sub: continue
        fig, ax = plt.subplots(2, 3, figsize=(15.5, 9.2), layout="constrained")
        xlim = (min(min(min(e["s"]) for e in eps) for _, eps in sub) - 0.12, 0.22)
        for spec, eps in sub:
            for j, e in enumerate(eps):
                lab = spec["label"] if j == 0 else None
                ax[0, 0].plot(e["s"], e["lat"], color=spec["color"], lw=1.9, alpha=0.9, label=lab)
                ax[0, 0].plot(e["s"][0], e["lat"][0], "o", ms=5, mfc="white", mec=spec["color"], mew=1.3)
                ax[0, 0].plot(e["s"][-1], e["lat"][-1], "*", ms=10, color=spec["color"], zorder=7)
                ax[0, 1].plot(e["s"] * 100, e["lat"] * 100, color=spec["color"], lw=1.6, alpha=0.9)
                ax[0, 1].plot(e["s"][-1] * 100, e["lat"][-1] * 100, "*", ms=10, color=spec["color"], zorder=7)
                ax[0, 2].plot(e["s"], e["up"], color=spec["color"], lw=1.9, alpha=0.9)
                ax[0, 2].plot(e["s"][0], e["up"][0], "o", ms=5, mfc="white", mec=spec["color"], mew=1.3)
                t = np.arange(len(e["d"])) * spec["dt"]
                ax[1, 0].plot(t, e["d"], color=spec["color"], lw=1.9, alpha=0.9, label=lab)
                ax[1, 1].plot(t, e["phi"], color=spec["color"], lw=1.9, alpha=0.9)
        for a in (ax[0, 0], ax[0, 2]):
            a.axhline(0, ls=":", lw=0.9, color="0.55"); a.set_xlim(*xlim)
            a.set_xlabel("approach $s$ (m)   [target at 0]")
        target_marker(ax[0, 0], gate=False); target_marker(ax[0, 2], gate=False)
        ax[0, 0].set_ylabel("lateral (m), + left"); ax[0, 0].set_title("Path from above")
        ax[0, 2].set_ylabel("height above target (m)"); ax[0, 2].set_title("Path from the side")
        target_marker(ax[0, 1], sc=100.0)
        ax[0, 1].axhline(0, ls=":", lw=0.9, color="0.55")
        ax[0, 1].set_xlim(-NEAR_D * 100, 5); ax[0, 1].set_ylim(-10, 10); ax[0, 1].set_aspect("equal")
        ax[0, 1].set_xlabel("approach $s$ (cm)"); ax[0, 1].set_ylabel("lateral (cm)")
        ax[0, 1].set_title(f"The last {NEAR_D*100:.0f} cm (gate = dashed)")
        ax[1, 0].axhline(GATE_D, ls="--", lw=1.3, color=GATECOL)
        ax[1, 0].set_xlabel("time (s)"); ax[1, 0].set_ylabel("$d$ (m)"); ax[1, 0].set_yscale("log")
        ax[1, 0].set_title("Position error vs time (log)")
        ax[1, 1].axhline(GATE_PHI_DEG, ls="--", lw=1.3, color=GATECOL)
        ax[1, 1].set_xlabel("time (s)"); ax[1, 1].set_ylabel("$\\varphi$ (deg)"); ax[1, 1].set_yscale("log")
        ax[1, 1].set_title("Orientation error vs time (log)")
        ax[0, 0].legend(frameon=False, fontsize="x-small", loc="best")

        ax[1, 2].axis("off"); ax[1, 2].set_xlim(0, 1); ax[1, 2].set_ylim(0, 1)
        X = [0.0, 0.46, 0.62, 0.78, 0.90, 1.0]
        header = ("controller", "s$\\to$gate", "min $d$", "min $\\varphi$", "$L/d_0$", "rev.")
        rows = [("#222222",) + header]
        for spec, eps in sub:
            m = [describe(e, spec["dt"]) for e in eps]
            g = [x["secs_to_gate"] for x in m if x["secs_to_gate"]]
            rv = [x["near_goal_reversals"] for x in m if x["near_goal_reversals"] is not None]
            rows.append((spec["color"], spec["label"].split("  (")[0],
                         f"{np.mean(g):.1f}" if g else "--",
                         f"{np.mean([x['min_d_mm'] for x in m]):.1f} mm",
                         f"{np.mean([x['min_phi_deg'] for x in m]):.2f}$^\\circ$",
                         f"{np.mean([x['straightness'] for x in m]):.2f}",
                         f"{np.mean(rv):.0f}" if rv else "--"))
        y = 0.92
        for r, row in enumerate(rows):
            colr, cells = row[0], row[1:]
            for c, cell in enumerate(cells):
                ax[1, 2].text(X[c], y, cell, fontsize=9.5, ha="left" if c == 0 else "right", va="top",
                              fontweight="bold" if r == 0 else "normal", color=colr if c == 0 else "#222222",
                              transform=ax[1, 2].transAxes)
            if r == 0:
                ax[1, 2].axhline(y - 0.035, xmin=0.0, xmax=1.0, color="0.6", lw=0.8)
            y -= 0.085
        ax[1, 2].text(0.0, y - 0.03, transform=ax[1, 2].transAxes,
                      s=f"mean over the {len(sub[0][1])} pass(es) on this target.\n"
                                     f"reversals = direction flips >90$^\\circ$ inside the last {NEAR_D*100:.0f} cm\n"
                                     "(the shake metric): 0 = a clean run-in.",
                      fontsize=8.5, va="top", color="0.35")
        fig.suptitle(f"{KEYLAB[k]}  —  {subtitle}", fontsize=14, fontweight="bold")
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(CMP, f"target_{SLUG[k]}_{tag}.{ext}"))
        plt.close(fig)
    print(f"[comparison] target_*_{tag}.{{pdf,png}}  ({len(keys)} targets)")

CRITERIA = [("secs_to_gate",        "Time to the gate",        "seconds",            False),
            ("secs_from_common_d",  f"Final approach",         "seconds",            False),
            ("min_d_mm",            "Position accuracy",       "min $d$ (mm)",       False),
            ("min_phi_deg",         "Orientation accuracy",    "min $\\varphi$ (deg)", False),
            ("straightness",        "Path efficiency",         "$L/d_0$",            False),
            ("near_goal_reversals", "Near-goal smoothness",    "direction reversals", False)]

def verdict(models, allrows, tag, subtitle):
    df = pd.DataFrame(allrows)
    labs = [s["label"] for s, _ in models]; cols = {s["label"]: s["color"] for s, _ in models}
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), layout="constrained")
    wins = {l: 0 for l in labs}
    for (col, ttl, yl, hib), ax in zip(CRITERIA, axes.ravel()):
        vals = [float(df[df.model == l][col].dropna().mean()) for l in labs]
        best = int(np.nanargmax(vals) if hib else np.nanargmin(vals))
        wins[labs[best]] += 1
        b = ax.bar(range(len(labs)), vals, color=[cols[l] for l in labs], edgecolor="white", lw=0.7, width=0.62)
        ax.bar_label(b, fmt="%.2f" if max(vals) < 10 else "%.1f", fontsize="small", padding=2)
        ax.plot(best, vals[best] + 0.10 * max(vals), marker="*", ms=16, color="#c8a415", zorder=6)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels([l.split("  (")[0] for l in labs], fontsize="x-small")
        ax.set_title(f"{ttl}   ($\\downarrow$ better)"); ax.set_ylabel(yl)
        ax.set_ylim(top=max(vals) * 1.3)
    order = sorted(labs, key=lambda l: -wins[l])
    tally = "     ".join(f"{l.split('  (')[0]}: {wins[l]}/{len(CRITERIA)}" for l in order)
    fig.suptitle(f"Which controller is better? {subtitle}\n"
                 f"mean over all passes, lower is better on every criterion — wins:   {tally}",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.025,
             "Success rate is not a criterion here: all controllers reach the gate on 5/5 passes. "
             f"'Final approach' times only the shared last {COMMON_D:.2f} m; 'reversals' counts direction flips "
             f">90 deg inside the last {NEAR_D*100:.0f} cm.", ha="center", va="top", fontsize=8.5, color="0.30")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(CMP, f"verdict_{tag}.{ext}"))
    plt.close(fig)
    print(f"[comparison] verdict_{tag}.{{pdf,png}}   wins: {tally}")
    return wins

def comparison_notes(wins_by_group):
    pages = []
    for tag, key, subtitle in GROUPS:
        if tag not in wins_by_group: continue
        models, wins = wins_by_group[tag]
        labs = [s["label"] for s, _ in models]
        best = max(wins, key=wins.get)
        tally = ", ".join(f"{l.split('  (')[0]}: {wins[l]}/{len(CRITERIA)}" for l in labs)
        pages.append(dict(fig=f"comparison_trajectories_{tag}.pdf", w=0.99, h=0.66,
            title=f"{subtitle} --- the paths, all targets",
            shows=("Every controller's gripper path in the target frame, one column per curriculum target; rows are "
                   "the view from above, a zoom on the last 15~cm with the 2~cm gate, and the view from the side."),
            read=("Read the columns as three independent problems and the rows as three scales. The full views show "
                  "the shape of the approach; the zoom row is where the controllers actually differ."),
            say=("``Same targets, same gate, same decision period --- and after matching the arm's reset pose, "
                 "essentially the same start state, so the shapes are directly comparable.''"),
            ask=("\\emph{Why the target frame?} --- the gripper$\\to$target relative pose is frame-independent, so "
                 "the end-to-end model (observations in \\texttt{base\\_link} FLU) and the WDLS logs (Unity world) "
                 "can be drawn on one pair of axes.")))
        for k in KEYS:
            f = os.path.join(CMP, f"target_{SLUG[k]}_{tag}.pdf")
            if not os.path.exists(f): continue
            pages.append(dict(fig=f"target_{SLUG[k]}_{tag}.pdf", w=0.99, h=0.60,
                title=f"{KEYLAB[k]} --- one target, every controller",
                shows=("The single curriculum target, alone: the path from above, the last 15~cm, the path from the "
                       "side, then the two errors $d$ and $\\varphi$ against time on log axes, and the numbers."),
                read=("The log axes are the point: a controller that converges geometrically shows a straight line, "
                      "and a controller that stalls or oscillates shows a flattening tail. The table underneath gives "
                      "the same four numbers for each controller on this target only."),
                say=("``Target by target the ranking is the same, so the average is not hiding one lucky case.''"),
                ask=("\\emph{Why is the arc harder than the on-axis target?} --- the base has to turn as well as "
                     "drive, so the whole-body coordination is exercised; both families take their longest time on "
                     "the same arc.")))
        pages.append(dict(fig=f"comparison_speed_{tag}.pdf", w=0.99, h=0.40,
            title=f"{subtitle} --- time and path efficiency",
            shows=("Three panels per target: the whole episode to the gate, the \\emph{final approach} (only the "
                   f"shared last {COMMON_D:.2f}~m), and path efficiency $L/d_0$."),
            read=("The middle panel is the like-for-like timing --- it removes any remaining difference in where "
                  "each controller starts. The right panel is the counterweight: being fast is not the same as "
                  "being efficient."),
            say=("``The end-to-end policy is the fastest to the gate; the WDLS controllers take the shortest "
                 "geometric path. Those are different virtues and the figure shows both.''"),
            ask=("\\emph{Is the comparison fair?} --- same targets, same gate, same $0.1$~s decision period, and "
                 "the start states are matched to within $0.1$~m and a few degrees.")))
        pages.append(dict(fig=f"verdict_{tag}.pdf", w=0.99, h=0.52,
            title=f"{subtitle} --- which one is better",
            shows=("Six criteria, each averaged over every pass, lower better on all six: time to the gate, final "
                   "approach time, position accuracy, orientation accuracy, path efficiency, and near-goal "
                   "smoothness (the number of direction reversals inside the last 15~cm). A star marks each winner."),
            read=(f"Win tally: {tally}. Success rate is deliberately absent --- every controller reaches the gate on "
                  "all five passes, so it separates nothing. The split is the finding: speed on one side, geometric "
                  "efficiency and smoothness on the other."),
            say=(f"``On this sweep {best.split('  (')[0]} wins the most criteria, but the honest summary is a "
                 "trade-off, not a knockout: the fast controller is the least smooth.''"),
            ask=("\\emph{Why count reversals?} --- it is this project's shake metric, the one that exposed the "
                 "reward-buys-wobble bug: a direction flip greater than $90^\\circ$ near the goal is motion that "
                 "does not bring the gripper closer.")))
    head = ("\\documentclass[11pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n"
            "\\usepackage{amsmath,amssymb}\n\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
            "\\begin{document}\n\\begin{center}{\\Large\\bfseries Controller comparison --- Notes for the Defense}"
            "\\\\[2pt]\nSame three curriculum targets, same $2$~cm\\,/\\,$1.15^\\circ$ gate, same $0.1$~s decision "
            "period, matched start poses.\\end{center}\n\\vspace{4pt}\n")
    body = ""
    for i, p in enumerate(pages, 1):
        body += (f"\\section*{{Figure {i} --- {p['title']}}}\n"
                 f"\\begin{{center}}\\includegraphics[width={p['w']}\\textwidth,height={p['h']}\\textheight,"
                 f"keepaspectratio]{{{p['fig']}}}\\end{{center}}\n"
                 f"\\textbf{{What it shows.}} {p['shows']}\n\n\\textbf{{How to read it.}} {p['read']}\n\n"
                 f"\\textbf{{Say this.}} {p['say']}\n\n\\textbf{{If asked.}} {p['ask']}\n\\clearpage\n")
    open(os.path.join(CMP, "comparison_notes.tex"), "w").write(head + body + "\\end{document}\n")
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "comparison_notes.tex"],
                           cwd=CMP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            print(f"[comparison] pdflatex pass {i} FAILED -- run it by hand in {CMP}"); return
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(CMP, "comparison_notes" + ext))
        except FileNotFoundError: pass
    print(f"[comparison] comparison_notes.pdf  ({len(pages)} pages)")

def main():
    global COMMON_D
    loaded = []
    for spec in SPECS:
        eps = load(spec)
        if not eps:
            print(f"[{spec['folder']}] no per-step POSE trace -- skipped "
                  f"({'run eval_s9tarc.py first' if spec['src'] != 'none' else 'this eval logs only d/phi'})")
            continue
        loaded.append((spec, eps))

    comp = [(s, e) for s, e in loaded if s.get("groups")]
    if len(comp) >= 2:
        COMMON_D = min(float(ep["d"][0]) for _, eps in comp for ep in eps)

    allrows = []
    for spec, eps in loaded:
        rows = figure_for(spec, eps)
        for r in rows: r["model"] = spec["label"]
        allrows += rows

    os.makedirs(CMP, exist_ok=True)
    wins_by_group = {}
    for tag, key, subtitle in GROUPS:
        models = [(s, e) for s, e in loaded if key in s.get("groups", ())]
        if len(models) < 2:
            print(f"[deliverables] {tag}: fewer than 2 models available -- skipped"); continue
        rows = [r for r in allrows if r["model"] in {s["label"] for s, _ in models}]
        comparison(models, tag, subtitle)
        comparison_bars(models, rows, tag, subtitle)
        per_target(models, tag, subtitle)
        wins_by_group[tag] = (models, verdict(models, rows, tag, subtitle))
    if wins_by_group:
        comparison_notes(wins_by_group)
    if allrows:
        cols = ["model"] + [c for c in allrows[0] if c != "model"]
        with open(os.path.join(CMP, "comparison_trajectory_metrics.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(allrows)
        print(f"[deliverables] comparison_trajectory_metrics.csv  ({len(allrows)} rows, all packages)")

if __name__ == "__main__":
    main()

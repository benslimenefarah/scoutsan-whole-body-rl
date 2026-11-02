#!/usr/bin/env python3

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/nyctophilia/UnityProjects/RL_Scoutsan/python/motor_log.csv"

def load(path):
    step, phase = [], []
    cmd, got, de, phi, vl, bv, bom = [], [], [], [], [], [], []
    with open(path) as f:
        f.readline()
        for ln in f:
            r = ln.rstrip("\n").split(",")
            if len(r) < 17: continue
            step.append(int(r[0])); phase.append(r[1])
            cmd.append([float(r[2]), float(r[3]), float(r[4])])
            got.append([float(r[6]), float(r[7]), float(r[8])])
            de.append(float(r[9])); phi.append(float(r[10]))
            vl.append([float(r[11]), float(r[12]), float(r[13])])
            bv.append(float(r[17]) if len(r) > 17 else 0.0)
            bom.append(float(r[18]) if len(r) > 18 else 0.0)
    return (np.array(step), phase, np.array(cmd), np.array(got),
            np.array(de), np.array(phi), np.array(vl), np.array(bv), np.array(bom))

def phase_spans(step, phase):
    spans, s0 = [], 0
    for i in range(1, len(phase)):
        if phase[i] != phase[i-1]:
            spans.append((step[s0], step[i-1], phase[i-1])); s0 = i
    if phase: spans.append((step[s0], step[-1], phase[-1]))
    return spans

def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"[plot] not found: {CSV}  (set motorCsvPath on the agent + run the motor test)")
    step, phase, cmd, got, de, phi, vl, bv, bom = load(CSV)
    if step.size == 0:
        raise SystemExit(f"[plot] no rows in {CSV} yet")
    print(f"[plot] {step.size} decisions, steps {step.min()}..{step.max()}  |basev|max={np.abs(bv).max():.3f} |baseom|max={np.abs(bom).max():.3f}")
    spans = phase_spans(step, phase)

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))
    def shade(a):
        for (s0, s1, nm) in spans:
            a.axvspan(s0, s1, color="k", alpha=0.03)
            a.text((s0+s1)/2, a.get_ylim()[1], nm, ha="center", va="top", fontsize=7, alpha=.6)

    names, cols = ["x", "y", "z"], ["tab:red", "tab:green", "tab:blue"]
    for i in range(3):
        ax[0, 0].plot(step, cmd[:, i], "--", c=cols[i], lw=1.2, label=f"cmd {names[i]}")
        ax[0, 0].plot(step, got[:, i], "-",  c=cols[i], lw=1.0, label=f"got {names[i]}")
    ax[0, 0].set_title("commanded (dashed) vs achieved (solid) per axis  [base-drive: x=fwd, z=lateral]")
    ax[0, 0].set_xlabel("decision"); ax[0, 0].set_ylabel("m"); ax[0, 0].legend(fontsize=7, ncol=3); ax[0, 0].grid(alpha=.3)

    ax[0, 1].plot(step, de, c="k", lw=1.0, label="track d (m)")
    ax2 = ax[0, 1].twinx(); ax2.plot(step, phi, c="tab:orange", lw=1.0, label="track phi (deg)")
    ax[0, 1].set_title("tracking error (gripper vs target) -- should fall to ~0 in each hold")
    ax[0, 1].set_xlabel("decision"); ax[0, 1].set_ylabel("d (m)"); ax2.set_ylabel("phi (deg)")
    ax[0, 1].legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8); ax[0, 1].grid(alpha=.3)

    for i in range(3):
        ax[1, 0].plot(step, vl[:, i], c=cols[i], lw=1.0, label=f"vEff_lin {names[i]}")
    ax[1, 0].set_title("PBVS linear error command v_eff"); ax[1, 0].set_xlabel("decision")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

    ax[1, 1].plot(step, bv,  c="tab:purple", lw=1.2, label="base v (m/s)")
    ax[1, 1].plot(step, bom, c="tab:brown",  lw=1.2, label="base om (rad/s)")
    ax[1, 1].axhline(0, c="k", lw=.6)
    ax[1, 1].set_title("COMMANDED BASE MOTION -- non-zero => the wheels drive / the base yaws")
    ax[1, 1].set_xlabel("decision"); ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=.3)
    for a in ax.flat: shade(a)

    out = os.path.splitext(CSV)[0] + "_motor.png"
    fig.tight_layout(); fig.savefig(out, dpi=110); print(f"[plot] wrote {out}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, FancyArrowPatch
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.patches import FancyArrowPatch as _FAP
import seaborn as sns

REPO  = os.environ.get("RL_SCOUTSAN_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT   = os.path.join(REPO, "deliverables", "common")

R_MIN, R_LINE = 0.75, 2.0
ARC_TOTAL     = 40.0
HEIGHT        = 0.25
ARM_REACH     = 0.9
GATE_D, GATE_PHI = 0.02, 0.02
GRIPPER_HOME  = (0.90, 0.52)
N_DIST_STEPS, N_ARC_STEPS = 500_000, 1_000_000

AX_COL = ("#d62728", "#2ca02c", "#1f77b4")

sns.set_theme(context="paper", style="ticks", font_scale=1.15, rc={
    "axes.linewidth": 0.9, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "mathtext.fontset": "cm", "axes.titleweight": "semibold",
})

def polar(r, az_deg):
    a = np.radians(az_deg)
    return r * np.sin(a), r * np.cos(a)

def frame_arrows_2d(ax, x, y, az_deg, L=0.22):
    a = np.radians(az_deg)
    zx, zy = np.sin(a) * L, np.cos(a) * L
    xx, xy = np.cos(a) * L * 0.7, -np.sin(a) * L * 0.7
    ax.arrow(x, y, zx, zy, head_width=0.045, head_length=0.055, fc=AX_COL[2], ec=AX_COL[2], lw=1.4, zorder=6)
    ax.arrow(x, y, xx, xy, head_width=0.045, head_length=0.055, fc=AX_COL[0], ec=AX_COL[0], lw=1.4, zorder=6)

def panel_top(ax):
    half = ARC_TOTAL / 2.0

    aa = np.linspace(-half, half, 200)
    xa, ya = polar(R_LINE, aa)
    ax.plot(xa, ya, color="#1f77b4", lw=3.4, solid_capstyle="round", zorder=4,
            label=f"phase 2 — ARC at {R_LINE:g} m,  $\\pm${half:g}$^\\circ$")
    ax.plot([0, 0], [R_MIN, R_LINE], color="#2e7d32", lw=3.4, solid_capstyle="round", zorder=4,
            label=f"phase 1 — on-axis LINE, {R_MIN:g} $\\to$ {R_LINE:g} m")
    t = np.linspace(-np.pi, np.pi, 200)
    for r in (R_MIN, R_LINE):
        ax.plot(r * np.sin(t), r * np.cos(t), ls=":", lw=0.9, color="0.72", zorder=1)
    ax.plot(ARM_REACH * np.sin(t), ARM_REACH * np.cos(t), ls="--", lw=1.2, color="#b5651d", zorder=2)
    ax.text(-ARM_REACH - 0.06, 0.42, f"arm reach\n$\\approx${ARM_REACH:g} m", color="#b5651d",
            fontsize="xx-small", ha="right", va="center")

    for az, lab, dx in ((-half, f"arc $-${half:g}$^\\circ$", -34), (0.0, "on-axis", 0),
                        (half, f"arc $+${half:g}$^\\circ$", 34)):
        x, y = polar(R_LINE, az)
        ax.plot(x, y, "o", ms=9, mfc="white", mec="#c0392b", mew=2.0, zorder=7)
        frame_arrows_2d(ax, x, y, az)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(dx, 16), ha="center",
                    fontsize="small", color="#c0392b")
    ax.annotate("the on-axis target is also\nthe s9_np FIXED target\n(targetSpawnOffset z=2, y=0.25)",
                polar(R_LINE, 0), textcoords="offset points", xytext=(-150, -58), ha="left", va="center",
                fontsize="xx-small", color="0.35",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="0.6",
                                connectionstyle="angle,angleA=0,angleB=-90,rad=4"))

    ax.add_patch(Rectangle((-0.35, -0.46), 0.70, 0.93, fc="#e8e8e8", ec="#444444", lw=1.2, zorder=5))
    ax.arrow(0, 0.47, 0, 0.30, head_width=0.09, head_length=0.10, fc="#444444", ec="#444444", lw=1.6, zorder=6)
    ax.plot(0, 0, "x", ms=8, mew=2, color="#444444", zorder=7)
    ax.text(0, -0.98, "robot start pose\n(spawn centre, heading $\\uparrow$)", ha="center", va="top",
            fontsize="x-small", color="#222222")

    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.15, 2.55); ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)   [robot's right $+$]"); ax.set_ylabel("forward (m)   [robot's heading]")
    ax.set_title("Top view — where a target can appear")
    ax.legend(frameon=False, fontsize="xx-small", loc="upper left", borderaxespad=0.3)
    ax.text(-1.68, -1.08, "the envelope is an L, not a fan:\nclose targets stay on-axis,\n"
                          f"wide angles exist only at {R_LINE:g} m\n"
                          f"\n{R_LINE:g} m out vs $\\approx${ARM_REACH:g} m of arm\n"
                          f"$\\Rightarrow$ the base must drive $\\approx${R_LINE-ARM_REACH:.1f} m",
            fontsize="xx-small", va="bottom", color="#1f4e79",
            bbox=dict(fc="#eaf2fb", ec="#b8d2ea", lw=0.6, boxstyle="round,pad=0.35"))

def fig_spawn():
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 7.4), layout="constrained")
    panel_top(ax)
    fig.suptitle("The T-arc target curriculum — the spawn envelope shared by every whole-body model\n"
                 f"minRadius {R_MIN:g} m, lineRadius {R_LINE:g} m, arcTotalDeg {ARC_TOTAL:g} "
                 f"($\\pm${ARC_TOTAL/2:g}$^\\circ$), height {HEIGHT:g} m, zAwayFromRobot on",
                 fontsize=12.5, fontweight="bold")
    fig.text(0.5, -0.015,
             f"Training ramp (s9_tarc): the radius grows over the first {N_DIST_STEPS:,} Academy steps,\n"
             f"then the arc opens over the next {N_ARC_STEPS:,}. Evaluation uses the deterministic\n"
             "set (count 4 / lineN 1) = the three red targets. Each model's observation origin\n"
             "(base_link) sits at its own offset from the spawn centre, which is why the per-model\n"
             "JSON spawn_dist / spawn_az read differently for the same physical target.",
             ha="center", va="top", fontsize=8.0, color="0.30")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"target_spawn_area.{ext}"))
    plt.close(fig)
    print(f"[common] target_spawn_area.{{pdf,png}}")

def euler_unity(x, y, z):
    rx, ry, rz = np.radians([x, y, z])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Ry @ Rx @ Rz

def triad(ax, origin, R, L=0.30, lw=2.6, labels=None, alpha=1.0):
    o = np.asarray(origin, float)
    for k in range(3):
        v = R[:, k] * L
        ax.quiver(o[0], o[1], o[2], v[0], v[1], v[2], color=AX_COL[k], lw=lw, arrow_length_ratio=0.18, alpha=alpha)
        if labels:
            ax.text(*(o + v * 1.18), labels[k], color=AX_COL[k], fontsize=8.5, ha="center", va="center")
    ax.scatter(*o, s=26, color="#222222", depthshade=False)

def setup3d(ax, title):
    ax.set_title(title, fontsize=11.5, pad=-2)
    ax.set_xlim(-0.72, 0.52); ax.set_ylim(-0.52, 0.72); ax.set_zlim(-0.32, 0.52)
    ax.set_box_aspect((1, 1, 0.78), zoom=1.28)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=22, azim=-58)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.03); pane.line.set_lw(0.0)
    ax.grid(False)

def fig_frames():
    fig = plt.figure(figsize=(14.0, 4.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.20])
    axA = fig.add_subplot(gs[0, 0], projection="3d")
    axB = fig.add_subplot(gs[0, 1], projection="3d")
    axC = fig.add_subplot(gs[0, 2]); axC.axis("off")

    Rt = np.eye(3)
    Rg = euler_unity(-18, 34, 12)
    pg = np.array([-0.52, 0.30, 0.26])
    L = 0.26

    setup3d(axA, "before: two separate frames")
    triad(axA, (0, 0, 0), Rt, L=L, labels=("$X_T$", "$Y_T$", "$Z_T$"))
    triad(axA, pg, Rg, L=L, labels=("$X_G$", "$Y_G$", "$Z_G$"))
    axA.plot(*zip(pg, (0, 0, 0)), ls="--", lw=1.4, color="#c0392b")
    mid = pg / 2
    axA.text(mid[0] - 0.04, mid[1] - 0.22, mid[2] - 0.12, "$d=\\|p_G-p_T\\|$", color="#c0392b", fontsize=9.5)
    axA.text(-0.70, 0.46, -0.30, "$\\varphi$ = the angle between\nthe two orientations",
             color="#7b3294", fontsize=9)

    setup3d(axB, "confondu: the frames coincide")
    triad(axB, (0, 0, 0), Rt, L=L, labels=("$X$", "$Y$", "$Z$"))
    triad(axB, (0.004, 0.004, 0.004), Rt, L=L, lw=5.4, alpha=0.32)
    axB.text(-0.10, 0.10, -0.28, f"$d<{GATE_D}$ m   and   $\\varphi<{GATE_PHI}$ rad "
                                 f"$({np.degrees(GATE_PHI):.2f}^\\circ)$",
             color="#2e7d32", fontsize=10, ha="center")

    txt = (
        r"$\bf{The\ gripper\ frame}$ (built in code, not a Transform):" "\n"
        r"    $q_G = R_{\mathrm{fingerLink1}}\cdot \mathrm{Euler}(-90,\,90,\,0)$" "\n"
        r"    $p_G = \frac{1}{2}(p_{f_1}+p_{f_2}) + q_G\cdot(0,0,0.02)$" "\n"
        "    (the grasp centre between the two fingers, pushed 2 cm along the\n"
        "     gripper's local $+Z$; the Euler offset is the Inspector value)\n\n"
        r"$\bf{The\ target\ frame}$:" "\n"
        r"    $q_T = R_{\mathrm{target}}\cdot \mathrm{Euler}(0,0,0)$,   $p_T = p_{\mathrm{target}}$" "\n"
        "    (targetFrameOffsetEuler is zero: the curriculum already authors\n"
        "     the target with $+Z$ pointing away from the robot)\n\n"
        r"$\bf{The\ two\ errors\ the\ gate\ is\ made\ of}$:" "\n"
        r"    $d = \|p_G - p_T\|$" "\n"
        r"    $\varphi = 2\arccos\,|\langle q_G, q_T\rangle| \;\in[0,\pi]$" "\n"
        "    the $|\\cdot|$ is the double-cover fix ($q$ and $-q$ are the same\n"
        "    rotation); $\\varphi$ is the geodesic angle, so it is frame-independent\n\n"
        r"$\bf{What\ the\ policy\ sees}$ (14 numbers, base_link FLU):" "\n"
        "    [0:3] $p_G$   [3:7] $q_G$   [7:10] $p_T$   [10:14] $q_T$\n"
        "    quaternions canonicalised $w>0$ before being observed\n\n"
        r"$\bf{Confondu}$ = both errors inside the gate at the same time." "\n"
        "    Position alone is not the task: the wrist must arrive aligned."
    )
    axC.text(0.0, 1.0, txt, va="top", ha="left", fontsize=9.2, linespacing=1.45, family="serif")

    fig.suptitle("Confondu — the gripper frame and the target frame — the definition every model is scored on",
                 fontsize=13.5, fontweight="bold")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"frames_confondu.{ext}"))
    plt.close(fig)
    print(f"[common] frames_confondu.{{pdf,png}}")

NOTES = [
    dict(fig="target_spawn_area.pdf", title="The T-arc spawn envelope and the robot start pose", w=0.72, h=0.62,
         shows=("Where a target can be created, relative to the robot's start pose, seen from above. The green segment "
                f"is the on-axis line ({R_MIN:g}$\\to${R_LINE:g} m), the blue arc is the far arc at {R_LINE:g} m "
                f"spanning $\\pm{ARC_TOTAL/2:g}^\\circ$, and the three red circles are the deterministic evaluation "
                f"targets, drawn with their frame axes. The target always sits {HEIGHT:g} m above the base."),
         read=("The envelope is an \\textbf{L}, not a filled fan: the curriculum first pushes the target out along the "
               "axis, and only then opens the angle at the far radius, so a close target is never also strongly "
               "off-axis. The dashed circle carries the reason this is a whole-body problem at all: the target is at "
               f"{R_LINE:g} m while the arm reaches about {ARM_REACH:g} m, so roughly {R_LINE-ARM_REACH:.1f} m has to "
               "be covered by driving the base."),
         say=("``Every whole-body model is trained and evaluated on this same envelope --- 2 m out, "
              "$\\pm$20\\textdegree, 25 cm up --- so the comparisons that follow are on identical targets.''"),
         ask=("\\emph{Why not sample the whole fan?} --- because close-and-wide targets would be behind the robot's "
              "shoulder, a qualitatively different (and much harder) problem than the one this stage is about. "
              "\\emph{Why do the JSON spawn distances differ per model?} --- they are measured from each robot's own "
              "\\texttt{base\\_link}, which sits at a different offset from the spawn centre; the physical geometry "
              "is the same, and that was verified by fitting a circle to the logged target positions.")),
    dict(fig="frames_confondu.pdf", title="What \\emph{confondu} means: the two frames and the two errors", w=0.99, h=0.46,
         shows=("The gripper frame and the target frame as RGB triads (X red, Y green, Z blue --- the same convention "
                "as the in-scene axis renderer), how each one is constructed from the serialized offsets, and the two "
                "error measures the success gate is made of."),
         read=("The left panel is a normal mid-episode state: the origins are $d$ apart and the orientations differ by "
               "$\\varphi$. The middle panel is the goal: the triads sit on top of each other. The gripper frame is "
               "\\emph{computed}, not a Transform in the scene --- it is finger link 1's rotation times the Euler "
               "offset $(-90,90,0)$, and the origin is the grasp centre pushed 2 cm forward."),
         say=("``The task is frame coincidence, not reaching: the gripper must arrive at the target's position "
              f"\\emph{{and}} in the target's orientation --- under {GATE_D*100:.0f}~cm and "
              f"{np.degrees(GATE_PHI):.2f}\\textdegree{{}} at the same time.''"),
         ask=("\\emph{Why the absolute value in $\\varphi$?} --- a quaternion and its negative are the same rotation; "
              "without $|\\cdot|$ the metric reports the long way round (e.g. 330\\textdegree{} instead of "
              "30\\textdegree). \\emph{Why is the gripper frame not just the finger's transform?} --- the finger link's "
              "authored axes do not match the target's convention; the Euler offset is what makes the two frames "
              "comparable, and getting it wrong makes the gate unreachable without any error message.")),
]

def esc(s):
    return str(s)

def notes_pdf():
    head = ("\\documentclass[11pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n"
            "\\usepackage{amsmath,amssymb}\n\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
            "\\newcommand{\\code}[1]{\\texttt{#1}}\n\\begin{document}\n"
            "\\begin{center}{\\Large\\bfseries Common figures --- Notes for the Defense}\\\\[2pt]\n"
            "The task-level figures: they describe the problem, not a model, so they come BEFORE any per-model "
            "result and are shared by every whole-body package.\\end{center}\n\\vspace{4pt}\n")
    body = ""
    for i, p in enumerate(NOTES, 1):
        body += (f"\\section*{{Figure {i} --- {p['title']}}}\n"
                 f"\\begin{{center}}\\includegraphics[width={p['w']}\\textwidth,height={p['h']}\\textheight,"
                 f"keepaspectratio]{{{p['fig']}}}\\end{{center}}\n"
                 f"\\textbf{{What it shows.}} {p['shows']}\n\n\\textbf{{How to read it.}} {p['read']}\n\n"
                 f"\\textbf{{Say this.}} {p['say']}\n\n\\textbf{{If asked.}} {p['ask']}\n\\clearpage\n")
    open(os.path.join(OUT, "common_notes.tex"), "w").write(head + body + "\\end{document}\n")
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "common_notes.tex"],
                           cwd=OUT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"pdflatex pass {i} failed in '{OUT}' -- run it manually to see the error")
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(OUT, "common_notes" + ext))
        except FileNotFoundError: pass
    print(f"[common] common_notes.pdf")

def main():
    os.makedirs(OUT, exist_ok=True)
    fig_spawn()
    fig_frames()
    notes_pdf()

if __name__ == "__main__":
    main()

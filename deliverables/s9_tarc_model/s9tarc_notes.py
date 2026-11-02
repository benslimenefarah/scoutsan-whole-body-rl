#!/usr/bin/env python

import os, json, csv, subprocess, sys
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
FIG   = os.path.join(HERE, "figures")
JSONP = os.path.join(HERE, "s9tarc_eval.json")
TRAJ  = os.path.join(FIG, "trajectory_metrics.csv")

def esc(s):
    return str(s).replace("\\", "").replace("_", "\\_").replace("%", "\\%")

def traj_summary():
    if not os.path.exists(TRAJ):
        return None
    rows = list(csv.DictReader(open(TRAJ)))
    if not rows: return None
    st = [float(r["straightness"]) for r in rows if r["straightness"]]
    lat = [float(r["max_abs_lat_m"]) for r in rows]
    return dict(n=len(rows), straight=np.mean(st), lat=max(lat))

def pages(m):
    s = m["summary"]; g = m["gate"]; gp = np.degrees(g["phi"])
    t = traj_summary()
    traj_read = ("The path leaves the spawn, swings out laterally for the two arc targets, and comes back onto the "
                 "target's approach axis; the zoom row shows the final centimetres entering the 2~cm gate."
                 if not t else
                 f"Mean path length is {t['straight']:.2f}$\\times$ the straight-line distance, with up to "
                 f"{t['lat']*100:.0f}~cm of lateral excursion on the arc targets; the zoom row shows the last 15~cm "
                 "entering the 2~cm gate. The grey dashed line is where the BASE went --- the base covers the long "
                 "haul, the arm does the final alignment.")
    return [
        dict(fig="s9tarc_reach.pdf", title="Reach: distance to target",
             shows=(f"Gripper$\\to$target distance $d$ vs decision step, one curve per pass, coloured by WHICH of the "
                    f"three curriculum targets it is (arc $-20^\\circ$ / on-axis / arc $+20^\\circ$); red dashed = the "
                    f"{g['d']}~m position gate."),
             read=(f"Every pass converges under the gate (reached rate {int(s['reached_rate']*100)}\\%). The curve "
                   "stops at the gate because the agent ends the episode there."),
             say=(f"``All {s['n_targets']} passes reach the {g['d']}~m gate, mean minimum distance "
                  f"{s['mean_min_d']*1000:.1f}~mm.''"),
             ask=("\\emph{Why do the curves end at different lengths?} --- each curve ends when its target is reached; "
                  "the on-axis target is far quicker than the $\\pm20^\\circ$ arc ones.")),
        dict(fig="s9tarc_orient.pdf", title="Confondu: orientation error",
             shows=(f"Orientation error $\\varphi$ (deg) vs step; gate = {gp:.2f}$^\\circ$ ({g['phi']}~rad). "
                    "$\\varphi = 2\\arccos|\\langle q_G,q_T\\rangle|$, the geodesic angle between the gripper and "
                    "target frames."),
             read=("$\\varphi$ starts near $90^\\circ$ (the gripper faces its home pose, the target faces away from the "
                   "robot) and falls under the gate together with the distance --- that simultaneity is what "
                   "\\emph{confondu} means."),
             say=(f"``Orientation converges to a mean minimum of {s['mean_min_phi_deg']:.2f}$^\\circ$, under the "
                  f"{gp:.2f}$^\\circ$ gate.''"),
             ask=("\\emph{Why is position alone not enough?} --- the task is frame coincidence: the gripper must arrive "
                  "with the right orientation, not just at the right point.")),
        dict(fig="s9tarc_basedrive.pdf", title="Base drives into arm reach",
             shows=("base\\_link$\\to$target distance vs step --- the mobile base's contribution, separated from the "
                    "arm's. The dotted line marks the $\\approx$0.9~m arm reach."),
             read=("The base closes from about 2~m to inside arm reach and then holds while the arm finishes; this is "
                   "the whole-body split the single policy learned on its own (no hand-coded hand-off)."),
             say=("``One policy drives the base AND the arm: the base brings the target inside the arm's reach, then "
                  "the arm performs the alignment.''"),
             ask=("\\emph{Is the split hand-coded?} --- no: the 8 actions are 6 joint deltas plus base $v,\\omega$; "
                  "the division of labour is learned.")),
        dict(fig="s9tarc_phase.pdf", title="Joint convergence into the gate corner",
             shows=("$\\varphi$ plotted against $d$ (a phase portrait); the open circle is the spawn, the red lines are "
                    "the two gate thresholds. The bottom-left corner is \\emph{confondu}."),
             read=("The trajectories sweep down-left into the corner instead of hugging one axis --- position and "
                   "orientation are reduced together, not one after the other."),
             say=("``Both errors enter the gate corner together --- the policy does not trade position for "
                  "orientation.''"),
             ask=("\\emph{What would a bad policy look like?} --- a curve that runs down the $d$ axis first and only "
                  "then drops in $\\varphi$ (reach, then fix the wrist), which wastes time and often stalls.")),
        dict(fig="s9tarc_speed.pdf", title="Speed per target",
             shows=("Decisions to first enter the gate, per pass, annotated in seconds (1 decision = 0.1~s)."),
             read=(f"Mean {s['mean_steps_to_gate']:.0f} decisions = {s['mean_secs_to_gate']:.1f}~s. The on-axis target "
                   "is reached in a fraction of the time the arc targets need."),
             say=(f"``Mean {s['mean_secs_to_gate']:.1f}~s to the gate over the sweep.''"),
             ask=("\\emph{Comparable to the WDLS regimes?} --- yes: same three targets, same gate, same 0.1~s decision "
                  "period. See \\code{comparison\\_speed.pdf} at the top of \\code{deliverables/}.")),
        dict(fig="trajectories.pdf", title="Trajectories in the target frame", w=0.99, h=0.66,
             shows=("The gripper's path expressed in the TARGET's own frame: $s$ = approach coordinate along the "
                    "target's outward axis (the robot starts at $s<0$, the target is $s=0$), lateral = sideways offset, "
                    "height = above the target. Colour = time within the episode. One column per curriculum target."),
             read=traj_read,
             say=("``Drawn in the target frame, the path is directly comparable with the other controllers --- same "
                  "targets, same axes, so the shapes can be read side by side.''"),
             ask=("\\emph{Why the target frame?} --- the gripper$\\to$target relative pose is frame-independent, so it "
                  "does not matter that the end-to-end model observes in base\\_link FLU while the WDLS logs are in "
                  "Unity world coordinates. It is the only frame in which the two are comparable.")),
        dict(fig="s9tarc_overview.pdf", title="Overview (one-slide summary)",
             shows="Reach, orientation, base drive, phase portrait and per-target speed on one figure, plus the numbers.",
             read="Use as the summary slide; the panels are detailed on the preceding pages.",
             say=(f"``s9\\_tarc: {int(s['reached_rate']*100)}\\% of the sweep reached, mean min $d$ "
                  f"{s['mean_min_d']*1000:.1f}~mm, mean min $\\varphi$ {s['mean_min_phi_deg']:.2f}$^\\circ$, "
                  f"{s['mean_secs_to_gate']:.1f}~s to the gate.''"),
             ask=("\\emph{What does T-arc add over the fixed-target s9?} --- the same agent and reward, trained on a "
                  "curriculum that grows the target from close and on-axis out to 2~m and $\\pm20^\\circ$; this eval is "
                  "the whole sweep, not a single pose.")),
    ]

def build_tex(m):
    s = m["summary"]; g = m["gate"]; gp = np.degrees(g["phi"])
    head = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n"
        "\\usepackage{amsmath,amssymb}\n\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
        "\\newcommand{\\code}[1]{\\texttt{#1}}\n\\graphicspath{{figures/}}\n\\begin{document}\n"
        f"\\begin{{center}}{{\\Large\\bfseries s9\\_tarc --- Figure Notes for the Defense}}\\\\[2pt]\n"
        f"{esc(m.get('lambda_desc',''))}. Gate {g['d']}~m / {gp:.2f}$^\\circ$; "
        f"{int(s['reached_rate']*100)}\\% reached; mean min $d$ {s['mean_min_d']*1000:.1f}~mm, "
        f"mean min $\\varphi$ {s['mean_min_phi_deg']:.2f}$^\\circ$, mean {s['mean_steps_to_gate']:.0f} decisions "
        f"({s['mean_secs_to_gate']:.1f}~s) to the gate. "
        f"Checkpoint {esc(m.get('checkpoint',''))}, {esc(m.get('date',''))}.\\end{{center}}\n\\vspace{{4pt}}\n")
    body = ""
    for i, p in enumerate(pages(m), 1):
        body += (f"\\section*{{Figure {i} --- {p['title']}}}\n"
                 f"\\begin{{center}}\\includegraphics[width={p.get('w', 0.80)}\\textwidth,"
                 f"height={p.get('h', 0.55)}\\textheight,keepaspectratio]{{{p['fig']}}}\\end{{center}}\n"
                 f"\\textbf{{What it shows.}} {p['shows']}\n\n\\textbf{{How to read it.}} {p['read']}\n\n"
                 f"\\textbf{{Say this.}} {p['say']}\n\n\\textbf{{If asked.}} {p['ask']}\n\\clearpage\n")
    return head + body + "\\end{document}\n"

def main():
    if not os.path.exists(JSONP):
        sys.exit("missing s9tarc_eval.json -- run eval_s9tarc.py first")
    m = json.load(open(JSONP))
    missing = [p["fig"] for p in pages(m) if not os.path.exists(os.path.join(FIG, p["fig"]))]
    if missing:
        sys.exit(f"missing figures {missing} -- run s9tarc_plots.py and make_trajectory_graphs.py first")
    open(os.path.join(FIG, "figure_notes.tex"), "w").write(build_tex(m))
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure_notes.tex"],
                           cwd=FIG, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"pdflatex pass {i} failed in '{FIG}' -- run it manually to see the error")
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(FIG, "figure_notes" + ext))
        except FileNotFoundError: pass
    print(f"wrote {os.path.join(FIG, 'figure_notes.pdf')}")

if __name__ == "__main__":
    main()

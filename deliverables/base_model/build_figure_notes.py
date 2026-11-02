#!/usr/bin/env python

import os, json, subprocess, sys
import numpy as np
import pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
FIG   = os.path.join(HERE, "figures")
JSONP = os.path.join(HERE, "base_eval.json")
CSV   = os.path.join(HERE, "base_eval_steps.csv")
GATE  = 0.6

def stats():
    s = json.load(open(JSONP))["summary"]
    df = pd.read_csv(CSV)
    spawn = df.groupby("episode")["d_m"].first()
    dtr = pd.Series({e["episode"]: e["decisions_to_reach"]
                     for e in json.load(open(JSONP))["episodes"] if e.get("decisions_to_reach")})
    common = spawn.index.intersection(dtr.index)
    x, y = spawn[common].to_numpy(float), dtr[common].to_numpy(float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else float("nan")
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) >= 2 else float("nan")
    return {
        "N": s["episodes"], "succ": s["successes"], "rate": round(s["success_rate"] * 100),
        "mean_dec": s["mean_decisions_to_reach"], "median_dec": s["median_decisions_to_reach"],
        "mean_min_d": s["mean_min_d"], "slope": round(slope, 1), "r": round(r, 2),
        "dmin": round(float(spawn.min()), 1), "dmax": round(float(spawn.max()), 1),
    }

def pages(k):
    N, succ, rate = k["N"], k["succ"], k["rate"]
    return [
        dict(fig="base_eval_convergence.pdf", title="Convergence to the target",
             shows=(f"For each of the {N} evaluation episodes, the distance from the base to the target versus "
                    "decision step. Colour encodes how far the target spawned (dark = far); the red dashed line "
                    f"is the {GATE}~m success gate."),
             read=(f"Every curve crosses below the gate: the base reaches the target in all {N} episodes "
                   f"({succ}/{N}). Each curve is roughly flat at first (the base turns in place, so distance "
                   "barely changes) then descends almost linearly (it drives at near-constant speed)."),
             say=(f"``The policy succeeds on {rate}\\% of episodes. The flat-then-linear shape is a clean "
                  "turn-then-drive strategy --- no curved, wasteful paths.''"),
             ask=("\\emph{Why the initial plateau?} --- The base first reduces its heading error (see the "
                  "alignment figure) before translating, so the distance stays roughly constant during the turn.")),
        dict(fig="base_eval_approach.pdf", title="Target approach in the robot frame",
             shows=("The target's position expressed in the base's own frame over each episode. The base sits at "
                    f"the origin (triangle) facing $+$forward; the red circle is the {GATE}~m gate; dots mark the "
                    "target at spawn."),
             read=("Targets start in every direction --- ahead (positive forward offset), behind (negative), left "
                   "or right --- and every trace funnels into the gate circle. The wide arcs are episodes where "
                   "the target was behind: as the base rotates to face it, the target sweeps across the frame."),
             say=("``In the robot's own reference frame, targets from any direction --- including directly behind "
                  "--- are brought into the success gate. The FLU policy covers the full 360$^\\circ$.''"),
             ask=("\\emph{Why do paths curve instead of going straight?} --- This plot is in the \\emph{moving} "
                  "robot frame; the target appears to arc because the robot is rotating. In the world frame the "
                  "base path is turn-then-straight.")),
        dict(fig="base_eval_scaling.pdf", title="Effort scales with spawn distance",
             shows=(f"Decisions needed to reach the gate versus spawn distance, one point per episode "
                    f"(spawns {k['dmin']}--{k['dmax']}~m), with a least-squares line. "
                    f"Pearson $r={k['r']}$, slope $\\approx {k['slope']}$ steps/m."),
             read=("Points trend upward --- farther targets take more decisions --- and the positive slope "
                   f"quantifies it: about {k['slope']} decision steps per additional metre."),
             say=("``Time-to-goal grows roughly linearly with distance, exactly what a constant-speed driver "
                  "should do --- the policy adds no overhead for far targets.''"),
             ask=(f"\\emph{{Is {N} episodes enough to claim linearity?}} --- It is a qualitative trend on {N} "
                  "deterministic episodes; I report it as indicative and would expand to more episodes for a "
                  "tight regression estimate.")),
        dict(fig="base_eval_bearing.pdf", title="Alignment: heading error over time",
             shows=("The absolute heading error $|\\beta|$ (angle between the base's facing and the target "
                    "direction) versus decision step, per episode."),
             read=("All curves fall to near $0^\\circ$ and stay there: the base aligns with the target and holds "
                   "that alignment while driving in. Curves that first climb toward $180^\\circ$ are targets that "
                   "spawned behind --- the base swings around, briefly pointing away, before settling."),
             say=("``Heading error converges to near zero early and stays there --- the policy locks onto the "
                  "target heading, then translates. That is why the distance curves are so straight.''"),
             ask=("\\emph{What is $\\beta$ exactly?} --- The angle between the base's forward (or rear, whichever "
                  "is nearer) axis and the direction to the target; $0^\\circ$ means facing the target.")),
        dict(fig="base_eval_overview.pdf", title="Overview (one-slide summary)",
             shows="The four diagnostics on a single figure --- convergence, robot-frame approach, scaling, alignment.",
             read="Use this as the summary slide; the four panels are described on the preceding pages.",
             say=(f"``The base driver reaches the target in {rate}\\% of episodes (top-left), from any relative "
                  "direction (top-right), in time that scales linearly with distance (bottom-left), by aligning "
                  "first and driving second (bottom-right).''"),
             ask=("\\emph{One figure to remember?} --- This one: it captures success, generality, efficiency, "
                  "and the turn-then-drive mechanism together.")),
    ]

def build_tex(k):
    head = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n\\usepackage{amsmath,amssymb}\n"
        "\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
        "\\graphicspath{{./}}\n\\begin{document}\n"
        "\\begin{center}{\\Large\\bfseries Base Driver --- Figure Notes for the Defense}\\\\[2pt]\n"
        f"{k['N']} deterministic episodes, gate {GATE}~m, {k['succ']}/{k['N']} success "
        f"($={k['rate']}\\%$); mean {k['mean_dec']} decisions to reach (median {k['median_dec']}), "
        f"mean min distance {k['mean_min_d']}~m. Date 2026-09-24.\\end{{center}}\n\\vspace{{4pt}}\n"
    )
    body = ""
    for i, p in enumerate(pages(k), 1):
        body += (
            f"\\section*{{Figure {i} --- {p['title']}}}\n"
            f"\\begin{{center}}\\includegraphics[width=0.80\\textwidth]{{{p['fig']}}}\\end{{center}}\n"
            f"\\textbf{{What it shows.}} {p['shows']}\n\n"
            f"\\textbf{{How to read it.}} {p['read']}\n\n"
            f"\\textbf{{Say this.}} {p['say']}\n\n"
            f"\\textbf{{If asked.}} {p['ask']}\n"
            "\\clearpage\n"
        )
    return head + body + "\\end{document}\n"

def main():
    if not (os.path.exists(JSONP) and os.path.exists(CSV)):
        sys.exit("missing base_eval.json/csv -- run eval_base_flu.py + eval_base_plots.py first")
    missing = [p["fig"] for p in pages(stats()) if not os.path.exists(os.path.join(FIG, p["fig"]))]
    if missing:
        sys.exit(f"missing figures {missing} -- run eval_base_plots.py first")
    tex = build_tex(stats())
    texpath = os.path.join(FIG, "figure_notes.tex")
    open(texpath, "w").write(tex)
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure_notes.tex"],
                           cwd=FIG, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"pdflatex pass {i} failed -- run it manually in {FIG}")
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(FIG, "figure_notes" + ext))
        except FileNotFoundError: pass
    print(f"wrote {os.path.join(FIG, 'figure_notes.pdf')}")

if __name__ == "__main__":
    main()

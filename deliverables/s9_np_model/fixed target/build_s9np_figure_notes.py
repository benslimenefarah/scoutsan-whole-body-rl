#!/usr/bin/env python

import os, json, subprocess, sys

HERE  = os.path.dirname(os.path.abspath(__file__))
FIG   = os.path.join(HERE, "figures")
JSONP = os.path.join(HERE, "s9np_eval.json")

def stats():
    s = json.load(open(JSONP))["summary"]
    import math
    return {
        "N": s["episodes"], "succ": s["successes"], "rate": round(s["success_rate"] * 100),
        "gd": s["gate_pos_m"], "gp_deg": round(math.degrees(s["gate_orient_rad"]), 1),
        "mind": s["mean_min_d"], "minp_deg": round(math.degrees(s["mean_min_phi_rad"]), 1),
    }

def pages(k):
    N, rate = k["N"], k["rate"]
    return [
        dict(fig="s9np_reach.pdf", title="Reach: distance to target",
             shows=(f"Gripper$\\to$target distance $d$ vs decision step, one line per episode ({N} episodes); the red "
                    f"dashed line is the {k['gd']}~m success gate."),
             read=(f"All {N} curves fall from $\\sim$2~m to inside the gate in $\\sim$26 decisions and overlap almost "
                   "exactly. The target is FIXED and the policy deterministic, so the overlap is a reproducibility check."),
             say=(f"``The gripper closes from 2~m to inside the {k['gd']*100:.0f}~cm gate in about 26 decisions, "
                  "identically every episode --- deterministic and repeatable.''"),
             ask=("\\emph{Why do the lines overlap?} --- Fixed target + \\texttt{deterministic=True} give the same "
                  "trajectory each episode; the overlap confirms the run is reproducible.")),
        dict(fig="s9np_orient.pdf", title="Confondu: orientation error",
             shows=(f"Orientation error $\\varphi$ (geodesic angle between the gripper frame and the target frame), in "
                    f"degrees, vs step; gate = {k['gp_deg']}$^\\circ$ ({0.05}~rad)."),
             read=("$\\varphi$ starts near $90^\\circ$ (arm home far from the target frame), drops steeply below the "
                   "gate by $\\sim$step 13, then holds with small residual jitter."),
             say=(f"``Orientation reaches confondu ($<{k['gp_deg']}^\\circ$) by $\\sim$step 13 and holds --- the wrist "
                  "aligns well before the reach finishes.''"),
             ask=("\\emph{What is $\\varphi$?} --- the single geodesic angle between the gripper and target frames "
                  "($0$ = coincident = confondu), from the quaternion dot product $2\\arccos|\\langle q_G,q_T\\rangle|$.")),
        dict(fig="s9np_basedrive.pdf", title="Base drives into arm reach",
             shows=("base\\_link$\\to$target distance (baseD) vs step; the dotted line marks the arm reach $\\approx$0.9~m."),
             read=("baseD falls from $\\sim$1.7~m to $\\sim$1.05~m --- the mobile base drives the target into the arm's "
                   "working range, then the arm does the fine reach (baseD need not reach 0)."),
             say=("``The base brings the target from 1.7~m into the arm's range, then hands off to the arm --- the "
                  "whole-body division of labour.''"),
             ask=("\\emph{Why doesn't baseD reach 0?} --- it shouldn't; the base only has to bring the target within "
                  "arm reach, the gripper (not the base) touches the target.")),
        dict(fig="s9np_phase.pdf", title="Joint convergence into the gate corner",
             shows=(f"The trajectory in $(d,\\varphi)$ space; the dashed lines mark the gate corner "
                    f"($d<{k['gd']}$~m, $\\varphi<{k['gp_deg']}^\\circ$)."),
             read=("Each run sweeps from the top-right (far, mis-oriented) into the bottom-left gate corner: $\\varphi$ "
                   "collapses early (by $d\\approx0.8$~m), then $d$ closes the rest at low $\\varphi$."),
             say=("``Both errors converge into the gate corner; orientation is solved first, then position closes --- "
                  "a clean, non-competing convergence.''"),
             ask=("\\emph{Does solving orientation first cost reach time?} --- no; $\\varphi$ is already under gate while "
                  "$d\\approx0.7$~m, so the wrist just holds alignment while the base finishes driving in.")),
        dict(fig="s9np_overview.pdf", title="Overview (one-slide summary)",
             shows="The four diagnostics on one figure --- reach, orientation, base drive, and the $(d,\\varphi)$ phase plot.",
             read="Use as the summary slide; the panels are detailed on the preceding pages.",
             say=(f"``s9\\_np reaches the {k['gd']*100:.0f}~cm gate in $\\sim$26 steps, aligns to $<{k['gp_deg']}^\\circ$, "
                  f"drives the base into arm range, and converges both errors into the gate corner --- {k['succ']}/{N} "
                  "episodes, identically.''"),
             ask=("\\emph{One figure to remember?} --- this one: reach, confondu, base drive and joint convergence together.")),
    ]

def build_tex(k):
    head = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{graphicx}\n"
        "\\usepackage{amsmath,amssymb}\n\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
        "\\graphicspath{{figures/}}\n\\begin{document}\n"
        "\\begin{center}{\\Large\\bfseries s9\\_np (fixed target) --- Figure Notes for the Defense}\\\\[2pt]\n"
        f"{k['N']} deterministic episodes, gate {k['gd']}~m / {k['gp_deg']}$^\\circ$, {k['succ']}/{k['N']} success "
        f"($={k['rate']}\\%$); mean min $d$ = {k['mind']}~m, mean min $\\varphi$ = {k['minp_deg']}$^\\circ$. Date 2026-09-24.\\end{{center}}\n\\vspace{{4pt}}\n"
    )
    body = ""
    for i, p in enumerate(pages(k), 1):
        body += (
            f"\\section*{{Figure {i} --- {p['title']}}}\n"
            f"\\begin{{center}}\\includegraphics[width=0.80\\textwidth]{{{p['fig']}}}\\end{{center}}\n"
            f"\\textbf{{What it shows.}} {p['shows']}\n\n"
            f"\\textbf{{How to read it.}} {p['read']}\n\n"
            f"\\textbf{{Say this.}} {p['say']}\n\n"
            f"\\textbf{{If asked.}} {p['ask']}\n\\clearpage\n"
        )
    return head + body + "\\end{document}\n"

def main():
    if not os.path.exists(JSONP):
        sys.exit("missing s9np_eval.json -- run eval_s9np.py first")
    k = stats()
    missing = [p["fig"] for p in pages(k) if not os.path.exists(os.path.join(FIG, p["fig"]))]
    if missing:
        sys.exit(f"missing figures {missing} -- run eval_s9np_plots.py first")
    open(os.path.join(FIG, "figure_notes.tex"), "w").write(build_tex(k))
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "figure_notes.tex"],
                           cwd=FIG, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"pdflatex pass {i} failed -- run it manually in '{FIG}'")
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(FIG, "figure_notes" + ext))
        except FileNotFoundError: pass
    print(f"wrote {os.path.join(FIG, 'figure_notes.pdf')}")

if __name__ == "__main__":
    main()

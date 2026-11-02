#!/usr/bin/env python

import os, json, subprocess, sys
import numpy as np

HERE  = os.path.dirname(os.path.abspath(__file__))
JSONP = os.path.join(HERE, "s9tarc_eval.json")

def esc(s):
    return str(s).replace("\\", "").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")

def build_tex(m):
    s = m["summary"]; g = m["gate"]; gp = np.degrees(g["phi"]); ts = m.get("target_set", {})
    rows = "\n".join(
        f"{{[{t['idx']}]}} & {t.get('target_key','?')} & {t['spawn_dist']:.2f} & {t['spawn_az']:+.0f} & "
        f"{t['steps_to_gate'] if t['steps_to_gate'] else '--'} & "
        f"{t['secs_to_gate'] if t.get('secs_to_gate') else '--'} & "
        f"{t['min_d']*1000:.1f} & {t['min_phi_deg']:.2f}\\\\" for t in m["targets"])
    return (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=2.2cm]{geometry}\n\\usepackage{amsmath,amssymb}\n"
        "\\usepackage{array}\n\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{5pt}\n"
        "\\newcommand{\\code}[1]{\\texttt{#1}}\n"
        "\\title{\\vspace{-1.5em}s9\\_tarc --- Model Card\\\\[2pt]\\large End-to-end DRL whole-body control, "
        "\\emph{confondu} on the T-arc target curriculum}\n"
        f"\\author{{RL\\_Scoutsan project}}\\date{{{esc(m.get('date',''))}}}\n"
        "\\begin{document}\\maketitle\\vspace{-1em}\n"

        "\\section{Task and robot}\n"
        "A Scout~V2 mobile base carrying a 6-DOF Doosan arm must bring its gripper frame \\emph{confondu} "
        "(coincident in position AND orientation) with a target frame. Unlike the Path~B models, this policy is "
        "\\textbf{end-to-end}: it outputs joint and base velocities directly --- there is no analytic inverse-kinematics "
        "controller in the loop. It is warm-started from \\code{s9\\_np} (the same agent trained on one fixed target) "
        "and retrained on the T-arc target curriculum, so that it faces the same target distribution as the WDLS "
        "models and can be compared with them.\n"

        "\\section{Observations (14) and actions (8)}\n"
        "Observations, in the \\code{base\\_link} FLU frame: [0:3] gripper position, [3:7] gripper quaternion, "
        "[7:10] target position, [10:14] target quaternion (quaternions canonicalised $w>0$). "
        "Actions: 6 joint increments ($\\pm$3\\textdegree{} per decision, shrunk near the goal by the adaptive action "
        "scale) plus the base linear and angular velocity commands. One decision every 5 physics steps = 0.1~s.\n"

        "\\section{Target curriculum (T-arc)}\n"
        "The target is placed on an on-axis LINE that grows from \\code{minRadius} 0.75~m to \\code{lineRadius} 2.0~m "
        "over the first 500\\,000 Academy steps, then on an ARC at 2.0~m opening to $\\pm$20\\textdegree{} "
        "(\\code{arcTotalDeg} 40) over the next 1\\,000\\,000 steps; height 0.25~m above the base, target frame "
        "$+Z$ pointing away from the robot (\\code{zAwayFromRobot}). "
        f"The evaluation cycles the deterministic \\code{{SampleLineArc}} set: {ts.get('distinct','3')} distinct targets "
        f"(on-axis, arc $-$20\\textdegree, arc $+$20\\textdegree) at {ts.get('radius_m',2.0)}~m --- the same set the "
        "wb\\_1am regimes were evaluated on.\n"

        "\\section{Physics, gate, checkpoint}\n\\begin{center}\\begin{tabular}{ll}\n\\hline item & value\\\\\\hline\n"
        "joint step / velocity cap & 3\\textdegree{} per decision; \\code{maxJointVelocityDeg} 30\\\\\n"
        "base caps & \\code{maxLinearVel} 0.3 m/s, \\code{maxAngularVel} 0.4 rad/s\\\\\n"
        "wheel cap & \\code{maxWheelVelocityDeg} 150\\\\\n"
        f"gate (scene Inspector) & $d<{g['d']}$ m AND $\\varphi<{g['phi']}$ rad ({gp:.2f}\\textdegree), no hold\\\\\n"
        "episode end & the gate ends the episode immediately (no settle-hold is measurable)\\\\\n"
        f"checkpoint & \\code{{{esc(m.get('checkpoint',''))}}}\\\\\n"
        "training & 4.0M steps, 8 robots, warm start from \\code{s9\\_np}, port 5025\\\\\n"
        "obs / act & 14 / 8 (continuous)\\\\\n\\hline\\end{tabular}\\end{center}\n"

        f"\\section{{Evaluation --- line+arc sweep ({s['n_targets']} passes, deterministic policy)}}\n"
        "\\begin{center}\\begin{tabular}{cccccccc}\n\\hline\n"
        "pass & target & dist (m) & az (\\textdegree) & steps$\\to$gate & s$\\to$gate & min $d$ (mm) & "
        "min $\\varphi$ (\\textdegree)\\\\\n\\hline\n"
        f"{rows}\n\\hline\\end{{tabular}}\\end{{center}}\n"
        "The distance and azimuth columns are measured from \\code{base\\_link}, which sits about 0.3~m ahead of the "
        "spawn centre, so they read below the curriculum's 2.0~m / $\\pm$20\\textdegree{} nominal values; the true "
        "geometry is identical for every model evaluated here.\n\n"
        f"Summary: reached {int(s['reached_rate']*100)}\\%, mean min $d$ {s['mean_min_d']*1000:.1f}~mm, mean min "
        f"$\\varphi$ {s['mean_min_phi_deg']:.2f}\\textdegree, mean {s['mean_steps_to_gate']:.0f} decisions "
        f"({s['mean_secs_to_gate']:.1f}~s) to the gate. Figures and speaker notes in \\code{{figures/}}; per-step trace "
        "(including the full gripper and target pose) in \\code{s9tarc\\_eval\\_steps.csv}.\n"

        "\\section*{Reproducibility}\n\\small\\begin{tabular}{ll}\n"
        f"Model dir & \\code{{scoutsan\\_Baseline/models/model\\_scoutsan\\_s9\\_tarc}}\\\\\n"
        f"Model prefix & \\code{{{esc(m.get('model_prefix',''))}}}\\\\\n"
        "Trainer & \\code{scoutsan\\_Baseline/training/train\\_scoutsan\\_s9\\_tarc.py}\\\\\n"
        "Scene & \\code{scoutsan\\_Baseline/scenes/s9.unity}\\\\\n"
        "Scene switches & \\code{useTArcCurriculum} 1, \\code{tArcLineArc} 1 (count 4 / lineN 1)\\\\\n"
        "Eval & \\code{eval\\_s9tarc.py} (Editor, 5004) $\\to$ \\code{s9tarc\\_plots.py} $\\to$ "
        "\\code{s9tarc\\_notes.py}\\\\\n"
        "Video & 3 cams (FollowCam/upcam/sidecam) + FrameAxesRenderer triads\\\\\n"
        "Env & \\code{unity-rl-gpu} (SB3 2.3.2, mlagents\\_envs 0.30.0)\\\\\n"
        f"Date & {esc(m.get('date',''))}\\\\\n\\end{{tabular}}\\normalsize\n\\end{{document}}\n")

def main():
    if not os.path.exists(JSONP):
        sys.exit("missing s9tarc_eval.json -- run eval_s9tarc.py first")
    m = json.load(open(JSONP))
    open(os.path.join(HERE, "s9tarc_model_card.tex"), "w").write(build_tex(m))
    for i in (1, 2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "s9tarc_model_card.tex"],
                           cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode != 0:
            sys.exit(f"pdflatex pass {i} failed in '{HERE}' -- run it manually to see the error")
    for ext in (".aux", ".log"):
        try: os.remove(os.path.join(HERE, "s9tarc_model_card" + ext))
        except FileNotFoundError: pass
    print(f"wrote {os.path.join(HERE, 's9tarc_model_card.pdf')}")

if __name__ == "__main__":
    main()
